"""Embedding engine for transforming chunks into vectors via Ollama."""

from __future__ import annotations

import logging
import time
from typing import Any

import ollama
from ollama import Client, ResponseError

from oracle_flexcube_copilot.chunking.models import Chunk
from oracle_flexcube_copilot.config import settings
from oracle_flexcube_copilot.embedding.cache import EmbeddingCache
from oracle_flexcube_copilot.embedding.models import EmbeddedChunk, EmbeddingMetrics

logger = logging.getLogger("oracle_flexcube_copilot.embedding.engine")


class EmbeddingEngine:
    """Handles vector embedding generation with caching, batching, and retries."""

    def __init__(
        self,
        client: Client | None = None,
        cache: EmbeddingCache | None = None,
        batch_size: int = 64,
        max_retries: int = 3,
        base_backoff: float = 2.0,
    ):
        """Initialize the embedding engine.
        
        Args:
            client: Optional Ollama client (defaults to connected to settings url).
            cache: Optional cache manager.
            batch_size: How many chunks to process in one API call.
            max_retries: Max retries for failed API calls.
            base_backoff: Base sleep time for exponential backoff.
        """
        self.client = client or Client(host=settings.ollama_base_url)
        self.cache = cache or EmbeddingCache()
        self.model_name = settings.embedding_model
        self.version = settings.embedding_version
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.base_backoff = base_backoff

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """Call Ollama embed with exponential backoff on failure."""
        for attempt in range(self.max_retries):
            try:
                # ollama.embed expects input: str | Sequence[str]
                response = self.client.embed(model=self.model_name, input=texts)
                return response.embeddings
            except ResponseError as e:
                # Usually raised if model is missing or context window exceeded
                logger.error("Ollama ResponseError: %s", e)
                # If model is explicitly missing, we should probably fail immediately, 
                # but we'll try to let retry handle transient issues.
                if attempt == self.max_retries - 1:
                    raise
            except Exception as e:
                logger.warning("Ollama connection/timeout error: %s (attempt %d/%d)", e, attempt + 1, self.max_retries)
                if attempt == self.max_retries - 1:
                    raise
            
            # Exponential backoff
            time.sleep(self.base_backoff * (2 ** attempt))
            
        return []

    def embed_chunks(self, chunks: list[Chunk]) -> tuple[list[EmbeddedChunk], EmbeddingMetrics]:
        """Embed a list of chunks, utilizing cache and batching.
        
        Args:
            chunks: List of Chunks to embed.
            
        Returns:
            Tuple of (embedded chunks, metrics).
        """
        start_time = time.time()
        embedded_chunks: list[EmbeddedChunk] = []
        metrics = EmbeddingMetrics()
        
        # We will separate chunks into cached and missing
        # To preserve order and relationship, we process sequentially using batches for missing.
        
        missing_chunks: list[Chunk] = []
        
        for chunk in chunks:
            # 1. Check cache
            cached_vector = self.cache.get(chunk.text, self.model_name)
            if cached_vector:
                metrics.cache_hits += 1
                embedded_chunks.append(self._create_embedded_chunk(chunk, cached_vector, 0.0))
            else:
                missing_chunks.append(chunk)
                
        metrics.cache_misses = len(missing_chunks)
        
        # 2. Batch process missing chunks
        for i in range(0, len(missing_chunks), self.batch_size):
            batch = missing_chunks[i : i + self.batch_size]
            texts = [c.text for c in batch]
            
            batch_start = time.time()
            try:
                vectors = self._embed_batch_with_retry(texts)
                batch_latency = (time.time() - batch_start) / len(batch) if batch else 0.0
                
                for chunk, vector in zip(batch, vectors):
                    self.cache.set(chunk.text, self.model_name, vector, chunk_id=chunk.id)
                    embedded_chunks.append(self._create_embedded_chunk(chunk, vector, batch_latency))
                    metrics.chunks_embedded += 1
                    
            except Exception as e:
                logger.error("Failed to embed batch starting at chunk index %d: %s", i, e)
                metrics.failures += len(batch)

        # 3. Calculate final metrics
        metrics.total_time_seconds = time.time() - start_time
        total_success = metrics.cache_hits + metrics.chunks_embedded
        if total_success > 0:
            metrics.average_latency = metrics.total_time_seconds / total_success
            metrics.vectors_per_second = total_success / metrics.total_time_seconds if metrics.total_time_seconds > 0 else 0.0
            
        return embedded_chunks, metrics

    def _create_embedded_chunk(self, chunk: Chunk, vector: list[float], latency: float) -> EmbeddedChunk:
        """Helper to create EmbeddedChunk and update Chunk fields."""
        # Also mutate original chunk for convenience in pipelines that pass the chunk object
        chunk.embedding = vector
        chunk.embedding_model = self.model_name
        chunk.embedding_status = "completed"
        
        return EmbeddedChunk(
            chunk=chunk,
            embedding=vector,
            embedding_model=self.model_name,
            embedding_dimension=len(vector),
            embedding_time=latency,
            embedding_version=self.version,
        )

    def embed(self, text: str) -> list[float]:
        """Embed a single query string.
        
        Args:
            text: The text to embed.
            
        Returns:
            The embedding vector.
        """
        return self._embed_batch_with_retry([text])[0]
