"""Tests for EmbeddingEngine."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from ollama import ResponseError

from oracle_flexcube_copilot.chunking.models import Chunk
from oracle_flexcube_copilot.embedding.cache import EmbeddingCache
from oracle_flexcube_copilot.embedding.engine import EmbeddingEngine
from oracle_flexcube_copilot.embedding.models import EmbeddedChunk


class MockEmbedResponse:
    def __init__(self, embeddings: list[list[float]]):
        self.embeddings = embeddings


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    # By default, return a successful response with mock embeddings
    def _embed(model, input, **kwargs):
        if isinstance(input, str):
            input = [input]
        # Return a simple [1.0, 0.0] for every text in batch
        return MockEmbedResponse(embeddings=[[1.0, 0.0] for _ in input])
    
    client.embed.side_effect = _embed
    return client


@pytest.fixture
def dummy_chunks() -> list[Chunk]:
    return [
        Chunk(id="1", document_id="doc1", text="Chunk one text"),
        Chunk(id="2", document_id="doc1", text="Chunk two text"),
        Chunk(id="3", document_id="doc1", text="Chunk three text"),
    ]


class TestEmbeddingEngine:
    """Tests for EmbeddingEngine functionality."""

    def test_successful_batch_embedding(self, mock_client: MagicMock, dummy_chunks: list[Chunk], tmp_path) -> None:
        """Engine should process chunks in batches successfully."""
        cache = EmbeddingCache(cache_dir=tmp_path)
        engine = EmbeddingEngine(client=mock_client, cache=cache, batch_size=2)
        
        embedded, metrics = engine.embed_chunks(dummy_chunks)
        
        # 3 chunks, batch size 2 -> 2 calls to embed
        assert mock_client.embed.call_count == 2
        assert len(embedded) == 3
        assert metrics.chunks_embedded == 3
        assert metrics.cache_misses == 3
        assert metrics.cache_hits == 0
        assert metrics.failures == 0
        
        # Verify chunk fields were mutated
        for chunk in dummy_chunks:
            assert chunk.embedding == [1.0, 0.0]
            assert chunk.embedding_status == "completed"

    def test_cache_hits(self, mock_client: MagicMock, dummy_chunks: list[Chunk], tmp_path) -> None:
        """Engine should use cache when available and skip API call."""
        cache = EmbeddingCache(cache_dir=tmp_path)
        # Pre-seed cache for chunk "1"
        cache.set(dummy_chunks[0].text, "nomic-embed-text", [0.5, 0.5])
        
        engine = EmbeddingEngine(client=mock_client, cache=cache, batch_size=2)
        embedded, metrics = engine.embed_chunks(dummy_chunks)
        
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 2
        assert metrics.chunks_embedded == 2
        
        # API called only for the remaining 2 chunks (fits in 1 batch)
        assert mock_client.embed.call_count == 1
        
        assert embedded[0].embedding == [0.5, 0.5]
        assert embedded[1].embedding == [1.0, 0.0]

    def test_retry_on_transient_failure(self, mock_client: MagicMock, dummy_chunks: list[Chunk], tmp_path) -> None:
        """Engine should retry upon exception and succeed eventually."""
        cache = EmbeddingCache(cache_dir=tmp_path)
        
        # Fail first time, succeed second time
        call_count = 0
        def _flaky_embed(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Timeout")
            return MockEmbedResponse(embeddings=[[2.0, 2.0] for _ in kwargs.get('input', [])])
            
        mock_client.embed.side_effect = _flaky_embed
        
        # Use fast backoff for test
        engine = EmbeddingEngine(client=mock_client, cache=cache, batch_size=3, base_backoff=0.01)
        embedded, metrics = engine.embed_chunks(dummy_chunks)
        
        assert mock_client.embed.call_count == 2
        assert metrics.chunks_embedded == 3
        assert metrics.failures == 0
        assert embedded[0].embedding == [2.0, 2.0]

    def test_failure_after_max_retries(self, mock_client: MagicMock, dummy_chunks: list[Chunk], tmp_path) -> None:
        """Engine should record failures if retries are exhausted."""
        cache = EmbeddingCache(cache_dir=tmp_path)
        
        # Always fail
        mock_client.embed.side_effect = ResponseError("Model not found")
        
        engine = EmbeddingEngine(client=mock_client, cache=cache, batch_size=3, max_retries=2, base_backoff=0.01)
        embedded, metrics = engine.embed_chunks(dummy_chunks)
        
        assert mock_client.embed.call_count == 2
        assert metrics.failures == 3
        assert metrics.chunks_embedded == 0
        assert len(embedded) == 0

    def test_metrics_calculation(self, mock_client: MagicMock, dummy_chunks: list[Chunk], tmp_path) -> None:
        """Engine should calculate throughput and latency correctly."""
        cache = EmbeddingCache(cache_dir=tmp_path)
        
        # Simulate slight delay
        def _delayed_embed(*args, **kwargs):
            time.sleep(0.05)
            return MockEmbedResponse(embeddings=[[1.0] for _ in kwargs.get('input', [])])
            
        mock_client.embed.side_effect = _delayed_embed
        
        engine = EmbeddingEngine(client=mock_client, cache=cache, batch_size=3)
        _, metrics = engine.embed_chunks(dummy_chunks)
        
        assert metrics.total_time_seconds >= 0.05
        assert metrics.average_latency > 0
        assert metrics.vectors_per_second > 0
