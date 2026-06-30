"""Pydantic models for the embedding engine."""

from __future__ import annotations

from pydantic import BaseModel, Field
from oracle_flexcube_copilot.chunking.models import Chunk


class EmbeddedChunk(BaseModel):
    """A chunk enriched with its vector embedding."""

    chunk: Chunk = Field(description="The source chunk")
    embedding: list[float] = Field(description="The vector representation of the chunk text")
    embedding_model: str = Field(description="Model used for embedding (e.g. nomic-embed-text)")
    embedding_dimension: int = Field(description="Dimensionality of the embedding vector")
    embedding_time: float = Field(description="Time taken to embed this chunk (latency in seconds)")
    embedding_version: str = Field(description="Embedding version (e.g. v1)")


class EmbeddingMetrics(BaseModel):
    """Metrics collected during an embedding batch process."""

    chunks_embedded: int = Field(default=0, description="Number of chunks successfully embedded")
    total_time_seconds: float = Field(default=0.0, description="Total duration of the embedding process")
    average_latency: float = Field(default=0.0, description="Average time per chunk embedding")
    vectors_per_second: float = Field(default=0.0, description="Throughput: vectors generated per second")
    failures: int = Field(default=0, description="Number of failed chunks")
    cache_hits: int = Field(default=0, description="Number of chunks served from cache")
    cache_misses: int = Field(default=0, description="Number of chunks that required generation")
