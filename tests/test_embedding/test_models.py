"""Tests for embedding models."""

from __future__ import annotations

from oracle_flexcube_copilot.chunking.models import Chunk
from oracle_flexcube_copilot.embedding.models import EmbeddedChunk, EmbeddingMetrics


class TestEmbeddingModels:
    """Tests for embedding data models."""

    def test_metrics_defaults(self) -> None:
        """Metrics should default to zero values."""
        metrics = EmbeddingMetrics()
        assert metrics.chunks_embedded == 0
        assert metrics.total_time_seconds == 0.0
        assert metrics.average_latency == 0.0
        assert metrics.vectors_per_second == 0.0
        assert metrics.failures == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0

    def test_embedded_chunk_creation(self) -> None:
        """EmbeddedChunk should store vector data alongside Chunk."""
        chunk = Chunk(id="c1", document_id="d1", text="text")
        embedded = EmbeddedChunk(
            chunk=chunk,
            embedding=[0.1, 0.2, 0.3],
            embedding_model="test-model",
            embedding_dimension=3,
            embedding_time=0.5,
            embedding_version="v1",
        )
        assert embedded.embedding == [0.1, 0.2, 0.3]
        assert embedded.embedding_dimension == 3
        assert embedded.chunk.id == "c1"
