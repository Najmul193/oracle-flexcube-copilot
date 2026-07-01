"""Tests for embedding cache."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from oracle_flexcube_copilot.embedding.cache import EmbeddingCache


@pytest.fixture
def temp_cache_dir() -> Path:
    """Provide a temporary directory for cache testing."""
    with TemporaryDirectory() as d:
        yield Path(d)


class TestEmbeddingCache:
    """Tests for file-based embedding cache."""

    def test_cache_miss(self, temp_cache_dir: Path) -> None:
        """Cache should return None for missing entries."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        assert cache.get("hello", "model_x") is None

    def test_cache_hit(self, temp_cache_dir: Path) -> None:
        """Cache should return vector after it is set, and store rich metadata."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        cache.set("hello", "model_x", [0.1, 0.2], chunk_id="test-chunk-1")
        vector = cache.get("hello", "model_x")
        assert vector == [0.1, 0.2]

        # Verify metadata
        path = cache._get_cache_path("hello", "model_x")
        data = json.loads(path.read_text("utf-8"))
        assert data["embedding_model"] == "model_x"
        assert data["embedding_dimension"] == 2
        assert data["chunk_id"] == "test-chunk-1"
        assert "created_at" in data
        assert "pipeline_version" in data

    def test_cache_model_isolation(self, temp_cache_dir: Path) -> None:
        """Cache should isolate entries by model name."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        cache.set("hello", "model_x", [0.1])
        assert cache.get("hello", "model_y") is None

    def test_cache_corrupted_file(self, temp_cache_dir: Path) -> None:
        """Cache should return None and not crash on corrupted JSON."""
        cache = EmbeddingCache(cache_dir=temp_cache_dir)
        cache.set("hello", "model_x", [0.1])
        # Corrupt it
        path = cache._get_cache_path("hello", "model_x")
        path.write_text("invalid json")

        assert cache.get("hello", "model_x") is None
