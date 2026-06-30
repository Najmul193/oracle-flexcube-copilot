"""File-based caching for generated embeddings."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from oracle_flexcube_copilot.config import settings

logger = logging.getLogger("oracle_flexcube_copilot.embedding.cache")


class EmbeddingCache:
    """Manages local file-based caching of embeddings."""

    def __init__(self, cache_dir: Path | None = None):
        """Initialize the cache directory.
        
        Args:
            cache_dir: Optional custom cache dir, defaults to settings.resolved_cache_dir / "embeddings".
        """
        self.cache_dir = cache_dir or settings.resolved_cache_dir / "embeddings"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, chunk_text: str, model_name: str) -> Path:
        """Derive a stable cache file path based on chunk text and model.
        
        Args:
            chunk_text: The text to be embedded.
            model_name: The embedding model name.
            
        Returns:
            The path to the cache file.
        """
        # Hash text + model to ensure different models don't collide
        hash_input = f"{model_name}:{chunk_text}".encode("utf-8")
        cache_key = hashlib.sha256(hash_input).hexdigest()
        return self.cache_dir / f"{cache_key}.json"

    def get(self, chunk_text: str, model_name: str) -> list[float] | None:
        """Retrieve an embedding from cache if it exists.
        
        Args:
            chunk_text: The text that was embedded.
            model_name: The model used.
            
        Returns:
            The embedding vector or None if not found.
        """
        path = self._get_cache_path(chunk_text, model_name)
        if path.exists():
            try:
                data = json.loads(path.read_text("utf-8"))
                # Handle legacy format which was {"embedding": [...]}
                if isinstance(data, dict):
                    return data.get("embedding")
                return None
            except Exception as e:
                logger.warning("Failed to read cache at %s: %s", path, e)
        return None

    def set(self, chunk_text: str, model_name: str, embedding: list[float], chunk_id: str = "") -> None:
        """Save an embedding to the cache with rich metadata.
        
        Args:
            chunk_text: The text that was embedded.
            model_name: The model used.
            embedding: The embedding vector to save.
            chunk_id: The ID of the chunk being embedded.
        """
        path = self._get_cache_path(chunk_text, model_name)
        try:
            data = {
                "embedding_model": model_name,
                "embedding_dimension": len(embedding),
                "chunk_id": chunk_id,
                "pipeline_version": settings.pipeline_version,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "embedding": embedding
            }
            path.write_text(json.dumps(data), "utf-8")
        except Exception as e:
            logger.warning("Failed to write cache at %s: %s", path, e)
