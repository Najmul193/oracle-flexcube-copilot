"""Application configuration via pydantic-settings."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project paths
    project_root: Path = Path.cwd()
    data_dir: Path = Path("data")
    chroma_db_dir: Path = Path("chroma_db")
    cache_dir: Path = Path("cache")
    log_dir: Path = Path("logs")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    llm_model: str = "qwen3:8b"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 2048

    # Chunking
    chunk_size: int = 800
    chunk_overlap: int = 100

    # Retrieval
    top_k_retrieval: int = 5
    retrieval_alpha: float = 0.5  # 0 = pure BM25, 1 = pure dense

    # Logging
    log_level: str = "INFO"
    log_format: Literal["json", "text"] = "text"

    # ChromaDB collection name
    chroma_collection_name: str = "oracle_flexcube_docs"

    # Pipeline Versioning
    pipeline_version: str = "1.0.0"
    chunking_version: str = "1.0.0"
    embedding_version: str = "v1"

    @property
    def resolved_data_dir(self) -> Path:
        """Return the absolute path for the data directory."""
        return self._resolve(self.data_dir)

    @property
    def resolved_chroma_db_dir(self) -> Path:
        """Return the absolute path for the ChromaDB directory."""
        return self._resolve(self.chroma_db_dir)

    @property
    def resolved_cache_dir(self) -> Path:
        """Return the absolute path for the cache directory."""
        return self._resolve(self.cache_dir)

    @property
    def resolved_log_dir(self) -> Path:
        """Return the absolute path for the logs directory."""
        return self._resolve(self.log_dir)

    def _resolve(self, path: Path) -> Path:
        """Resolve a path relative to project_root if it's relative."""
        if path.is_absolute():
            return path
        return (self.project_root / path).resolve()


# Global singleton
settings = Settings()  # type: ignore[call-arg]