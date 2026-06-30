"""Custom exception classes for the application."""

from __future__ import annotations


class OracleCopilotError(Exception):
    """Base exception for all application errors."""


class ConfigurationError(OracleCopilotError):
    """Raised when there is a configuration issue."""


class IngestionError(OracleCopilotError):
    """Raised when PDF ingestion fails."""


class PDFProcessingError(IngestionError):
    """Raised when a specific PDF cannot be processed."""


class RetrievalError(OracleCopilotError):
    """Raised when document retrieval fails."""


class LLMError(OracleCopilotError):
    """Raised when the LLM call fails."""


class EmbeddingError(OracleCopilotError):
    """Raised when embedding generation fails."""


class CacheError(OracleCopilotError):
    """Raised when cache operations fail."""