"""LLM Layer Custom Exceptions."""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for all LLM layer errors."""


class LLMConnectionError(LLMError):
    """Ollama server is unreachable."""


class LLMTimeoutError(LLMError):
    """Request to Ollama timed out."""


class LLMModelNotFoundError(LLMError):
    """Requested model is not available in Ollama."""


class LLMEmptyResponseError(LLMError):
    """LLM returned an empty response."""


class LLMContextOverflowError(LLMError):
    """Prompt exceeds the model's context window."""
