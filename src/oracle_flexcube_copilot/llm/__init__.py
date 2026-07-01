"""LLM Integration Module — answer generation, streaming, formatting."""

from __future__ import annotations

from oracle_flexcube_copilot.llm.client import OllamaLLMClient
from oracle_flexcube_copilot.llm.exceptions import (
    LLMConnectionError,
    LLMContextOverflowError,
    LLMEmptyResponseError,
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
)
from oracle_flexcube_copilot.llm.formatter import ConsoleAnswerFormatter
from oracle_flexcube_copilot.llm.generator import RAGAnswerGenerator
from oracle_flexcube_copilot.llm.models import (
    AnswerMetadata,
    AnswerMetrics,
    AnswerResponse,
    Citation,
    LLMConfig,
)
from oracle_flexcube_copilot.llm.stream import StreamHandler

__all__ = [
    "AnswerMetadata",
    "AnswerMetrics",
    "AnswerResponse",
    "Citation",
    "ConsoleAnswerFormatter",
    "LLMConfig",
    "LLMConnectionError",
    "LLMContextOverflowError",
    "LLMEmptyResponseError",
    "LLMError",
    "LLMModelNotFoundError",
    "LLMTimeoutError",
    "OllamaLLMClient",
    "RAGAnswerGenerator",
    "StreamHandler",
]
