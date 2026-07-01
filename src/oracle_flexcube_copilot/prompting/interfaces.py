from __future__ import annotations

from typing import Protocol, runtime_checkable

from oracle_flexcube_copilot.indexing.models import SearchResult
from oracle_flexcube_copilot.prompting.models import (
    Citation,
    ContextBlock,
    ContextConfig,
    PromptRequest,
)


@runtime_checkable
class TokenEstimator(Protocol):
    """Estimates the token count of a text string."""

    def estimate(self, text: str) -> int:
        """Return estimated token count for *text*."""
        ...


@runtime_checkable
class ContextFormatter(Protocol):
    """Formats retrieved chunks into a structured context string with metadata."""

    def format(
        self,
        chunks: list[SearchResult],
    ) -> tuple[str, list[ContextBlock], list[Citation]]:
        """Transform search results into (xml_context, blocks, citations).

        Args:
            chunks: Ranked search results from the retrieval engine.

        Returns:
            A tuple of:
            - XML-formatted context string
            - Structured context blocks (one per chunk group)
            - Citation metadata for each block
        """
        ...


@runtime_checkable
class SystemPromptBuilder(Protocol):
    """Builds system prompts for different modes (strict, support, benchmark, ...)."""

    def build(self, mode: str = "strict") -> str:
        """Return a system prompt string for the given *mode*."""
        ...


@runtime_checkable
class PromptBuilder(Protocol):
    """Assembles a complete PromptRequest from a question and retrieved context."""

    def build(
        self,
        question: str,
        context: list[SearchResult],
        config: ContextConfig | None = None,
    ) -> PromptRequest:
        """Build a fully structured prompt.

        Args:
            question: The user's original question.
            context: Retrieved search results.
            config: Optional context configuration overrides.

        Returns:
            A PromptRequest ready for the LLM.
        """
        ...
