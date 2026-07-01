"""LLM Layer Protocols."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from oracle_flexcube_copilot.llm.models import AnswerResponse
from oracle_flexcube_copilot.prompting.models import PromptRequest


@runtime_checkable
class LLMClient(Protocol):
    """Protocol for the underlying LLM API client."""

    def generate(self, prompt: str, **kwargs: int | float | str | bool) -> str:
        """Generate a complete response.

        Args:
            prompt: The full prompt string.
            **kwargs: Additional generation options.

        Returns:
            The generated text.
        """
        ...

    def stream(self, prompt: str, **kwargs: int | float | str | bool) -> Iterator[str]:
        """Stream tokens from the LLM.

        Args:
            prompt: The full prompt string.
            **kwargs: Additional generation options.

        Yields:
            Each token as a string.
        """
        ...

    @property
    def model_name(self) -> str:
        """Return the model name."""
        ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """Protocol for generating answers from prompts."""

    def generate(
        self,
        prompt_request: PromptRequest,
        mode: str = "concise",
    ) -> AnswerResponse:
        """Generate an answer from a fully assembled PromptRequest.

        Args:
            prompt_request: The assembled prompt with context and citations.
            mode: Answer mode - concise, detailed, or expert.

        Returns:
            An AnswerResponse with answer, citations, confidence, and metadata.
        """
        ...

    def stream(
        self,
        prompt_request: PromptRequest,
        mode: str = "concise",
    ) -> Iterator[str]:
        """Stream tokens from the LLM.

        Args:
            prompt_request: The assembled prompt with context and citations.
            mode: Answer mode - concise, detailed, or expert.

        Yields:
            Each token as a string.
        """
        ...


@runtime_checkable
class AnswerFormatter(Protocol):
    """Protocol for formatting AnswerResponse for display."""

    def format(self, response: AnswerResponse, question: str) -> str:
        """Format an AnswerResponse for console display.

        Args:
            response: The answer response to format.
            question: The original user question.

        Returns:
            A formatted string ready for console output.
        """
        ...


@runtime_checkable
class ConversationMemory(Protocol):
    """Protocol for conversation memory (future use)."""

    def add_turn(self, question: str, answer: str) -> None:
        """Record a question-answer pair."""
        ...

    def get_history(self) -> list[dict[str, str]]:
        """Return conversation history."""
        ...

    def clear(self) -> None:
        """Clear conversation history."""
        ...
