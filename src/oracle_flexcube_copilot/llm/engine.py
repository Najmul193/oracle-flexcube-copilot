"""Qwen LLM Integration Engine."""

import logging
from collections.abc import Iterator

from ollama import Client

from oracle_flexcube_copilot.config import settings

logger = logging.getLogger(__name__)


class LLMEngine:
    """Thin, focused wrapper around the Ollama Qwen LLM."""

    def __init__(
        self,
        client: Client | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        """Initialize the LLM engine.

        Args:
            client: Optional pre-built Ollama Client.
            model: Override the model name (defaults to settings.llm_model).
            temperature: Override the temperature (defaults to settings.llm_temperature).
            max_tokens: Override the max output tokens (defaults to settings.llm_max_tokens).
        """
        self.client = client or Client(host=settings.ollama_base_url)
        self.model = model or settings.llm_model
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self.max_tokens = max_tokens or settings.llm_max_tokens

    def generate(self, prompt: str) -> str:
        """Generate a complete response from a prompt.

        Args:
            prompt: The fully assembled prompt string.

        Returns:
            The model's text response.
        """
        logger.info(f"Sending prompt to {self.model}")
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        )
        return response.response

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream tokens from the LLM one-by-one for real-time output.

        Args:
            prompt: The fully assembled prompt string.

        Yields:
            Each token as a string.
        """
        logger.info(f"Streaming response from {self.model}")
        for chunk in self.client.generate(
            model=self.model,
            prompt=prompt,
            stream=True,
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        ):
            if chunk.response:
                yield chunk.response
