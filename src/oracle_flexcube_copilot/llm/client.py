"""Ollama LLM Client with retry, streaming, and error handling."""

from __future__ import annotations

import logging
import time
from collections.abc import Iterator

from ollama import Client as OllamaClient

from oracle_flexcube_copilot.llm.exceptions import (
    LLMConnectionError,
    LLMContextOverflowError,
    LLMEmptyResponseError,
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
)
from oracle_flexcube_copilot.llm.models import LLMConfig

logger = logging.getLogger("oracle_flexcube_copilot.llm.client")

_RETRYABLE = (LLMConnectionError, LLMTimeoutError)
_MAX_RETRIES = 3
_BACKOFF = [1.0, 4.0, 10.0]


class OllamaLLMClient:
    """Production-grade Ollama client with retry, timeout, and error recovery."""

    def __init__(self, config: LLMConfig | None = None) -> None:
        self.config = config or LLMConfig()
        self._client = OllamaClient(host=self.config.base_url)
        self._model = self.config.model
        self._verify_model()

    def _verify_model(self) -> None:
        """Verify the model is available in Ollama on init."""
        try:
            models = self._client.list()
            available = {m.model for m in models.get("models", [])}
            # Ollama returns names like "qwen3:8b"
            if self._model not in available and not any(
                m.startswith(self._model) for m in available
            ):
                logger.warning(
                    "Model '%s' not found in Ollama. Available: %s",
                    self._model,
                    ", ".join(sorted(available)),
                )
        except Exception as exc:
            logger.warning("Could not verify model list: %s", exc)

    @property
    def model_name(self) -> str:
        return self._model

    def _build_options(self, **overrides: int | float | str | bool) -> dict[str, int | float | str | bool]:
        options: dict[str, int | float | str | bool] = {
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "repeat_penalty": self.config.repeat_penalty,
            "num_ctx": self.config.num_ctx,
            "num_predict": self.config.num_predict,
        }
        options.update(overrides)
        return options

    def generate(self, prompt: str, **kwargs: int | float | str | bool) -> str:
        """Generate a complete response with automatic retry on failures.

        Args:
            prompt: The full prompt string.
            **kwargs: Override any LLMConfig option.

        Returns:
            The generated text.

        Raises:
            LLMConnectionError: Ollama server unreachable after retries.
            LLMTimeoutError: Request timed out after retries.
            LLMModelNotFoundError: Model not found on server.
            LLMEmptyResponseError: Server returned empty response.
        """
        last_error: Exception | None = None
        options = self._build_options(**kwargs)

        for attempt in range(_MAX_RETRIES):
            try:
                logger.debug(
                    "LLM generate attempt %d/%d to %s",
                    attempt + 1, _MAX_RETRIES, self._model,
                )
                response = self._client.generate(
                    model=self._model,
                    prompt=prompt,
                    options=options,
                )
                result = response.get("response", "")
                if not result:
                    raise LLMEmptyResponseError("Ollama returned empty response")
                return result

            except LLMEmptyResponseError:
                raise
            except Exception as exc:
                last_error = self._classify_error(exc)
                if not isinstance(last_error, _RETRYABLE):
                    raise last_error from exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF[attempt]
                    logger.warning(
                        "LLM attempt %d failed: %s. Retrying in %.1fs...",
                        attempt + 1, last_error, delay,
                    )
                    time.sleep(delay)

        raise last_error  # type: ignore[misc]

    def stream(self, prompt: str, **kwargs: int | float | str | bool) -> Iterator[str]:
        """Stream tokens from the LLM with automatic retry on failures.

        Args:
            prompt: The full prompt string.
            **kwargs: Override any LLMConfig option.

        Yields:
            Each token as a string.

        Raises:
            LLMConnectionError: Ollama server unreachable after retries.
            LLMTimeoutError: Request timed out after retries.
            LLMModelNotFoundError: Model not found on server.
        """
        options = self._build_options(**kwargs)
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                stream = self._client.generate(
                    model=self._model,
                    prompt=prompt,
                    stream=True,
                    options=options,
                )
                token_count = 0
                for chunk in stream:
                    token = chunk.get("response", "")
                    if token:
                        token_count += 1
                        yield token
                if token_count == 0:
                    raise LLMEmptyResponseError("Ollama returned empty stream")
                return

            except LLMEmptyResponseError:
                raise
            except Exception as exc:
                last_error = self._classify_error(exc)
                if not isinstance(last_error, _RETRYABLE):
                    raise last_error from exc
                if attempt < _MAX_RETRIES - 1:
                    delay = _BACKOFF[attempt]
                    logger.warning(
                        "LLM stream attempt %d failed: %s. Retrying in %.1fs...",
                        attempt + 1, last_error, delay,
                    )
                    time.sleep(delay)

        raise last_error  # type: ignore[misc]

    def _classify_error(self, exc: Exception) -> LLMError:
        """Classify a raw exception into an LLMError subclass."""
        msg = str(exc).lower()
        if "connection" in msg or "refused" in msg or "unreachable" in msg:
            return LLMConnectionError(str(exc))
        if "timeout" in msg or "timed out" in msg:
            return LLMTimeoutError(str(exc))
        if "not found" in msg and "model" in msg:
            return LLMModelNotFoundError(str(exc))
        if "context" in msg and ("overflow" in msg or "exceed" in msg or "too large" in msg):
            return LLMContextOverflowError(str(exc))
        return LLMConnectionError(str(exc))
