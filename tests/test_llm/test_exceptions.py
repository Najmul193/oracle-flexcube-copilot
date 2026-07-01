"""Tests for llm/exceptions.py."""

from __future__ import annotations

from oracle_flexcube_copilot.llm.exceptions import (
    LLMConnectionError,
    LLMContextOverflowError,
    LLMEmptyResponseError,
    LLMError,
    LLMModelNotFoundError,
    LLMTimeoutError,
)


class TestLLMExceptions:
    def test_base_exception(self) -> None:
        exc = LLMError("Base error")
        assert str(exc) == "Base error"
        assert isinstance(exc, Exception)

    def test_connection_error(self) -> None:
        exc = LLMConnectionError("Connection refused")
        assert str(exc) == "Connection refused"
        assert isinstance(exc, LLMError)

    def test_timeout_error(self) -> None:
        exc = LLMTimeoutError("Request timed out")
        assert str(exc) == "Request timed out"
        assert isinstance(exc, LLMError)

    def test_model_not_found_error(self) -> None:
        exc = LLMModelNotFoundError("Model not found")
        assert str(exc) == "Model not found"
        assert isinstance(exc, LLMError)

    def test_empty_response_error(self) -> None:
        exc = LLMEmptyResponseError("Empty response")
        assert str(exc) == "Empty response"
        assert isinstance(exc, LLMError)

    def test_context_overflow_error(self) -> None:
        exc = LLMContextOverflowError("Context too large")
        assert str(exc) == "Context too large"
        assert isinstance(exc, LLMError)
