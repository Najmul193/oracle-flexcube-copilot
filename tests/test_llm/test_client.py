"""Tests for llm/client.py."""

# ruff: noqa: ARG002

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from oracle_flexcube_copilot.llm.client import OllamaLLMClient
from oracle_flexcube_copilot.llm.exceptions import (
    LLMConnectionError,
    LLMContextOverflowError,
    LLMEmptyResponseError,
    LLMModelNotFoundError,
    LLMTimeoutError,
)
from oracle_flexcube_copilot.llm.models import LLMConfig


@pytest.fixture
def mock_ollama_client() -> MagicMock:
    """Mock the underlying Ollama Client."""
    with patch("oracle_flexcube_copilot.llm.client.OllamaClient") as mock:
        client_instance = MagicMock()
        mock.return_value = client_instance
        client_instance.list.return_value = {"models": []}
        yield client_instance


class TestOllamaLLMClientInit:
    def test_init_with_default_config(self, mock_ollama_client: MagicMock) -> None:
        client = OllamaLLMClient()
        assert client.model_name == "qwen3:8b"
        assert client.config.base_url == "http://localhost:11434"

    def test_init_with_custom_config(self, mock_ollama_client: MagicMock) -> None:
        config = LLMConfig(model="llama3:8b", base_url="http://custom:11434")
        client = OllamaLLMClient(config=config)
        assert client.model_name == "llama3:8b"
        assert client.config.base_url == "http://custom:11434"


class TestOllamaLLMClientGenerate:
    def test_successful_generate(self, mock_ollama_client: MagicMock) -> None:
        mock_ollama_client.generate.return_value = {"response": "This is the answer."}
        client = OllamaLLMClient()
        result = client.generate("What is GL?")
        assert result == "This is the answer."
        mock_ollama_client.generate.assert_called_once()

    def test_empty_response_raises_error(self, mock_ollama_client: MagicMock) -> None:
        mock_ollama_client.generate.return_value = {"response": ""}
        client = OllamaLLMClient()
        with pytest.raises(LLMEmptyResponseError):
            client.generate("What is GL?")

    def test_connection_error_retries(self, mock_ollama_client: MagicMock) -> None:
        mock_ollama_client.generate.side_effect = ConnectionError("Connection refused")
        client = OllamaLLMClient()
        with pytest.raises(LLMConnectionError):
            client.generate("What is GL?")
        assert mock_ollama_client.generate.call_count == 3

    def test_timeout_error_retries(self, mock_ollama_client: MagicMock) -> None:
        mock_ollama_client.generate.side_effect = TimeoutError("timed out")
        client = OllamaLLMClient()
        with pytest.raises(LLMTimeoutError):
            client.generate("What is GL?")
        assert mock_ollama_client.generate.call_count == 3

    def test_model_not_found(self, mock_ollama_client: MagicMock) -> None:
        mock_ollama_client.generate.side_effect = Exception("model 'bad' not found")
        client = OllamaLLMClient()
        with pytest.raises(LLMModelNotFoundError):
            client.generate("What is GL?")

    def test_eventual_success_after_retries(self, mock_ollama_client: MagicMock) -> None:
        mock_ollama_client.generate.side_effect = [
            ConnectionError("Connection refused"),
            {"response": "Success after retry."},
        ]
        client = OllamaLLMClient()
        result = client.generate("What is GL?")
        assert result == "Success after retry."
        assert mock_ollama_client.generate.call_count == 2


class TestOllamaLLMClientStream:
    def test_successful_stream(self, mock_ollama_client: MagicMock) -> None:
        mock_ollama_client.generate.return_value = iter(
            [{"response": "Hello "}, {"response": "World"}]
        )
        client = OllamaLLMClient()
        tokens = list(client.stream("What is GL?"))
        assert tokens == ["Hello ", "World"]

    def test_empty_stream_raises_error(self, mock_ollama_client: MagicMock) -> None:
        mock_ollama_client.generate.return_value = iter([])
        client = OllamaLLMClient()
        with pytest.raises(LLMEmptyResponseError):
            list(client.stream("What is GL?"))

    def test_stream_connection_error_retries(self, mock_ollama_client: MagicMock) -> None:
        mock_ollama_client.generate.side_effect = ConnectionError("Connection refused")
        client = OllamaLLMClient()
        with pytest.raises(LLMConnectionError):
            list(client.stream("What is GL?"))
        assert mock_ollama_client.generate.call_count == 3


class TestClassifyError:
    def test_connection_refused(self, mock_ollama_client: MagicMock) -> None:
        client = OllamaLLMClient()
        exc = client._classify_error(ConnectionError("Connection refused"))
        assert isinstance(exc, LLMConnectionError)

    def test_timeout(self, mock_ollama_client: MagicMock) -> None:
        client = OllamaLLMClient()
        exc = client._classify_error(TimeoutError("request timed out"))
        assert isinstance(exc, LLMTimeoutError)

    def test_model_not_found(self, mock_ollama_client: MagicMock) -> None:
        client = OllamaLLMClient()
        exc = client._classify_error(Exception("model 'x' not found"))
        assert isinstance(exc, LLMModelNotFoundError)

    def test_context_overflow(self, mock_ollama_client: MagicMock) -> None:
        client = OllamaLLMClient()
        exc = client._classify_error(Exception("context overflow"))
        assert isinstance(exc, LLMContextOverflowError)

    def test_default_connection_error(self, mock_ollama_client: MagicMock) -> None:
        client = OllamaLLMClient()
        exc = client._classify_error(Exception("Unknown error"))
        assert isinstance(exc, LLMConnectionError)
