"""Tests for LLMEngine."""

from unittest.mock import Mock

from oracle_flexcube_copilot.llm.engine import LLMEngine


def test_generate_returns_text() -> None:
    """LLMEngine.generate() should return the model's response text."""
    mock_client = Mock()
    mock_client.generate.return_value = Mock(response="This is the answer.")

    engine = LLMEngine(client=mock_client)
    result = engine.generate("What is CASA?")

    assert result == "This is the answer."
    mock_client.generate.assert_called_once()


def test_stream_yields_tokens() -> None:
    """LLMEngine.stream() should yield each token from the model."""
    mock_client = Mock()
    mock_client.generate.return_value = iter(
        [
            Mock(response="This "),
            Mock(response="is "),
            Mock(response="the answer."),
        ]
    )

    engine = LLMEngine(client=mock_client)
    tokens = list(engine.stream("What is STTM_PRODUCT?"))

    assert tokens == ["This ", "is ", "the answer."]


def test_uses_settings_defaults() -> None:
    """LLMEngine should use configured model and temperature by default."""
    mock_client = Mock()
    mock_client.generate.return_value = Mock(response="ok")

    engine = LLMEngine(client=mock_client)
    engine.generate("test")

    call_kwargs = mock_client.generate.call_args
    assert call_kwargs.kwargs["options"]["temperature"] == engine.temperature
    assert call_kwargs.kwargs["model"] == engine.model
