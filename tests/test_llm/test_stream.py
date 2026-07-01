"""Tests for llm/stream.py."""

from __future__ import annotations

from oracle_flexcube_copilot.llm.stream import StreamHandler


class TestStreamHandler:
    def test_handles_tokens(self) -> None:
        handler = StreamHandler()
        tokens = ["Hello ", "World", "!"]
        result = list(handler.handle(iter(tokens)))
        assert result == tokens
        assert handler.token_count == 3
        assert handler.text == "Hello World!"

    def test_empty_stream(self) -> None:
        handler = StreamHandler()
        result = list(handler.handle(iter([])))
        assert result == []
        assert handler.token_count == 0
        assert handler.text == ""

    def test_reuse(self) -> None:
        handler = StreamHandler()
        list(handler.handle(iter(["first"])))
        assert handler.text == "first"
        list(handler.handle(iter(["second"])))
        assert handler.token_count == 1
        assert handler.text == "second"
