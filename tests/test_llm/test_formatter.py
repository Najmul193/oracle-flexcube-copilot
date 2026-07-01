"""Tests for llm/formatter.py."""

from __future__ import annotations

from oracle_flexcube_copilot.llm.formatter import ConsoleAnswerFormatter
from oracle_flexcube_copilot.llm.models import AnswerMetadata, AnswerResponse, Citation


class TestConsoleAnswerFormatter:
    def test_format_contains_answer(self) -> None:
        formatter = ConsoleAnswerFormatter()
        response = AnswerResponse(
            answer="This is the GL Balance Transfer procedure.",
            citations=[
                Citation(document="GL.pdf", section="7.1.1", page=44, score=0.95),
            ],
            confidence="High",
            confidence_percentage=85.0,
            metadata=AnswerMetadata(
                prompt_tokens=500,
                completion_tokens=120,
                total_tokens=620,
                retrieval_time=0.3,
                generation_time=2.1,
                model_name="qwen3:8b",
            ),
        )
        output = formatter.format(response, "What is GL Balance Transfer?")
        assert "What is GL Balance Transfer?" in output
        assert "GL Balance Transfer procedure" in output
        assert "GL.pdf" in output
        assert "High" in output
        assert "85%" in output
        assert "2.1" in output or "2100ms" in output
        assert "qwen3:8b" in output

    def test_format_no_citations(self) -> None:
        formatter = ConsoleAnswerFormatter()
        response = AnswerResponse(
            answer="No relevant data found.",
            citations=[],
            confidence="Low",
            confidence_percentage=10.0,
        )
        output = formatter.format(response, "test question")
        assert "No relevant data" in output
        assert "Sources" not in output

    def test_format_section_optional(self) -> None:
        formatter = ConsoleAnswerFormatter()
        response = AnswerResponse(
            answer="Answer without section.",
            citations=[Citation(document="GL.pdf", page=44, score=0.9)],
            confidence="Medium",
            confidence_percentage=60.0,
        )
        output = formatter.format(response, "test")
        assert "GL.pdf" in output

    def test_format_stream_start(self) -> None:
        formatter = ConsoleAnswerFormatter()
        header = formatter.format_stream_start("What is GL?", "detailed")
        assert "What is GL?" in header
        assert "detailed" in header

    def test_format_stream_end(self) -> None:
        formatter = ConsoleAnswerFormatter()
        response = AnswerResponse(
            answer="Streamed answer.",
            citations=[Citation(document="CASA.pdf", page=10, score=0.8)],
            confidence="Medium",
            confidence_percentage=65.0,
            metadata=AnswerMetadata(
                prompt_tokens=400,
                completion_tokens=80,
                total_tokens=480,
                retrieval_time=0.2,
                generation_time=0.0,
                model_name="qwen3:8b",
            ),
        )
        footer = formatter.format_stream_end(response)
        assert "CASA.pdf" in footer
        assert "Medium" in footer
        assert "qwen3:8b" in footer
