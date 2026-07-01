"""Tests for llm/models.py."""

from __future__ import annotations

from oracle_flexcube_copilot.llm.models import (
    AnswerMetadata,
    AnswerMetrics,
    AnswerResponse,
    Citation,
    LLMConfig,
)


class TestLLMConfig:
    def test_defaults(self) -> None:
        c = LLMConfig()
        assert c.base_url == "http://localhost:11434"
        assert c.model == "qwen3:8b"
        assert c.temperature == 0.1
        assert c.top_p == 0.9
        assert c.repeat_penalty == 1.1
        assert c.num_ctx == 8192
        assert c.num_predict == 2048
        assert c.timeout == 120

    def test_custom_values(self) -> None:
        c = LLMConfig(
            base_url="http://custom:11434",
            model="llama3:8b",
            temperature=0.5,
            top_p=0.8,
            repeat_penalty=1.2,
            num_ctx=16384,
            num_predict=4096,
            timeout=60,
        )
        assert c.base_url == "http://custom:11434"
        assert c.model == "llama3:8b"
        assert c.temperature == 0.5
        assert c.top_p == 0.8
        assert c.repeat_penalty == 1.2
        assert c.num_ctx == 16384
        assert c.num_predict == 4096
        assert c.timeout == 60

    def test_validation_rejects_invalid_values(self) -> None:
        import pytest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            LLMConfig(temperature=-1.0)
        with pytest.raises(ValidationError):
            LLMConfig(top_p=5.0)


class TestCitation:
    def test_defaults(self) -> None:
        cit = Citation(document="GL.pdf", page=44, score=0.95)
        assert cit.document == "GL.pdf"
        assert cit.section is None
        assert cit.page == 44
        assert cit.score == 0.95

    def test_with_section(self) -> None:
        cit = Citation(document="CASA.pdf", section="Account Opening", page=10, score=1.0)
        assert cit.document == "CASA.pdf"
        assert cit.section == "Account Opening"


class TestAnswerMetadata:
    def test_defaults(self) -> None:
        m = AnswerMetadata()
        assert m.prompt_tokens == 0
        assert m.completion_tokens == 0
        assert m.total_tokens == 0
        assert m.retrieval_time == 0.0
        assert m.generation_time == 0.0
        assert m.model_name == ""

    def test_custom(self) -> None:
        m = AnswerMetadata(
            prompt_tokens=500,
            completion_tokens=200,
            total_tokens=700,
            retrieval_time=0.5,
            generation_time=2.3,
            model_name="qwen3:8b",
        )
        assert m.prompt_tokens == 500
        assert m.completion_tokens == 200
        assert m.generation_time == 2.3
        assert m.model_name == "qwen3:8b"


class TestAnswerResponse:
    def test_defaults(self) -> None:
        r = AnswerResponse(answer="Test answer")
        assert r.answer == "Test answer"
        assert r.citations == []
        assert r.confidence == "Low"
        assert r.confidence_percentage == 0.0
        assert r.mode == "concise"
        assert r.metadata.prompt_tokens == 0

    def test_with_citations(self) -> None:
        r = AnswerResponse(
            answer="Detailed answer",
            citations=[Citation(document="GL.pdf", page=44, score=0.95)],
            confidence="High",
            confidence_percentage=85.0,
            mode="detailed",
        )
        assert len(r.citations) == 1
        assert r.citations[0].document == "GL.pdf"
        assert r.confidence == "High"
        assert r.confidence_percentage == 85.0
        assert r.mode == "detailed"


class TestAnswerMetrics:
    def test_defaults(self) -> None:
        m = AnswerMetrics()
        assert m.total_questions == 0
        assert m.avg_generation_time == 0.0
        assert m.total_tokens == 0
        assert m.model_name == ""
