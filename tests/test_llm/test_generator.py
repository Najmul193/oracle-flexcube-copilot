"""Tests for llm/generator.py."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from oracle_flexcube_copilot.llm.generator import RAGAnswerGenerator, _calculate_confidence
from oracle_flexcube_copilot.prompting.models import ContextBlock, PromptRequest


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.generate.return_value = "This is a test answer."
    client.model_name = "qwen3:8b"
    return client


@pytest.fixture
def sample_prompt_request() -> PromptRequest:
    return PromptRequest(
        system_prompt="You are an Oracle assistant.",
        user_prompt="What is GL Balance Transfer?",
        formatted_context="<context><text>GL Balance Transfer details.</text></context>",
        context_blocks=[
            ContextBlock(
                chunk_id="c1",
                document="GL.pdf",
                document_id="doc1",
                section="7.1.1",
                page=44,
                score=0.95,
                entities=["STDGLTRF"],
                text="GL Balance Transfer details.",
                index=1,
            ),
            ContextBlock(
                chunk_id="c2",
                document="GL.pdf",
                document_id="doc1",
                section="7.1",
                page=44,
                score=0.85,
                entities=[],
                text="GL Balance Transfer overview.",
                index=2,
            ),
        ],
        estimated_tokens=500,
        citations=[],
    )


class TestRAGAnswerGeneratorInit:
    def test_default_init(self) -> None:
        generator = RAGAnswerGenerator()
        assert generator._client.config.model == "qwen3:8b"

    def test_custom_client(self, mock_client: MagicMock) -> None:
        generator = RAGAnswerGenerator(client=mock_client)
        assert generator._client == mock_client


class TestRAGAnswerGeneratorGenerate:
    def test_concise_mode(self, mock_client: MagicMock, sample_prompt_request: PromptRequest) -> None:
        generator = RAGAnswerGenerator(client=mock_client)
        response = generator.generate(sample_prompt_request, mode="concise")
        assert response.answer == "This is a test answer."
        assert response.mode == "concise"
        assert len(response.citations) > 0
        assert response.confidence in ("High", "Medium", "Low")
        assert mock_client.generate.called

    def test_detailed_mode(self, mock_client: MagicMock, sample_prompt_request: PromptRequest) -> None:
        generator = RAGAnswerGenerator(client=mock_client)
        response = generator.generate(sample_prompt_request, mode="detailed")
        assert response.mode == "detailed"
        assert response.answer == "This is a test answer."

    def test_expert_mode(self, mock_client: MagicMock, sample_prompt_request: PromptRequest) -> None:
        generator = RAGAnswerGenerator(client=mock_client)
        response = generator.generate(sample_prompt_request, mode="expert")
        assert response.mode == "expert"

    def test_unknown_mode_falls_back(self, mock_client: MagicMock, sample_prompt_request: PromptRequest) -> None:
        generator = RAGAnswerGenerator(client=mock_client)
        response = generator.generate(sample_prompt_request, mode="unknown")
        assert response.mode == "concise"

    def test_citations_are_deduplicated(self, mock_client: MagicMock) -> None:
        prompt = PromptRequest(
            system_prompt="",
            user_prompt="test",
            formatted_context="<context/>",
            context_blocks=[
                ContextBlock(
                    chunk_id="c1", document="GL.pdf", document_id="d1",
                    section="7.1", page=44, score=0.9, entities=[],
                    text="a", index=1,
                ),
                ContextBlock(
                    chunk_id="c2", document="GL.pdf", document_id="d1",
                    section="7.1", page=44, score=0.8, entities=[],
                    text="b", index=2,
                ),
            ],
            estimated_tokens=100,
            citations=[],
        )
        generator = RAGAnswerGenerator(client=mock_client)
        response = generator.generate(prompt)
        assert len(response.citations) == 1

    def test_confidence_high_with_good_scores(self, mock_client: MagicMock, sample_prompt_request: PromptRequest) -> None:
        generator = RAGAnswerGenerator(client=mock_client)
        response = generator.generate(sample_prompt_request)
        # High scores + entities + 1 doc → should be High
        assert response.confidence == "High"

    def test_metadata_populated(self, mock_client: MagicMock, sample_prompt_request: PromptRequest) -> None:
        generator = RAGAnswerGenerator(client=mock_client)
        response = generator.generate(sample_prompt_request)
        assert response.metadata.model_name == "qwen3:8b"
        assert response.metadata.prompt_tokens == 500
        assert response.metadata.completion_tokens > 0
        assert response.metadata.generation_time > 0

    def test_empty_context_blocks(self, mock_client: MagicMock) -> None:
        prompt = PromptRequest(
            system_prompt="", user_prompt="test",
            formatted_context="", context_blocks=[],
            estimated_tokens=10, citations=[],
        )
        generator = RAGAnswerGenerator(client=mock_client)
        response = generator.generate(prompt)
        assert response.confidence == "Low"
        assert response.confidence_percentage == 0.0


class TestRAGAnswerGeneratorStream:
    def test_stream_yields_tokens(self, mock_client: MagicMock, sample_prompt_request: PromptRequest) -> None:
        mock_client.stream.return_value = iter(["Hello ", "World"])
        generator = RAGAnswerGenerator(client=mock_client)
        tokens = list(generator.stream(sample_prompt_request))
        assert tokens == ["Hello ", "World"]

    def test_stream_unknown_mode(self, mock_client: MagicMock, sample_prompt_request: PromptRequest) -> None:
        mock_client.stream.return_value = iter(["test"])
        generator = RAGAnswerGenerator(client=mock_client)
        tokens = list(generator.stream(sample_prompt_request, mode="unknown"))
        assert tokens == ["test"]


class TestCalculateConfidence:
    def test_empty_blocks_returns_low(self) -> None:
        prompt = PromptRequest(
            system_prompt="", user_prompt="",
            formatted_context="", context_blocks=[],
            estimated_tokens=0, citations=[],
        )
        label, pct = _calculate_confidence(prompt)
        assert label == "Low"
        assert pct == 0.0

    def test_high_scores_with_entities(self) -> None:
        prompt = PromptRequest(
            system_prompt="", user_prompt="",
            formatted_context="",
            context_blocks=[
                ContextBlock(
                    chunk_id="c1", document="GL.pdf", document_id="d1",
                    section="s1", page=1, score=0.95, entities=["STDGLTRF"],
                    text="t", index=1,
                ),
                ContextBlock(
                    chunk_id="c2", document="GL.pdf", document_id="d1",
                    section="s2", page=2, score=0.90, entities=["STDGLTRF"],
                    text="t", index=2,
                ),
            ],
            estimated_tokens=0, citations=[],
        )
        label, pct = _calculate_confidence(prompt)
        assert label == "High"
        assert pct >= 80.0

    def test_medium_scores(self) -> None:
        prompt = PromptRequest(
            system_prompt="", user_prompt="",
            formatted_context="",
            context_blocks=[
                ContextBlock(
                    chunk_id="c1", document="GL.pdf", document_id="d1",
                    section="s1", page=1, score=0.75, entities=[],
                    text="t", index=1,
                ),
                ContextBlock(
                    chunk_id="c2", document="CASA.pdf", document_id="d2",
                    section="s2", page=2, score=0.7, entities=[],
                    text="t", index=2,
                ),
            ],
            estimated_tokens=0, citations=[],
        )
        label, pct = _calculate_confidence(prompt)
        assert label == "Medium"
        assert 50.0 <= pct < 80.0

    def test_low_scores(self) -> None:
        prompt = PromptRequest(
            system_prompt="", user_prompt="",
            formatted_context="",
            context_blocks=[
                ContextBlock(
                    chunk_id="c1", document="GL.pdf", document_id="d1",
                    section="s1", page=1, score=0.3, entities=[],
                    text="t", index=1,
                ),
            ],
            estimated_tokens=0, citations=[],
        )
        label, pct = _calculate_confidence(prompt)
        assert label == "Low"
        assert pct < 50.0
