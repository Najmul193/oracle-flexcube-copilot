"""Tests for PromptBuilder."""

from oracle_flexcube_copilot.indexing.models import SearchResult
from oracle_flexcube_copilot.prompts.builder import PromptBuilder


def test_build_rag_prompt_with_context() -> None:
    builder = PromptBuilder()
    context = [
        SearchResult(
            chunk_id="1",
            score=0.9,
            source_document="test.pdf",
            page=42,
            heading="Section A",
            oracle_entities=["STTM_PRODUCT"],
            text="This is the text.",
            retrieval_method="vector",
        )
    ]

    prompt = builder.build_rag_prompt("What is it?", context)

    assert "You are an expert Oracle FLEXCUBE documentation assistant." in prompt
    assert "test.pdf" in prompt
    assert "Page: 42" in prompt
    assert "Section A" in prompt
    assert "This is the text." in prompt
    assert "What is it?" in prompt


def test_build_rag_prompt_empty_context() -> None:
    builder = PromptBuilder()
    prompt = builder.build_rag_prompt("What is it?", [])

    assert "No context retrieved." in prompt
    assert "What is it?" in prompt
