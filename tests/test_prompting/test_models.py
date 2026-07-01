from __future__ import annotations

from oracle_flexcube_copilot.prompting.models import (
    Citation,
    ContextBlock,
    ContextConfig,
    PromptRequest,
)


def test_context_block_defaults() -> None:
    block = ContextBlock(
        chunk_id="abc123",
        document="test.pdf",
        page=5,
        score=0.95,
        text="Some content",
        index=1,
    )
    assert block.section is None
    assert block.section_id is None
    assert block.page_end is None
    assert block.entities == []


def test_citation_round_trip() -> None:
    citation = Citation(
        chunk_id="abc",
        document="GL.pdf",
        section="GL Transfer",
        page=42,
        score=0.9,
    )
    assert citation.chunk_id == "abc"
    assert citation.score == 0.9

    dumped = citation.model_dump()
    restored = Citation.model_validate(dumped)
    assert restored == citation


def test_context_config_defaults() -> None:
    cfg = ContextConfig()
    assert cfg.max_tokens == 4096
    assert cfg.include_system_prompt is True
    assert cfg.include_date is True


def test_context_config_custom() -> None:
    cfg = ContextConfig(max_tokens=2048, include_system_prompt=False)
    assert cfg.max_tokens == 2048
    assert cfg.include_system_prompt is False
    assert cfg.include_date is True


def test_prompt_request_round_trip() -> None:
    block = ContextBlock(
        chunk_id="c1",
        document="doc.pdf",
        page=1,
        score=0.9,
        text="chunk text",
        index=1,
    )
    citation = Citation(
        chunk_id="c1",
        document="doc.pdf",
        page=1,
        score=0.9,
    )
    req = PromptRequest(
        system_prompt="sys prompt",
        user_prompt="user question",
        formatted_context="<context>...</context>",
        context_blocks=[block],
        estimated_tokens=100,
        citations=[citation],
    )
    assert req.estimated_tokens == 100
    assert len(req.context_blocks) == 1
    assert len(req.citations) == 1

    dumped = req.model_dump()
    restored = PromptRequest.model_validate(dumped)
    assert restored == req


def test_prompt_request_empty_lists() -> None:
    req = PromptRequest(
        system_prompt="sys",
        user_prompt="q",
        formatted_context="",
        context_blocks=[],
        estimated_tokens=0,
        citations=[],
    )
    assert req.context_blocks == []
    assert req.citations == []
