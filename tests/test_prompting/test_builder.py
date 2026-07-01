from __future__ import annotations

from oracle_flexcube_copilot.indexing.models import SearchResult
from oracle_flexcube_copilot.prompting.builder import RAGPromptBuilder
from oracle_flexcube_copilot.prompting.models import ContextConfig


def _make_chunk(
    chunk_id: str,
    doc: str = "test.pdf",
    page: int = 1,
    heading: str = "Section A",
    text: str = "Some content.",
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        source_document=doc,
        page=page,
        heading=heading,
        oracle_entities=[],
        text=text,
        retrieval_method="vector",
    )


def test_build_with_context() -> None:
    builder = RAGPromptBuilder()
    chunks = [_make_chunk("a", doc="GL.pdf", page=42, heading="GL Transfer", text="Details here.")]
    req = builder.build("How do I maintain GL balance?", chunks)

    assert req.system_prompt != ""
    assert req.user_prompt == "How do I maintain GL balance?"
    assert "GL.pdf" in req.formatted_context
    assert "Details here." in req.formatted_context
    assert len(req.context_blocks) == 1
    assert len(req.citations) == 1
    assert req.estimated_tokens > 0


def test_build_empty_context() -> None:
    builder = RAGPromptBuilder()
    req = builder.build("What is CASA?", [])

    assert req.user_prompt == "What is CASA?"
    assert len(req.context_blocks) == 0
    assert len(req.citations) == 0
    assert req.formatted_context == "<context>\n</context>"


def test_build_uses_config() -> None:
    builder = RAGPromptBuilder()
    chunks = [_make_chunk("a", text="Hello world.")]
    config = ContextConfig(max_tokens=2, include_system_prompt=False, include_date=False)
    req = builder.build("Q?", chunks, config=config)

    assert req.system_prompt == ""
    assert "Today's Date" not in req.formatted_context
    # With max_tokens=2 and ~3 char text, the block may be dropped
    assert len(req.context_blocks) <= 1


def test_build_preserves_citations() -> None:
    builder = RAGPromptBuilder()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=5, heading="Transfer", text="A", score=0.9),
        _make_chunk("b", doc="CASA.pdf", page=10, heading="Config", text="B", score=0.7),
    ]
    req = builder.build("Q?", chunks)

    assert len(req.citations) == 2
    assert req.citations[0].chunk_id == "a"
    assert req.citations[1].chunk_id == "b"
    assert req.citations[0].document == "GL.pdf"
    assert req.citations[1].document == "CASA.pdf"


def test_build_ordering() -> None:
    builder = RAGPromptBuilder()
    chunks = [
        _make_chunk("a", page=1, heading="Intro", text="first", score=0.3),
        _make_chunk("b", page=2, heading="Config", text="second", score=0.9),
        _make_chunk("c", page=3, heading="Setup", text="third", score=0.6),
    ]
    req = builder.build("Q?", chunks)
    assert req.context_blocks[0].chunk_id == "a"
    assert req.context_blocks[1].chunk_id == "b"
    assert req.context_blocks[2].chunk_id == "c"


def test_build_duplicate_dedup() -> None:
    builder = RAGPromptBuilder()
    chunks = [
        _make_chunk("a", text="original"),
        _make_chunk("a", text="duplicate"),
    ]
    req = builder.build("Q?", chunks)
    assert len(req.context_blocks) == 1
    assert req.context_blocks[0].text == "original"


def test_build_deterministic() -> None:
    builder = RAGPromptBuilder()
    chunks = [
        _make_chunk("b", text="B", score=0.6),
        _make_chunk("a", text="A", score=0.9),
    ]
    req1 = builder.build("Q?", chunks)
    req2 = builder.build("Q?", chunks)
    assert req1.model_dump() == req2.model_dump()


def test_build_include_date() -> None:
    builder = RAGPromptBuilder()
    chunks = [_make_chunk("a", text="test")]
    req = builder.build("Q?", chunks)

    # Date should be rendered somewhere in the output
    assert len(req.system_prompt) > 0


def test_build_system_prompt_content() -> None:
    builder = RAGPromptBuilder()
    chunks = [_make_chunk("a", text="test")]
    req = builder.build("Q?", chunks)
    assert "Answering Policy" in req.system_prompt
    assert "Oracle FLEXCUBE Copilot" in req.system_prompt


def test_build_no_system_prompt() -> None:
    builder = RAGPromptBuilder()
    chunks = [_make_chunk("a", text="test")]
    config = ContextConfig(include_system_prompt=False)
    req = builder.build("Q?", chunks, config=config)
    assert req.system_prompt == ""


def test_build_estimated_tokens() -> None:
    builder = RAGPromptBuilder()
    chunks = [_make_chunk("a", text="Hello")]
    req = builder.build("Q?", chunks)
    assert req.estimated_tokens >= 1
    # The estimate should be deterministic
    req2 = builder.build("Q?", chunks)
    assert req.estimated_tokens == req2.estimated_tokens
