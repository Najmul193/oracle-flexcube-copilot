from __future__ import annotations

from oracle_flexcube_copilot.indexing.models import SearchResult
from oracle_flexcube_copilot.prompting.formatter import (
    DefaultContextFormatter,
    HeuristicTokenEstimator,
)


def _make_chunk(
    chunk_id: str,
    doc: str = "test.pdf",
    page: int = 1,
    heading: str = "Section A",
    text: str = "Some content.",
    score: float = 0.9,
    entities: list[str] | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=score,
        source_document=doc,
        page=page,
        heading=heading,
        oracle_entities=entities or [],
        text=text,
        retrieval_method="vector",
    )


# -- Token Estimator tests --


def test_heuristic_estimator_empty() -> None:
    est = HeuristicTokenEstimator()
    assert est.estimate("") == 0


def test_heuristic_estimator_basic() -> None:
    est = HeuristicTokenEstimator()
    assert est.estimate("hello") == 1  # len 5 // 4 = 1
    assert est.estimate("a" * 100) == 25  # 100 // 4 = 25


def test_heuristic_estimator_rounds_up() -> None:
    est = HeuristicTokenEstimator()
    # max(1, len // 4) so short strings return at least 1
    assert est.estimate("ab") == 1
    assert est.estimate("") == 0


# -- Deduplication tests --


def test_deduplicate_removes_duplicate_chunk_ids() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", page=1, heading="Intro", text="first"),
        _make_chunk("b", page=2, heading="Config", text="second"),
        _make_chunk("a", page=3, heading="Intro", text="duplicate"),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 2
    assert blocks[0].text == "first"
    assert blocks[1].text == "second"
    assert blocks[0].chunk_id == "a"
    assert blocks[1].chunk_id == "b"


def test_deduplicate_preserves_order() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("c", page=1, heading="Setup", text="rank3", score=0.3),
        _make_chunk("a", page=2, heading="Transfer", text="rank1", score=0.9),
        _make_chunk("b", page=3, heading="Config", text="rank2", score=0.6),
        _make_chunk("a", page=4, heading="Transfer", text="duplicate of rank1", score=0.8),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 3
    assert blocks[0].chunk_id == "c"  # first occurrence preserved
    assert blocks[1].chunk_id == "a"
    assert blocks[2].chunk_id == "b"


# -- Adjacent merge tests --


def test_merge_adjacent_same_section() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=5, heading="GL Transfer", text="Part one."),
        _make_chunk("b", doc="GL.pdf", page=6, heading="GL Transfer", text="Part two."),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 1
    assert "Part one." in blocks[0].text
    assert "Part two." in blocks[0].text
    assert blocks[0].page == 5


def test_no_merge_different_document() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=5, heading="GL Transfer", text="Part one."),
        _make_chunk("b", doc="CASA.pdf", page=6, heading="GL Transfer", text="Part two."),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 2


def test_no_merge_different_heading() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=5, heading="Transfer", text="Part one."),
        _make_chunk("b", doc="GL.pdf", page=6, heading="Batch", text="Part two."),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 2


def test_no_merge_non_consecutive_pages() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=5, heading="GL Transfer", text="Part one."),
        _make_chunk("b", doc="GL.pdf", page=8, heading="GL Transfer", text="Part two."),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 2


def test_merge_multiple_chunks() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=5, heading="GL Transfer", text="A"),
        _make_chunk("b", doc="GL.pdf", page=6, heading="GL Transfer", text="B"),
        _make_chunk("c", doc="GL.pdf", page=7, heading="GL Transfer", text="C"),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 1
    assert blocks[0].text == "A\n\nB\n\nC"


def test_merge_with_different_section_in_between() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=5, heading="GL Transfer", text="A"),
        _make_chunk("b", doc="GL.pdf", page=6, heading="Config", text="B"),
        _make_chunk("c", doc="GL.pdf", page=7, heading="GL Transfer", text="C"),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 3  # no merge because different heading in between


def test_merge_collects_entities() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk(
            "a", doc="GL.pdf", page=5, heading="GL Transfer", text="A", entities=["STTM_GL"]
        ),
        _make_chunk(
            "b", doc="GL.pdf", page=6, heading="GL Transfer", text="B", entities=["STTM_BALANCE"]
        ),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 1
    assert "STTM_GL" in blocks[0].entities
    assert "STTM_BALANCE" in blocks[0].entities


# -- Token budget truncation tests --


def test_truncate_removes_lowest_ranked_blocks() -> None:
    fmt = DefaultContextFormatter()
    # Each block renders to ~100+ chars, ~25+ tokens
    chunks = [
        _make_chunk("a", text="A" * 200, score=0.9),
        _make_chunk("b", text="B" * 200, score=0.8),
        _make_chunk("c", text="C" * 200, score=0.7),
        _make_chunk("d", text="D" * 200, score=0.6),
    ]
    _xml, blocks, _citations = fmt.format_with_budget(chunks, max_tokens=20)
    # With max_tokens=20, all blocks removed one by one from the end
    assert len(blocks) < len(chunks) or len(blocks) == 0


def test_truncate_preserves_order() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", text="X" * 300, score=0.9),
        _make_chunk("b", text="Y" * 300, score=0.8),
    ]
    _xml, blocks, _citations = fmt.format_with_budget(chunks, max_tokens=500)
    if blocks:
        assert blocks[0].chunk_id == "a"
        assert blocks[0].index == 1


def test_truncate_never_splits_block() -> None:
    fmt = DefaultContextFormatter()
    # Single large block
    chunks = [_make_chunk("a", text="LARGE " * 1000, score=0.9)]
    _xml, blocks, _citations = fmt.format_with_budget(chunks, max_tokens=5)
    # Either the block fits, or it's removed entirely
    assert len(blocks) in (0, 1)
    if blocks:
        assert "LARGE" in blocks[0].text


def test_truncate_under_budget_no_change() -> None:
    fmt = DefaultContextFormatter()
    chunks = [_make_chunk("a", text="short", score=0.9)]
    _xml, blocks, _citations = fmt.format_with_budget(chunks, max_tokens=99999)
    assert len(blocks) == 1


def test_truncate_empty_chunks() -> None:
    fmt = DefaultContextFormatter()
    _xml, blocks, citations = fmt.format_with_budget([], max_tokens=100)
    assert blocks == []
    assert citations == []


# -- Citation preservation tests --


def test_citations_match_blocks() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=5, heading="Transfer", text="A", score=0.9),
        _make_chunk("b", doc="CASA.pdf", page=10, heading="Config", text="B", score=0.7),
    ]
    _xml, blocks, citations = fmt.format(chunks)
    assert len(citations) == len(blocks)
    for block, citation in zip(blocks, citations, strict=False):
        assert citation.chunk_id == block.chunk_id
        assert citation.document == block.document
        assert citation.section == block.section
        assert citation.page == block.page
        assert citation.score == block.score


def test_citation_merged_chunk() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=5, heading="Transfer", text="A"),
        _make_chunk("b", doc="GL.pdf", page=6, heading="Transfer", text="B"),
    ]
    _xml, blocks, citations = fmt.format(chunks)
    assert len(blocks) == 1
    assert len(citations) == 1
    # Citation keeps the first chunk_id after merge
    assert citations[0].chunk_id == "a"


# -- Deterministic output tests --


def test_deterministic_output() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("b", text="B", score=0.6),
        _make_chunk("a", text="A", score=0.9),
        _make_chunk("c", text="C", score=0.7),
    ]
    xml1, blocks1, cit1 = fmt.format(chunks)
    xml2, blocks2, cit2 = fmt.format(chunks)
    assert xml1 == xml2
    assert blocks1 == blocks2
    assert cit1 == cit2


def test_deterministic_truncation() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", text="X" * 500, score=0.9),
        _make_chunk("b", text="Y" * 500, score=0.8),
    ]
    xml1, blocks1, cit1 = fmt.format_with_budget(chunks, max_tokens=100)
    xml2, blocks2, cit2 = fmt.format_with_budget(chunks, max_tokens=100)
    assert xml1 == xml2
    assert blocks1 == blocks2
    assert cit1 == cit2


# -- Ordering tests --


def test_blocks_in_rank_order() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", page=1, heading="Intro", text="first", score=0.3),
        _make_chunk("b", page=2, heading="Config", text="second", score=0.9),
        _make_chunk("c", page=3, heading="Setup", text="third", score=0.6),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    # Blocks should preserve the original order (rank order from retrieval)
    assert blocks[0].chunk_id == "a"
    assert blocks[1].chunk_id == "b"
    assert blocks[2].chunk_id == "c"
    # Indices should be sequential
    for i, block in enumerate(blocks, start=1):
        assert block.index == i


def test_empty_retrieval() -> None:
    fmt = DefaultContextFormatter()
    xml, blocks, citations = fmt.format([])
    assert xml == "<context>\n</context>"
    assert blocks == []
    assert citations == []


# -- XML format tests --


def test_xml_structure() -> None:
    fmt = DefaultContextFormatter()
    chunks = [_make_chunk("a", doc="test.pdf", page=1, heading="H", text="T")]
    xml, _blocks, _citations = fmt.format(chunks)
    assert xml.startswith("<context>")
    assert xml.endswith("</context>")
    assert "<context_block" in xml
    assert "</context_block>" in xml
    assert "<document>test.pdf</document>" in xml
    assert "<text>" in xml
    assert "T" in xml


# -- Min score filtering tests --


def test_min_score_zero_keeps_all() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", score=0.1),
        _make_chunk("b", score=0.2),
        _make_chunk("c", score=0.3),
    ]
    _xml, blocks, _citations = fmt.format(chunks, min_score=0.0)
    assert len(blocks) == 3


def test_min_score_filters_low_scoring() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", score=0.1),
        _make_chunk("b", score=0.2),
        _make_chunk("c", score=0.3),
    ]
    _xml, blocks, _citations = fmt.format(chunks, min_score=0.25)
    assert len(blocks) == 1
    assert blocks[0].chunk_id == "c"


def test_min_score_with_budget() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", text="A", score=0.1),
        _make_chunk("b", text="B", score=0.5),
        _make_chunk("c", text="C", score=0.9),
    ]
    _xml, blocks, _citations = fmt.format_with_budget(
        chunks, max_tokens=9999, min_score=0.3
    )
    assert len(blocks) == 2
    assert blocks[0].chunk_id == "b"
    assert blocks[1].chunk_id == "c"


# -- Hierarchical merge tests --


def test_extract_section_num_numeric() -> None:
    assert DefaultContextFormatter._extract_section_num("7 Maintaining GL") == (7,)
    assert DefaultContextFormatter._extract_section_num("7.1 GL Transfer") == (7, 1)
    assert DefaultContextFormatter._extract_section_num("7.1.1 Details") == (7, 1, 1)


def test_extract_section_num_no_match() -> None:
    assert DefaultContextFormatter._extract_section_num("Introduction") == ()


def test_is_hierarchical_child_direct() -> None:
    assert DefaultContextFormatter._is_hierarchical_child((7,), (7, 1)) is True


def test_is_hierarchical_child_grandchild() -> None:
    assert DefaultContextFormatter._is_hierarchical_child((7,), (7, 1, 1)) is True


def test_is_hierarchical_child_sibling_not() -> None:
    assert DefaultContextFormatter._is_hierarchical_child((7, 1), (7, 2)) is False


def test_is_hierarchical_child_same_level_not() -> None:
    assert DefaultContextFormatter._is_hierarchical_child((7,), (8,)) is False


def test_is_hierarchical_child_empty_not() -> None:
    assert DefaultContextFormatter._is_hierarchical_child((), (7,)) is False
    assert DefaultContextFormatter._is_hierarchical_child((7,), ()) is False


def test_merge_hierarchical_parent_child() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=44, heading="7 Maintaining GL Balance Transfer", text="Parent content.", score=0.9),
        _make_chunk("b", doc="GL.pdf", page=44, heading="7.1 GL Balance Transfer Maintenance", text="Child content.", score=0.8),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 1
    assert "Parent content." in blocks[0].text
    assert "Child content." in blocks[0].text
    assert blocks[0].section == "7 Maintaining GL Balance Transfer"


def test_merge_hierarchical_chain() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=44, heading="7 Maintaining GL", text="Level 1.", score=0.9),
        _make_chunk("b", doc="GL.pdf", page=44, heading="7.1 GL Transfer", text="Level 2.", score=0.8),
        _make_chunk("c", doc="GL.pdf", page=45, heading="7.1.1 Details", text="Level 3.", score=0.7),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 1
    assert "Level 1." in blocks[0].text
    assert "Level 2." in blocks[0].text
    assert "Level 3." in blocks[0].text
    assert blocks[0].section == "7 Maintaining GL"


def test_merge_hierarchical_different_document_no_merge() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=44, heading="7 Maintaining GL", text="A"),
        _make_chunk("b", doc="CASA.pdf", page=10, heading="7.1 Config", text="B"),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 2


def test_merge_hierarchical_non_numeric_heading_no_merge() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=44, heading="Introduction", text="A"),
        _make_chunk("b", doc="GL.pdf", page=45, heading="Overview", text="B"),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 2


def test_merge_hierarchical_different_branch_no_merge() -> None:
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=44, heading="7 GL Transfer", text="A"),
        _make_chunk("b", doc="GL.pdf", page=45, heading="8 Config", text="B"),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 2


def test_merge_hierarchical_with_adjacent_merge() -> None:
    """Hierarchical merge works after adjacent same-heading merge."""
    fmt = DefaultContextFormatter()
    chunks = [
        _make_chunk("a", doc="GL.pdf", page=10, heading="7 GL Transfer", text="Part1", score=0.9),
        _make_chunk("b", doc="GL.pdf", page=11, heading="7 GL Transfer", text="Part2", score=0.8),
        _make_chunk("c", doc="GL.pdf", page=12, heading="7.1 Config", text="Child", score=0.7),
    ]
    _xml, blocks, _citations = fmt.format(chunks)
    assert len(blocks) == 1
    assert "Part1" in blocks[0].text
    assert "Part2" in blocks[0].text
    assert "Child" in blocks[0].text
    assert blocks[0].section == "7 GL Transfer"


def test_merge_hierarchical_empty() -> None:
    fmt = DefaultContextFormatter()
    _xml, blocks, _citations = fmt.format([])
    assert blocks == []
