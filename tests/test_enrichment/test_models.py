"""Tests for enrichment Pydantic models."""

from __future__ import annotations

from oracle_flexcube_copilot.enrichment.models import (
    EnrichedBlock,
    EnrichedDocument,
    HeadingNode,
    Reference,
    Section,
    TableData,
)


class TestEnrichedDocument:
    """Tests for :class:`EnrichedDocument`."""

    def test_construction_with_all_fields(self) -> None:
        """Should construct with all fields provided."""
        doc = EnrichedDocument(
            document_id="sha256_abc",
            filename="test.pdf",
            title="Test Doc",
            total_pages=10,
            total_words=5000,
        )
        assert doc.document_id == "sha256_abc"
        assert doc.filename == "test.pdf"
        assert doc.total_pages == 10
        assert doc.total_words == 5000

    def test_default_lists_are_empty(self) -> None:
        """All list fields should default to empty lists."""
        doc = EnrichedDocument(document_id="x", filename="x.pdf")
        assert doc.toc == []
        assert doc.heading_tree == []
        assert doc.sections == []
        assert doc.enriched_blocks == []
        assert doc.cross_references == []
        assert doc.tables == []

    def test_ingestion_timestamp_is_iso_string(self) -> None:
        """ingestion_timestamp should be a valid ISO 8601 string."""
        doc = EnrichedDocument(document_id="x", filename="x.pdf")
        assert isinstance(doc.ingestion_timestamp, str)
        assert "T" in doc.ingestion_timestamp  # ISO 8601 has T separator


class TestSection:
    """Tests for :class:`Section`."""

    def test_defaults(self) -> None:
        """Section should have correct defaults."""
        section = Section(title="Test", level=1, page_start=1, page_end=5)
        assert section.id == ""
        assert section.number == ""
        assert section.parent_id is None
        assert section.child_ids == []
        assert section.block_ids == []
        assert section.word_count == 0


class TestReference:
    """Tests for :class:`Reference`."""

    def test_defaults(self) -> None:
        """Reference should have correct defaults."""
        ref = Reference(text="See Chapter 8")
        assert ref.id == ""
        assert ref.target == ""
        assert ref.reference_type == "cross_ref"
        assert ref.source_block_id == ""
        assert ref.source_page is None


class TestTableData:
    """Tests for :class:`TableData`."""

    def test_defaults(self) -> None:
        """TableData should have correct defaults."""
        td = TableData()
        assert td.id == ""
        assert td.source_block_id == ""
        assert td.page == 0
        assert td.title == ""
        assert td.headers == []
        assert td.rows == []
        assert td.num_rows == 0
        assert td.num_cols == 0


class TestHeadingNode:
    """Tests for :class:`HeadingNode`."""

    def test_children_default_empty(self) -> None:
        """Children should default to empty list."""
        node = HeadingNode(title="Chapter 1", level=1)
        assert node.children == []
        assert node.block_ids == []

    def test_nested_children(self) -> None:
        """Should support nested children."""
        child = HeadingNode(title="Section 1.1", level=2)
        parent = HeadingNode(title="Chapter 1", level=1, children=[child])
        assert len(parent.children) == 1
        assert parent.children[0].title == "Section 1.1"


class TestEnrichedBlock:
    """Tests for :class:`EnrichedBlock`."""

    def test_defaults(self) -> None:
        """EnrichedBlock should have correct defaults."""
        eb = EnrichedBlock(id="abc:p1:b0")
        assert eb.section_id is None
        assert eb.heading_path == []
        assert eb.depth == 0
