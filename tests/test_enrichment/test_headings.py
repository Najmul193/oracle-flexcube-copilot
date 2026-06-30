"""Tests for heading normalization."""

from __future__ import annotations

from oracle_flexcube_copilot.enrichment.headings import (
    get_heading_path,
    heading_tree_to_flat,
    normalize_headings,
)
from oracle_flexcube_copilot.ingestion.models import Document


class TestNormalizeHeadings:
    """Tests for :func:`normalize_headings`."""

    def test_returns_heading_tree(self, sample_document: Document) -> None:
        """Should return a nested heading tree."""
        tree = normalize_headings(sample_document)
        assert len(tree) > 0
        # Top-level nodes
        assert tree[0].title == "Chapter 1: Product Definition"
        assert tree[0].level == 1

    def test_heading_tree_nesting(self, sample_document: Document) -> None:
        """Child headings should be nested under parents."""
        tree = normalize_headings(sample_document)
        ch1 = tree[0]
        assert len(ch1.children) == 2
        assert ch1.children[0].title == "1.1 Product Code"
        assert ch1.children[1].title == "1.2 Rate Code"

    def test_heading_numbers(self, sample_document: Document) -> None:
        """Headings should have normalized section numbers."""
        tree = normalize_headings(sample_document)
        assert tree[0].normalized_number == "1"
        assert tree[0].children[0].normalized_number == "1.1"
        assert tree[0].children[1].normalized_number == "1.2"

    def test_multiple_top_level(self, sample_document: Document) -> None:
        """Multiple top-level headings should be separate roots."""
        tree = normalize_headings(sample_document)
        assert len(tree) == 2
        assert tree[1].title == "Chapter 2: Configuration"
        assert tree[1].normalized_number == "2"

    def test_empty_document(self) -> None:
        """A document with no headings should return empty tree."""
        from oracle_flexcube_copilot.ingestion.models import Block, Document, DocumentMetadata, Page, Paragraph
        from datetime import datetime, timezone
        doc = Document(
            id="test", filename="empty.pdf", absolute_path="/tmp/empty.pdf",
            sha256="abc", file_size_bytes=0,
            last_modified=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            metadata=DocumentMetadata(page_count=1),
            pages=[Page(id="test:p1", page_number=1, blocks=[
                Block(id="test:p1:b0", type="text", block_index=0, paragraphs=[Paragraph(text="Some text", index=0)])
            ])],
        )
        tree = normalize_headings(doc)
        assert tree == []


class TestHeadingTreeToFlat:
    """Tests for :func:`heading_tree_to_flat`."""

    def test_flattens_tree(self, sample_document: Document) -> None:
        """Should flatten nested tree into document order."""
        tree = normalize_headings(sample_document)
        flat = heading_tree_to_flat(tree)
        assert len(flat) == 5
        assert flat[0].title == "Chapter 1: Product Definition"
        assert flat[1].title == "1.1 Product Code"
        assert flat[2].title == "1.2 Rate Code"
        assert flat[3].title == "Chapter 2: Configuration"
        assert flat[4].title == "2.1 Setup"


class TestGetHeadingPath:
    """Tests for :func:`get_heading_path`."""

    def test_returns_path_to_heading_block(self, sample_document: Document) -> None:
        """Should return heading path for a heading block."""
        tree = normalize_headings(sample_document)
        # Use a heading block ID
        heading_block_id = sample_document.pages[0].blocks[0].id
        path = get_heading_path(tree, heading_block_id)
        assert len(path) >= 1
        assert "Chapter 1: Product Definition" in path

    def test_unknown_block(self, sample_document: Document) -> None:
        """Unknown block should return empty path."""
        tree = normalize_headings(sample_document)
        path = get_heading_path(tree, "nonexistent")
        assert path == []


class TestHeadingTreeEdgeCases:
    """Additional edge case tests for heading normalization."""

    def test_three_levels_deep(self) -> None:
        """Should handle 3 levels of heading nesting."""
        from oracle_flexcube_copilot.ingestion.models import Block, Document, DocumentMetadata, Page, Paragraph
        from datetime import datetime, timezone
        sha = "threelevel"
        doc = Document(
            id=sha, filename="deep.pdf", absolute_path="/tmp/deep.pdf",
            sha256=sha, file_size_bytes=0,
            last_modified=datetime(2024, 1, 1, tzinfo=timezone.utc),
            created_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            metadata=DocumentMetadata(page_count=1),
            pages=[Page(id=f"{sha}:p1", page_number=1, blocks=[
                Block(id=f"{sha}:p1:b0", type="heading", level=1, block_index=0, paragraphs=[Paragraph(text="Chapter 1", index=0)]),
                Block(id=f"{sha}:p1:b1", type="heading", level=2, block_index=1, paragraphs=[Paragraph(text="Section 1.1", index=0)]),
                Block(id=f"{sha}:p1:b2", type="heading", level=3, block_index=2, paragraphs=[Paragraph(text="Subsection 1.1.1", index=0)]),
            ])],
        )
        tree = normalize_headings(doc)
        assert len(tree) == 1
        assert len(tree[0].children) == 1
        assert len(tree[0].children[0].children) == 1
        assert tree[0].children[0].children[0].normalized_number == "1.1.1"

    def test_flatten_empty_tree(self) -> None:
        """Flattening an empty tree should return an empty list."""
        assert heading_tree_to_flat([]) == []

    def test_block_ids_preserved_in_nodes(self, sample_document: Document) -> None:
        """Heading nodes should carry the block IDs from the source blocks."""
        tree = normalize_headings(sample_document)
        flat = heading_tree_to_flat(tree)
        for node in flat:
            assert len(node.block_ids) >= 1
            # Block IDs should be from the sample_document
            assert all(isinstance(bid, str) and len(bid) > 0 for bid in node.block_ids)