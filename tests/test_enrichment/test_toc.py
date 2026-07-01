"""Tests for TOC extraction."""

from __future__ import annotations

from oracle_flexcube_copilot.enrichment.toc import extract_toc, get_toc_depth, get_toc_section_count
from oracle_flexcube_copilot.ingestion.models import Document, TOCEntry


class TestExtractTOC:
    """Tests for :func:`extract_toc`."""

    def test_uses_existing_toc(self, sample_document: Document) -> None:
        """Should return the existing TOC when present."""
        toc = extract_toc(sample_document)
        assert len(toc) == 5
        assert toc[0].title == "Chapter 1: Product Definition"
        assert toc[0].level == 1

    def test_toc_entries_have_required_fields(self, sample_document: Document) -> None:
        """Each TOC entry should have level, title, and page."""
        toc = extract_toc(sample_document)
        for entry in toc:
            assert isinstance(entry.level, int)
            assert isinstance(entry.title, str)
            assert isinstance(entry.page, int)

    def test_reconstructs_from_headings(self, sample_document: Document) -> None:
        """Should reconstruct TOC from heading blocks when no TOC exists."""
        doc = sample_document
        doc.table_of_contents = []  # Clear TOC
        toc = extract_toc(doc)
        assert len(toc) > 0
        assert all(isinstance(e, TOCEntry) for e in toc)


class TestGetTOCDepth:
    """Tests for :func:`get_toc_depth`."""

    def test_returns_max_level(self, sample_document: Document) -> None:
        """Should return the maximum level in the TOC."""
        toc = extract_toc(sample_document)
        depth = get_toc_depth(toc)
        assert depth == 2

    def test_empty_toc(self) -> None:
        """Empty TOC should return 0."""
        assert get_toc_depth([]) == 0


class TestGetTOCSectionCount:
    """Tests for :func:`get_toc_section_count`."""

    def test_counts_all(self, sample_document: Document) -> None:
        """Should count all entries."""
        toc = extract_toc(sample_document)
        assert get_toc_section_count(toc) == 5

    def test_counts_by_level(self, sample_document: Document) -> None:
        """Should filter by level."""
        toc = extract_toc(sample_document)
        assert get_toc_section_count(toc, level=1) == 2
        assert get_toc_section_count(toc, level=2) == 3

    def test_empty_list_count(self) -> None:
        """Empty list should return 0."""
        assert get_toc_section_count([]) == 0
        assert get_toc_section_count([], level=1) == 0


class TestTOCReconstruction:
    """Tests for TOC reconstruction from heading blocks."""

    def test_reconstructed_toc_preserves_page_order(self, sample_document: Document) -> None:
        """Reconstructed TOC entries should be ordered by page number."""
        doc = sample_document
        doc.table_of_contents = []
        toc = extract_toc(doc)
        pages = [e.page for e in toc]
        assert pages == sorted(pages)

    def test_single_level_toc_depth(self) -> None:
        """TOC with only level-1 entries should have depth 1."""
        toc = [
            TOCEntry(level=1, title="Chapter 1", page=1),
            TOCEntry(level=1, title="Chapter 2", page=5),
        ]
        assert get_toc_depth(toc) == 1
