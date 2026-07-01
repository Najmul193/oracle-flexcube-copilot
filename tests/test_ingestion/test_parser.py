"""Tests for the PDF parser."""

from __future__ import annotations

from pathlib import Path

import fitz

from oracle_flexcube_copilot.ingestion.parser import parse_document_metadata, parse_pages


class TestParseDocumentMetadata:
    """Tests for :func:`parse_document_metadata`."""

    def test_extracts_metadata(self, valid_pdf_path: Path) -> None:
        """Metadata should be extracted from a valid PDF."""
        doc = fitz.open(str(valid_pdf_path))
        metadata = parse_document_metadata(doc)
        doc.close()

        assert metadata.page_count == 3
        assert isinstance(metadata.title, str)
        assert isinstance(metadata.author, str)

    def test_empty_metadata(self, multi_page_pdf_path: Path) -> None:
        """A PDF without metadata should return defaults."""
        doc = fitz.open(str(multi_page_pdf_path))
        metadata = parse_document_metadata(doc)
        doc.close()

        assert metadata.page_count == 5
        assert metadata.title == ""
        assert metadata.author == ""

    def test_page_count_matches(self, valid_pdf_path: Path) -> None:
        """Page count should match the actual number of pages."""
        doc = fitz.open(str(valid_pdf_path))
        metadata = parse_document_metadata(doc)
        doc.close()
        assert metadata.page_count == 3


class TestParsePages:
    """Tests for :func:`parse_pages`."""

    def test_parses_all_pages(self, valid_pdf_path: Path) -> None:
        """All pages should be parsed."""
        doc = fitz.open(str(valid_pdf_path))
        pages = list(parse_pages(doc))
        doc.close()
        assert len(pages) == 3

    def test_page_numbers_are_one_based(self, valid_pdf_path: Path) -> None:
        """Page numbers should start at 1."""
        doc = fitz.open(str(valid_pdf_path))
        pages = list(parse_pages(doc))
        doc.close()
        assert pages[0].page_number == 1
        assert pages[1].page_number == 2
        assert pages[2].page_number == 3

    def test_pages_have_blocks(self, valid_pdf_path: Path) -> None:
        """Each page should have at least one block."""
        doc = fitz.open(str(valid_pdf_path))
        pages = list(parse_pages(doc))
        doc.close()
        for page in pages:
            assert len(page.blocks) > 0

    def test_blocks_have_paragraphs(self, valid_pdf_path: Path) -> None:
        """Blocks should contain paragraphs with text."""
        doc = fitz.open(str(valid_pdf_path))
        pages = list(parse_pages(doc))
        doc.close()
        for page in pages:
            for block in page.blocks:
                assert len(block.paragraphs) > 0
                for para in block.paragraphs:
                    assert len(para.text) > 0

    def test_word_and_character_counts(self, valid_pdf_path: Path) -> None:
        """Word and character counts should be positive."""
        doc = fitz.open(str(valid_pdf_path))
        pages = list(parse_pages(doc))
        doc.close()
        for page in pages:
            assert page.word_count > 0
            assert page.character_count > 0

    def test_heading_detection(self, multi_page_pdf_path: Path) -> None:
        """Larger font blocks should be detected as headings."""
        doc = fitz.open(str(multi_page_pdf_path))
        pages = list(parse_pages(doc))
        doc.close()

        # First page should have a heading (fontsize 18 vs body 11)
        first_page = pages[0]
        heading_blocks = [b for b in first_page.blocks if b.type == "heading"]
        assert len(heading_blocks) > 0

    def test_multi_page_parsing(self, multi_page_pdf_path: Path) -> None:
        """A 5-page PDF should yield 5 pages."""
        doc = fitz.open(str(multi_page_pdf_path))
        pages = list(parse_pages(doc))
        doc.close()
        assert len(pages) == 5

    def test_blocks_have_block_index(self, valid_pdf_path: Path) -> None:
        """Each block should have a sequential block_index."""
        doc = fitz.open(str(valid_pdf_path))
        pages = list(parse_pages(doc))
        doc.close()
        for page in pages:
            for i, block in enumerate(page.blocks):
                assert block.block_index == i
