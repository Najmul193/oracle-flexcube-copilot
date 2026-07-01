"""Tests for section hierarchy builder."""

from __future__ import annotations

from datetime import UTC

from oracle_flexcube_copilot.enrichment.hierarchy import build_sections
from oracle_flexcube_copilot.ingestion.models import Document


class TestBuildSections:
    """Tests for :func:`build_sections`."""

    def test_returns_sections(self, sample_document: Document) -> None:
        """Should return a list of sections."""
        sections = build_sections(sample_document)
        assert len(sections) > 0

    def test_sections_have_titles(self, sample_document: Document) -> None:
        """Each section should have a title."""
        sections = build_sections(sample_document)
        for section in sections:
            assert section.title != ""

    def test_sections_have_page_ranges(self, sample_document: Document) -> None:
        """Each section should have page start and end."""
        sections = build_sections(sample_document)
        for section in sections:
            assert section.page_start >= 1
            assert section.page_end >= section.page_start

    def test_section_numbers(self, sample_document: Document) -> None:
        """Sections should have normalized numbers."""
        sections = build_sections(sample_document)
        numbers = [s.number for s in sections]
        assert "1" in numbers
        assert "1.1" in numbers
        assert "1.2" in numbers
        assert "2" in numbers

    def test_sections_have_block_references(self, sample_document: Document) -> None:
        """Sections should reference their blocks."""
        sections = build_sections(sample_document)
        for section in sections:
            assert len(section.block_ids) > 0

    def test_parent_child_relationships(self, sample_document: Document) -> None:
        """Sections should have parent/child relationships."""
        sections = build_sections(sample_document)
        # Get top-level section
        top_level = [s for s in sections if s.level == 1]
        for tl in top_level:
            for child_id in tl.child_ids:
                child = next(s for s in sections if s.id == child_id)
                assert child.parent_id == tl.id

    def test_word_counts(self, sample_document: Document) -> None:
        """Sections should have word counts."""
        sections = build_sections(sample_document)
        for section in sections:
            assert section.word_count >= 0

    def test_empty_document_creates_default(self) -> None:
        """A document with no headings should create a default section."""
        from datetime import datetime

        from oracle_flexcube_copilot.ingestion.models import (
            Block,
            Document,
            DocumentMetadata,
            Page,
            Paragraph,
        )

        doc = Document(
            id="test",
            filename="no_headings.pdf",
            absolute_path="/tmp/empty.pdf",
            sha256="abc",
            file_size_bytes=0,
            last_modified=datetime(2024, 1, 1, tzinfo=UTC),
            created_time=datetime(2024, 1, 1, tzinfo=UTC),
            metadata=DocumentMetadata(page_count=1, title="No Headings"),
            pages=[
                Page(
                    id="test:p1",
                    page_number=1,
                    blocks=[
                        Block(
                            id="test:p1:b0",
                            type="text",
                            block_index=0,
                            paragraphs=[Paragraph(text="Some content", index=0)],
                        )
                    ],
                )
            ],
        )
        sections = build_sections(doc)
        assert len(sections) == 1
        assert sections[0].title == "No Headings"

    def test_section_ids_are_unique(self, sample_document: Document) -> None:
        """All section IDs should be unique."""
        sections = build_sections(sample_document)
        ids = [s.id for s in sections]
        assert len(ids) == len(set(ids))

    def test_last_section_covers_end_of_document(self, sample_document: Document) -> None:
        """Last section's page_end should reach or exceed the document page count."""
        sections = build_sections(sample_document)
        last_section = sections[-1]
        assert last_section.page_end >= sample_document.metadata.page_count
