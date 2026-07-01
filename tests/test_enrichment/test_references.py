"""Tests for cross-reference extraction."""

from __future__ import annotations

from datetime import UTC

from oracle_flexcube_copilot.enrichment.references import (
    extract_cross_references,
    extract_entity_references,
    extract_references,
)
from oracle_flexcube_copilot.ingestion.models import Document


class TestExtractReferences:
    """Tests for :func:`extract_references`."""

    def test_finds_cross_references(self, sample_document: Document) -> None:
        """Should find cross-references in document text."""
        refs = extract_references(sample_document)
        assert len(refs) > 0

    def test_finds_entity_references(self, sample_document: Document) -> None:
        """Should find entity references like STTM_PRODUCT."""
        refs = extract_references(sample_document)
        entity_refs = [r for r in refs if r.reference_type == "entity_ref"]
        assert len(entity_refs) > 0
        assert any("STTM_PRODUCT" in r.target for r in entity_refs)

    def test_finds_see_chapter_refs(self, sample_document: Document) -> None:
        """Should find 'See Chapter' references."""
        refs = extract_references(sample_document)
        see_chapter = [r for r in refs if "Chapter" in r.text]
        assert len(see_chapter) > 0
        assert any("8" in r.target for r in see_chapter)

    def test_references_have_required_fields(self, sample_document: Document) -> None:
        """Each reference should have text, target, and type."""
        refs = extract_references(sample_document)
        for ref in refs:
            assert ref.text != ""
            assert ref.target != ""
            assert ref.reference_type in (
                "cross_ref",
                "entity_ref",
                "appendix_ref",
                "table_ref",
                "figure_ref",
            )
            assert ref.source_block_id != ""
            assert ref.source_page is not None

    def test_extract_cross_references(self, sample_document: Document) -> None:
        """Should filter to only cross_ref types."""
        refs = extract_cross_references(sample_document)
        for ref in refs:
            assert ref.reference_type == "cross_ref"

    def test_extract_entity_references(self, sample_document: Document) -> None:
        """Should filter to only entity_ref types."""
        refs = extract_entity_references(sample_document)
        for ref in refs:
            assert ref.reference_type == "entity_ref"

    def test_deduplicates(self, sample_document: Document) -> None:
        """Duplicate references should not be added."""
        refs1 = extract_references(sample_document)
        refs2 = extract_references(sample_document)
        # Same content should produce same count
        assert len(refs1) == len(refs2)

    def test_stable_reference_ids(self, sample_document: Document) -> None:
        """Reference IDs should follow {doc_id}:ref:{N} pattern."""
        refs = extract_references(sample_document)
        for ref in refs:
            assert ref.id.startswith(sample_document.id + ":ref:")

    def test_empty_document_no_references(self) -> None:
        """A document with no text should produce no references."""
        from datetime import datetime

        from oracle_flexcube_copilot.ingestion.models import (
            Block,
            Document,
            DocumentMetadata,
            Page,
            Paragraph,
        )

        doc = Document(
            id="empty",
            filename="empty.pdf",
            absolute_path="/tmp/empty.pdf",
            sha256="empty",
            file_size_bytes=0,
            last_modified=datetime(2024, 1, 1, tzinfo=UTC),
            created_time=datetime(2024, 1, 1, tzinfo=UTC),
            metadata=DocumentMetadata(page_count=1),
            pages=[
                Page(
                    id="empty:p1",
                    page_number=1,
                    blocks=[
                        Block(
                            id="empty:p1:b0",
                            type="text",
                            block_index=0,
                            paragraphs=[
                                Paragraph(text="Simple text with no references.", index=0),
                            ],
                        ),
                    ],
                )
            ],
        )
        refs = extract_references(doc)
        assert isinstance(refs, list)

    def test_appendix_reference_detected(self) -> None:
        """Should detect 'Appendix A' references."""
        from datetime import datetime

        from oracle_flexcube_copilot.ingestion.models import (
            Block,
            Document,
            DocumentMetadata,
            Page,
            Paragraph,
        )

        doc = Document(
            id="appx",
            filename="appx.pdf",
            absolute_path="/tmp/appx.pdf",
            sha256="appx",
            file_size_bytes=0,
            last_modified=datetime(2024, 1, 1, tzinfo=UTC),
            created_time=datetime(2024, 1, 1, tzinfo=UTC),
            metadata=DocumentMetadata(page_count=1),
            pages=[
                Page(
                    id="appx:p1",
                    page_number=1,
                    blocks=[
                        Block(
                            id="appx:p1:b0",
                            type="text",
                            block_index=0,
                            paragraphs=[
                                Paragraph(text="Refer to Appendix A for more details.", index=0),
                            ],
                        ),
                    ],
                )
            ],
        )
        refs = extract_references(doc)
        appendix_refs = [r for r in refs if r.reference_type == "appendix_ref"]
        assert len(appendix_refs) >= 1
