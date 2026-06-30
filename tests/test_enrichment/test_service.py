"""Tests for the DocumentEnrichmentService."""

from __future__ import annotations

from oracle_flexcube_copilot.enrichment.service import DocumentEnrichmentService
from oracle_flexcube_copilot.enrichment.models import EnrichedDocument, Section, HeadingNode
from oracle_flexcube_copilot.enrichment.interfaces import DocumentEnricher
from oracle_flexcube_copilot.ingestion.models import Document


class TestDocumentEnrichmentService:
    """Tests for :class:`DocumentEnrichmentService`."""

    def test_enrich_returns_enriched_document(self, sample_document: Document) -> None:
        """Enrich should return an EnrichedDocument."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert isinstance(enriched, EnrichedDocument)

    def test_enriched_has_sections(self, sample_document: Document) -> None:
        """Enriched document should have sections."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert len(enriched.sections) > 0

    def test_enriched_has_toc(self, sample_document: Document) -> None:
        """Enriched document should have TOC."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert len(enriched.toc) > 0

    def test_enriched_has_heading_tree(self, sample_document: Document) -> None:
        """Enriched document should have heading tree."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert len(enriched.heading_tree) > 0

    def test_enriched_has_cross_references(self, sample_document: Document) -> None:
        """Enriched document should have cross-references."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert len(enriched.cross_references) > 0

    def test_enriched_metadata(self, sample_document: Document) -> None:
        """Enriched document should retain document metadata."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert enriched.document_id == sample_document.id
        assert enriched.filename == sample_document.filename
        assert enriched.title == sample_document.metadata.title
        assert enriched.total_pages == sample_document.metadata.page_count

    def test_enriched_blocks_have_section_context(self, sample_document: Document) -> None:
        """Enriched blocks should have section_id assigned."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        for eb in enriched.enriched_blocks:
            assert eb.section_id is not None

    def test_enriched_blocks_have_heading_path(self, sample_document: Document) -> None:
        """Enriched blocks should have heading paths."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        for eb in enriched.enriched_blocks:
            assert isinstance(eb.heading_path, list)

    def test_enriched_retains_pages(self, sample_document: Document) -> None:
        """Enriched document should retain raw pages."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert len(enriched.pages) == len(sample_document.pages)

    def test_entity_extraction_works(self, sample_document: Document) -> None:
        """Enriched document should have oracle_entities extracted."""
        # sample_document has "STTM_PRODUCT" in it (from conftest.py)
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert len(enriched.oracle_entities) >= 1
        assert any(e.name == "STTM_PRODUCT" for e in enriched.oracle_entities)

    def test_classification_works(self, sample_document: Document) -> None:
        """Document should be classified."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert isinstance(enriched.module_classification, str)

    def test_implements_protocol(self, sample_document: Document) -> None:
        """DocumentEnrichmentService should satisfy the DocumentEnricher protocol."""
        service = DocumentEnrichmentService()
        assert isinstance(service, DocumentEnricher)

    def test_section_parent_child(self, sample_document: Document) -> None:
        """Sections should have proper parent/child relationships."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        for section in enriched.sections:
            if section.parent_id:
                parent = next(s for s in enriched.sections if s.id == section.parent_id)
                assert section.id in parent.child_ids

    def test_total_words_populated(self, sample_document: Document) -> None:
        """Enriched document should have a non-zero total_words value."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert enriched.total_words > 0

    def test_ingestion_timestamp_is_iso(self, sample_document: Document) -> None:
        """ingestion_timestamp should be a valid ISO format string."""
        service = DocumentEnrichmentService()
        enriched = service.enrich(sample_document)
        assert isinstance(enriched.ingestion_timestamp, str)
        assert "T" in enriched.ingestion_timestamp

    def test_idempotent_enrichment(self, sample_document: Document) -> None:
        """Enriching the same document twice should produce consistent results."""
        service = DocumentEnrichmentService()
        enriched1 = service.enrich(sample_document)
        enriched2 = service.enrich(sample_document)
        assert enriched1.document_id == enriched2.document_id
        assert len(enriched1.sections) == len(enriched2.sections)
        assert len(enriched1.cross_references) == len(enriched2.cross_references)
        assert len(enriched1.enriched_blocks) == len(enriched2.enriched_blocks)
        # Section IDs should match
        for s1, s2 in zip(enriched1.sections, enriched2.sections):
            assert s1.id == s2.id
            assert s1.title == s2.title