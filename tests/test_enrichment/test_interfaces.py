"""Tests for enrichment protocol definitions."""

from __future__ import annotations

from oracle_flexcube_copilot.enrichment.interfaces import (
    DocumentEnricher,
    HeadingNormalizer,
    HierarchyBuilder,
    ReferenceExtractor,
    SectionBuilder,
    TOCExtractor,
    TableExtractor,
)
from oracle_flexcube_copilot.enrichment.service import DocumentEnrichmentService
from oracle_flexcube_copilot.enrichment.models import EnrichedDocument
from oracle_flexcube_copilot.ingestion.models import Document


class TestProtocolsAreRuntimeCheckable:
    """All protocols should be decorated with @runtime_checkable."""

    def test_document_enricher_is_runtime_checkable(self) -> None:
        """DocumentEnricher should support isinstance checks."""
        assert isinstance(DocumentEnrichmentService(), DocumentEnricher)

    def test_toc_extractor_is_runtime_checkable(self) -> None:
        """TOCExtractor protocol should be runtime-checkable."""
        class _Impl:
            def extract(self, document: Document) -> list[dict]:
                return []

        assert isinstance(_Impl(), TOCExtractor)

    def test_heading_normalizer_is_runtime_checkable(self) -> None:
        """HeadingNormalizer protocol should be runtime-checkable."""
        class _Impl:
            def normalize(self, document: Document) -> list[dict]:
                return []

        assert isinstance(_Impl(), HeadingNormalizer)

    def test_section_builder_is_runtime_checkable(self) -> None:
        """SectionBuilder protocol should be runtime-checkable."""
        class _Impl:
            def build(self, document: Document) -> list[dict]:
                return []

        assert isinstance(_Impl(), SectionBuilder)

    def test_reference_extractor_is_runtime_checkable(self) -> None:
        """ReferenceExtractor protocol should be runtime-checkable."""
        class _Impl:
            def extract(self, document: Document) -> list[dict]:
                return []

        assert isinstance(_Impl(), ReferenceExtractor)

    def test_table_extractor_is_runtime_checkable(self) -> None:
        """TableExtractor protocol should be runtime-checkable."""
        class _Impl:
            def extract(self, document: Document) -> list[dict]:
                return []

        assert isinstance(_Impl(), TableExtractor)

    def test_hierarchy_builder_is_runtime_checkable(self) -> None:
        """HierarchyBuilder protocol should be runtime-checkable."""
        class _Impl:
            def build(self, document: Document) -> list[dict]:
                return []

        assert isinstance(_Impl(), HierarchyBuilder)


class TestProtocolConformance:
    """Service implementations should satisfy their protocols."""

    def test_service_satisfies_enricher_protocol(self, sample_document: Document) -> None:
        """DocumentEnrichmentService should produce an EnrichedDocument."""
        service: DocumentEnricher = DocumentEnrichmentService()
        result = service.enrich(sample_document)
        assert isinstance(result, EnrichedDocument)

    def test_non_conforming_class_fails(self) -> None:
        """A class without the right method should not satisfy the protocol."""
        class _NotAnEnricher:
            def wrong_method(self) -> None:
                pass

        assert not isinstance(_NotAnEnricher(), DocumentEnricher)
