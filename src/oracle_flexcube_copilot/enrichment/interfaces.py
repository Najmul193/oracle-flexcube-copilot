"""Protocol definitions for the document enrichment pipeline.

Allows swapping implementations (e.g. PyMuPDF -> other parser, rule-based -> LLM-based).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from oracle_flexcube_copilot.enrichment.models import EnrichedDocument
from oracle_flexcube_copilot.ingestion.models import Document


@runtime_checkable
class DocumentEnricher(Protocol):
    """Main enrichment interface — transforms a parsed Document into an EnrichedDocument."""

    def enrich(self, document: Document) -> EnrichedDocument:
        """Enrich a parsed document with semantic structure.

        Args:
            document: A fully parsed ``Document`` from the ingestion pipeline.

        Returns:
            An ``EnrichedDocument`` with sections, heading tree, references, and tables.
        """
        ...


@runtime_checkable
class TOCExtractor(Protocol):
    """Extracts table of contents from a document."""

    def extract(self, document: Document) -> list[dict]:
        """Extract TOC entries from a document.

        Args:
            document: A parsed ``Document``.

        Returns:
            List of TOC entry dicts with level, title, page.
        """
        ...


@runtime_checkable
class HeadingNormalizer(Protocol):
    """Normalizes heading hierarchy across a document."""

    def normalize(self, document: Document) -> list[dict]:
        """Normalize headings and return a heading tree.

        Args:
            document: A parsed ``Document``.

        Returns:
            A nested heading tree structure.
        """
        ...


@runtime_checkable
class SectionBuilder(Protocol):
    """Builds sections by assigning blocks to their parent headings."""

    def build(self, document: Document) -> list[dict]:
        """Build sections from document headings and blocks.

        Args:
            document: A parsed ``Document``.

        Returns:
            List of section dicts with title, level, blocks, page range.
        """
        ...


@runtime_checkable
class ReferenceExtractor(Protocol):
    """Extracts cross-references from document text."""

    def extract(self, document: Document) -> list[dict]:
        """Find all cross-references in the document.

        Args:
            document: A parsed ``Document``.

        Returns:
            List of reference dicts with text, target, page.
        """
        ...


@runtime_checkable
class TableExtractor(Protocol):
    """Extracts table structures from document blocks."""

    def extract(self, document: Document) -> list[dict]:
        """Extract table structures from a document.

        Args:
            document: A parsed ``Document``.

        Returns:
            List of table dicts with headers, rows, page, title.
        """
        ...


@runtime_checkable
class HierarchyBuilder(Protocol):
    """Builds hierarchical sections from document headings and blocks."""

    def build(self, document: Document) -> list[dict]:
        """Build hierarchical sections from document structure.

        Args:
            document: A parsed ``Document``.

        Returns:
            List of section dicts with title, level, parent, children, page range.
        """
        ...
