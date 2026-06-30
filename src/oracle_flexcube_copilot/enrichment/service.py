"""Document enrichment service — orchestrates the full enrichment pipeline."""

from __future__ import annotations

import logging
import time
from typing import Any

from oracle_flexcube_copilot.enrichment.headings import (
    get_heading_path,
    heading_tree_to_flat,
    normalize_headings,
)
from oracle_flexcube_copilot.enrichment.hierarchy import build_sections
from oracle_flexcube_copilot.enrichment.models import (
    EnrichedBlock,
    EnrichedDocument,
    HeadingNode,
    Reference,
    Section,
    TableData,
)
from oracle_flexcube_copilot.enrichment.references import extract_references
from oracle_flexcube_copilot.enrichment.tables import extract_tables
from oracle_flexcube_copilot.enrichment.toc import extract_toc
from oracle_flexcube_copilot.ingestion.models import Document

logger = logging.getLogger("oracle_flexcube_copilot.enrichment.service")


class DocumentEnrichmentService:
    """Orchestrates the document enrichment pipeline.

    Transforms a parsed ``Document`` into an ``EnrichedDocument`` with:
    - Extracted table of contents
    - Normalized heading tree
    - Hierarchical sections
    - Cross-references
    - Table structures
    - Blocks with section context

    Usage::

        service = DocumentEnrichmentService()
        enriched = service.enrich(document)
    """

    def enrich(self, document: Document) -> EnrichedDocument:
        """Run the full enrichment pipeline on a parsed document.

        Args:
            document: A parsed ``Document`` from the ingestion pipeline.

        Returns:
            An ``EnrichedDocument`` with all enrichment data.
        """
        logger.info("Enriching document: %s", document.filename)
        start = time.time()

        # Step 1: Extract TOC
        toc = extract_toc(document)

        # Step 2: Build heading tree
        heading_tree = normalize_headings(document)

        # Step 3: Build sections
        sections = build_sections(document)

        # Step 4: Extract cross-references
        references = extract_references(document)

        # Step 5: Extract tables
        tables = extract_tables(document)

        # Step 6: Build enriched blocks with section context
        enriched_blocks = self._build_enriched_blocks(document, sections, heading_tree)

        enriched = EnrichedDocument(
            document_id=document.id,
            filename=document.filename,
            title=document.metadata.title or document.filename,
            total_pages=document.metadata.page_count,
            total_words=document.total_words,
            toc=toc,
            heading_tree=heading_tree,
            sections=sections,
            enriched_blocks=enriched_blocks,
            cross_references=references,
            tables=tables,
        )

        elapsed = time.time() - start
        logger.info(
            "Enriched %s — %d sections, %d refs, %d tables in %.2fs",
            document.filename,
            len(sections),
            len(references),
            len(tables),
            elapsed,
        )

        return enriched

    def _build_enriched_blocks(
        self,
        document: Document,
        sections: list[Section],
        heading_tree: list[HeadingNode],
    ) -> list[EnrichedBlock]:
        """Build enriched blocks by assigning section context and heading paths.

        Args:
            document: The parsed document.
            sections: Built sections.
            heading_tree: The heading tree.

        Returns:
            List of EnrichedBlock instances.
        """
        # Build lookup: block_id -> section_id
        block_to_section: dict[str, str] = {}
        for section in sections:
            for bid in section.block_ids:
                block_to_section[bid] = section.id

        enriched: list[EnrichedBlock] = []
        flat_headings = heading_tree_to_flat(heading_tree)

        for page in document.pages:
            for block in page.blocks:
                section_id = block_to_section.get(block.id)
                heading_path = get_heading_path(heading_tree, block.id)
                depth = len(heading_path)

                enriched.append(
                    EnrichedBlock(
                        id=block.id,
                        section_id=section_id,
                        heading_path=heading_path,
                        depth=depth,
                    )
                )

        return enriched