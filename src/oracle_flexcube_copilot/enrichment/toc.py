"""TOC extraction — extracts and normalizes table of contents from parsed documents."""

from __future__ import annotations

import logging

from oracle_flexcube_copilot.ingestion.models import Document, TOCEntry

logger = logging.getLogger("oracle_flexcube_copilot.enrichment.toc")


def extract_toc(document: Document) -> list[TOCEntry]:
    """Extract the table of contents from a parsed document.

    Uses the TOC already parsed during ingestion. If the TOC is empty,
    attempts to reconstruct it from heading blocks.

    Args:
        document: A parsed ``Document`` with optional ``table_of_contents``.

    Returns:
        A list of ``TOCEntry`` instances.
    """
    if document.table_of_contents:
        logger.info("Using existing TOC with %d entries", len(document.table_of_contents))
        return document.table_of_contents

    # Reconstruct TOC from heading blocks
    logger.info("No TOC found, reconstructing from heading blocks")
    toc: list[TOCEntry] = []
    for page in document.pages:
        for block in page.blocks:
            if block.type == "heading" and block.level is not None:
                title = " ".join(p.text for p in block.paragraphs)
                toc.append(TOCEntry(level=block.level, title=title, page=page.page_number))

    return toc


def get_toc_depth(toc: list[TOCEntry]) -> int:
    """Return the maximum depth of the TOC hierarchy.

    Args:
        toc: List of TOC entries.

    Returns:
        Maximum level value (0 if empty).
    """
    if not toc:
        return 0
    return max(e.level for e in toc)


def get_toc_section_count(toc: list[TOCEntry], level: int | None = None) -> int:
    """Count TOC entries, optionally filtered by level.

    Args:
        toc: List of TOC entries.
        level: If provided, only count entries at this level.

    Returns:
        Number of matching entries.
    """
    if level is not None:
        return sum(1 for e in toc if e.level == level)
    return len(toc)
