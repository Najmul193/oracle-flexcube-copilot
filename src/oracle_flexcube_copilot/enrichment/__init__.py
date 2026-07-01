"""Document enrichment engine — transforms parsed documents into semantically enriched documents.

Adds TOC, heading trees, sections, cross-references, and table structure.
"""

from __future__ import annotations

from oracle_flexcube_copilot.enrichment.models import (
    EnrichedBlock,
    EnrichedDocument,
    HeadingNode,
    Reference,
    Section,
    TableData,
)
from oracle_flexcube_copilot.enrichment.service import DocumentEnrichmentService

__all__ = [
    "DocumentEnrichmentService",
    "EnrichedBlock",
    "EnrichedDocument",
    "HeadingNode",
    "Reference",
    "Section",
    "TableData",
]
