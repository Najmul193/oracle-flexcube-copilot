"""Document enrichment engine — transforms parsed documents into semantically enriched documents.

Adds TOC, heading trees, sections, cross-references, and table structure.
"""

from __future__ import annotations

from oracle_flexcube_copilot.enrichment.service import DocumentEnrichmentService
from oracle_flexcube_copilot.enrichment.models import (
    EnrichedDocument,
    Section,
    EnrichedBlock,
    HeadingNode,
    Reference,
    TableData,
)

__all__ = [
    "DocumentEnrichmentService",
    "EnrichedDocument",
    "Section",
    "EnrichedBlock",
    "HeadingNode",
    "Reference",
    "TableData",
]
