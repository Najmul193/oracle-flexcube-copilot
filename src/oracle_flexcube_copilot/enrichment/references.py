"""Cross-reference extraction — detects cross-references in document text."""

from __future__ import annotations

import logging
import re

from oracle_flexcube_copilot.enrichment.models import Reference
from oracle_flexcube_copilot.ingestion.models import Document

logger = logging.getLogger("oracle_flexcube_copilot.enrichment.references")

# Patterns for detecting Oracle FLEXCUBE manual cross-references
REFERENCE_PATTERNS: list[tuple[str, str]] = [
    (r"See\s+(Chapter|Section|Appendix)\s+(\d+(?:\.\d+)*)", "cross_ref"),
    (r"Refer\s+to\s+([A-Z][A-Z_0-9]+)", "entity_ref"),
    (r"(?:see|refer to)\s+(Chapter|Section)\s+(\d+(?:\.\d+)*)", "cross_ref"),
    (r"Refer\s+Chapter\s+(\d+(?:\.\d+)*)", "cross_ref"),
    (r"as\s+described\s+in\s+(Chapter|Section)\s+(\d+(?:\.\d+)*)", "cross_ref"),
    (r"For\s+more\s+information.*?(?:Chapter|Section)\s+(\d+(?:\.\d+)*)", "cross_ref"),
    (
        r"(?:'s\s*interface|'s\s*module|'s\s*functionality)\s+(?:is\s+)?(?:explained|described|covered)\s+in\s+(Chapter|Section)\s+(\d+(?:\.\d+)*)",
        "cross_ref",
    ),
    (r"[A-Z]{2,}_[A-Z0-9_]+(?:-[A-Z0-9_]+)*", "entity_ref"),  # STTM_PRODUCT, etc.
    (r"Appendix\s+([A-Z])", "appendix_ref"),
    (r"Table\s+(\d+(?:\.\d+)*)", "table_ref"),
    (r"Figure\s+(\d+(?:\.\d+)*)", "figure_ref"),
]


def extract_references(document: Document) -> list[Reference]:
    """Find all cross-references in a document by scanning block text.

    Args:
        document: A parsed ``Document``.

    Returns:
        A list of ``Reference`` instances.
    """
    references: list[Reference] = []
    seen: set[str] = set()
    ref_counter = 0

    for page in document.pages:
        for block in page.blocks:
            for para in block.paragraphs:
                refs = _find_references_in_text(para.text, block.id, page.page_number)
                for ref in refs:
                    dedup_key = f"{ref.target}:{ref.reference_type}:{ref.source_page}"
                    if dedup_key not in seen:
                        seen.add(dedup_key)
                        ref_counter += 1
                        ref.id = f"{document.id}:ref:{ref_counter}"
                        references.append(ref)

    logger.info("Found %d cross-references", len(references))
    return references


def _find_references_in_text(text: str, block_id: str, page: int) -> list[Reference]:
    """Find all references in a single text string.

    Args:
        text: The text to scan.
        block_id: The source block ID.
        page: The page number.

    Returns:
        List of Reference instances found.
    """
    refs: list[Reference] = []

    for pattern, ref_type in REFERENCE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            groups = match.groups()
            if ref_type == "entity_ref" and len(match.groups()) >= 1:
                target = (
                    match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                )
            elif len(groups) >= 2:
                target = f"{groups[0]} {groups[1]}"
            else:
                target = groups[0] if groups else match.group(0)

            refs.append(
                Reference(
                    id="",
                    text=match.group(0).strip(),
                    target=target.strip(),
                    reference_type=ref_type,
                    source_block_id=block_id,
                    source_page=page,
                )
            )

    return refs


def extract_entity_references(document: Document) -> list[Reference]:
    """Extract only entity references (Oracle FLEXCUBE table/field names).

    Args:
        document: A parsed ``Document``.

    Returns:
        List of entity-type Reference instances.
    """
    return [ref for ref in extract_references(document) if ref.reference_type == "entity_ref"]


def extract_cross_references(document: Document) -> list[Reference]:
    """Extract only cross-references (See Chapter, Refer to Section, etc.).

    Args:
        document: A parsed ``Document``.

    Returns:
        List of cross_ref-type Reference instances.
    """
    return [ref for ref in extract_references(document) if ref.reference_type == "cross_ref"]
