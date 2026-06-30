"""Oracle document classification and entity extraction."""

from __future__ import annotations

import logging
import re
from typing import Any

from oracle_flexcube_copilot.enrichment.models import OracleEntity
from oracle_flexcube_copilot.ingestion.models import Document

logger = logging.getLogger("oracle_flexcube_copilot.enrichment.classification")

# Common Oracle FLEXCUBE modules
MODULE_KEYWORDS = {
    "CASA": ["Current Accounts", "Savings Accounts", "CASA"],
    "Loans": ["Loans", "Lending", "CL", "Consumer Lending"],
    "Treasury": ["Treasury", "Money Market", "Foreign Exchange", "FX", "Derivatives"],
    "Trade Finance": ["Letters of Credit", "Trade Finance", "Bills", "Collections"],
    "Islamic Banking": ["Islamic", "Mudarabah", "Murabaha", "Ijara"],
    "Payments": ["Payments", "Funds Transfer", "SWIFT", "RTGS"],
    "Core": ["Core Services", "Security Management", "User Guide", "Installation"],
}

# Entity extraction patterns
ENTITY_PATTERNS = [
    # Match standard screen/module codes (e.g., STTM_PRODUCT, CLDPROD)
    (re.compile(r"\b([A-Z0-9_]{4,20})\b"), "MODULE/SCREEN"),
]

SPECIFIC_ENTITIES = {
    "CASA": "MODULE",
    "EOD": "PROCESS",
    "BOD": "PROCESS",
    "GL": "MODULE",
    "CIF": "MODULE",
    "ICCF": "MODULE",
}


def classify_document(document: Document) -> str:
    """Classify the document into a high-level Oracle module based on title and content.

    Args:
        document: A parsed Document.

    Returns:
        The classification string (e.g., 'CASA', 'Loans', 'Unknown').
    """
    search_text = f"{document.filename} {document.metadata.title or ''}".lower()
    
    # Also grab the first few pages of text to help with classification
    content_sample = ""
    for page in document.pages[:3]:
        for block in page.blocks:
            for para in block.paragraphs:
                content_sample += para.text.lower() + " "

    search_text += " " + content_sample

    best_match = "Unknown"
    max_hits = 0

    for module, keywords in MODULE_KEYWORDS.items():
        hits = sum(1 for kw in keywords if kw.lower() in search_text)
        if hits > max_hits:
            max_hits = hits
            best_match = module

    return best_match


def extract_oracle_entities(document: Document, block_to_section: dict[str, str]) -> list[OracleEntity]:
    """Extract Oracle-specific entities from the document text.

    Args:
        document: A parsed Document.
        block_to_section: Dictionary mapping block IDs to their section IDs.

    Returns:
        List of OracleEntity objects.
    """
    entities: list[OracleEntity] = []
    seen: set[str] = set()

    for page in document.pages:
        for block in page.blocks:
            section_id = block_to_section.get(block.id)
            for para in block.paragraphs:
                text = para.text
                
                # Check for specific entities
                for entity_name, entity_type in SPECIFIC_ENTITIES.items():
                    if re.search(rf"\b{entity_name}\b", text):
                        key = f"{entity_name}:{page.page_number}"
                        if key not in seen:
                            seen.add(key)
                            entities.append(
                                OracleEntity(
                                    name=entity_name,
                                    entity_type=entity_type,
                                    page=page.page_number,
                                    section_id=section_id,
                                    context=text[:200],  # store some context
                                )
                            )

                # Check for patterned entities (like STTM_PRODUCT)
                for pattern, entity_type in ENTITY_PATTERNS:
                    for match in pattern.finditer(text):
                        entity_name = match.group(1)
                        # Filter out common false positives (purely numeric, too short)
                        if len(entity_name) >= 4 and any(c.isalpha() for c in entity_name) and "_" in entity_name:
                            key = f"{entity_name}:{page.page_number}"
                            if key not in seen:
                                seen.add(key)
                                entities.append(
                                    OracleEntity(
                                        name=entity_name,
                                        entity_type=entity_type,
                                        page=page.page_number,
                                        section_id=section_id,
                                        context=text[:200],
                                    )
                                )

    logger.info("Found %d Oracle entities", len(entities))
    return entities
