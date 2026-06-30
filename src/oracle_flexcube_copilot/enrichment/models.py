"""Pydantic models for the enriched document representation.

Represents the hierarchy:

    EnrichedDocument
        ├── TOC entries
        ├── Heading Tree (nested)
        ├── Sections (flat list with parent/child refs)
        ├── Cross-references
        ├── Tables
        └── EnrichedBlocks (blocks with section context)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, ConfigDict

from oracle_flexcube_copilot.ingestion.models import TOCEntry


class HeadingNode(BaseModel):
    """A node in the normalized heading tree."""

    title: str = Field(description="Heading text")
    level: int = Field(description="Heading level (1 = top-level chapter)")
    normalized_number: str = Field(default="", description="Normalized section number (e.g. 1.2.3)")
    page: int = Field(default=0, description="Page number where this heading starts")
    children: list[HeadingNode] = Field(default_factory=list, description="Child headings")
    block_ids: list[str] = Field(default_factory=list, description="Block IDs belonging to this heading")


class Section(BaseModel):
    """A document section defined by a heading and its content blocks."""

    id: str = Field(default="", description="Stable section identifier")
    title: str = Field(description="Section title (heading text)")
    number: str = Field(default="", description="Section number (e.g. 1.2.3)")
    level: int = Field(description="Heading level (1 = top-level)")
    parent_id: str | None = Field(default=None, description="ID of parent section")
    child_ids: list[str] = Field(default_factory=list, description="IDs of child sections")
    page_start: int = Field(description="First page of this section")
    page_end: int = Field(description="Last page of this section")
    block_ids: list[str] = Field(default_factory=list, description="Block IDs belonging to this section")
    word_count: int = Field(default=0, description="Total word count in this section")


class EnrichedBlock(BaseModel):
    """A block with enriched context: section membership and heading path."""

    id: str = Field(description="Block ID (stable, matches ingestion Block.id)")
    section_id: str | None = Field(default=None, description="Section this block belongs to")
    heading_path: list[str] = Field(default_factory=list, description="Hierarchical heading path to this block")
    depth: int = Field(default=0, description="Depth in the heading hierarchy")


class Reference(BaseModel):
    """A cross-reference extracted from document text."""

    id: str = Field(default="", description="Stable reference identifier")
    text: str = Field(description="The raw reference text as found")
    target: str = Field(default="", description="Extracted target (e.g. 'Chapter 8', 'STTM_PRODUCT')")
    reference_type: str = Field(default="cross_ref", description="Type: cross_ref, see_also, appendix_ref")
    source_block_id: str = Field(default="", description="Block ID where this reference was found")
    source_page: int | None = Field(default=None, description="Page where this reference appears")


class TableData(BaseModel):
    """An extracted table with headers and rows."""

    id: str = Field(default="", description="Stable table identifier")
    source_block_id: str = Field(default="", description="Block ID where this table was found")
    page: int = Field(default=0, description="Page number")
    title: str = Field(default="", description="Table title or caption")
    headers: list[str] = Field(default_factory=list, description="Column headers")
    rows: list[list[str]] = Field(default_factory=list, description="Data rows")
    num_rows: int = Field(default=0, description="Number of data rows")
    num_cols: int = Field(default=0, description="Number of columns")


class EnrichedDocument(BaseModel):
    """A fully enriched document ready for chunking and retrieval."""

    document_id: str = Field(description="Source Document ID (SHA-256)")
    filename: str = Field(description="Original filename")
    title: str = Field(default="", description="Document title from metadata")
    total_pages: int = Field(default=0)
    total_words: int = Field(default=0)

    # Hierarchical structure
    toc: list[TOCEntry] = Field(default_factory=list, description="Table of contents entries")
    heading_tree: list[HeadingNode] = Field(default_factory=list, description="Nested heading hierarchy")
    sections: list[Section] = Field(default_factory=list, description="Flat list of all sections")

    # Enriched data
    enriched_blocks: list[EnrichedBlock] = Field(default_factory=list, description="Blocks with section context")
    cross_references: list[Reference] = Field(default_factory=list, description="All cross-references found")
    tables: list[TableData] = Field(default_factory=list, description="All tables found")

    ingestion_timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


# Re-export from ingestion models for convenience
# TOCEntry is re-exported from ingestion.models for API convenience.
# Import it via enrichment/__init__.py which handles cross-module imports.
