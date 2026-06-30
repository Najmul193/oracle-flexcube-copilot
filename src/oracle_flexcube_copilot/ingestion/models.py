"""Pydantic models for the document ingestion pipeline.

Represents the hierarchical structure:

    Document
        ├── TOC
        ├── Pages
        │       └── Blocks (heading, text, table, figure)
        │               └── Paragraphs
        └── DocumentMetadata

All IDs are deterministic content-based hashes for stable referencing
across re-ingestion and incremental indexing.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field, ConfigDict


def make_block_id(doc_sha256: str, page_number: int, block_index: int) -> str:
    """Derive a stable block ID from document SHA-256, page number, and block index."""
    return f"{doc_sha256}:p{page_number}:b{block_index}"


def make_page_id(doc_sha256: str, page_number: int) -> str:
    """Derive a stable page ID from document SHA-256 and page number."""
    return f"{doc_sha256}:p{page_number}"


class ImageInfo(BaseModel):
    """Metadata about an image on a page."""

    bbox: tuple[float, float, float, float] = Field(
        description="Bounding box (x0, y0, x1, y1) in PDF coordinates"
    )
    width: float = Field(default=0, description="Image width in points")
    height: float = Field(default=0, description="Image height in points")

    model_config = ConfigDict(frozen=True)


class TableData(BaseModel):
    """Structured table data extracted from a block."""

    headers: list[str] = Field(default_factory=list, description="Table header row")
    rows: list[list[str]] = Field(default_factory=list, description="Table data rows")
    title: str = Field(default="", description="Table title or caption")

    model_config = ConfigDict(frozen=True)


class Reference(BaseModel):
    """A cross-reference found within text (e.g. 'See Chapter 8', 'Refer to STTM_PRODUCT')."""

    text: str = Field(description="The full reference text as it appears")
    target: str | None = Field(default=None, description="Extracted target identifier")
    page: int | None = Field(default=None, description="Referenced page number, if available")

    model_config = ConfigDict(frozen=True)


class Paragraph(BaseModel):
    """A single paragraph of text within a block."""

    text: str = Field(description="The plain-text content of the paragraph")
    index: int = Field(description="Zero-based position of this paragraph within its parent block")

    model_config = ConfigDict(frozen=True)


def _temp_id() -> str:
    """Generate a temporary block/page ID before stable IDs are assigned."""
    from uuid import uuid4
    return f"tmp:{uuid4().hex[:12]}"


class Block(BaseModel):
    """A structural block on a page (heading, text body, list, table, figure, etc.)."""

    id: str = Field(default_factory=_temp_id, description="Stable block ID: doc_sha256:p{page}:b{block}")
    type: str = Field(description="Block type: heading, text, list, table, figure, etc.")
    level: int | None = Field(default=None, description="Heading level (1-6) if type is heading, else None")
    paragraphs: list[Paragraph] = Field(default_factory=list, description="Paragraphs contained in this block")
    block_index: int = Field(description="Zero-based position of this block on the page")
    image: ImageInfo | None = Field(default=None, description="Image metadata if type is figure")
    table: TableData | None = Field(default=None, description="Table data if type is table")
    cross_references: list[Reference] = Field(default_factory=list, description="Cross-references found in this block")

    model_config = ConfigDict(frozen=True)


class Page(BaseModel):
    """A single page extracted from a PDF document."""

    id: str = Field(default_factory=_temp_id, description="Stable page ID: doc_sha256:p{page_number}")
    page_number: int = Field(description="One-based page number within the document")
    blocks: list[Block] = Field(default_factory=list, description="Structural blocks on this page")
    word_count: int = Field(default=0, description="Total number of words on this page")
    character_count: int = Field(default=0, description="Total number of characters on this page")

    model_config = ConfigDict(frozen=True)


class DocumentMetadata(BaseModel):
    """PDF document metadata extracted from the file's info dictionary."""

    title: str = Field(default="", description="Document title")
    author: str = Field(default="", description="Document author")
    subject: str = Field(default="", description="Document subject")
    keywords: str = Field(default="", description="Document keywords")
    producer: str = Field(default="", description="PDF producer (e.g. iText, Apache)")
    creator: str = Field(default="", description="Application that created the document")
    creation_date: datetime | None = Field(default=None, description="Document creation timestamp")
    modification_date: datetime | None = Field(default=None, description="Last modification timestamp")
    page_count: int = Field(default=0, description="Total number of pages in the document")


class TOCEntry(BaseModel):
    """A single entry in the document's table of contents."""

    level: int = Field(description="Hierarchy level (1 = chapter, 2 = section, etc.)")
    title: str = Field(description="Entry title text")
    page: int = Field(description="One-based page number where the entry begins")


class Document(BaseModel):
    """A fully parsed PDF document with metadata, pages, blocks, and paragraphs."""

    id: str = Field(description="Stable document identifier (SHA-256 of file contents)")
    filename: str = Field(description="PDF filename (e.g. CASA.pdf)")
    absolute_path: str = Field(description="Absolute filesystem path to the PDF")
    sha256: str = Field(description="SHA-256 checksum of the file contents")
    file_size_bytes: int = Field(description="File size in bytes")
    last_modified: datetime = Field(description="File last-modified timestamp")
    created_time: datetime = Field(description="File creation timestamp")
    mime_type: str = Field(default="application/pdf", description="MIME type of the file")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="PDF document metadata")
    table_of_contents: list[TOCEntry] = Field(default_factory=list, description="Document table of contents")
    pages: list[Page] = Field(default_factory=list, description="List of pages in the document")

    @property
    def total_words(self) -> int:
        """Return the total word count across all pages."""
        return sum(p.word_count for p in self.pages)

    @property
    def total_characters(self) -> int:
        """Return the total character count across all pages."""
        return sum(p.character_count for p in self.pages)