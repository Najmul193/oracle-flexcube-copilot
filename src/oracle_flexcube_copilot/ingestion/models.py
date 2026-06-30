"""Pydantic models for the document ingestion pipeline.

Represents the hierarchical structure:

    Document
        ├── Pages
        │       └── Blocks
        │               └── Paragraphs
        └── DocumentMetadata
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, Field


class Paragraph(BaseModel):
    """A single paragraph of text within a block."""

    text: str = Field(description="The plain-text content of the paragraph")
    index: int = Field(description="Zero-based position of this paragraph within its parent block")

    class Config:
        frozen = True


class Block(BaseModel):
    """A structural block on a page (heading, text body, list, table, etc.)."""

    type: str = Field(description="Block type: heading, text, list, table, figure, etc.")
    level: int | None = Field(default=None, description="Heading level (1-6) if type is heading, else None")
    paragraphs: list[Paragraph] = Field(default_factory=list, description="Paragraphs contained in this block")
    block_index: int = Field(description="Zero-based position of this block on the page")

    class Config:
        frozen = True


class Page(BaseModel):
    """A single page extracted from a PDF document."""

    page_number: int = Field(description="One-based page number within the document")
    blocks: list[Block] = Field(default_factory=list, description="Structural blocks on this page")
    word_count: int = Field(default=0, description="Total number of words on this page")
    character_count: int = Field(default=0, description="Total number of characters on this page")

    class Config:
        frozen = True


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


class Document(BaseModel):
    """A fully parsed PDF document with metadata, pages, blocks, and paragraphs."""

    id: str = Field(default_factory=lambda: uuid4().hex, description="Unique document identifier (UUID hex)")
    filename: str = Field(description="PDF filename (e.g. CASA.pdf)")
    absolute_path: str = Field(description="Absolute filesystem path to the PDF")
    sha256: str = Field(description="SHA-256 checksum of the file contents")
    file_size_bytes: int = Field(description="File size in bytes")
    last_modified: datetime = Field(description="File last-modified timestamp")
    created_time: datetime = Field(description="File creation timestamp")
    mime_type: str = Field(default="application/pdf", description="MIME type of the file")
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata, description="PDF document metadata")
    pages: list[Page] = Field(default_factory=list, description="List of pages in the document")

    @property
    def total_words(self) -> int:
        """Return the total word count across all pages."""
        return sum(p.word_count for p in self.pages)

    @property
    def total_characters(self) -> int:
        """Return the total character count across all pages."""
        return sum(p.character_count for p in self.pages)