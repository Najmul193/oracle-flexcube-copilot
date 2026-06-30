"""Shared fixtures for enrichment tests."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from oracle_flexcube_copilot.ingestion.models import (
    Block,
    Document,
    DocumentMetadata,
    Page,
    Paragraph,
    TOCEntry,
)


@pytest.fixture
def sample_document() -> Document:
    """Create a minimal Document with headings and text blocks for enrichment testing."""
    sha256 = "a8f23c1d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1"
    toc = [
        TOCEntry(level=1, title="Chapter 1: Product Definition", page=1),
        TOCEntry(level=2, title="1.1 Product Code", page=1),
        TOCEntry(level=2, title="1.2 Rate Code", page=2),
        TOCEntry(level=1, title="Chapter 2: Configuration", page=3),
        TOCEntry(level=2, title="2.1 Setup", page=3),
    ]

    pages = [
        Page(
            id=f"{sha256}:p1",
            page_number=1,
            blocks=[
                Block(id=f"{sha256}:p1:b0", type="heading", level=1, block_index=0, paragraphs=[Paragraph(text="Chapter 1: Product Definition", index=0)]),
                Block(id=f"{sha256}:p1:b1", type="heading", level=2, block_index=1, paragraphs=[Paragraph(text="1.1 Product Code", index=0)]),
                Block(id=f"{sha256}:p1:b2", type="text", block_index=2, paragraphs=[
                    Paragraph(text="The product code identifies the financial product.", index=0),
                    Paragraph(text="Refer to STTM_PRODUCT for details.", index=1),
                ]),
            ],
            word_count=15, character_count=100,
        ),
        Page(
            id=f"{sha256}:p2",
            page_number=2,
            blocks=[
                Block(id=f"{sha256}:p2:b0", type="heading", level=2, block_index=0, paragraphs=[Paragraph(text="1.2 Rate Code", index=0)]),
                Block(id=f"{sha256}:p2:b1", type="text", block_index=1, paragraphs=[
                    Paragraph(text="See Chapter 8 for interest rate configuration.", index=0),
                    Paragraph(text="Rate codes are defined in the system.", index=1),
                ]),
            ],
            word_count=18, character_count=120,
        ),
        Page(
            id=f"{sha256}:p3",
            page_number=3,
            blocks=[
                Block(id=f"{sha256}:p3:b0", type="heading", level=1, block_index=0, paragraphs=[Paragraph(text="Chapter 2: Configuration", index=0)]),
                Block(id=f"{sha256}:p3:b1", type="heading", level=2, block_index=1, paragraphs=[Paragraph(text="2.1 Setup", index=0)]),
                Block(id=f"{sha256}:p3:b2", type="text", block_index=2, paragraphs=[Paragraph(text="Follow these steps to configure the system.", index=0)]),
            ],
            word_count=10, character_count=60,
        ),
    ]

    return Document(
        id=sha256,
        filename="test_doc.pdf",
        absolute_path="/tmp/test_doc.pdf",
        sha256=sha256,
        file_size_bytes=1024,
        last_modified=datetime(2024, 1, 1, tzinfo=timezone.utc),
        created_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
        mime_type="application/pdf",
        metadata=DocumentMetadata(title="Test Document", page_count=3),
        table_of_contents=toc,
        pages=pages,
    )