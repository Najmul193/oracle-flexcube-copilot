"""Tests for SemanticSectionChunker."""

from __future__ import annotations

import pytest
from datetime import datetime, timezone

from oracle_flexcube_copilot.chunking.strategy import SemanticSectionChunker
from oracle_flexcube_copilot.enrichment.models import EnrichedBlock, EnrichedDocument, Section, OracleEntity, Reference
from oracle_flexcube_copilot.ingestion.models import Block, Document, DocumentMetadata, Page, Paragraph, TableData


def _make_dummy_document() -> Document:
    return Document(
        id="doc_test",
        filename="test.pdf",
        absolute_path="/tmp/test.pdf",
        sha256="doc_test",
        file_size_bytes=100,
        last_modified=datetime.now(timezone.utc),
        created_time=datetime.now(timezone.utc),
        metadata=DocumentMetadata(page_count=2, title="Test Doc"),
        pages=[
            Page(
                id="doc_test:p1",
                page_number=1,
                blocks=[
                    Block(
                        id="doc_test:p1:b0",
                        type="heading",
                        block_index=0,
                        paragraphs=[Paragraph(text="Section 1", index=0)]
                    ),
                    Block(
                        id="doc_test:p1:b1",
                        type="text",
                        block_index=1,
                        paragraphs=[Paragraph(text=" ".join(f"WA{i}" for i in range(400)), index=0)]
                    ),
                    Block(
                        id="doc_test:p1:b2",
                        type="text",
                        block_index=2,
                        paragraphs=[Paragraph(text=" ".join(f"WB{i}" for i in range(400)), index=0)]
                    ),
                ],
                word_count=802
            ),
            Page(
                id="doc_test:p2",
                page_number=2,
                blocks=[
                    Block(
                        id="doc_test:p2:b0",
                        type="text",
                        block_index=0,
                        paragraphs=[
                            Paragraph(text="Step 1: Do this", index=0),
                            Paragraph(text="Step 2: Do that", index=1),
                            Paragraph(text="Step 3: Finish", index=2),
                        ]
                    ),
                    Block(
                        id="doc_test:p2:b1",
                        type="table",
                        block_index=1,
                        paragraphs=[],
                        table=TableData(headers=["A", "B"], rows=[["1", "2"]])
                    ),
                ],
                word_count=10
            )
        ]
    )


def _make_dummy_enriched_doc(doc: Document) -> EnrichedDocument:
    section1 = Section(
        id="doc_test:sec:1",
        title="Section 1",
        level=1,
        page_start=1,
        page_end=2,
        block_ids=["doc_test:p1:b0", "doc_test:p1:b1", "doc_test:p1:b2", "doc_test:p2:b0", "doc_test:p2:b1"]
    )
    enriched_blocks = [
        EnrichedBlock(id=b.id, section_id=section1.id, heading_path=["Section 1"])
        for p in doc.pages for b in p.blocks
    ]
    return EnrichedDocument(
        document_id=doc.id,
        filename=doc.filename,
        title=doc.metadata.title,
        total_pages=doc.metadata.page_count,
        total_words=doc.total_words,
        sections=[section1],
        enriched_blocks=enriched_blocks,
        pages=doc.pages,
    )


class TestSemanticSectionChunker:
    """Tests for SemanticSectionChunker."""

    def test_chunking_respects_max_tokens(self) -> None:
        """Chunker should split large sections to respect max_tokens."""
        doc = _make_dummy_document()
        enriched = _make_dummy_enriched_doc(doc)
        
        # Max 900 tokens. 400 words is ~520 tokens.
        # Two 400 word blocks should be split across two chunks.
        chunker = SemanticSectionChunker(target_tokens=600, max_tokens=900, overlap_tokens=50)
        chunks = chunker.chunk(enriched)
        
        # Expecting at least 2 chunks for the 800 words + table + procedure
        assert len(chunks) >= 2
        for chunk in chunks:
            assert chunk.token_count <= 900
            assert chunk.heading_path == ["Section 1"]
            assert chunk.metadata is not None
            assert chunk.metadata.pipeline_version != ""

    def test_unbreakable_procedures(self) -> None:
        """Procedure steps should not be split."""
        doc = _make_dummy_document()
        enriched = _make_dummy_enriched_doc(doc)
        
        chunker = SemanticSectionChunker(target_tokens=600, max_tokens=900, overlap_tokens=0)
        chunks = chunker.chunk(enriched)
        
        # Find the chunk containing the procedure
        proc_chunk = next(c for c in chunks if "Step 1" in c.text)
        assert "Step 1" in proc_chunk.text
        assert "Step 2" in proc_chunk.text
        assert "Step 3" in proc_chunk.text

    def test_table_representation(self) -> None:
        """Tables should be represented as markdown in the chunk text."""
        doc = _make_dummy_document()
        enriched = _make_dummy_enriched_doc(doc)
        
        chunker = SemanticSectionChunker(target_tokens=600, max_tokens=900, overlap_tokens=0)
        chunks = chunker.chunk(enriched)
        
        table_chunk = next(c for c in chunks if "|" in c.text)
        assert "A" in table_chunk.text
        assert "---" in table_chunk.text
        assert "1" in table_chunk.text

    def test_overlap_logic(self) -> None:
        """Chunks should overlap if overlapping is enabled."""
        doc = _make_dummy_document()
        enriched = _make_dummy_enriched_doc(doc)
        
        chunker = SemanticSectionChunker(target_tokens=600, max_tokens=900, overlap_tokens=600)
        chunks = chunker.chunk(enriched)
        
        if len(chunks) > 1:
            # Check if there is common text between consecutive chunks
            text1_words = set(chunks[0].text.split())
            text2_words = set(chunks[1].text.split())
            intersection = text1_words.intersection(text2_words)
            # The overlap is typically the last block of the previous chunk
            assert len(intersection) > 10  # Significant overlap
