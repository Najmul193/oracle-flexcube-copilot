"""Tests for Chunking Pydantic models."""

from __future__ import annotations

from oracle_flexcube_copilot.chunking.models import Chunk, ChunkMetadata
from oracle_flexcube_copilot.enrichment.models import OracleEntity, Reference


class TestChunkModels:
    """Tests for Chunk and ChunkMetadata."""

    def test_chunk_metadata_defaults(self) -> None:
        """ChunkMetadata should have default empty strings for embeddings."""
        meta = ChunkMetadata(pipeline_version="1.0", chunking_version="1.0")
        assert meta.embedding_model == ""
        assert meta.embedding_version == ""

    def test_chunk_construction(self) -> None:
        """Chunk should populate correctly with minimal data."""
        chunk = Chunk(
            id="doc1:chunk:1",
            document_id="doc1",
            text="Sample chunk text",
        )
        assert chunk.id == "doc1:chunk:1"
        assert chunk.document_id == "doc1"
        assert chunk.text == "Sample chunk text"
        assert chunk.heading_path == []
        assert chunk.section_title == ""
        assert chunk.page_start == 0
        assert chunk.page_end == 0
        assert chunk.oracle_entities == []
        assert chunk.references == []
        assert chunk.table_ids == []
        assert chunk.token_count == 0
        assert chunk.word_count == 0
        assert chunk.embedding_status == "pending"
        assert chunk.metadata is None
