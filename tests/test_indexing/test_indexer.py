"""Tests for the ChromaIndexer."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from oracle_flexcube_copilot.chunking.models import Chunk, ChunkMetadata
from oracle_flexcube_copilot.embedding.models import EmbeddedChunk
from oracle_flexcube_copilot.enrichment.models import OracleEntity
from oracle_flexcube_copilot.indexing.indexer import ChromaIndexer


@pytest.fixture
def temp_db_dir() -> Path:
    """Provide a temporary directory for ChromaDB storage."""
    with TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def dummy_chunks() -> list[EmbeddedChunk]:
    c1 = Chunk(id="chunk1", document_id="doc1", text="First chunk text", page_start=1, page_end=1)
    c1.metadata = ChunkMetadata(
        document_name="doc1.pdf",
        module_classification="CASA",
        pipeline_version="1.0.0",
        chunking_version="1.0",
    )
    c1.heading_path = ["Chapter 1", "Section A"]
    c1.oracle_entities = [OracleEntity(name="STTM_PRODUCT", entity_type="SCREEN")]
    e1 = EmbeddedChunk(
        chunk=c1,
        embedding=[0.1, 0.2, 0.3],
        embedding_model="test",
        embedding_dimension=3,
        embedding_time=0.1,
        embedding_version="v1",
    )

    c2 = Chunk(id="chunk2", document_id="doc1", text="Second chunk text", page_start=2, page_end=2)
    e2 = EmbeddedChunk(
        chunk=c2,
        embedding=[0.4, 0.5, 0.6],
        embedding_model="test",
        embedding_dimension=3,
        embedding_time=0.1,
        embedding_version="v1",
    )

    return [e1, e2]


class TestChromaIndexer:
    """Tests for ChromaDB indexer."""

    def test_add_and_get_chunks(self, temp_db_dir: Path, dummy_chunks: list[EmbeddedChunk]) -> None:
        """Should add chunks and retrieve them with correct metadata."""
        indexer = ChromaIndexer(db_dir=str(temp_db_dir), collection_name="test_col")

        metrics = indexer.add_chunks(dummy_chunks)
        assert metrics.chunks_added == 2
        assert metrics.chunks_skipped == 0

        # Test get_chunk
        retrieved = indexer.get_chunk("chunk1")
        assert retrieved is not None
        assert retrieved["id"] == "chunk1"
        assert retrieved["document"] == "First chunk text"
        import numpy as np

        assert np.allclose(retrieved["embedding"], [0.1, 0.2, 0.3])

        meta = retrieved["metadata"]
        assert meta["document_id"] == "doc1"
        assert meta["module"] == "CASA"

        # Verify JSON serialization survived
        heading_path = json.loads(meta["heading_path"])
        assert heading_path == ["Chapter 1", "Section A"]

        entities = json.loads(meta["oracle_entities"])
        assert entities[0]["name"] == "STTM_PRODUCT"

    def test_duplicate_prevention(
        self, temp_db_dir: Path, dummy_chunks: list[EmbeddedChunk]
    ) -> None:
        """Should not add duplicate chunks if they already exist."""
        indexer = ChromaIndexer(db_dir=str(temp_db_dir), collection_name="test_col")

        indexer.add_chunks(dummy_chunks)

        # Try to add again
        metrics2 = indexer.add_chunks(dummy_chunks)
        assert metrics2.chunks_added == 0
        assert metrics2.chunks_skipped == 2

    def test_update_chunks(self, temp_db_dir: Path, dummy_chunks: list[EmbeddedChunk]) -> None:
        """Should upsert chunks."""
        indexer = ChromaIndexer(db_dir=str(temp_db_dir), collection_name="test_col")

        indexer.add_chunks(dummy_chunks)

        # Modify chunk 1 text
        dummy_chunks[0].chunk.text = "Modified text"

        metrics = indexer.update_chunks([dummy_chunks[0]])
        assert metrics.chunks_updated == 1

        retrieved = indexer.get_chunk("chunk1")
        assert retrieved is not None
        assert retrieved["document"] == "Modified text"

    def test_delete_document(self, temp_db_dir: Path, dummy_chunks: list[EmbeddedChunk]) -> None:
        """Should purge all chunks associated with a document_id."""
        indexer = ChromaIndexer(db_dir=str(temp_db_dir), collection_name="test_col")
        indexer.add_chunks(dummy_chunks)

        health_before = indexer.health_check()
        assert health_before.total_chunks == 2

        metrics = indexer.delete_document("doc1")
        assert metrics.chunks_deleted == 2
        assert metrics.documents_deleted == 1

        health_after = indexer.health_check()
        assert health_after.total_chunks == 0

        # doc1 shouldn't exist
        assert len(indexer.get_document("doc1")) == 0

    def test_search(self, temp_db_dir: Path, dummy_chunks: list[EmbeddedChunk]) -> None:
        """Search should return hits sorted by distance."""
        indexer = ChromaIndexer(db_dir=str(temp_db_dir), collection_name="test_col")
        indexer.add_chunks(dummy_chunks)

        # Dummy search vector close to [0.1, 0.2, 0.3]
        hits = indexer.search(query_embedding=[0.11, 0.21, 0.31], top_k=1)

        assert len(hits) == 1
        assert hits[0].chunk_id == "chunk1"
        assert hits[0].retrieval_method == "vector"
        assert hits[0].source_document == "doc1.pdf"
        assert hits[0].heading == "Section A"
        assert hits[0].oracle_entities == ["STTM_PRODUCT"]

    def test_collection_reset(self, temp_db_dir: Path, dummy_chunks: list[EmbeddedChunk]) -> None:
        """Reset should clear everything."""
        indexer = ChromaIndexer(db_dir=str(temp_db_dir), collection_name="test_col")
        indexer.add_chunks(dummy_chunks)

        assert indexer.health_check().total_chunks == 2

        indexer.reset_collection()

        assert indexer.health_check().total_chunks == 0

    def test_persistence_across_instances(
        self, temp_db_dir: Path, dummy_chunks: list[EmbeddedChunk]
    ) -> None:
        """Ensure data persists when client is recreated."""
        indexer1 = ChromaIndexer(db_dir=str(temp_db_dir), collection_name="test_col")
        indexer1.add_chunks(dummy_chunks)
        assert indexer1.health_check().total_chunks == 2

        # Simulate closing and reopening the db
        indexer2 = ChromaIndexer(db_dir=str(temp_db_dir), collection_name="test_col")
        assert indexer2.health_check().total_chunks == 2
        retrieved_again = indexer2.get_chunk("chunk1")
        assert retrieved_again is not None
        import numpy as np

        assert np.allclose(retrieved_again["embedding"], [0.1, 0.2, 0.3])
