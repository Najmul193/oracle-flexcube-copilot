"""Tests for Oracle Entity Index."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from oracle_flexcube_copilot.chunking.models import Chunk, ChunkMetadata
from oracle_flexcube_copilot.enrichment.models import OracleEntity
from oracle_flexcube_copilot.indexing.entity_index import OracleEntityIndex


@pytest.fixture
def temp_db_dir() -> Path:
    """Provide a temporary directory for SQLite DB."""
    with TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def dummy_chunk() -> Chunk:
    c = Chunk(id="chunk1", document_id="doc1", text="Text", page_start=1, page_end=1)
    c.metadata = ChunkMetadata(
        document_name="doc1.pdf", pipeline_version="1.0", chunking_version="1.0"
    )
    c.heading_path = ["Chapter 1", "Section A"]
    c.oracle_entities = [
        OracleEntity(name="STTM_PRODUCT", entity_type="SCREEN"),
        OracleEntity(name="CASA", entity_type="MODULE"),
    ]
    return c


class TestOracleEntityIndex:
    """Tests for exact-match Oracle Entity Index."""

    def test_index_and_lookup(self, temp_db_dir: Path, dummy_chunk: Chunk) -> None:
        """Should index entities and retrieve them via exact lookup."""
        index = OracleEntityIndex(db_dir=str(temp_db_dir))

        inserted = index.index_chunk(dummy_chunk)
        assert inserted == 2

        # Exact match
        locations = index.lookup("sttm_product")
        assert len(locations) == 1
        assert locations[0].document_id == "doc1"
        assert locations[0].document_name == "doc1.pdf"
        assert locations[0].page == 1
        assert locations[0].section == "Section A"
        assert locations[0].chunk_id == "chunk1"

    def test_ignore_duplicates(self, temp_db_dir: Path, dummy_chunk: Chunk) -> None:
        """Should ignore if same chunk is indexed twice."""
        index = OracleEntityIndex(db_dir=str(temp_db_dir))

        inserted1 = index.index_chunk(dummy_chunk)
        assert inserted1 == 2

        inserted2 = index.index_chunk(dummy_chunk)
        assert inserted2 == 0

        assert len(index.lookup("CASA")) == 1

    def test_remove_document(self, temp_db_dir: Path, dummy_chunk: Chunk) -> None:
        """Should purge all entries for a document."""
        index = OracleEntityIndex(db_dir=str(temp_db_dir))
        index.index_chunk(dummy_chunk)

        assert len(index.lookup("CASA")) == 1

        removed = index.remove_document("doc1")
        assert removed == 2

        assert len(index.lookup("CASA")) == 0

    def test_remove_chunk(self, temp_db_dir: Path, dummy_chunk: Chunk) -> None:
        """Should purge entries by chunk ID."""
        index = OracleEntityIndex(db_dir=str(temp_db_dir))
        index.index_chunk(dummy_chunk)

        removed = index.remove_chunk("chunk1")
        assert removed == 2
        assert len(index.lookup("CASA")) == 0

    def test_clear(self, temp_db_dir: Path, dummy_chunk: Chunk) -> None:
        """Should truncate table."""
        index = OracleEntityIndex(db_dir=str(temp_db_dir))
        index.index_chunk(dummy_chunk)

        index.clear()
        assert len(index.lookup("CASA")) == 0
