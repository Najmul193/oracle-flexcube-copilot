from pathlib import Path

from oracle_flexcube_copilot.chunking.models import Chunk, ChunkMetadata
from oracle_flexcube_copilot.retrieval.bm25 import BM25Indexer, BM25Retriever


def test_bm25_indexer_retriever(tmp_path: Path):
    index_path = tmp_path / "bm25.pkl"

    # 1. Create fake chunks
    meta = ChunkMetadata(
        pipeline_version="1",
        chunking_version="1",
        embedding_model="x",
        embedding_version="1",
        document_name="doc.pdf",
    )

    chunk1 = Chunk(
        id="c1", document_id="d1", text="Oracle FLEXCUBE STTM_PRODUCT module", metadata=meta
    )
    chunk2 = Chunk(
        id="c2", document_id="d1", text="Retail Lending and CASA integration", metadata=meta
    )
    chunk3 = Chunk(
        id="c3", document_id="d1", text="Some unrelated text without keywords", metadata=meta
    )

    # 2. Build index
    indexer = BM25Indexer(index_path=index_path)
    indexer.build([chunk1, chunk2, chunk3])

    assert index_path.exists()

    # 3. Retrieve
    retriever = BM25Retriever(index_path=index_path)
    results = retriever.retrieve("STTM_PRODUCT", top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "c1"
    assert results[0].retrieval_method == "bm25"
    assert "STTM_PRODUCT" in results[0].text
