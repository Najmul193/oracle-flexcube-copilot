"""Tests for Vector Retrieval Engine."""

from unittest.mock import Mock

from oracle_flexcube_copilot.indexing.models import SearchResult
from oracle_flexcube_copilot.retrieval.engine import VectorRetriever


def test_vector_retriever() -> None:
    """It should orchestrate embedding and index search."""
    mock_embedder = Mock()
    mock_embedder.embed.return_value = [0.1, 0.2, 0.3]

    mock_indexer = Mock()
    mock_indexer.search.return_value = [
        SearchResult(
            chunk_id="1",
            score=0.9,
            source_document="doc.pdf",
            page=1,
            heading="Section",
            oracle_entities=[],
            text="text",
            retrieval_method="vector",
        )
    ]

    retriever = VectorRetriever(embedder=mock_embedder, indexer=mock_indexer)
    results = retriever.retrieve("What is CASA?", top_k=5)

    # Assert orchestrator behavior
    mock_embedder.embed.assert_called_once_with("What is CASA?")
    mock_indexer.search.assert_called_once_with([0.1, 0.2, 0.3], top_k=5)

    assert len(results) == 1
    assert results[0].chunk_id == "1"
