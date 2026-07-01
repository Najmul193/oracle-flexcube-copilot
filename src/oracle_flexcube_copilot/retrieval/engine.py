"""Vector Retrieval Engine."""

import logging

from oracle_flexcube_copilot.embedding.engine import EmbeddingEngine
from oracle_flexcube_copilot.indexing.indexer import ChromaIndexer
from oracle_flexcube_copilot.indexing.models import SearchResult

logger = logging.getLogger(__name__)


class VectorRetriever:
    """Retrieves context using pure vector similarity search."""

    def __init__(self, embedder: EmbeddingEngine, indexer: ChromaIndexer):
        """Initialize with necessary embedding and indexing engines."""
        self.embedder = embedder
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform semantic search for a query string."""
        logger.info(f"Vector search for query: '{query}'")

        # 1. Embed the query
        query_vector = self.embedder.embed(query)

        # 2. Search ChromaDB
        results = self.indexer.search(query_vector, top_k=top_k)

        logger.debug(f"Retrieved {len(results)} chunks.")
        return results
