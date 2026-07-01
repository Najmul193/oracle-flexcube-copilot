"""Sparse retrieval using BM25 for exact keyword matching."""

import logging
import pickle
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from oracle_flexcube_copilot.chunking.models import Chunk
from oracle_flexcube_copilot.config import settings
from oracle_flexcube_copilot.indexing.models import SearchResult

logger = logging.getLogger(__name__)

# Simple tokenizer for BM25
TOKENIZER_REGEX = re.compile(r"\w+")


def tokenize(text: str) -> list[str]:
    """Tokenize text into words for BM25."""
    return TOKENIZER_REGEX.findall(text.lower())


class BM25Indexer:
    """Builds and serializes a BM25 sparse index."""

    def __init__(self, index_path: Path | None = None):
        """Initialize the indexer.

        Args:
            index_path: Path to save the index pickle. Defaults to settings.data_dir / 'bm25_index.pkl'.
        """
        self.index_path = index_path or (settings.data_dir / "bm25_index.pkl")
        self.index_path.parent.mkdir(parents=True, exist_ok=True)

    def build(self, chunks: list[Chunk]) -> None:
        """Build the BM25 index from a list of chunks and save it to disk."""
        logger.info("Building BM25 index for %d chunks...", len(chunks))

        corpus_tokens = []
        metadata_store = []

        for chunk in chunks:
            # Tokenize text
            corpus_tokens.append(tokenize(chunk.text))

            # Extract entity names
            entity_names = [e.name for e in chunk.oracle_entities] if chunk.oracle_entities else []

            # Get document name
            doc_name = (
                chunk.metadata.document_name
                if chunk.metadata and chunk.metadata.document_name
                else ""
            )

            # Extract heading
            heading = chunk.section_title
            if not heading and chunk.heading_path:
                heading = chunk.heading_path[-1]

            # Store necessary metadata for SearchResult reconstruction
            metadata_store.append(
                {
                    "chunk_id": chunk.id,
                    "text": chunk.text,
                    "source_document": doc_name,
                    "page": chunk.page_start,
                    "heading": heading,
                    "oracle_entities": entity_names,
                }
            )

        bm25 = BM25Okapi(corpus_tokens)

        # Save both the model and the metadata
        payload = {"bm25": bm25, "metadata_store": metadata_store}

        with open(self.index_path, "wb") as f:
            pickle.dump(payload, f)

        logger.info("BM25 index saved to %s", self.index_path)


class BM25Retriever:
    """Retrieves context using BM25 sparse index."""

    def __init__(self, index_path: Path | None = None):
        """Initialize and load the BM25 index."""
        self.index_path = index_path or (settings.data_dir / "bm25_index.pkl")
        self.bm25: BM25Okapi | None = None
        self.metadata_store: list[dict[str, Any]] = []
        self._load_index()

    def _load_index(self) -> None:
        if not self.index_path.exists():
            logger.warning(
                "BM25 index not found at %s. Sparse retrieval will return empty results.",
                self.index_path,
            )
            return

        logger.info("Loading BM25 index from %s", self.index_path)
        with open(self.index_path, "rb") as f:
            payload = pickle.load(f)
            self.bm25 = payload["bm25"]
            self.metadata_store = payload["metadata_store"]

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform BM25 search for a query string."""
        if not self.bm25:
            return []

        logger.info(f"BM25 search for query: '{query}'")
        query_tokens = tokenize(query)

        # Get scores for all documents
        scores = self.bm25.get_scores(query_tokens)

        # Sort and get top_k indices
        # We need argsort in descending order
        # Fallback to pure python since numpy isn't in dependencies
        scored_indices = [(i, score) for i, score in enumerate(scores) if score > 0]
        scored_indices.sort(key=lambda x: x[1], reverse=True)

        top_indices = scored_indices[:top_k]

        results = []
        for idx, score in top_indices:
            meta = self.metadata_store[idx]
            page = int(meta["page"])
            if page < 1:
                logger.warning("Chunk %s has invalid page=%d in BM25 index; using 1", meta["chunk_id"], page)
                page = 1
            results.append(
                SearchResult(
                    chunk_id=meta["chunk_id"],
                    score=float(score),
                    source_document=meta["source_document"],
                    page=page,
                    heading=meta["heading"],
                    oracle_entities=meta["oracle_entities"],
                    text=meta["text"],
                    retrieval_method="bm25",
                )
            )

        logger.debug(f"Retrieved {len(results)} BM25 chunks.")
        return results
