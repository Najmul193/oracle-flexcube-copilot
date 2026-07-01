"""Oracle Entity exact-match retrieval for hybrid search."""

from __future__ import annotations

import logging
import re

from oracle_flexcube_copilot.indexing.entity_index import OracleEntityIndex
from oracle_flexcube_copilot.indexing.indexer import ChromaIndexer
from oracle_flexcube_copilot.indexing.models import SearchResult

logger = logging.getLogger(__name__)

# Oracle entity identifier pattern (same as classification.py)
_ENTITY_RE = re.compile(r'\b([A-Z0-9_]{4,20})\b')


def extract_entity_names(query: str) -> list[str]:
    """Extract potential Oracle entity identifiers from a query string.

    Returns uppercase entity names matching the Oracle identifier pattern.
    """
    matches = _ENTITY_RE.findall(query)
    seen: set[str] = set()
    result: list[str] = []
    for m in matches:
        if m not in seen:
            seen.add(m)
            result.append(m)
    return result


class EntityRetriever:
    """Retrieves chunks by exact Oracle entity name lookup.

    Extracts entity identifiers from a query, looks them up in the
    OracleEntityIndex, and returns matching chunks with full text.
    """

    def __init__(
        self,
        entity_index: OracleEntityIndex,
        indexer: ChromaIndexer,
    ) -> None:
        self._entity_index = entity_index
        self._indexer = indexer

    def retrieve(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Search by extracting entity names and looking them up in the index.

        Returns up to top_k results.
        """
        entity_names = extract_entity_names(query)
        if not entity_names:
            return []

        logger.info("Entity search for names: %s", entity_names)

        # Collect chunk IDs from entity index
        seen_chunks: dict[str, float] = {}
        for name in entity_names:
            locations = self._entity_index.lookup(name)
            for loc in locations:
                if loc.chunk_id not in seen_chunks:
                    seen_chunks[loc.chunk_id] = 1.0

        if not seen_chunks:
            return []

        # Fetch full text from ChromaDB by IDs
        chunk_ids = list(seen_chunks.keys())
        try:
            collection = self._indexer.collection
            fetched = collection.get(ids=chunk_ids, include=["documents", "metadatas"])
        except Exception:
            logger.warning("Failed to fetch entity-matched chunks from ChromaDB", exc_info=True)
            return []

        results: list[SearchResult] = []
        if fetched and fetched.get("ids"):
            for i, cid in enumerate(fetched["ids"][0]):
                meta = fetched["metadatas"][0][i] if fetched.get("metadatas") else {}
                doc_text = fetched["documents"][0][i] if fetched.get("documents") else ""
                results.append(
                    SearchResult(
                        chunk_id=cid,
                        score=1.0,
                        source_document=str(meta.get("document_name", "")),
                        page=int(meta.get("page_start", 0)),
                        heading=str(meta.get("section", "")),
                        oracle_entities=[],
                        text=str(doc_text),
                        retrieval_method="entity",
                        document_id=str(meta.get("document_id", "")),
                        module=str(meta.get("module", "")),
                        section_id=str(meta.get("section_id", "")),
                    )
                )

        logger.info("Entity search returned %d results", len(results))
        return results[:top_k]
