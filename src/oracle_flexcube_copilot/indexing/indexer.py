"""Indexer module wrapping ChromaDB persistent client."""

from __future__ import annotations

import json
import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from oracle_flexcube_copilot.chunking.models import ChunkMetadata
from oracle_flexcube_copilot.config import settings
from oracle_flexcube_copilot.embedding.models import EmbeddedChunk
from oracle_flexcube_copilot.indexing.entity_index import OracleEntityIndex
from oracle_flexcube_copilot.indexing.models import IndexHealth, IndexMetrics, SearchResult

logger = logging.getLogger("oracle_flexcube_copilot.indexing.indexer")


class ChromaIndexer:
    """Manages persistent ChromaDB vector indexing and retrieval."""

    def __init__(self, db_dir: str | None = None, collection_name: str | None = None):
        """Initialize persistent ChromaDB client."""
        self.db_dir = db_dir or str(settings.resolved_cache_dir.parent / "chroma_db")
        self.collection_name = collection_name or settings.chroma_collection_name

        self.client = chromadb.PersistentClient(
            path=self.db_dir, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.entity_index = OracleEntityIndex(db_dir=self.db_dir)

    def _serialize_metadata(self, chunk: EmbeddedChunk) -> dict[str, Any]:
        """Convert a chunk to flat ChromaDB-compatible metadata."""
        c = chunk.chunk

        # Safe defaults if metadata is somehow missing
        module_class = ""
        chunking_ver = ""
        embedding_ver = chunk.embedding_version

        if c.metadata:
            module_class = c.metadata.module_classification or ""
            chunking_ver = c.metadata.chunking_version or ""

        # Serialize nested lists to JSON strings
        heading_path_str = json.dumps(c.heading_path) if c.heading_path else "[]"

        # Extract basic info from OracleEntity list
        entities = []
        if c.oracle_entities:
            entities = [{"name": e.name, "type": e.entity_type} for e in c.oracle_entities]
        entities_str = json.dumps(entities)

        raw = {
            "chunk_id": c.id,
            "document_id": c.document_id,
            "document_name": c.metadata.document_name if c.metadata else "",
            "module": module_class,
            "section": c.section_title or "",
            "section_id": c.section_id or "",
            "heading_path": heading_path_str,
            "page_start": c.page_start,
            "page_end": c.page_end,
            "oracle_entities": entities_str,
            "processing_stage": "INDEXED",
            "chunking_version": chunking_ver,
            "embedding_version": embedding_ver,
            "pipeline_version": settings.pipeline_version,
        }
        # ChromaDB only accepts str/int/float/bool — sanitize every value
        return {k: self._to_meta_value(v) for k, v in raw.items()}

    @staticmethod
    def _to_meta_value(v: Any) -> str | int | float | bool:
        """Convert any value to a ChromaDB-safe metadata type."""
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return v
        if isinstance(v, float):
            return v
        if isinstance(v, str):
            return v
        if v is None:
            return ""
        # Lists, dicts, etc. → JSON string
        try:
            return json.dumps(v)
        except Exception:
            return str(v)

    def _deserialize_metadata(self, meta: dict[str, Any]) -> ChunkMetadata:
        """Helper to unpack JSON strings back to python objects (not fully reconstructing Chunk here)."""
        # Usually search results return this metadata dict.
        return meta

    def add_chunks(self, chunks: list[EmbeddedChunk]) -> IndexMetrics:
        """Insert new chunks. Skips ones that already exist."""
        metrics = IndexMetrics()
        if not chunks:
            return metrics

        # Check existing ids
        chunk_ids = [c.chunk.id for c in chunks]
        existing = self.collection.get(ids=chunk_ids, include=[])
        existing_set = set(existing.get("ids", []))

        to_insert = [c for c in chunks if c.chunk.id not in existing_set]
        metrics.chunks_skipped = len(chunks) - len(to_insert)

        if to_insert:
            ids = [c.chunk.id for c in to_insert]
            embeddings = [c.embedding for c in to_insert]
            documents = [c.chunk.text for c in to_insert]
            metadatas = [self._serialize_metadata(c) for c in to_insert]

            self.collection.add(
                ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
            )
            metrics.chunks_added = len(to_insert)

            # Update the indexed status in the objects and index entities
            for c in to_insert:
                c.chunk.indexed = True
                self.entity_index.index_chunk(c.chunk)

        return metrics

    def update_chunks(self, chunks: list[EmbeddedChunk]) -> IndexMetrics:
        """Upsert chunks (update if exist, insert if not)."""
        metrics = IndexMetrics()
        if not chunks:
            return metrics

        ids = [c.chunk.id for c in chunks]
        embeddings = [c.embedding for c in chunks]
        documents = [c.chunk.text for c in chunks]
        metadatas = [self._serialize_metadata(c) for c in chunks]

        self.collection.upsert(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

        # It's an upsert so Chroma doesn't tell us exactly how many were added vs updated
        # But we treat it as updated
        metrics.chunks_updated = len(chunks)

        for c in chunks:
            c.chunk.indexed = True
            # For entity index, simplest is to remove the chunk and re-add to avoid dups/stale
            self.entity_index.remove_chunk(c.chunk.id)
            self.entity_index.index_chunk(c.chunk)

        return metrics

    def delete_document(self, document_id: str) -> IndexMetrics:
        """Delete all chunks belonging to a document."""
        metrics = IndexMetrics()

        # Find all chunks for this doc
        results = self.collection.get(where={"document_id": document_id}, include=[])
        ids = results.get("ids", [])

        if ids:
            self.collection.delete(ids=ids)
            metrics.chunks_deleted = len(ids)
            metrics.documents_deleted = 1

            # Sync with Entity Index
            self.entity_index.remove_document(document_id)

        return metrics

    def delete_chunk(self, chunk_id: str) -> IndexMetrics:
        """Delete a single chunk by ID."""
        metrics = IndexMetrics()
        self.collection.delete(ids=[chunk_id])
        # Chroma API doesn't return count of deleted, assume 1
        metrics.chunks_deleted = 1

        # Sync with Entity Index
        self.entity_index.remove_chunk(chunk_id)

        return metrics

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        """Retrieve a raw chunk payload from Chroma."""
        results = self.collection.get(
            ids=[chunk_id], include=["metadatas", "documents", "embeddings"]
        )
        if not results or not results.get("ids"):
            return None

        return {
            "id": results["ids"][0],
            "document": results["documents"][0] if results.get("documents") is not None else "",
            "metadata": results["metadatas"][0] if results.get("metadatas") is not None else {},
            "embedding": results["embeddings"][0] if results.get("embeddings") is not None else [],
        }

    def get_document(self, document_id: str) -> list[dict[str, Any]]:
        """Retrieve all raw chunk payloads for a document."""
        results = self.collection.get(
            where={"document_id": document_id}, include=["metadatas", "documents"]
        )
        ids = results.get("ids", [])
        docs = []
        for i, cid in enumerate(ids):
            docs.append(
                {
                    "id": cid,
                    "document": results["documents"][i]
                    if results.get("documents") is not None
                    else "",
                    "metadata": results["metadatas"][i]
                    if results.get("metadatas") is not None
                    else {},
                }
            )
        return docs

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[SearchResult]:
        """Perform similarity search."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )

        if not results or not results.get("ids") or not results["ids"][0]:
            return []

        hits = []
        for i, cid in enumerate(results["ids"][0]):
            meta = results["metadatas"][0][i] if results.get("metadatas") is not None else {}
            doc_text = results["documents"][0][i] if results.get("documents") is not None else ""
            distance = results["distances"][0][i] if results.get("distances") is not None else 0.0

            # Parse JSON fields safely
            try:
                entities_list = json.loads(meta.get("oracle_entities", "[]"))
                entity_names = [e.get("name") for e in entities_list if "name" in e]
            except Exception:
                entity_names = []

            page = int(meta.get("page_start", 0))
            if page == 0:
                logger.warning(
                    "Chunk %s has page_start=0 in vector index — stale data from before "
                    "Block.page_number fix. Re-index to correct.",
                    cid,
                )
            heading = meta.get("section", "")
            if not heading:
                try:
                    h_path = json.loads(meta.get("heading_path", "[]"))
                    if h_path:
                        heading = h_path[-1]
                except Exception:
                    pass

            hits.append(
                SearchResult(
                    chunk_id=cid,
                    score=float(distance),
                    source_document=str(meta.get("document_name", "")),
                    page=int(page),
                    heading=str(heading),
                    oracle_entities=entity_names,
                    text=str(doc_text),
                    retrieval_method="vector",
                    document_id=str(meta.get("document_id", "")),
                    module=str(meta.get("module", "")),
                    section_id=str(meta.get("section_id", "")),
                )
            )

        return hits

    def reset_collection(self) -> None:
        """Drop and recreate the entire collection."""
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name, metadata={"hnsw:space": "cosine"}
        )
        self.entity_index.clear()

    def health_check(self) -> IndexHealth:
        """Validate collection accessibility and return stats."""
        try:
            total_chunks = self.collection.count()
            # Approximation for total docs by doing a limit query
            # A true total docs query is expensive, we'd need distinct document_ids
            # We'll just return the chunk count and true
            return IndexHealth(
                collection_name=self.collection_name,
                total_documents=-1,  # Complex to calculate natively
                total_chunks=total_chunks,
                is_accessible=True,
            )
        except Exception as e:
            return IndexHealth(
                collection_name=self.collection_name,
                total_documents=0,
                total_chunks=0,
                is_accessible=False,
                error=str(e),
            )
