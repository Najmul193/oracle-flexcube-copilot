"""Pydantic models for indexing and vector search."""

from __future__ import annotations

from pydantic import BaseModel, Field


class IndexMetrics(BaseModel):
    """Metrics tracking indexing operations."""

    chunks_added: int = Field(default=0, description="Number of new chunks inserted")
    chunks_updated: int = Field(default=0, description="Number of existing chunks updated")
    chunks_skipped: int = Field(default=0, description="Number of chunks skipped (duplicates)")
    chunks_deleted: int = Field(default=0, description="Number of chunks deleted")
    documents_deleted: int = Field(default=0, description="Number of documents completely purged")


class IndexHealth(BaseModel):
    """Health metrics for the vector index."""

    collection_name: str = Field(description="Name of the ChromaDB collection")
    total_documents: int = Field(description="Total unique documents in index")
    total_chunks: int = Field(description="Total chunks in index")
    is_accessible: bool = Field(description="True if ChromaDB is responding")
    error: str | None = Field(default=None, description="Error message if unhealthy")


class SearchResult(BaseModel):
    """Unified search result model across all retrieval engines."""

    chunk_id: str = Field(description="Unique ID of the chunk")
    score: float = Field(description="Relevance score (e.g. cosine distance, BM25 score)")
    source_document: str = Field(description="Name of the source document")
    page: int = Field(description="Page number of the chunk")
    heading: str = Field(description="Closest heading or section title")
    oracle_entities: list[str] = Field(default_factory=list, description="Extracted Oracle entities in chunk")
    text: str = Field(description="The text content of the chunk")
    retrieval_method: str = Field(description="The retrieval engine used (e.g. vector, bm25, exact)")

