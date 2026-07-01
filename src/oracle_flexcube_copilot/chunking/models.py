"""Pydantic models for the document chunking pipeline.

Represents the chunked data structure ready for embedding.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from oracle_flexcube_copilot.enrichment.models import OracleEntity, Reference


class ChunkMetadata(BaseModel):
    """Metadata attached to a chunk, preserving versioning and source context."""

    pipeline_version: str = Field(description="Pipeline version used for processing")
    chunking_version: str = Field(description="Chunking strategy version")
    embedding_model: str = Field(default="", description="Target embedding model (populated later)")
    embedding_version: str = Field(default="", description="Embedding version (populated later)")
    document_name: str | None = Field(default=None, description="Source document name")
    module_classification: str | None = Field(default=None, description="Module classification")


class Chunk(BaseModel):
    """A semantic text chunk ready for embedding."""

    id: str = Field(description="Unique chunk identifier (e.g., doc_id:chunk:N)")
    document_id: str = Field(description="Source document ID")
    text: str = Field(description="The text content of the chunk")

    # Hierarchical context
    heading_path: list[str] = Field(default_factory=list, description="Hierarchical heading path")
    section_title: str = Field(default="", description="Immediate section title")
    section_id: str = Field(default="", description="Stable section identifier")

    # Page range
    page_start: int = Field(default=0, description="Start page number")
    page_end: int = Field(default=0, description="End page number")

    # Enrichment
    oracle_entities: list[OracleEntity] = Field(
        default_factory=list, description="Entities found in this chunk"
    )
    references: list[Reference] = Field(
        default_factory=list, description="Cross-references in this chunk"
    )
    table_ids: list[str] = Field(
        default_factory=list, description="IDs of tables referenced/included"
    )

    # Metrics
    token_count: int = Field(default=0, description="Estimated token count")
    word_count: int = Field(default=0, description="Actual word count")

    # Status
    embedding_status: str = Field(
        default="pending", description="Status of embedding (pending/completed)"
    )

    # Indexing & Retrieval
    retrieval_score: float | None = Field(
        default=None, description="Similarity score during retrieval"
    )
    embedding: list[float] | None = Field(default=None, description="Vector representation")
    embedding_model: str | None = Field(default=None, description="Model used for embedding")
    indexed: bool = Field(default=False, description="Whether this chunk is in ChromaDB")

    # Metadata
    metadata: ChunkMetadata | None = Field(default=None, description="Pipeline metadata")
