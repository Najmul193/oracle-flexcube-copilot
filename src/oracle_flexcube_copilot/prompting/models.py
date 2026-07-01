from __future__ import annotations

from pydantic import BaseModel, Field


class ContextBlock(BaseModel):
    """A single retrieved context block with full metadata.

    No text parsing needed — all citation data is available as structured fields.
    """

    chunk_id: str = Field(description="Unique chunk identifier")
    document: str = Field(description="Source document filename")
    document_id: str = Field(
        default="",
        description="Stable document identifier (SHA-256)",
    )
    section: str | None = Field(default=None, description="Section heading")
    section_id: str | None = Field(
        default=None,
        description="Stable section identifier for merge disambiguation",
    )
    page: int = Field(description="Start page number")
    module: str = Field(
        default="",
        description="Oracle module classification (e.g. General Ledger)",
    )
    page_end: int | None = Field(
        default=None,
        description="End page number (may differ from page if multiple pages)",
    )
    score: float = Field(description="Retrieval relevance score")
    entities: list[str] = Field(
        default_factory=list,
        description="Oracle entities in this block",
    )
    text: str = Field(description="Chunk text content")
    index: int = Field(
        description="Context number used for citation references (e.g. [Context 1])",
    )


class Citation(BaseModel):
    """Citation metadata preserved for answer generation."""

    chunk_id: str = Field(description="Chunk identifier")
    document: str = Field(description="Source document filename")
    section: str | None = Field(default=None, description="Section heading")
    page: int = Field(description="Page number")
    score: float = Field(description="Retrieval score used for ranking")


class ContextConfig(BaseModel):
    """Configuration for context assembly and prompt building."""

    max_tokens: int = Field(
        default=4096,
        description="Maximum allowed prompt tokens",
    )
    min_score: float = Field(
        default=0.0,
        description="Minimum similarity score to include a chunk (0 = no filter)",
    )
    include_system_prompt: bool = Field(
        default=True,
        description="Whether to include the system prompt",
    )
    include_date: bool = Field(
        default=True,
        description="Whether to include today's date in the prompt",
    )


class PromptRequest(BaseModel):
    """The fully assembled prompt ready for the LLM.

    Contains both the formatted string and structured metadata
    so downstream components (citation engine, answer generator, reranker)
    can operate without reparsing.
    """

    system_prompt: str = Field(
        description="System instructions and answering policy",
    )
    user_prompt: str = Field(description="The user's original question")
    formatted_context: str = Field(
        description="Complete XML-formatted context string",
    )
    context_blocks: list[ContextBlock] = Field(
        description="Structured context blocks, one per retrieved chunk group",
    )
    estimated_tokens: int = Field(
        description="Estimated total token count for the assembled prompt",
    )
    citations: list[Citation] = Field(
        description="All citations for answer referencing",
    )
