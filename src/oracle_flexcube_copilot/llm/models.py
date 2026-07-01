"""LLM Integration Models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class LLMConfig(BaseModel):
    """Configuration for the LLM client."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama base URL")
    model: str = Field(default="qwen3:8b", description="Model name to use")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    repeat_penalty: float = Field(
        default=1.1, ge=0.0, description="Penalty for repeating tokens"
    )
    num_ctx: int = Field(default=8192, ge=1, description="Context window size")
    num_predict: int = Field(default=2048, ge=1, description="Max tokens to generate")
    timeout: int = Field(default=120, ge=1, description="Request timeout in seconds")


class Citation(BaseModel):
    """A single citation referencing a retrieved chunk."""

    document: str = Field(description="Source document filename")
    section: str | None = Field(default=None, description="Section heading")
    page: int = Field(description="Page number")
    score: float = Field(description="Retrieval score")


class AnswerMetadata(BaseModel):
    """Token usage and timing metadata for an answer."""

    prompt_tokens: int = Field(default=0, description="Tokens in the prompt")
    completion_tokens: int = Field(default=0, description="Tokens in the completion")
    total_tokens: int = Field(default=0, description="Total tokens used")
    retrieval_time: float = Field(default=0.0, description="Retrieval time in seconds")
    generation_time: float = Field(default=0.0, description="Generation time in seconds")
    model_name: str = Field(default="", description="Model used for generation")


class AnswerResponse(BaseModel):
    """Complete response from the answer generator."""

    answer: str = Field(description="Generated answer text")
    citations: list[Citation] = Field(
        default_factory=list, description="Attached citations from retrieved chunks"
    )
    confidence: str = Field(default="Low", description="Confidence level (High/Medium/Low)")
    confidence_percentage: float = Field(
        default=0.0, ge=0.0, le=100.0, description="Confidence as percentage"
    )
    reasoning_time: float = Field(
        default=0.0, description="Total reasoning time (retrieval + generation)"
    )
    metadata: AnswerMetadata = Field(
        default_factory=AnswerMetadata, description="Token and timing metadata"
    )
    mode: str = Field(default="concise", description="Answer mode used (concise/detailed/expert)")


class AnswerMetrics(BaseModel):
    """Aggregated answer metrics for evaluation."""

    total_questions: int = Field(default=0, description="Number of questions answered")
    avg_generation_time: float = Field(default=0.0, description="Average generation time")
    avg_total_latency: float = Field(default=0.0, description="Average total latency")
    avg_confidence: float = Field(default=0.0, description="Average confidence percentage")
    total_tokens: int = Field(default=0, description="Total completion tokens")
    model_name: str = Field(default="", description="Model used")
