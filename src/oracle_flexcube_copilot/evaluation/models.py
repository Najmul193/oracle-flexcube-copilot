"""Models for the evaluation framework."""

from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    """A single evaluation query and its expected ground truth."""
    
    question: str = Field(description="The user query")
    expected_document: str = Field(description="Filename of the document containing the answer")
    expected_section: str | None = Field(default=None, description="Optional expected section title")
    expected_keywords: list[str] = Field(default_factory=list, description="Keywords expected in the answer")


class EvalMetrics(BaseModel):
    """Aggregated evaluation metrics for a benchmark run."""
    
    total_queries: int
    top_1_accuracy: float
    top_3_accuracy: float
    mrr: float
    recall_at_5: float
    recall_at_10: float
    avg_embedding_latency: float
    avg_retrieval_latency: float
