"""Models for the evaluation framework."""

from pydantic import BaseModel, Field


class EvalQuery(BaseModel):
    """A single evaluation query and its expected ground truth."""

    question: str = Field(description="The user query")
    expected_document: str = Field(description="Filename of the document containing the answer")
    expected_section: str | None = Field(
        default=None, description="Optional expected section title"
    )
    expected_page: int | None = Field(
        default=None, description="Optional expected page number"
    )
    expected_screen: str | None = Field(
        default=None, description="Optional expected Oracle screen/transaction code"
    )
    expected_keywords: list[str] = Field(
        default_factory=list, description="Keywords expected in the answer"
    )


class EvalMetrics(BaseModel):
    """Aggregated evaluation metrics for a benchmark run."""

    total_queries: int = Field(description="Number of queries evaluated")
    top_1_accuracy: float = Field(description="Hit@1 — fraction where correct doc is rank 1")
    top_3_accuracy: float = Field(description="Hit@3 — fraction where correct doc is in top 3")
    mrr: float = Field(description="Mean Reciprocal Rank")
    recall_at_5: float = Field(description="Recall@5 — fraction where correct doc is in top 5")
    recall_at_10: float = Field(description="Recall@10 — fraction where correct doc is in top 10")
    hit_at_1: float = Field(description="Hit@1 (alias for top_1_accuracy)")
    ndcg_at_5: float = Field(
        description="NDCG@5 — Normalized Discounted Cumulative Gain at rank 5"
    )
    ndcg_at_10: float = Field(
        description="NDCG@10 — Normalized Discounted Cumulative Gain at rank 10"
    )
    avg_embedding_latency: float = Field(
        description="Average embedding time per query in seconds"
    )
    avg_retrieval_latency: float = Field(
        description="Average retrieval time per query in seconds"
    )
