"""Report formatting."""

from oracle_flexcube_copilot.evaluation.models import EvalMetrics


def generate_markdown_report(metrics: EvalMetrics) -> str:
    """Format EvalMetrics as a Markdown report."""
    return f"""# Retrieval Evaluation Report

| Metric | Score |
| ------ | ----- |
| Total Queries | {metrics.total_queries} |
| Top-1 Accuracy | {metrics.top_1_accuracy:.2%} |
| Top-3 Accuracy | {metrics.top_3_accuracy:.2%} |
| Recall@5 | {metrics.recall_at_5:.2%} |
| Recall@10 | {metrics.recall_at_10:.2%} |
| MRR | {metrics.mrr:.3f} |

### Performance
* **Avg Embedding Latency**: {metrics.avg_embedding_latency * 1000:.1f} ms
* **Avg Retrieval Latency**: {metrics.avg_retrieval_latency * 1000:.1f} ms
"""
