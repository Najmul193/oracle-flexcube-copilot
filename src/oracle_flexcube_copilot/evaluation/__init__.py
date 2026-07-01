"""Evaluation Framework."""

from __future__ import annotations

from oracle_flexcube_copilot.evaluation.benchmark import RetrievalEvaluator
from oracle_flexcube_copilot.evaluation.dataset import load_dataset
from oracle_flexcube_copilot.evaluation.metrics import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
    is_match,
)
from oracle_flexcube_copilot.evaluation.models import EvalMetrics, EvalQuery
from oracle_flexcube_copilot.evaluation.report import generate_markdown_report

__all__ = [
    "EvalMetrics",
    "EvalQuery",
    "RetrievalEvaluator",
    "calculate_hit_at_k",
    "calculate_mrr",
    "calculate_ndcg_at_k",
    "calculate_recall_at_k",
    "generate_markdown_report",
    "is_match",
    "load_dataset",
]
