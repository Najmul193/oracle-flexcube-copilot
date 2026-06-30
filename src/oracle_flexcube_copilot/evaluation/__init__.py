"""Evaluation Framework."""

from __future__ import annotations

from oracle_flexcube_copilot.evaluation.models import EvalQuery, EvalMetrics
from oracle_flexcube_copilot.evaluation.dataset import load_dataset
from oracle_flexcube_copilot.evaluation.metrics import calculate_mrr, calculate_recall_at_k
from oracle_flexcube_copilot.evaluation.benchmark import RetrievalEvaluator
from oracle_flexcube_copilot.evaluation.report import generate_markdown_report

__all__ = [
    "EvalQuery",
    "EvalMetrics",
    "load_dataset",
    "calculate_mrr",
    "calculate_recall_at_k",
    "RetrievalEvaluator",
    "generate_markdown_report",
]
