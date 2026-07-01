"""Metrics calculation for retrieval evaluation."""

import math

from oracle_flexcube_copilot.evaluation.models import EvalQuery
from oracle_flexcube_copilot.indexing.models import SearchResult


def is_match(result: SearchResult, query: EvalQuery) -> bool:
    """Determine if a search result matches the expected ground truth."""
    if query.expected_document and query.expected_document != result.source_document:
        return False

    if query.expected_section and query.expected_section.lower() not in result.heading.lower():
        return False

    if query.expected_page and query.expected_page > 0 and result.page != query.expected_page:
        return False

    if query.expected_keywords:
        text_lower = result.text.lower()
        for kw in query.expected_keywords:
            if kw.lower() not in text_lower:
                return False

    return True


def calculate_hit_at_k(results: list[SearchResult], query: EvalQuery, k: int) -> float:
    """Hit@k — 1 if correct document appears in top k, else 0."""
    for res in results[:k]:
        if is_match(res, query):
            return 1.0
    return 0.0


def calculate_mrr(results: list[SearchResult], query: EvalQuery) -> float:
    """Mean Reciprocal Rank for a single query."""
    for i, res in enumerate(results):
        if is_match(res, query):
            return 1.0 / (i + 1)
    return 0.0


def calculate_recall_at_k(results: list[SearchResult], query: EvalQuery, k: int) -> float:
    """Recall@k — 1 if correct document is in top k, else 0."""
    return calculate_hit_at_k(results, query, k)


def calculate_ndcg_at_k(results: list[SearchResult], query: EvalQuery, k: int) -> float:
    """NDCG@k — Normalized Discounted Cumulative Gain at rank k.

    With binary relevance (single relevant document), this equals:
        DCG@k = 1 / log2(pos + 1)  if relevant doc at position pos <= k, else 0
        IDCG = 1.0                 (best case: relevant doc at rank 1)
        NDCG = DCG / IDCG
    """
    for i, res in enumerate(results[:k], start=1):
        if is_match(res, query):
            return 1.0 / math.log2(i + 1)
    return 0.0
