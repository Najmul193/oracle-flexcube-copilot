"""Metrics calculation for retrieval evaluation."""

from oracle_flexcube_copilot.evaluation.models import EvalQuery
from oracle_flexcube_copilot.indexing.models import SearchResult


def is_match(result: SearchResult, query: EvalQuery) -> bool:
    """Determine if a search result matches the expected ground truth."""
    if query.expected_document and query.expected_document != result.source_document:
        return False
        
    if query.expected_section:
        # A simple check: either the exact section name, or it's part of the heading path
        if query.expected_section.lower() not in result.heading.lower():
            return False
            
    if query.expected_keywords:
        text_lower = result.text.lower()
        for kw in query.expected_keywords:
            if kw.lower() not in text_lower:
                return False
                
    return True


def calculate_mrr(results: list[SearchResult], query: EvalQuery) -> float:
    """Calculate Mean Reciprocal Rank for a single query."""
    for i, res in enumerate(results):
        if is_match(res, query):
            return 1.0 / (i + 1)
    return 0.0


def calculate_recall_at_k(results: list[SearchResult], query: EvalQuery, k: int) -> float:
    """Calculate Recall at K (1.0 if correct document is in top K, else 0.0)."""
    for res in results[:k]:
        if is_match(res, query):
            return 1.0
    return 0.0
