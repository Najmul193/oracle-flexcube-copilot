"""Tests for evaluation metrics."""

from oracle_flexcube_copilot.evaluation.metrics import (
    calculate_mrr,
    calculate_recall_at_k,
    is_match,
)
from oracle_flexcube_copilot.evaluation.models import EvalQuery
from oracle_flexcube_copilot.indexing.models import SearchResult


def test_is_match_exact_document() -> None:
    query = EvalQuery(question="q", expected_document="doc1.pdf")
    res1 = SearchResult(
        chunk_id="1",
        score=1.0,
        source_document="doc1.pdf",
        page=1,
        heading="",
        text="",
        retrieval_method="",
    )
    res2 = SearchResult(
        chunk_id="2",
        score=1.0,
        source_document="doc2.pdf",
        page=1,
        heading="",
        text="",
        retrieval_method="",
    )

    assert is_match(res1, query) is True
    assert is_match(res2, query) is False


def test_is_match_with_section_and_keywords() -> None:
    query = EvalQuery(
        question="q",
        expected_document="doc1.pdf",
        expected_section="Introduction",
        expected_keywords=["product", "code"],
    )

    res1 = SearchResult(
        chunk_id="1",
        score=1.0,
        source_document="doc1.pdf",
        page=1,
        heading="Chapter 1 - Introduction",
        text="The product code is 123",
        retrieval_method="",
    )
    res2 = SearchResult(
        chunk_id="2",
        score=1.0,
        source_document="doc1.pdf",
        page=1,
        heading="Chapter 2",
        text="The product code is 123",
        retrieval_method="",
    )
    res3 = SearchResult(
        chunk_id="3",
        score=1.0,
        source_document="doc1.pdf",
        page=1,
        heading="Introduction",
        text="No related info here.",
        retrieval_method="",
    )

    assert is_match(res1, query) is True
    assert is_match(res2, query) is False  # Wrong section
    assert is_match(res3, query) is False  # Missing keywords


def test_calculate_mrr() -> None:
    query = EvalQuery(question="q", expected_document="target.pdf")

    r1 = SearchResult(
        chunk_id="1",
        score=1.0,
        source_document="wrong.pdf",
        page=1,
        heading="",
        text="",
        retrieval_method="",
    )
    r2 = SearchResult(
        chunk_id="2",
        score=1.0,
        source_document="target.pdf",
        page=1,
        heading="",
        text="",
        retrieval_method="",
    )
    r3 = SearchResult(
        chunk_id="3",
        score=1.0,
        source_document="target.pdf",
        page=1,
        heading="",
        text="",
        retrieval_method="",
    )

    mrr = calculate_mrr([r1, r2, r3], query)
    assert mrr == 0.5  # Found at rank 2

    mrr_none = calculate_mrr([r1], query)
    assert mrr_none == 0.0


def test_calculate_recall_at_k() -> None:
    query = EvalQuery(question="q", expected_document="target.pdf")

    r1 = SearchResult(
        chunk_id="1",
        score=1.0,
        source_document="wrong.pdf",
        page=1,
        heading="",
        text="",
        retrieval_method="",
    )
    r2 = SearchResult(
        chunk_id="2",
        score=1.0,
        source_document="wrong2.pdf",
        page=1,
        heading="",
        text="",
        retrieval_method="",
    )
    r3 = SearchResult(
        chunk_id="3",
        score=1.0,
        source_document="target.pdf",
        page=1,
        heading="",
        text="",
        retrieval_method="",
    )

    assert calculate_recall_at_k([r1, r2, r3], query, k=1) == 0.0
    assert calculate_recall_at_k([r1, r2, r3], query, k=2) == 0.0
    assert calculate_recall_at_k([r1, r2, r3], query, k=3) == 1.0
