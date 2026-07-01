"""Benchmark execution engine."""

import time

from oracle_flexcube_copilot.embedding.engine import EmbeddingEngine
from oracle_flexcube_copilot.evaluation.metrics import calculate_mrr, calculate_recall_at_k
from oracle_flexcube_copilot.evaluation.models import EvalMetrics, EvalQuery
from oracle_flexcube_copilot.indexing.indexer import ChromaIndexer


class RetrievalEvaluator:
    """Evaluates the retrieval pipeline against a dataset."""

    def __init__(self, embedder: EmbeddingEngine, indexer: ChromaIndexer):
        """Initialize with an embedder and indexer."""
        self.embedder = embedder
        self.indexer = indexer

    def evaluate(self, queries: list[EvalQuery], top_k: int = 10) -> EvalMetrics:
        """Run all queries and calculate aggregate metrics."""
        total = len(queries)
        if total == 0:
            raise ValueError("Dataset is empty")

        mrr_sum = 0.0
        r5_sum = 0.0
        r10_sum = 0.0
        top1_sum = 0.0
        top3_sum = 0.0

        embed_time_sum = 0.0
        retrieve_time_sum = 0.0

        for q in queries:
            # Measure Embedding
            t0 = time.perf_counter()
            query_vector = self.embedder.embed(q.question)
            t1 = time.perf_counter()
            embed_time_sum += t1 - t0

            # Measure Retrieval
            t0 = time.perf_counter()
            results = self.indexer.search(query_vector, top_k=top_k)
            t1 = time.perf_counter()
            retrieve_time_sum += t1 - t0

            # Metrics
            mrr_sum += calculate_mrr(results, q)
            r5_sum += calculate_recall_at_k(results, q, 5)
            r10_sum += calculate_recall_at_k(results, q, 10)
            top1_sum += calculate_recall_at_k(results, q, 1)
            top3_sum += calculate_recall_at_k(results, q, 3)

        return EvalMetrics(
            total_queries=total,
            top_1_accuracy=top1_sum / total,
            top_3_accuracy=top3_sum / total,
            mrr=mrr_sum / total,
            recall_at_5=r5_sum / total,
            recall_at_10=r10_sum / total,
            avg_embedding_latency=embed_time_sum / total,
            avg_retrieval_latency=retrieve_time_sum / total,
        )
