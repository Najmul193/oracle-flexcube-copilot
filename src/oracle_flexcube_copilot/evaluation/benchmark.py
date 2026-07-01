"""Benchmark execution engine for the full hybrid retrieval pipeline."""

import time

from oracle_flexcube_copilot.embedding.engine import EmbeddingEngine
from oracle_flexcube_copilot.evaluation.metrics import (
    calculate_hit_at_k,
    calculate_mrr,
    calculate_ndcg_at_k,
    calculate_recall_at_k,
)
from oracle_flexcube_copilot.evaluation.models import EvalMetrics, EvalQuery
from oracle_flexcube_copilot.indexing.entity_index import OracleEntityIndex
from oracle_flexcube_copilot.indexing.indexer import ChromaIndexer
from oracle_flexcube_copilot.retrieval.bm25 import BM25Retriever
from oracle_flexcube_copilot.retrieval.engine import VectorRetriever
from oracle_flexcube_copilot.retrieval.entity import EntityRetriever
from oracle_flexcube_copilot.retrieval.fusion import RRFFuser


class RetrievalEvaluator:
    """Evaluates the full hybrid retrieval pipeline against a dataset."""

    def __init__(
        self,
        embedder: EmbeddingEngine,
        indexer: ChromaIndexer,
        top_k_per_source: int = 20,
    ) -> None:
        self.embedder = embedder
        self.indexer = indexer
        self.top_k_per_source = top_k_per_source

        self._vector_retriever = VectorRetriever(embedder=embedder, indexer=indexer)
        self._bm25_retriever = BM25Retriever()
        self._entity_index = OracleEntityIndex()
        self._entity_retriever = EntityRetriever(
            entity_index=self._entity_index, indexer=indexer
        )
        self._fuser = RRFFuser()

    def evaluate(self, queries: list[EvalQuery], top_k: int = 10) -> EvalMetrics:
        """Run all queries through the full hybrid pipeline and calculate metrics."""
        total = len(queries)
        if total == 0:
            raise ValueError("Dataset is empty")

        mrr_sum = 0.0
        r5_sum = 0.0
        r10_sum = 0.0
        top1_sum = 0.0
        top3_sum = 0.0
        ndcg5_sum = 0.0
        ndcg10_sum = 0.0

        embed_time_sum = 0.0
        retrieve_time_sum = 0.0

        for q in queries:
            # Embedding
            t0 = time.perf_counter()
            _ = self.embedder.embed(q.question)
            t1 = time.perf_counter()
            embed_time_sum += t1 - t0

            # Hybrid retrieval
            t0 = time.perf_counter()
            vector_results = self._vector_retriever.retrieve(
                q.question, top_k=self.top_k_per_source
            )
            bm25_results = self._bm25_retriever.retrieve(
                q.question, top_k=self.top_k_per_source
            )
            entity_results = self._entity_retriever.retrieve(
                q.question, top_k=self.top_k_per_source
            )
            results = self._fuser.fuse(
                [vector_results, bm25_results, entity_results], top_k=top_k
            )
            t1 = time.perf_counter()
            retrieve_time_sum += t1 - t0

            # Metrics
            mrr_sum += calculate_mrr(results, q)
            r5_sum += calculate_recall_at_k(results, q, 5)
            r10_sum += calculate_recall_at_k(results, q, 10)
            top1_sum += calculate_hit_at_k(results, q, 1)
            top3_sum += calculate_hit_at_k(results, q, 3)
            ndcg5_sum += calculate_ndcg_at_k(results, q, 5)
            ndcg10_sum += calculate_ndcg_at_k(results, q, 10)

        n = float(total)
        return EvalMetrics(
            total_queries=total,
            top_1_accuracy=top1_sum / n,
            top_3_accuracy=top3_sum / n,
            mrr=mrr_sum / n,
            recall_at_5=r5_sum / n,
            recall_at_10=r10_sum / n,
            hit_at_1=top1_sum / n,
            ndcg_at_5=ndcg5_sum / n,
            ndcg_at_10=ndcg10_sum / n,
            avg_embedding_latency=embed_time_sum / n,
            avg_retrieval_latency=retrieve_time_sum / n,
        )
