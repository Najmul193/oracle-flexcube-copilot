"""Reciprocal Rank Fusion (RRF) for combining multiple retrieval strategies."""

import logging
from collections import defaultdict

from oracle_flexcube_copilot.indexing.models import SearchResult

logger = logging.getLogger(__name__)


class RRFFuser:
    """Combines multiple search result lists using Reciprocal Rank Fusion."""

    def __init__(self, k: int = 60):
        """Initialize the fuser.

        Args:
            k: The constant 'k' used in RRF (default 60 is standard).
        """
        self.k = k

    def fuse(self, result_lists: list[list[SearchResult]], top_k: int = 5) -> list[SearchResult]:
        """Apply Reciprocal Rank Fusion to multiple result lists.

        Formula: rrf_score = sum(1 / (k + rank_in_list))

        Args:
            result_lists: A list of lists of SearchResult objects.
            top_k: Number of final results to return.

        Returns:
            The combined top_k SearchResult objects, sorted by RRF score.
        """
        # Dictionary to accumulate scores for each unique chunk_id
        rrf_scores: dict[str, float] = defaultdict(float)

        # Dictionary to keep the best SearchResult object for each chunk_id
        # We prefer keeping the one with the original score/metadata if needed,
        # but since metadata is identical, any instance is fine. We will update
        # its score to the fused score later.
        chunk_map: dict[str, SearchResult] = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                chunk_id = result.chunk_id

                # Calculate RRF score for this list
                score = 1.0 / (self.k + rank)
                rrf_scores[chunk_id] += score

                if chunk_id not in chunk_map:
                    # Make a shallow copy so we don't mutate the original result's score yet
                    chunk_map[chunk_id] = SearchResult(
                        chunk_id=result.chunk_id,
                        score=result.score,
                        source_document=result.source_document,
                        page=result.page,
                        heading=result.heading,
                        oracle_entities=result.oracle_entities,
                        text=result.text,
                        retrieval_method=result.retrieval_method,
                    )
                else:
                    # If seen again from a different method, update the retrieval_method string
                    # to indicate it was found by multiple methods
                    current = chunk_map[chunk_id]
                    if result.retrieval_method not in current.retrieval_method:
                        current.retrieval_method += f"+{result.retrieval_method}"

        # Create a list of results with updated scores
        fused_results = []
        for chunk_id, total_score in rrf_scores.items():
            res = chunk_map[chunk_id]
            # Replace the original similarity score with the RRF score
            res.score = total_score
            fused_results.append(res)

        # Sort by RRF score descending
        fused_results.sort(key=lambda x: x.score, reverse=True)

        top_results = fused_results[:top_k]

        # Normalize fused scores to [0, 1] so citations are comparable
        if top_results:
            max_score = max(r.score for r in top_results)
            min_score = min(r.score for r in top_results)
            score_range = max_score - min_score
            if score_range > 0:
                for r in top_results:
                    r.score = (r.score - min_score) / score_range

        logger.info("Fused %d unique results down to top %d.", len(fused_results), len(top_results))

        return top_results
