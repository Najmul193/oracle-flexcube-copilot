from __future__ import annotations

import logging

from oracle_flexcube_copilot.indexing.models import SearchResult
from oracle_flexcube_copilot.prompting.models import Citation, ContextBlock
from oracle_flexcube_copilot.prompting.templates import render_context_xml

logger = logging.getLogger("oracle_flexcube_copilot.prompting.formatter")


class HeuristicTokenEstimator:
    """Simple token estimator using the 4-char-per-token heuristic.

    Deterministic and requires no external dependencies.
    Swap for a Qwen tokenizer later without changing other code.
    """

    def estimate(self, text: str) -> int:
        """Return estimated token count for *text*."""
        if not text:
            return 0
        return max(1, len(text) // 4)


class DefaultContextFormatter:
    """Formats retrieved chunks into XML context with dedup, merge, and truncation.

    Processing pipeline:
    1. Deduplicate by chunk_id (keep highest-ranked).
    2. Merge adjacent chunks from same document + section + section_id + consecutive pages.
    3. Assign sequential indices.
    4. Build ContextBlocks with full metadata.
    5. Render to XML.
    6. Truncate lowest-indexed blocks first if over token budget.
    """

    def __init__(self, token_estimator: HeuristicTokenEstimator | None = None) -> None:
        self._estimator = token_estimator or HeuristicTokenEstimator()

    def format(
        self,
        chunks: list[SearchResult],
        min_score: float = 0.0,
    ) -> tuple[str, list[ContextBlock], list[Citation]]:
        """Transform search results into (xml_context, blocks, citations).

        Args:
            chunks: Ranked search results from the retrieval engine.
            min_score: Minimum similarity score to include a chunk (0 = no filter).

        Returns:
            A tuple of:
            - XML-formatted context string
            - Structured context blocks (one per chunk group)
            - Citation metadata for each block
        """
        filtered = self._filter_by_score(chunks, min_score)
        deduped = self._deduplicate(filtered)
        merged = self._merge_adjacent(deduped)
        blocks = self._build_blocks(merged)
        xml = render_context_xml(blocks)
        citations = self._build_citations(blocks)
        return xml, blocks, citations

    def format_with_budget(
        self,
        chunks: list[SearchResult],
        max_tokens: int = 4096,
        min_score: float = 0.0,
    ) -> tuple[str, list[ContextBlock], list[Citation]]:
        """Format and truncate to fit within a token budget.

        Removes lowest-ranked blocks first. Never splits a block.

        Args:
            chunks: Ranked search results from the retrieval engine.
            max_tokens: Maximum allowed prompt tokens.
            min_score: Minimum similarity score to include a chunk (0 = no filter).
        """
        xml, blocks, citations = self.format(chunks, min_score=min_score)
        current_tokens = self._estimator.estimate(xml)

        if current_tokens <= max_tokens or not blocks:
            return xml, blocks, citations

        # Remove lowest-ranked (highest-indexed) blocks first
        trimmed_blocks = list(blocks)
        while trimmed_blocks and current_tokens > max_tokens:
            removed = trimmed_blocks.pop()
            current_tokens = self._estimator.estimate(
                render_context_xml(trimmed_blocks),
            )
            logger.info(
                "Truncated context: removed block %d (%s) to stay under %d tokens",
                removed.index,
                removed.chunk_id,
                max_tokens,
            )

        xml = render_context_xml(trimmed_blocks)
        citations = self._build_citations(trimmed_blocks)
        return xml, trimmed_blocks, citations

    @staticmethod
    def _filter_by_score(
        chunks: list[SearchResult],
        min_score: float,
    ) -> list[SearchResult]:
        """Remove chunks below the minimum score threshold."""
        if min_score <= 0.0:
            return chunks
        result = [c for c in chunks if c.score >= min_score]
        if len(result) < len(chunks):
            logger.info(
                "Filtered %d chunks below min_score=%s; %d remaining",
                len(chunks) - len(result),
                min_score,
                len(result),
            )
        return result

    def _deduplicate(self, chunks: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate chunk_ids, keeping the first (highest-ranked) occurrence."""
        seen: set[str] = set()
        result: list[SearchResult] = []
        for chunk in chunks:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                result.append(chunk)
        return result

    def _merge_adjacent(self, chunks: list[SearchResult]) -> list[SearchResult]:
        """Merge consecutive chunks from the same document, heading, and section_id.

        Requires consecutive or overlapping page ranges (strictly increasing,
        adjacent pages). Tracks the last page of merged groups so chained
        merges work correctly (e.g. pages 5, 6, 7).
        """
        if not chunks:
            return []

        merged: list[SearchResult] = []
        current = chunks[0]
        current_last_page = current.page

        for next_chunk in chunks[1:]:
            same_doc = current.source_document == next_chunk.source_document
            same_heading = current.heading == next_chunk.heading
            pages_adjacent = (
                next_chunk.page > current_last_page and next_chunk.page - current_last_page <= 1
            )

            if same_doc and same_heading and pages_adjacent:
                current_last_page = next_chunk.page
                current = SearchResult(
                    chunk_id=current.chunk_id,
                    score=current.score,
                    source_document=current.source_document,
                    page=current.page,
                    heading=current.heading,
                    oracle_entities=list(
                        set(current.oracle_entities) | set(next_chunk.oracle_entities),
                    ),
                    text=current.text + "\n\n" + next_chunk.text,
                    retrieval_method=current.retrieval_method,
                )
            else:
                merged.append(current)
                current = next_chunk
                current_last_page = current.page

        merged.append(current)
        return merged

    def _build_blocks(self, chunks: list[SearchResult]) -> list[ContextBlock]:
        """Convert SearchResults into ContextBlocks with sequential indices."""
        blocks: list[ContextBlock] = []
        for idx, chunk in enumerate(chunks, start=1):
            blocks.append(
                ContextBlock(
                    chunk_id=chunk.chunk_id,
                    document=chunk.source_document,
                    section=chunk.heading,
                    page=chunk.page,
                    score=chunk.score,
                    entities=list(chunk.oracle_entities),
                    text=chunk.text,
                    index=idx,
                ),
            )
        return blocks

    def _build_citations(self, blocks: list[ContextBlock]) -> list[Citation]:
        """Extract Citation metadata from ContextBlocks."""
        return [
            Citation(
                chunk_id=b.chunk_id,
                document=b.document,
                section=b.section,
                page=b.page,
                score=b.score,
            )
            for b in blocks
        ]
