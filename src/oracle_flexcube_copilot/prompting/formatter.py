from __future__ import annotations

import logging
import re

from oracle_flexcube_copilot.indexing.models import SearchResult
from oracle_flexcube_copilot.prompting.models import Citation, ContextBlock
from oracle_flexcube_copilot.prompting.templates import render_context_xml

# Pattern to extract section numbers like "7", "7.1", "7.1.1" from heading text
_SECTION_NUM_RE = re.compile(r"^(\d+(?:\.\d+)*)\b")

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
        merged = self._merge_hierarchical(merged)
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
                    document_id=current.document_id or next_chunk.document_id,
                    module=current.module or next_chunk.module,
                    section_id=current.section_id or next_chunk.section_id,
                )
            else:
                merged.append(current)
                current = next_chunk
                current_last_page = current.page

        merged.append(current)
        return merged

    @staticmethod
    def _extract_section_num(heading: str) -> tuple[int, ...]:
        """Extract numeric section number from heading text, e.g. '7.1.1 -> ...'."""
        m = _SECTION_NUM_RE.match(heading)
        if not m:
            return ()
        return tuple(int(p) for p in m.group(1).split("."))

    @staticmethod
    def _is_hierarchical_child(
        parent_num: tuple[int, ...],
        child_num: tuple[int, ...],
    ) -> bool:
        """Check if child_num is a hierarchical descendant of parent_num."""
        if not parent_num or not child_num:
            return False
        if len(child_num) <= len(parent_num):
            return False
        return child_num[: len(parent_num)] == parent_num

    def _merge_hierarchical(self, chunks: list[SearchResult]) -> list[SearchResult]:
        """Merge consecutive chunks whose headings form a parent-child hierarchy.

        For example, "7 Maintaining GL Balance Transfer" (parent),
        "7.1 GL Balance Transfer Maintenance" (child), and
        "7.1.1 Maintaining GL Balance Transfer Details" (grandchild)
        are merged into a single block with the parent heading.

        Operates on the output of _merge_adjacent (same-heading chunks are
        already combined). Only merges if consecutive in the ranked list.
        """
        if not chunks:
            return []

        merged: list[SearchResult] = []
        current = chunks[0]
        current_num = self._extract_section_num(current.heading)

        for next_chunk in chunks[1:]:
            same_doc = current.source_document == next_chunk.source_document
            next_num = self._extract_section_num(next_chunk.heading)
            is_child = (
                same_doc
                and current_num
                and next_num
                and self._is_hierarchical_child(current_num, next_num)
            )

            if is_child:
                current_last_page = current.page if current.page > 0 else next_chunk.page
                merged_page = min(current.page, next_chunk.page) if current.page > 0 and next_chunk.page > 0 else current_last_page
                merged_score = max(current.score, next_chunk.score)
                merged_entities = list(
                    set(current.oracle_entities) | set(next_chunk.oracle_entities),
                )
                current = SearchResult(
                    chunk_id=current.chunk_id,
                    score=merged_score,
                    source_document=current.source_document,
                    page=merged_page,
                    heading=current.heading,
                    oracle_entities=merged_entities,
                    text=current.text + "\n\n" + next_chunk.text,
                    retrieval_method=current.retrieval_method,
                    document_id=current.document_id or next_chunk.document_id,
                    module=current.module or next_chunk.module,
                    section_id=current.section_id or next_chunk.section_id,
                )
                current_num = self._extract_section_num(current.heading)
            else:
                merged.append(current)
                current = next_chunk
                current_num = next_num

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
                    document_id=chunk.document_id,
                    section=chunk.heading,
                    section_id=chunk.section_id or chunk.chunk_id,
                    page=chunk.page,
                    module=chunk.module,
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
