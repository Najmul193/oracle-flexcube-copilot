"""Section builder — assigns blocks to hierarchical sections based on heading structure."""

from __future__ import annotations

import logging

from oracle_flexcube_copilot.enrichment.headings import heading_tree_to_flat, normalize_headings
from oracle_flexcube_copilot.enrichment.models import HeadingNode, Section
from oracle_flexcube_copilot.ingestion.models import Document

logger = logging.getLogger("oracle_flexcube_copilot.enrichment.hierarchy")


def build_sections(document: Document) -> list[Section]:
    """Build hierarchical sections by assigning blocks to their parent headings.

    Each section is defined by a heading and includes all text blocks
    between that heading and the next heading of the same or higher level.

    Args:
        document: A parsed ``Document``.

    Returns:
        A list of ``Section`` instances with parent/child relationships.
    """
    heading_tree = normalize_headings(document)
    flat_headings = heading_tree_to_flat(heading_tree)

    if not flat_headings:
        # No headings found — create a single default section
        return _create_default_section(document)

    # Build sections
    sections: list[Section] = []
    for i, heading_node in enumerate(flat_headings):
        # Determine page range
        page_start = heading_node.page
        if i + 1 < len(flat_headings):
            page_end = max(page_start, flat_headings[i + 1].page - 1)
        else:
            page_end = document.metadata.page_count

        # Collect block IDs for this section (from heading blocks + text blocks up to next heading)
        block_ids = list(heading_node.block_ids)
        _collect_block_ids(
            document,
            heading_node,
            flat_headings[i + 1] if i + 1 < len(flat_headings) else None,
            block_ids,
        )

        section = Section(
            id=f"{document.id}:sec:{i + 1}",
            title=heading_node.title,
            number=heading_node.normalized_number,
            level=heading_node.level,
            page_start=page_start,
            page_end=page_end,
            block_ids=block_ids,
        )

        # Compute word count
        section.word_count = _count_words_in_blocks(document, block_ids)

        sections.append(section)

    # Set parent/child relationships
    _assign_parent_child(sections)

    return sections


def _collect_block_ids(
    document: Document,
    current_heading: HeadingNode,
    next_heading: HeadingNode | None,
    block_ids: list[str],
) -> None:
    """Collect block IDs between this heading and the next one.

    Args:
        document: The parsed document.
        current_heading: The current heading node.
        next_heading: The next heading node (or None).
        block_ids: List to append block IDs to.
    """
    collecting = False
    for page in document.pages:
        for block in page.blocks:
            # Start collecting after the current heading
            if block.id in current_heading.block_ids:
                collecting = True
                continue

            # Stop at the next heading
            if next_heading and block.id in next_heading.block_ids:
                return

            if collecting and block.type != "heading":
                block_ids.append(block.id)


def _create_default_section(document: Document) -> list[Section]:
    """Create a single default section for documents with no headings.

    Args:
        document: The parsed document.

    Returns:
        A list with one default section.
    """
    block_ids = [b.id for page in document.pages for b in page.blocks]
    word_count = sum(
        len(p.text.split()) for page in document.pages for b in page.blocks for p in b.paragraphs
    )

    return [
        Section(
            id=f"{document.id}:sec:1",
            title=document.metadata.title or document.filename,
            number="1",
            level=1,
            page_start=1,
            page_end=document.metadata.page_count,
            block_ids=block_ids,
            word_count=word_count,
        )
    ]


def _assign_parent_child(sections: list[Section]) -> None:
    """Assign parent_id and child_ids based on heading level hierarchy.

    Args:
        sections: List of sections to update in place.
    """
    stack: list[Section] = []
    for section in sections:
        # Pop sections at the same or deeper level
        while stack and stack[-1].level >= section.level:
            stack.pop()

        if stack:
            parent = stack[-1]
            section.parent_id = parent.id
            parent.child_ids.append(section.id)

        stack.append(section)


def _count_words_in_blocks(document: Document, block_ids: list[str]) -> int:
    """Count total words across a set of blocks.

    Args:
        document: The parsed document.
        block_ids: Block IDs to count.

    Returns:
        Total word count.
    """
    word_count = 0
    block_id_set = set(block_ids)
    for page in document.pages:
        for block in page.blocks:
            if block.id in block_id_set:
                for para in block.paragraphs:
                    word_count += len(para.text.split())
    return word_count
