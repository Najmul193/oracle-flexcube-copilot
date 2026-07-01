"""Heading normalization — builds a normalized heading tree from parsed documents."""

from __future__ import annotations

import logging

from oracle_flexcube_copilot.enrichment.models import HeadingNode
from oracle_flexcube_copilot.ingestion.models import Block, Document

logger = logging.getLogger("oracle_flexcube_copilot.enrichment.headings")


def normalize_headings(document: Document) -> list[HeadingNode]:
    """Build a normalized heading tree from document headings.

    Extracts heading blocks from all pages, sorts them by page and position,
    and builds a nested tree structure with normalized section numbers.

    Args:
        document: A parsed ``Document``.

    Returns:
        A list of top-level ``HeadingNode`` instances with nested children.
    """
    # Collect all heading blocks across pages
    heading_blocks: list[tuple[int, int, Block]] = []
    for page in document.pages:
        for block in page.blocks:
            if block.type == "heading" and block.level is not None:
                heading_blocks.append((page.page_number, block.block_index, block))

    # Sort by page, then by block index
    heading_blocks.sort(key=lambda x: (x[0], x[1]))

    # Build tree
    root = _build_heading_tree(heading_blocks, document.id)
    return root


def _build_heading_tree(headings: list[tuple[int, int, Block]], doc_id: str) -> list[HeadingNode]:
    """Build a nested heading tree from a flat sorted list of heading blocks.

    Args:
        headings: Sorted list of (page_number, block_index, Block) tuples.
        doc_id: Document ID for stable block IDs.

    Returns:
        List of top-level HeadingNode instances.
    """
    # Convert to nodes first
    nodes: list[HeadingNode] = []
    for page_num, block_idx, block in headings:
        title = " ".join(p.text for p in block.paragraphs)
        nodes.append(
            HeadingNode(
                title=title,
                level=block.level or 1,
                page=page_num,
                block_ids=[block.id],
            )
        )

    if not nodes:
        return []

    # Assign normalized numbers
    _assign_numbers(nodes)

    # Build tree hierarchy
    return _nest_headings(nodes)


def _assign_numbers(nodes: list[HeadingNode]) -> None:
    """Assign hierarchical section numbers (e.g. 1, 1.1, 1.2, 2, 2.1.1).

    Args:
        nodes: Flat list of heading nodes in document order.
    """
    counters: list[int] = [0] * 10  # Max depth of 10

    for node in nodes:
        level = node.level
        # Reset counters for deeper levels
        counters[level - 1] += 1
        for i in range(level, len(counters)):
            counters[i] = 0

        parts = [str(counters[i]) for i in range(level)]
        node.normalized_number = ".".join(parts)


def _nest_headings(nodes: list[HeadingNode]) -> list[HeadingNode]:
    """Nest flat heading list into a tree hierarchy.

    Args:
        nodes: Flat list of heading nodes in document order.

    Returns:
        Top-level nodes with children nested.
    """
    root: list[HeadingNode] = []
    stack: list[HeadingNode] = []

    for node in nodes:
        # Pop stack until we find a parent
        while stack and stack[-1].level >= node.level:
            stack.pop()

        if stack:
            stack[-1].children.append(node)
        else:
            root.append(node)

        stack.append(node)

    return root


def heading_tree_to_flat(heading_tree: list[HeadingNode]) -> list[HeadingNode]:
    """Flatten a nested heading tree into a list in document order.

    Args:
        heading_tree: Nested heading tree.

    Returns:
        Flat list of HeadingNode instances.
    """
    result: list[HeadingNode] = []

    def _flatten(nodes: list[HeadingNode]) -> None:
        for node in nodes:
            result.append(node)
            _flatten(node.children)

    _flatten(heading_tree)
    return result


def get_heading_path(heading_tree: list[HeadingNode], target_id: str) -> list[str]:
    """Get the hierarchical heading path to the block containing target_id.

    Args:
        heading_tree: Nested heading tree.
        target_id: Block ID to find.

    Returns:
        List of heading titles from root to the containing heading.
    """
    path: list[str] = []

    def _search(nodes: list[HeadingNode]) -> bool:
        for node in nodes:
            if target_id in node.block_ids:
                path.append(node.title)
                return True
            if _search(node.children):
                path.insert(0, node.title)
                return True
        return False

    _search(heading_tree)
    return path
