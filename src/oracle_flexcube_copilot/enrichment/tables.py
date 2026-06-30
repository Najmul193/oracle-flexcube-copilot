"""Table extraction — detects and preserves tabular structures from document blocks."""

from __future__ import annotations

import logging
import re
from typing import Any

from oracle_flexcube_copilot.enrichment.models import TableData
from oracle_flexcube_copilot.ingestion.models import Document

logger = logging.getLogger("oracle_flexcube_copilot.enrichment.tables")

# Heuristic: if a block's text lines contain consistent delimiters, it's likely a table
TABLE_DELIMITER_PATTERNS = [
    re.compile(r"\s{2,}"),  # Two or more spaces
    re.compile(r"\t"),       # Tabs
    re.compile(r"\|"),       # Pipe
]


def extract_tables(document: Document) -> list[TableData]:
    """Extract table structures from document blocks.

    Scans blocks for tabular patterns (consistent delimiters, multiple rows
    with similar structure) and preserves them as structured TableData.

    Args:
        document: A parsed ``Document``.

    Returns:
        A list of ``TableData`` instances.
    """
    tables: list[TableData] = []
    table_counter = 0

    for page in document.pages:
        for block in page.blocks:
            if block.type == "table":
                # Block was already identified as a table during parsing
                if block.table:
                    table_counter += 1
                    tables.append(
                        TableData(
                            id=f"{document.id}:tbl:{table_counter}",
                            source_block_id=block.id,
                            page=page.page_number,
                            title=block.table.title,
                            headers=block.table.headers,
                            rows=block.table.rows,
                            num_rows=len(block.table.rows),
                            num_cols=len(block.table.headers) if block.table.headers else (
                                len(block.table.rows[0]) if block.table.rows else 0
                            ),
                        )
                    )
            elif block.type == "text":
                # Try to detect tables in text blocks
                table = _detect_table_in_text(block, page.page_number, document.id)
                if table:
                    table_counter += 1
                    table.id = f"{document.id}:tbl:{table_counter}"
                    tables.append(table)

    logger.info("Extracted %d tables", len(tables))
    return tables


def _detect_table_in_text(block: Any, page_number: int, doc_id: str) -> TableData | None:
    """Try to detect a table structure within a text block.

    Args:
        block: A Block from the ingestion pipeline.
        page_number: The page number.
        doc_id: The document ID.

    Returns:
        A TableData instance if a table is detected, None otherwise.
    """
    if not block.paragraphs:
        return None

    # Join all paragraph text
    full_text = "\n".join(p.text for p in block.paragraphs)
    lines = full_text.strip().split("\n")

    if len(lines) < 2:
        return None

    # Check for consistent delimiter patterns
    for delimiter_pattern in TABLE_DELIMITER_PATTERNS:
        if _is_likely_table(lines, delimiter_pattern):
            return _parse_table_lines(lines, block.id, page_number)

    return None


def _is_likely_table(lines: list[str], delimiter: re.Pattern) -> bool:
    """Check if a set of lines looks like a table based on delimiter consistency.

    Args:
        lines: The text lines to check.
        delimiter: The delimiter pattern to test.

    Returns:
        True if the lines look like a table.
    """
    if len(lines) < 2:
        return False

    # Count columns per line using the delimiter
    col_counts: list[int] = []
    for line in lines:
        if line.strip():
            cols = len(delimiter.split(line.strip()))
            col_counts.append(cols)

    if len(col_counts) < 2:
        return False

    # Check if most lines have the same number of columns
    from statistics import mode, StatisticsError
    try:
        most_common = mode(col_counts)
        consistent = sum(1 for c in col_counts if c == most_common)
        return consistent >= len(col_counts) * 0.6 and most_common >= 2
    except StatisticsError:
        return False


def _parse_table_lines(lines: list[str], block_id: str, page_number: int) -> TableData | None:
    """Parse table lines into headers and rows.

    Args:
        lines: The text lines.
        block_id: The source block ID.
        page_number: The page number.

    Returns:
        A TableData instance.
    """
    # Try space-splitting first, then tab-splitting
    for delimiter in [re.compile(r"\s{2,}"), re.compile(r"\t")]:
        rows: list[list[str]] = []
        for line in lines:
            if line.strip():
                cells = [cell.strip() for cell in delimiter.split(line.strip()) if cell.strip()]
                if cells:
                    rows.append(cells)

        if len(rows) >= 2:
            # First row is likely headers
            headers = rows[0]
            data_rows = rows[1:]

            return TableData(
                id="",
                source_block_id=block_id,
                page=page_number,
                title="",
                headers=headers,
                rows=data_rows,
                num_rows=len(data_rows),
                num_cols=len(headers),
            )

    return None