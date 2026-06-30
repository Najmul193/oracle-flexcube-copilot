"""PDF parser — extracts structured content from PyMuPDF documents.

Produces a hierarchical representation:

    Page
        └── Blocks (heading, text, list, table, figure)
                └── Paragraphs
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Generator

import fitz

from oracle_flexcube_copilot.exceptions import PDFProcessingError
from oracle_flexcube_copilot.ingestion.models import Block, DocumentMetadata, Page, Paragraph

logger = logging.getLogger("oracle_flexcube_copilot.ingestion.parser")

# Font sizes above this threshold (relative to the page's median) are considered headings
HEADING_FONT_THRESHOLD = 1.2


def parse_document_metadata(doc: fitz.Document) -> DocumentMetadata:
    """Extract document-level metadata from a PyMuPDF document.

    Args:
        doc: An opened PyMuPDF document.

    Returns:
        A ``DocumentMetadata`` instance with the extracted fields.
    """
    meta: dict[str, Any] = doc.metadata or {}

    creation_date = _parse_pdf_date(meta.get("creationDate"))
    modification_date = _parse_pdf_date(meta.get("modDate"))

    return DocumentMetadata(
        title=meta.get("title", "") or "",
        author=meta.get("author", "") or "",
        subject=meta.get("subject", "") or "",
        keywords=meta.get("keywords", "") or "",
        producer=meta.get("producer", "") or "",
        creator=meta.get("creator", "") or "",
        creation_date=creation_date,
        modification_date=modification_date,
        page_count=doc.page_count,
    )


def parse_pages(doc: fitz.Document) -> Generator[Page, None, None]:
    """Yield ``Page`` objects for every page in the document.

    Each page is parsed into structural blocks (headings, text) and paragraphs.
    Page-level word and character counts are computed.

    Args:
        doc: An opened PyMuPDF document.

    Yields:
        ``Page`` objects with populated blocks.

    Raises:
        PDFProcessingError: If a page cannot be parsed.
    """
    for page_number in range(doc.page_count):
        try:
            page = doc.load_page(page_number)
            blocks = _parse_page_blocks(page)
            text = " ".join(p.text for b in blocks for p in b.paragraphs)
            word_count = len(text.split())
            character_count = len(text)

            yield Page(
                page_number=page_number + 1,
                blocks=blocks,
                word_count=word_count,
                character_count=character_count,
            )
        except Exception as e:
            raise PDFProcessingError(
                f"Failed to parse page {page_number + 1}: {e}"
            ) from e


def _parse_page_blocks(page: fitz.Page) -> list[Block]:
    """Parse a single PyMuPDF page into a list of ``Block`` objects.

    Uses ``page.get_text("dict")`` to access the page's structure:
    blocks → lines → spans. Font size/weight heuristics are used to
    identify headings.

    Args:
        page: A PyMuPDF page object.

    Returns:
        A list of ``Block`` instances.
    """
    text_dict: dict[str, Any] = page.get_text("dict")
    page_blocks: list[dict[str, Any]] = text_dict.get("blocks", [])
    result: list[Block] = []

    # Collect font sizes to compute a median for heading detection
    font_sizes: list[float] = []
    for fb in page_blocks:
        for line in fb.get("lines", []):
            for span in line.get("spans", []):
                font_sizes.append(span.get("size", 0))
    median_font = _median(font_sizes) if font_sizes else 12.0

    block_index = 0
    for fb in page_blocks:
        block_type = fb.get("type", 0)
        block: Block | None = None

        if block_type == 0:  # Text block
            block = _build_text_block(fb, median_font, block_index)
        elif block_type == 1:  # Image block
            block = Block(type="figure", block_index=block_index, level=None)
        else:
            continue

        if block is not None:
            result.append(block)
            block_index += 1

    return result


def _build_text_block(
    fb: dict[str, Any], median_font: float, block_index: int
) -> Block | None:
    """Build a ``Block`` from a PyMuPDF text block dictionary.

    Joins lines into paragraphs and detects heading status based on
    font size relative to the page median.

    Args:
        fb: A PyMuPDF text block dictionary.
        median_font: Median font size on the page.
        block_index: Zero-based index of this block on the page.

    Returns:
        A ``Block`` instance, or ``None`` if the block has no text.
    """
    lines = fb.get("lines", [])
    if not lines:
        return None

    # Collect all spans and find the maximum font size in this block
    all_spans: list[dict[str, Any]] = []
    max_font = 0.0
    first_font = 0.0
    for line in lines:
        spans = line.get("spans", [])
        for span in spans:
            font = span.get("size", 0)
            if first_font == 0.0:
                first_font = font
            max_font = max(max_font, font)
            all_spans.append(span)

    if not all_spans:
        return None

    # Join spans into paragraph text
    paragraph_text = _join_span_text(all_spans)
    if not paragraph_text.strip():
        return None

    # Heuristic: if the first or max font is significantly larger than median → heading
    is_heading = max_font >= median_font * HEADING_FONT_THRESHOLD or first_font >= median_font * HEADING_FONT_THRESHOLD
    # Also check if text looks like a heading (short, no sentence-ending punctuation)
    looks_like_heading = (
        len(paragraph_text.strip()) < 200
        and not paragraph_text.strip().endswith((".", ":", ";"))
        and paragraph_text.strip().isupper() is False  # not just all-caps
    )

    block_type = "heading" if is_heading else "text"
    level = _estimate_heading_level(max_font, median_font) if is_heading else None

    # Split into paragraphs on double newlines
    raw_paragraphs = paragraph_text.split("\n\n")
    paragraphs: list[Paragraph] = []
    para_index = 0
    for raw_para in raw_paragraphs:
        cleaned = raw_para.strip()
        if cleaned:
            paragraphs.append(Paragraph(text=cleaned, index=para_index))
            para_index += 1

    if not paragraphs:
        return None

    return Block(
        type=block_type,
        level=level,
        block_index=block_index,
        paragraphs=paragraphs,
    )


def _join_span_text(spans: list[dict[str, Any]]) -> str:
    """Join a list of PyMuPDF span dictionaries into a single text string.

    Adds appropriate spacing between spans.

    Args:
        spans: List of PyMuPDF span dicts with a ``text`` key.

    Returns:
        A concatenated text string.
    """
    parts: list[str] = []
    for span in spans:
        text: str = span.get("text", "")
        if text:
            # Add space between spans unless the text starts with punctuation
            if parts and not text.startswith((".", ",", ":", ";", ")", "]", "}")):
                parts.append(" ")
            parts.append(text)
    return "".join(parts)


def _estimate_heading_level(font_size: float, median_font: float) -> int:
    """Estimate heading level (1-6) based on font size ratio.

    Args:
        font_size: The font size of the block.
        median_font: The median font size on the page.

    Returns:
        A heading level from 1 (largest) to 6 (smallest).
    """
    ratio = font_size / median_font if median_font > 0 else 1.0
    if ratio >= 2.5:
        return 1
    if ratio >= 2.0:
        return 2
    if ratio >= 1.6:
        return 3
    if ratio >= 1.3:
        return 4
    if ratio >= 1.1:
        return 5
    return 6


def _parse_pdf_date(date_str: str | None) -> datetime | None:
    """Parse a PDF date string (e.g. ``D:20230101120000+05'30'``) into a datetime.

    Args:
        date_str: A PDF date string or ``None``.

    Returns:
        A ``datetime`` object, or ``None`` if parsing fails.
    """
    if not date_str:
        return None

    # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
    try:
        cleaned = date_str.strip()
        if cleaned.startswith("D:"):
            cleaned = cleaned[2:]

        # Basic format: YYYYMMDDHHmmSS
        fmt = "%Y%m%d%H%M%S"
        # Handle timezone offset if present
        if len(cleaned) >= 14 and cleaned[14] in ("+", "-"):
            # Strip timezone info — store as naive UTC for simplicity
            cleaned = cleaned[:14]

        return datetime.strptime(cleaned, fmt)  # noqa: DTZ007
    except (ValueError, IndexError):
        logger.warning("Failed to parse PDF date string: %s", date_str)
        return None


def _median(values: list[float]) -> float:
    """Compute the median of a list of floats.

    Args:
        values: A list of floats.

    Returns:
        The median value.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n % 2 == 1:
        return sorted_vals[n // 2]
    return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0