"""Document ingestion engine — discover, load, parse, and represent PDFs as structured documents."""

from __future__ import annotations

from oracle_flexcube_copilot.ingestion.models import Document, Page, Block, Paragraph, DocumentMetadata
from oracle_flexcube_copilot.ingestion.scanner import scan_pdfs
from oracle_flexcube_copilot.ingestion.loader import load_pdf
from oracle_flexcube_copilot.ingestion.parser import parse_document_metadata, parse_pages
from oracle_flexcube_copilot.ingestion.service import DocumentIngestionService

__all__ = [
    "Document",
    "Page",
    "Block",
    "Paragraph",
    "DocumentMetadata",
    "scan_pdfs",
    "load_pdf",
    "parse_document_metadata",
    "parse_pages",
    "DocumentIngestionService",
]