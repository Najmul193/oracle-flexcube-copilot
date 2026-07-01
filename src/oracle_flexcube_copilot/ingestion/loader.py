"""PDF loader — safely opens PDF files using PyMuPDF (fitz)."""

from __future__ import annotations

from pathlib import Path

import fitz

from oracle_flexcube_copilot.exceptions import (
    CorruptedPDFError,
    EmptyPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFNotFoundError,
)


def load_pdf(path: Path) -> fitz.Document:
    """Open a PDF file and return a PyMuPDF ``Document``.

    Performs safety checks for:
    - File existence
    - Valid PDF structure
    - Encryption / password protection
    - Empty documents (zero pages)

    Args:
        path: Absolute or relative path to the PDF file.

    Returns:
        An opened ``fitz.Document`` ready for reading.

    Raises:
        PDFNotFoundError: If the file does not exist.
        InvalidPDFError: If the file is not a valid PDF.
        CorruptedPDFError: If the PDF data is corrupted.
        EmptyPDFError: If the PDF has zero pages.
        EncryptedPDFError: If the PDF is password-protected.
    """
    if not path.exists():
        raise PDFNotFoundError(f"PDF file not found: {path}")
    if not path.is_file():
        raise InvalidPDFError(f"Path is not a file: {path}")

    try:
        doc: fitz.Document = fitz.open(str(path))
    except fitz.FileDataError as e:
        raise CorruptedPDFError(f"Corrupted or invalid PDF: {path} — {e}") from e
    except Exception as e:
        raise InvalidPDFError(f"Failed to open PDF: {path} — {e}") from e

    if doc.needs_pass:
        doc.close()
        raise EncryptedPDFError(f"PDF is password-protected: {path}")

    if doc.page_count == 0:
        doc.close()
        raise EmptyPDFError(f"PDF has zero pages: {path}")

    return doc


def load_pdf_safe(path: Path) -> fitz.Document | None:
    """Open a PDF safely, returning ``None`` instead of raising on error.

    This is a convenience wrapper around :func:`load_pdf` for cases where
    you want to skip problematic files without interrupting the pipeline.

    Args:
        path: Path to the PDF file.

    Returns:
        An opened ``fitz.Document``, or ``None`` if the file could not be loaded.
    """
    try:
        return load_pdf(path)
    except PDFNotFoundError, InvalidPDFError, CorruptedPDFError, EmptyPDFError, EncryptedPDFError:
        return None
