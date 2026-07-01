"""Tests for the PDF loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from oracle_flexcube_copilot.exceptions import (
    CorruptedPDFError,
    EmptyPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFNotFoundError,
)
from oracle_flexcube_copilot.ingestion.loader import load_pdf, load_pdf_safe


class TestLoadPDF:
    """Tests for :func:`load_pdf`."""

    def test_loads_valid_pdf(self, valid_pdf_path: Path) -> None:
        """A valid PDF should be opened successfully."""
        doc = load_pdf(valid_pdf_path)
        assert doc is not None
        assert doc.page_count > 0
        doc.close()

    def test_missing_file(self, missing_pdf_path: Path) -> None:
        """A missing file should raise PDFNotFoundError."""
        with pytest.raises(PDFNotFoundError, match="PDF file not found"):
            load_pdf(missing_pdf_path)

    def test_empty_pdf(self, empty_pdf_path: Path) -> None:
        """An empty PDF (0 pages) should raise EmptyPDFError."""
        with pytest.raises(EmptyPDFError):
            load_pdf(empty_pdf_path)

    def test_encrypted_pdf(self, encrypted_pdf_path: Path) -> None:
        """An encrypted PDF should raise EncryptedPDFError."""
        with pytest.raises(EncryptedPDFError, match="password-protected"):
            load_pdf(encrypted_pdf_path)

    def test_corrupted_pdf(self, corrupted_pdf_path: Path) -> None:
        """A corrupted file should raise CorruptedPDFError."""
        with pytest.raises(CorruptedPDFError, match="Corrupted or invalid PDF"):
            load_pdf(corrupted_pdf_path)

    def test_directory_path(self, tmp_data_dir: Path) -> None:
        """A directory path should raise InvalidPDFError."""
        with pytest.raises(InvalidPDFError, match="Path is not a file"):
            load_pdf(tmp_data_dir)


class TestLoadPDFSafe:
    """Tests for :func:`load_pdf_safe`."""

    def test_valid_pdf_returns_doc(self, valid_pdf_path: Path) -> None:
        """A valid PDF should return the document."""
        doc = load_pdf_safe(valid_pdf_path)
        assert doc is not None
        doc.close()

    def test_missing_file_returns_none(self, missing_pdf_path: Path) -> None:
        """A missing file should return None."""
        assert load_pdf_safe(missing_pdf_path) is None

    def test_encrypted_file_returns_none(self, encrypted_pdf_path: Path) -> None:
        """An encrypted file should return None."""
        assert load_pdf_safe(encrypted_pdf_path) is None

    def test_corrupted_file_returns_none(self, corrupted_pdf_path: Path) -> None:
        """A corrupted file should return None."""
        assert load_pdf_safe(corrupted_pdf_path) is None

    def test_empty_file_returns_none(self, empty_pdf_path: Path) -> None:
        """An empty file should return None."""
        assert load_pdf_safe(empty_pdf_path) is None
