"""Tests for the PDF scanner."""

from __future__ import annotations

from pathlib import Path

import pytest

from oracle_flexcube_copilot.exceptions import ScannerError
from oracle_flexcube_copilot.ingestion.scanner import _is_hidden, scan_pdfs


class TestScanPDFs:
    """Tests for :func:`scan_pdfs`."""

    def test_scans_valid_pdfs(self, tmp_data_dir: Path, valid_pdf_path: Path) -> None:
        """A directory with a valid PDF should return that PDF."""
        results = list(scan_pdfs(tmp_data_dir))
        assert len(results) == 1
        assert results[0] == valid_pdf_path.resolve()

    def test_scans_multiple_pdfs(self, tmp_data_dir: Path, valid_pdf_path: Path, multi_page_pdf_path: Path) -> None:
        """Multiple PDFs should all be discovered."""
        results = list(scan_pdfs(tmp_data_dir))
        assert len(results) == 2

    def test_returns_deterministic_order(self, tmp_data_dir: Path) -> None:
        """Results should be sorted alphabetically."""
        # Create PDFs with names that sort out of creation order
        for name in ("b.pdf", "a.pdf", "c.pdf"):
            path = tmp_data_dir / name
            doc = __import__("fitz").open()
            doc.new_page()
            doc.save(str(path))
            doc.close()

        results = list(scan_pdfs(tmp_data_dir))
        assert [p.name for p in results] == ["a.pdf", "b.pdf", "c.pdf"]

    def test_empty_directory(self, tmp_data_dir: Path) -> None:
        """An empty directory should yield no results."""
        results = list(scan_pdfs(tmp_data_dir))
        assert results == []

    def test_nonexistent_directory(self) -> None:
        """A non-existent directory should raise ScannerError."""
        with pytest.raises(ScannerError, match="Directory does not exist"):
            list(scan_pdfs(Path("/nonexistent/path")))

    def test_file_path_instead_of_directory(self, valid_pdf_path: Path) -> None:
        """Passing a file instead of directory should raise ScannerError."""
        with pytest.raises(ScannerError, match="Path is not a directory"):
            list(scan_pdfs(valid_pdf_path))

    def test_ignores_hidden_directories(self, tmp_data_dir: Path) -> None:
        """PDFs inside hidden directories should be skipped."""
        hidden_dir = tmp_data_dir / ".hidden"
        hidden_dir.mkdir()
        doc = __import__("fitz").open()
        doc.new_page()
        doc.save(str(hidden_dir / "hidden.pdf"))
        doc.close()

        results = list(scan_pdfs(tmp_data_dir))
        assert results == []

    def test_ignores_non_pdf_files(self, tmp_data_dir: Path) -> None:
        """Non-PDF files should be ignored."""
        (tmp_data_dir / "readme.txt").write_text("hello")
        (tmp_data_dir / "image.png").write_bytes(b"\x89PNG\r\n")

        results = list(scan_pdfs(tmp_data_dir))
        assert results == []


class TestIsHidden:
    """Tests for :func:`_is_hidden`."""

    def test_hidden_file(self) -> None:
        """A file starting with a dot is hidden."""
        assert _is_hidden(Path("/tmp/.hidden/file.pdf"))

    def test_hidden_directory(self) -> None:
        """A file inside a hidden directory is hidden."""
        assert _is_hidden(Path("/.config/data/file.pdf"))

    def test_visible_file(self) -> None:
        """A normal file is not hidden."""
        assert not _is_hidden(Path("/data/file.pdf"))

    def test_mixed_path(self) -> None:
        """A path with both visible and hidden components is hidden."""
        assert _is_hidden(Path("/data/.cache/file.pdf"))