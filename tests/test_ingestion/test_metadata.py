"""Tests for file-level metadata utilities."""

from __future__ import annotations

from pathlib import Path

import pytest

from oracle_flexcube_copilot.ingestion.metadata import (
    compute_sha256,
    get_creation_time,
    get_file_size,
    get_last_modified,
    get_mime_type,
)


class TestComputeSHA256:
    """Tests for :func:`compute_sha256`."""

    def test_returns_64_char_hex(self, valid_pdf_path: Path) -> None:
        """SHA-256 should be a 64-character hex string."""
        sha = compute_sha256(valid_pdf_path)
        assert len(sha) == 64
        assert all(c in "0123456789abcdef" for c in sha)

    def test_deterministic(self, valid_pdf_path: Path) -> None:
        """Same file should produce the same hash."""
        sha1 = compute_sha256(valid_pdf_path)
        sha2 = compute_sha256(valid_pdf_path)
        assert sha1 == sha2

    def test_different_files_different_hash(
        self, valid_pdf_path: Path, multi_page_pdf_path: Path
    ) -> None:
        """Different files should produce different hashes."""
        sha1 = compute_sha256(valid_pdf_path)
        sha2 = compute_sha256(multi_page_pdf_path)
        assert sha1 != sha2

    def test_raises_on_missing_file(self, missing_pdf_path: Path) -> None:
        """A missing file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            compute_sha256(missing_pdf_path)


class TestGetFileSize:
    """Tests for :func:`get_file_size`."""

    def test_returns_positive_int(self, valid_pdf_path: Path) -> None:
        """File size should be a positive integer."""
        size = get_file_size(valid_pdf_path)
        assert isinstance(size, int)
        assert size > 0


class TestGetLastModified:
    """Tests for :func:`get_last_modified`."""

    def test_returns_datetime(self, valid_pdf_path: Path) -> None:
        """Last modified should be a datetime."""
        dt = get_last_modified(valid_pdf_path)
        assert dt is not None
        assert dt.tzinfo is not None


class TestGetCreationTime:
    """Tests for :func:`get_creation_time`."""

    def test_returns_datetime(self, valid_pdf_path: Path) -> None:
        """Creation time should be a datetime."""
        dt = get_creation_time(valid_pdf_path)
        assert dt is not None
        assert dt.tzinfo is not None


class TestGetMimeType:
    """Tests for :func:`get_mime_type`."""

    def test_pdf_mime_type(self, valid_pdf_path: Path) -> None:
        """PDF files should return application/pdf."""
        mime = get_mime_type(valid_pdf_path)
        assert mime == "application/pdf"

    def test_unknown_extension(self, tmp_data_dir: Path) -> None:
        """Files with unknown extensions should not raise errors."""
        path = tmp_data_dir / "test.xyz"
        path.write_text("test")
        mime = get_mime_type(path)
        assert isinstance(mime, str)
