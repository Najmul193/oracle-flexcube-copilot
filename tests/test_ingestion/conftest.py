"""Shared fixtures and PDF generators for ingestion tests."""

from __future__ import annotations

from pathlib import Path

import fitz
import pytest


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temporary data directory for test PDFs."""
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def valid_pdf_path(tmp_data_dir: Path) -> Path:
    """Create a valid 3-page PDF with headings, paragraphs, and a table."""
    path = tmp_data_dir / "valid_doc.pdf"
    doc = fitz.open()
    pages_content = [
        ("Title Page", "# Document Title\n\nThis is the first paragraph."),
        (
            "Chapter 1",
            "## Introduction\n\nThis is the introduction paragraph.\n\nMore details here.",
        ),
        ("Chapter 2", "## Configuration\n\nStep 1: Do this.\nStep 2: Do that."),
    ]
    for title, content in pages_content:
        page = doc.new_page()
        # Insert title as large text
        page.insert_text(fitz.Point(72, 100), title, fontsize=24)
        # Insert content
        page.insert_text(fitz.Point(72, 150), content, fontsize=11)
    doc.save(str(path))
    doc.close()
    return path


@pytest.fixture
def empty_pdf_path(tmp_data_dir: Path) -> Path:
    """Create an empty PDF (0 pages) using a minimal valid PDF structure."""
    path = tmp_data_dir / "empty.pdf"
    # A minimal valid PDF with zero pages
    content = b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n2 0 obj\n<< /Type /Pages /Kids [] /Count 0 >>\nendobj\nxref\n0 3\n0000000000 65535 f \n0000000009 00000 n \n0000000058 00000 n \ntrailer\n<< /Size 3 /Root 1 0 R >>\nstartxref\n129\n%%%%EOF"
    path.write_bytes(content)
    return path


@pytest.fixture
def encrypted_pdf_path(tmp_data_dir: Path) -> Path:
    """Create a password-protected PDF."""
    path = tmp_data_dir / "encrypted.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(
        str(path), encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="password", user_pw="password"
    )
    doc.close()
    return path


@pytest.fixture
def corrupted_pdf_path(tmp_data_dir: Path) -> Path:
    """Create a file with .pdf extension but random binary content."""
    path = tmp_data_dir / "corrupted.pdf"
    with open(path, "wb") as f:
        f.write(b"\x00\x01\x02\x03\xff\xfe\xfd\xfc")
    return path


@pytest.fixture
def missing_pdf_path(tmp_data_dir: Path) -> Path:
    """Return a path that does not exist on disk."""
    return tmp_data_dir / "nonexistent.pdf"


@pytest.fixture
def multi_page_pdf_path(tmp_data_dir: Path) -> Path:
    """Create a 5-page PDF with varied content for block/paragraph testing."""
    path = tmp_data_dir / "multi_page.pdf"
    doc = fitz.open()
    for i in range(5):
        page = doc.new_page()
        page.insert_text(fitz.Point(72, 72), f"Page {i + 1} Heading", fontsize=18)
        page.insert_text(fitz.Point(72, 120), f"Body text on page {i + 1}. " * 10, fontsize=11)
        page.insert_text(fitz.Point(72, 200), "More content here.", fontsize=11)
    doc.save(str(path))
    doc.close()
    return path
