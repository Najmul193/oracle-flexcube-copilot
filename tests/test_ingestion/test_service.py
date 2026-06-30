"""Tests for the DocumentIngestionService."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from oracle_flexcube_copilot.ingestion.models import Document, DocumentMetadata
from oracle_flexcube_copilot.ingestion.service import DocumentIngestionService


class TestDocumentIngestionService:
    """Tests for :class:`DocumentIngestionService`."""

    def test_scan_returns_pdf_paths(self, tmp_data_dir: Path, valid_pdf_path: Path) -> None:
        """Scan should return discovered PDF paths."""
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        paths = service.scan(tmp_data_dir)
        assert len(paths) == 1
        assert paths[0].suffix == ".pdf"

    def test_load_valid_pdf(self, tmp_data_dir: Path, valid_pdf_path: Path) -> None:
        """Load should return a PyMuPDF document."""
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        doc = service.load(valid_pdf_path)
        assert doc is not None
        assert doc.page_count == 3
        doc.close()

    def test_parse_returns_document_model(self, tmp_data_dir: Path, valid_pdf_path: Path) -> None:
        """Parse should return a Document instance."""
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        doc = service.load(valid_pdf_path)
        document = service.parse(doc, valid_pdf_path)
        doc.close()

        assert isinstance(document, Document)
        assert document.filename == "valid_doc.pdf"
        assert document.metadata.page_count == 3
        assert len(document.pages) == 3

    def test_ingest_document_success(self, tmp_data_dir: Path, valid_pdf_path: Path) -> None:
        """Ingest document should return a fully populated Document."""
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        document = service.ingest_document(valid_pdf_path)

        assert document is not None
        assert document.filename == "valid_doc.pdf"
        assert len(document.sha256) == 64
        assert document.file_size_bytes > 0
        assert document.mime_type == "application/pdf"

    def test_ingest_document_skips_encrypted(self, tmp_data_dir: Path, encrypted_pdf_path: Path) -> None:
        """Encrypted PDFs should be skipped (return None)."""
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        document = service.ingest_document(encrypted_pdf_path)
        assert document is None

    def test_ingest_document_skips_corrupted(self, tmp_data_dir: Path, corrupted_pdf_path: Path) -> None:
        """Corrupted PDFs should be skipped (return None)."""
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        document = service.ingest_document(corrupted_pdf_path)
        assert document is None

    def test_ingest_document_skips_empty(self, tmp_data_dir: Path, empty_pdf_path: Path) -> None:
        """Empty PDFs should be skipped (return None)."""
        # Use load_pdf_safe path since empty PDF may not raise
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        document = service.ingest_document(empty_pdf_path)
        assert document is None

    def test_ingest_document_skips_missing(self, tmp_data_dir: Path, missing_pdf_path: Path) -> None:
        """Missing PDFs should be skipped (return None)."""
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        document = service.ingest_document(missing_pdf_path)
        assert document is None

    def test_ingest_directory(self, tmp_data_dir: Path, valid_pdf_path: Path, multi_page_pdf_path: Path) -> None:
        """ingest_directory should process all PDFs and return Documents."""
        cache_dir = tmp_data_dir / "cache"
        service = DocumentIngestionService(manifest_dir=cache_dir)
        documents = service.ingest_directory(tmp_data_dir)

        assert len(documents) == 2
        assert all(isinstance(d, Document) for d in documents)

    def test_ingest_directory_creates_manifest(self, tmp_data_dir: Path, valid_pdf_path: Path) -> None:
        """The manifest should be written after ingestion."""
        cache_dir = tmp_data_dir / "cache"
        service = DocumentIngestionService(manifest_dir=cache_dir)
        service.ingest_directory(tmp_data_dir)

        manifest_path = cache_dir / "documents_manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert "ingestion_timestamp" in manifest
        assert "total_documents" in manifest
        assert "documents" in manifest
        assert len(manifest["documents"]) == 1

    def test_manifest_contents(self, tmp_data_dir: Path, valid_pdf_path: Path) -> None:
        """Manifest entries should have required fields."""
        cache_dir = tmp_data_dir / "cache"
        service = DocumentIngestionService(manifest_dir=cache_dir)
        service.ingest_directory(tmp_data_dir)

        manifest = service.load_manifest()
        assert manifest is not None

        doc_entry = manifest["documents"][0]
        assert "id" in doc_entry
        assert "filename" in doc_entry
        assert "sha256" in doc_entry
        assert "pages" in doc_entry
        assert "indexed" in doc_entry
        assert doc_entry["indexed"] is False

    def test_load_manifest_nonexistent(self, tmp_data_dir: Path) -> None:
        """Loading a non-existent manifest should return None."""
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        assert service.load_manifest() is None

    def test_document_has_blocks_and_paragraphs(self, tmp_data_dir: Path, valid_pdf_path: Path) -> None:
        """Parsed documents should have blocks with paragraphs."""
        service = DocumentIngestionService(manifest_dir=tmp_data_dir / "cache")
        document = service.ingest_document(valid_pdf_path)

        assert document is not None
        assert len(document.pages) > 0
        for page in document.pages:
            assert len(page.blocks) > 0
            for block in page.blocks:
                assert len(block.paragraphs) > 0