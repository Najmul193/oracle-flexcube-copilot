"""Document ingestion service — orchestrates scanning, loading, parsing, and manifest generation."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oracle_flexcube_copilot.exceptions import (
    CorruptedPDFError,
    EmptyPDFError,
    EncryptedPDFError,
    InvalidPDFError,
    PDFNotFoundError,
    PDFProcessingError,
)
from oracle_flexcube_copilot.ingestion.loader import load_pdf
from oracle_flexcube_copilot.ingestion.metadata import (
    compute_sha256,
    get_creation_time,
    get_file_size,
    get_last_modified,
    get_mime_type,
)
from oracle_flexcube_copilot.ingestion.models import Document, make_block_id, make_page_id
from oracle_flexcube_copilot.ingestion.parser import parse_document_metadata, parse_pages, parse_toc
from oracle_flexcube_copilot.ingestion.scanner import scan_pdfs

logger = logging.getLogger("oracle_flexcube_copilot.ingestion.service")


class DocumentIngestionService:
    """Orchestrates the end-to-end document ingestion pipeline.

    Typical usage::

        service = DocumentIngestionService()
        documents = service.ingest_directory(Path("data/"))
    """

    def __init__(self, manifest_dir: Path | None = None) -> None:
        """Initialize the service.

        Args:
            manifest_dir: Directory where ``documents_manifest.json`` will be
                written. Defaults to ``cache/`` relative to the current directory.
        """
        self.manifest_dir = manifest_dir or Path("cache")
        self.manifest_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scan(self, data_dir: Path | None = None) -> list[Path]:
        """Discover PDF files in *data_dir*.

        Args:
            data_dir: Root directory to scan. Defaults to ``data/``.

        Returns:
            A sorted list of absolute PDF file paths.
        """
        root = data_dir or Path("data")
        logger.info("Scanning directory: %s", root)
        paths = list(scan_pdfs(root))
        logger.info("Discovered %d PDF(s) in %s", len(paths), root)
        return paths

    def load(self, path: Path) -> Any:
        """Open a PDF file for reading.

        Args:
            path: Path to the PDF file.

        Returns:
            A PyMuPDF ``Document`` object.

        Raises:
            PDFNotFoundError, InvalidPDFError, CorruptedPDFError,
            EmptyPDFError, EncryptedPDFError: On load failure.
        """
        return load_pdf(path)

    def parse(self, doc: Any, path: Path) -> Document:
        """Parse an opened PDF document into a ``Document`` model.

        Args:
            doc: An opened PyMuPDF ``Document``.
            path: The original file path (used for metadata).

        Returns:
            A fully populated ``Document`` instance with stable IDs.
        """
        resolved = path.resolve()
        sha256 = compute_sha256(resolved)
        metadata = parse_document_metadata(doc)
        toc = parse_toc(doc)

        # Parse pages and assign stable IDs
        raw_pages = list(parse_pages(doc))
        pages = []
        for p in raw_pages:
            blocks = []
            for b in p.blocks:
                block_id = make_block_id(sha256, p.page_number, b.block_index)
                blocks.append(b.model_copy(update={"id": block_id}))
            page_id = make_page_id(sha256, p.page_number)
            pages.append(p.model_copy(update={"id": page_id, "blocks": blocks}))

        return Document(
            id=sha256,
            filename=resolved.name,
            absolute_path=str(resolved),
            sha256=sha256,
            file_size_bytes=get_file_size(resolved),
            last_modified=get_last_modified(resolved),
            created_time=get_creation_time(resolved),
            mime_type=get_mime_type(resolved),
            metadata=metadata,
            table_of_contents=toc,
            pages=pages,
        )

    def ingest_document(self, path: Path) -> Document | None:
        """Load, parse, and return a single PDF document.

        Args:
            path: Path to the PDF file.

        Returns:
            A ``Document`` instance, or ``None`` if the file could not be processed.
        """
        logger.info("Ingesting document: %s", path.name)
        start = time.time()

        try:
            doc = self.load(path)
        except (
            PDFNotFoundError,
            InvalidPDFError,
            CorruptedPDFError,
            EmptyPDFError,
            EncryptedPDFError,
        ) as e:
            logger.error("Skipping %s: %s", path.name, e)
            return None

        try:
            document = self.parse(doc, path)
            elapsed = time.time() - start
            logger.info(
                "Ingested %s — %d pages, %d words in %.2fs",
                path.name,
                document.metadata.page_count,
                document.total_words,
                elapsed,
            )
            return document
        except PDFProcessingError as e:
            logger.error("Failed to parse %s: %s", path.name, e)
            return None
        finally:
            doc.close()

    def ingest_directory(self, data_dir: Path | None = None) -> list[Document]:
        """Ingest all PDFs in a directory and generate the document manifest.

        Args:
            data_dir: Root directory containing PDFs. Defaults to ``data/``.

        Returns:
            A list of successfully ingested ``Document`` instances.
        """
        root = data_dir or Path("data")
        pdf_paths = self.scan(root)
        total = len(pdf_paths)
        logger.info("Starting ingestion of %d document(s) from %s", total, root)

        documents: list[Document] = []
        start = time.time()

        for i, path in enumerate(pdf_paths, start=1):
            logger.info("[%d/%d] Processing: %s", i, total, path.name)
            doc = self.ingest_document(path)
            if doc is not None:
                documents.append(doc)

        elapsed = time.time() - start
        logger.info(
            "Ingestion complete — %d/%d documents processed in %.2fs",
            len(documents),
            total,
            elapsed,
        )

        # Generate and save manifest
        manifest = self._generate_manifest(documents)
        self._save_manifest(manifest)

        return documents

    # ------------------------------------------------------------------
    # Manifest management
    # ------------------------------------------------------------------

    def _generate_manifest(self, documents: list[Document]) -> dict[str, Any]:
        """Build the document manifest dictionary.

        Args:
            documents: List of ingested documents.

        Returns:
            A serializable dictionary with ingestion metadata and document entries.
        """
        return {
            "ingestion_timestamp": datetime.now(UTC).isoformat(),
            "total_documents": len(documents),
            "total_pages": sum(d.metadata.page_count for d in documents),
            "documents": [
                {
                    "id": d.id,
                    "filename": d.filename,
                    "sha256": d.sha256,
                    "file_size_bytes": d.file_size_bytes,
                    "pages": d.metadata.page_count,
                    "indexed": False,
                }
                for d in documents
            ],
        }

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        """Write the manifest to ``cache/documents_manifest.json``.

        Args:
            manifest: The manifest dictionary.
        """
        manifest_path = self.manifest_dir / "documents_manifest.json"
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2, default=str)
            logger.info("Manifest saved to %s", manifest_path)
        except OSError as e:
            logger.error("Failed to save manifest: %s", e)

    def load_manifest(self) -> dict[str, Any] | None:
        """Load the document manifest from disk.

        Returns:
            The manifest dictionary, or ``None`` if the file does not exist
            or cannot be parsed.
        """
        manifest_path = self.manifest_dir / "documents_manifest.json"
        if not manifest_path.exists():
            logger.warning("Manifest not found: %s", manifest_path)
            return None
        try:
            with open(manifest_path, encoding="utf-8") as f:
                return json.load(f)  # type: ignore[no-any-return]
        except (OSError, json.JSONDecodeError) as e:
            logger.error("Failed to load manifest: %s", e)
            return None
