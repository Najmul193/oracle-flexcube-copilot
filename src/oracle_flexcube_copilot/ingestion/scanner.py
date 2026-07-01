"""PDF scanner — recursively discovers PDF files in a directory."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path

from oracle_flexcube_copilot.exceptions import ScannerError


def scan_pdfs(root_dir: Path) -> Generator[Path]:
    """Recursively yield PDF files under *root_dir* in deterministic order.

    Hidden files (directories or files whose name starts with ``.``) are
    silently skipped. Only files with a ``.pdf`` suffix are returned.

    Args:
        root_dir: The root directory to scan.

    Yields:
        Path objects for each discovered PDF file, sorted alphabetically.

    Raises:
        ScannerError: If *root_dir* does not exist or is not a directory.
    """
    if not root_dir.exists():
        raise ScannerError(f"Directory does not exist: {root_dir}")
    if not root_dir.is_dir():
        raise ScannerError(f"Path is not a directory: {root_dir}")

    try:
        # Walk sorted to guarantee deterministic ordering
        for path in sorted(root_dir.rglob("*.pdf")):
            # Skip files with any hidden component in the path
            if _is_hidden(path):
                continue
            if path.is_file():
                yield path.resolve()
    except OSError as e:
        raise ScannerError(f"Error scanning directory {root_dir}: {e}") from e


def _is_hidden(path: Path) -> bool:
    """Return ``True`` if any path component starts with a dot."""
    return any(part.startswith(".") for part in path.parts)
