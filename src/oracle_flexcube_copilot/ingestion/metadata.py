"""File-level metadata utilities — SHA-256 checksums, timestamps, MIME types."""

from __future__ import annotations

import hashlib
import mimetypes
from datetime import datetime, timezone
from pathlib import Path


def compute_sha256(path: Path) -> str:
    """Compute the SHA-256 hex digest of a file.

    Reads the file in 64 KiB chunks to handle large PDFs efficiently.

    Args:
        path: Path to the file.

    Returns:
        The SHA-256 checksum as a hex string.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
    """
    sha256 = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)  # 64 KiB
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def get_file_size(path: Path) -> int:
    """Return the file size in bytes.

    Args:
        path: Path to the file.

    Returns:
        File size in bytes.
    """
    return path.stat().st_size


def get_last_modified(path: Path) -> datetime:
    """Return the last-modified timestamp as a timezone-aware datetime.

    Args:
        path: Path to the file.

    Returns:
        A UTC timezone-aware datetime.
    """
    mtime = path.stat().st_mtime
    return datetime.fromtimestamp(mtime, tz=timezone.utc)


def get_creation_time(path: Path) -> datetime:
    """Return the file creation timestamp as a timezone-aware datetime.

    On macOS this is the birth time (``st_birthtime``). On other systems
    it falls back to the last metadata change time (``st_ctime``).

    Args:
        path: Path to the file.

    Returns:
        A UTC timezone-aware datetime.
    """
    stat = path.stat()
    # st_birthtime is available on macOS/BSD
    ctime = getattr(stat, "st_birthtime", stat.st_ctime)
    return datetime.fromtimestamp(ctime, tz=timezone.utc)


def get_mime_type(path: Path) -> str:
    """Guess the MIME type of a file based on its extension.

    Args:
        path: Path to the file.

    Returns:
        The MIME type string (e.g. ``"application/pdf"``).
    """
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"