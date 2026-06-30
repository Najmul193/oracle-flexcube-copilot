"""Structured logging configuration."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from pythonjsonlogger import jsonlogger

from oracle_flexcube_copilot.config import settings


def setup_logging(log_dir: Path | None = None) -> logging.Logger:
    """Configure and return the application root logger.

    Args:
        log_dir: Directory for log files. Defaults to settings.resolved_log_dir.

    Returns:
        Configured root logger instance.
    """
    log_dir = log_dir or settings.resolved_log_dir
    log_dir.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger("oracle_flexcube_copilot")
    root_logger.setLevel(settings.log_level.upper())

    # Remove existing handlers to avoid duplicates on re-initialization
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(settings.log_level.upper())

    if settings.log_format == "json":
        formatter = jsonlogger.JsonFormatter(
            fmt="%(asctime)s %(name)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    else:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (always text format for readability)
    log_file = log_dir / "oracle_copilot.log"
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module.

    Args:
        name: The module name (typically ``__name__``).

    Returns:
        A logger instance prefixed with the application namespace.
    """
    return logging.getLogger(f"oracle_flexcube_copilot.{name}")