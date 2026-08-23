"""
utils.py

Shared helper utilities: logging setup, safe directory creation, and small
formatting helpers used across the Cost / Infrastructure project. Trimmed
from the original combined project's utils.py: the corpus-missing message
helper (Zipf/corpus-specific) has been removed since this project never
reads a raw text corpus. The Heaps-missing message is kept since it's used
by this project's integration module.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a configured logger that writes to stdout with a simple,
    readable format. Safe to call multiple times for the same name
    (won't duplicate handlers).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="[%(levelname)s] %(name)s: %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = False
    return logger


def ensure_dir(path: Path) -> Path:
    """Create a directory (and parents) if it doesn't already exist."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_usd(value: float | None) -> str:
    """Format a number as a USD string, or return a placeholder marker."""
    if value is None:
        return "N/A (placeholder)"
    return f"${value:,.2f}"


def print_banner(title: str, char: str = "=", width: int = 78) -> None:
    """Print a simple banner line to stdout for readable CLI output."""
    print(char * width)
    print(title)
    print(char * width)


def print_missing_heaps_message() -> None:
    """Standard message when Heaps results are not yet available."""
    print("Heaps results not found. Skipping Heaps integration.")
