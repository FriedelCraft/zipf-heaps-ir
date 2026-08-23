"""
utils.py

Shared helper utilities: logging setup, safe directory creation, and small
formatting helpers used across the project. Kept deliberately small — this
is an undergraduate course project, not an enterprise codebase.
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


def print_missing_corpus_message(expected_path: Path) -> None:
    """Standard, friendly message when the processed corpus is missing."""
    print_banner("PROCESSED BENGALI CORPUS NOT FOUND", char="!")
    print(f"Expected file: {expected_path}")
    print()
    print("To continue, you can:")
    print("  1. Add the processed corpus at the path above.")
    print("  2. Run with --mode demo to test the project using synthetic data.")
    print("!" * 78)
