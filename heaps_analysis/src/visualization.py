"""
visualization.py

Matplotlib figure generation for the standalone Heaps' Law Analysis
project: the log-log fit plot, the vocabulary-growth curve, and the
residual plot. Uses matplotlib only (no seaborn), saves high-resolution
PNGs, and always includes titles and axis labels so figures are suitable
for a report or slide deck.

Visual style (colors, DPI, figure size, title conventions) is kept
consistent with the companion zipf_analysis_project's figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

DEFAULT_DPI = 200
DEFAULT_FIGSIZE = (8, 6)


def _new_figure(figsize: tuple[float, float] = DEFAULT_FIGSIZE):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def _save(fig, path: Path, dpi: int = DEFAULT_DPI) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    logger.info(f"Saved figure: {path}")


def plot_heaps_loglog(
    growth_df: pd.DataFrame,
    fit,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    label_prefix: str = "",
) -> None:
    """
    heaps_loglog.png — empirical vocabulary-growth checkpoints (log-log)
    with the fitted Heaps' Law power line overlaid.
    """
    fig, ax = _new_figure(figsize)

    if growth_df.empty:
        ax.text(0.5, 0.5, "No Heaps checkpoints available", ha="center", va="center")
    else:
        log_n = np.log(growth_df["token_count"].astype(float))
        log_v = np.log(growth_df["vocabulary_size"].astype(float))
        ax.scatter(
            log_n, log_v, s=20, alpha=0.7, label="Empirical checkpoints", color="tab:blue",
        )
        if fit is not None:
            x_line = np.linspace(log_n.min(), log_n.max(), 100)
            y_line = fit.intercept_log_k + fit.beta * x_line
            ax.plot(
                x_line, y_line, color="tab:red", linewidth=2,
                label=f"Fitted Heaps (beta={fit.beta:.3f}, "
                      f"R^2={fit.r_squared:.3f})",
            )

    ax.set_xlabel("log(token count N)")
    ax.set_ylabel("log(vocabulary size V)")
    ax.set_title(f"{label_prefix}Heaps' Law: Vocabulary Growth (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_path, dpi=dpi)


def plot_heaps_vocabulary_growth(
    growth_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    label_prefix: str = "",
) -> None:
    """
    heaps_vocabulary_growth.png — token count (x) vs vocabulary size (y).
    Token count uses a log scale since checkpoints are log-spaced and the
    corpus can span many orders of magnitude (thousands to hundreds of
    millions of tokens); vocabulary size stays linear so the characteristic
    concave "diminishing returns" Heaps curve is visible directly.
    """
    fig, ax = _new_figure(figsize)

    if growth_df.empty:
        ax.text(0.5, 0.5, "No Heaps checkpoints available", ha="center", va="center")
    else:
        ax.plot(
            growth_df["token_count"], growth_df["vocabulary_size"],
            marker="o", markersize=4, color="tab:green", linewidth=1.5,
        )
        ax.set_xscale("log")

    ax.set_xlabel("Token count N (log scale)")
    ax.set_ylabel("Vocabulary size V(N)")
    ax.set_title(f"{label_prefix}Vocabulary Growth vs Corpus Size")
    ax.grid(True, alpha=0.3, which="both")
    _save(fig, output_path, dpi=dpi)


def plot_heaps_residuals(
    residuals_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    label_prefix: str = "",
) -> None:
    """heaps_residuals.png — log-log fit residuals vs log(token count)."""
    fig, ax = _new_figure(figsize)

    if residuals_df.empty:
        ax.text(0.5, 0.5, "No residual data available", ha="center", va="center")
    else:
        ax.scatter(
            residuals_df["log_token_count"], residuals_df["residual"],
            s=20, alpha=0.7, color="tab:purple",
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")

    ax.set_xlabel("log(token count N)")
    ax.set_ylabel("residual (observed - predicted log-vocabulary)")
    ax.set_title(f"{label_prefix}Heaps' Law Fit Residuals vs log(token count)")
    ax.grid(True, alpha=0.3)
    _save(fig, output_path, dpi=dpi)
