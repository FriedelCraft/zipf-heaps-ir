"""
visualization.py

Matplotlib figure generation for the standalone Zipf Analysis project.
Trimmed from the original combined project's visualization.py: only the
three Zipf-related figures are kept here (log-log plot, residuals,
piecewise slopes). Cost/infrastructure/Heaps figure functions live in the
separate cost_infrastructure_project.

Uses matplotlib only (no seaborn), saves high-resolution PNGs, and always
includes titles and axis labels so figures are suitable for a report or
slide deck.
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


# =============================================================================
# ZIPF FIGURES
# =============================================================================

def plot_zipf_loglog(
    log_df: pd.DataFrame,
    overall_fit,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    label_prefix: str = "",
) -> None:
    """
    zipf_loglog.png — observed rank-frequency distribution (log-log) with
    the fitted single Zipf line overlaid.
    """
    fig, ax = _new_figure(figsize)

    if log_df.empty:
        ax.text(0.5, 0.5, "No data available", ha="center", va="center")
    else:
        ax.scatter(
            log_df["log_rank"], log_df["log_frequency"],
            s=8, alpha=0.5, label="Observed", color="tab:blue",
        )
        if overall_fit is not None:
            x_line = np.linspace(log_df["log_rank"].min(), log_df["log_rank"].max(), 100)
            y_line = overall_fit.intercept + overall_fit.slope * x_line
            ax.plot(
                x_line, y_line, color="tab:red", linewidth=2,
                label=f"Fitted Zipf (s={overall_fit.exponent_s:.3f}, "
                      f"R^2={overall_fit.r_squared:.3f})",
            )

    ax.set_xlabel("log(rank)")
    ax.set_ylabel("log(frequency)")
    ax.set_title(f"{label_prefix}Zipf Rank-Frequency Distribution (log-log)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _save(fig, output_path, dpi=dpi)


def plot_zipf_residuals(
    residuals_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    label_prefix: str = "",
) -> None:
    """zipf_residuals.png — residuals vs log(rank)."""
    fig, ax = _new_figure(figsize)

    if residuals_df.empty:
        ax.text(0.5, 0.5, "No residual data available", ha="center", va="center")
    else:
        ax.scatter(
            residuals_df["log_rank"], residuals_df["residual"],
            s=8, alpha=0.5, color="tab:purple",
        )
        ax.axhline(0, color="black", linewidth=1, linestyle="--")

    ax.set_xlabel("log(rank)")
    ax.set_ylabel("residual (observed - predicted log-frequency)")
    ax.set_title(f"{label_prefix}Zipf Fit Residuals vs log(rank)")
    ax.grid(True, alpha=0.3)
    _save(fig, output_path, dpi=dpi)


def plot_zipf_piecewise_slopes(
    piecewise_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
    label_prefix: str = "",
) -> None:
    """zipf_piecewise_slopes.png — compare exponents across rank regions."""
    fig, ax = _new_figure(figsize)

    if piecewise_df.empty:
        ax.text(0.5, 0.5, "No piecewise data available", ha="center", va="center")
    else:
        x = range(len(piecewise_df))
        ax.bar(x, piecewise_df["exponent_s"], color="tab:green", alpha=0.8)
        ax.set_xticks(list(x))
        ax.set_xticklabels(piecewise_df["region"], rotation=45, ha="right")
        for i, r2 in enumerate(piecewise_df["r_squared"]):
            ax.text(i, piecewise_df["exponent_s"].iloc[i], f"R^2={r2:.2f}",
                     ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Rank region")
    ax.set_ylabel("Estimated Zipf exponent (s)")
    ax.set_title(f"{label_prefix}Piecewise Zipf Exponents by Rank Region")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, output_path, dpi=dpi)
