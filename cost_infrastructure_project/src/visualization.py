"""
visualization.py

Matplotlib figure generation for the standalone Cost / Infrastructure /
Integration project. Uses matplotlib only (no seaborn), saves
high-resolution PNGs, and always includes titles and axis labels so
figures are suitable for a report or slide deck.

Every figure comparing Search_Oriented_System vs LLM_Oriented_System is
explicitly titled as a SCENARIO-BASED ARCHITECTURAL comparison — never
framed as real company cost data.
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

SYSTEM_COLORS = {
    "Search_Oriented_System": "tab:blue",
    "LLM_Oriented_System": "tab:red",
}
SYSTEM_LABELS = {
    "Search_Oriented_System": "Search-Oriented System",
    "LLM_Oriented_System": "LLM-Oriented System",
}


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
# ASSIGNMENT-PROVIDED DATA COST BASELINE
# =============================================================================

def plot_cost_vs_corpus_size(
    scenarios_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> None:
    """cost_vs_corpus_size.png — assignment-based data cost baseline by scenario."""
    fig, ax = _new_figure(figsize)

    ax.bar(scenarios_df["scenario"], scenarios_df["data_cost_usd"], color="tab:orange")
    for i, v in enumerate(scenarios_df["data_cost_usd"]):
        ax.text(i, v, f"${v:,.0f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Corpus size scenario")
    ax.set_ylabel("Assignment-provided data cost (USD)")
    ax.set_title("Data Cost Baseline vs Corpus Size\n(ASSIGNMENT-PROVIDED ASSUMPTION: $1,000 / 100,000 words — FACT, not modelled)")
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, output_path, dpi=dpi)


# =============================================================================
# A. INITIAL COST COMPARISON — Search-Oriented vs LLM-Oriented
# =============================================================================

def plot_initial_cost_comparison(
    totals_df: pd.DataFrame,
    scenario: str,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> None:
    """initial_cost_comparison.png — one-time/initial cost, both systems, one scenario."""
    fig, ax = _new_figure(figsize)

    systems = list(totals_df["system"])
    values = list(totals_df["initial_total_usd"])
    colors = [SYSTEM_COLORS.get(s, "gray") for s in systems]
    labels = [SYSTEM_LABELS.get(s, s) for s in systems]

    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"${v:,.0f}",
                 ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Initial (one-time) cost (USD)")
    ax.set_title(
        f"Initial Cost Comparison — {scenario} scenario\n"
        f"SCENARIO-BASED ARCHITECTURAL ESTIMATE (not real company costs)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, output_path, dpi=dpi)


# =============================================================================
# B. RECURRING ANNUAL COST COMPARISON — Search-Oriented vs LLM-Oriented
# =============================================================================

def plot_recurring_cost_comparison(
    totals_df: pd.DataFrame,
    scenario: str,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> None:
    """recurring_cost_comparison.png — recurring annual cost, both systems, one scenario."""
    fig, ax = _new_figure(figsize)

    systems = list(totals_df["system"])
    values = list(totals_df["recurring_annual_total_usd"])
    colors = [SYSTEM_COLORS.get(s, "gray") for s in systems]
    labels = [SYSTEM_LABELS.get(s, s) for s in systems]

    bars = ax.bar(labels, values, color=colors)
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v, f"${v:,.0f}",
                 ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Recurring annual cost (USD/year)")
    ax.set_title(
        f"Recurring Annual Cost Comparison — {scenario} scenario\n"
        f"SCENARIO-BASED ARCHITECTURAL ESTIMATE (not real company costs)"
    )
    ax.grid(True, alpha=0.3, axis="y")
    _save(fig, output_path, dpi=dpi)


# =============================================================================
# C. COST BREAKDOWN BY COMPONENT
# =============================================================================

def plot_cost_breakdown_by_component(
    breakdown_df: pd.DataFrame,
    scenario: str,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """cost_breakdown_by_component.png — every component, both systems, one scenario."""
    fig, ax = _new_figure(figsize)

    if breakdown_df.empty:
        ax.text(0.5, 0.5, "No component data available", ha="center", va="center")
    else:
        components = breakdown_df["component"]
        x = np.arange(len(components))
        width = 0.35
        search_vals = breakdown_df["search_oriented_usd"].fillna(0)
        llm_vals = breakdown_df["llm_oriented_usd"].fillna(0)

        ax.bar(x - width / 2, search_vals, width,
               label=SYSTEM_LABELS["Search_Oriented_System"],
               color=SYSTEM_COLORS["Search_Oriented_System"])
        ax.bar(x + width / 2, llm_vals, width,
               label=SYSTEM_LABELS["LLM_Oriented_System"],
               color=SYSTEM_COLORS["LLM_Oriented_System"])
        ax.set_xticks(x)
        ax.set_xticklabels(components, rotation=45, ha="right")
        ax.legend()
        ax.set_yscale("log")

    ax.set_xlabel("Cost component")
    ax.set_ylabel("Cost (USD, log scale)")
    ax.set_title(
        f"Cost Breakdown by Component — {scenario} scenario\n"
        f"SCENARIO-BASED ARCHITECTURAL ESTIMATE (log scale; data_acquisition dominates both)"
    )
    ax.grid(True, alpha=0.3, axis="y", which="both")
    _save(fig, output_path, dpi=dpi)


# =============================================================================
# D. COST SCALING ACROSS SMALL / MEDIUM / LARGE
# =============================================================================

def plot_cost_scaling_by_scenario(
    scaling_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = (10, 6),
) -> None:
    """
    cost_scaling_scenarios.png — two-panel figure: initial cost (left) and
    recurring annual cost (right), each showing both systems across
    SMALL/MEDIUM/LARGE.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    scenario_order = ["SMALL", "MEDIUM", "LARGE"]

    for system in ["Search_Oriented_System", "LLM_Oriented_System"]:
        sub = scaling_df[scaling_df["system"] == system].set_index("scenario")
        sub = sub.reindex(scenario_order)
        color = SYSTEM_COLORS[system]
        label = SYSTEM_LABELS[system]

        ax1.plot(scenario_order, sub["initial_total_usd"], marker="o", color=color, label=label)
        ax2.plot(scenario_order, sub["recurring_annual_total_usd"], marker="o", color=color, label=label)

    for ax, title, ylabel in [
        (ax1, "Initial Cost", "Initial cost (USD)"),
        (ax2, "Recurring Annual Cost", "Recurring annual cost (USD/year)"),
    ]:
        ax.set_xlabel("Corpus size scenario")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=8)

    fig.suptitle(
        "Cost Scaling Across Scenarios\nSCENARIO-BASED ARCHITECTURAL ESTIMATE (log scale)"
    )
    fig.tight_layout()
    _save(fig, output_path, dpi=dpi)


# =============================================================================
# E. SENSITIVITY / UNCERTAINTY ANALYSIS (LOW / BASE / HIGH)
# =============================================================================

def plot_sensitivity_range(
    sensitivity_df: pd.DataFrame,
    scenario: str,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = (9, 6),
) -> None:
    """
    sensitivity_range.png — LOW/BASE/HIGH initial + recurring totals for
    both systems at one scenario, shown as error-bar style ranges to make
    explicit that these are assumptions with uncertainty, not point-precise
    figures.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    systems = ["Search_Oriented_System", "LLM_Oriented_System"]
    x = np.arange(len(systems))

    for ax, cost_col, title, ylabel in [
        (ax1, "initial_total_usd", "Initial Cost", "Initial cost (USD)"),
        (ax2, "recurring_annual_total_usd", "Recurring Annual Cost", "Recurring annual cost (USD/year)"),
    ]:
        base_vals, low_errs, high_errs = [], [], []
        for system in systems:
            sub = sensitivity_df[sensitivity_df["system"] == system].set_index("sensitivity_level")
            low = float(sub.loc["LOW", cost_col]) if "LOW" in sub.index else np.nan
            base = float(sub.loc["BASE", cost_col]) if "BASE" in sub.index else np.nan
            high = float(sub.loc["HIGH", cost_col]) if "HIGH" in sub.index else np.nan
            base_vals.append(base)
            low_errs.append(max(base - low, 0))
            high_errs.append(max(high - base, 0))

        colors = [SYSTEM_COLORS[s] for s in systems]
        ax.bar(x, base_vals, color=colors, alpha=0.85,
               yerr=[low_errs, high_errs], capsize=8, ecolor="black")
        ax.set_xticks(x)
        ax.set_xticklabels([SYSTEM_LABELS[s] for s in systems], rotation=15, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"Sensitivity Analysis (LOW / BASE / HIGH) — {scenario} scenario\n"
        f"Error bars show per-component uncertainty ranges from cost_assumptions.csv"
    )
    fig.tight_layout()
    _save(fig, output_path, dpi=dpi)


# =============================================================================
# HEAPS / VOCABULARY GROWTH FIGURE
# =============================================================================

def plot_vocabulary_growth_projection(
    projection_df: pd.DataFrame,
    output_path: Path,
    dpi: int = DEFAULT_DPI,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE,
) -> None:
    """vocabulary_growth_projection.png — projected vocabulary size vs corpus multiplier."""
    fig, ax = _new_figure(figsize)

    if projection_df.empty:
        ax.text(0.5, 0.5, "No Heaps results available", ha="center", va="center")
    else:
        ax.plot(
            projection_df["multiplier"], projection_df["projected_vocabulary_size"],
            marker="o", color="tab:green",
        )
        for _, row in projection_df.iterrows():
            ax.annotate(
                f"{row['projected_vocabulary_size']:,.0f}",
                (row["multiplier"], row["projected_vocabulary_size"]),
                textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8,
            )

    ax.set_xlabel("Corpus size multiplier (x current corpus)")
    ax.set_ylabel("Projected vocabulary size")
    ax.set_title("Heaps' Law — Projected Vocabulary Growth\n(QUALITATIVE INTERPRETATION — not used to scale cost figures)")
    ax.grid(True, alpha=0.3)
    _save(fig, output_path, dpi=dpi)
