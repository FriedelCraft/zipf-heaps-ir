"""
zipf_analysis.py

Investigates whether a SINGLE power-law (Zipf) model adequately describes
the corpus's rank-frequency distribution.

    f(r) = C * r^(-s)          =>      log(f(r)) = log(C) - s * log(r)

Implements:
    STEP 1  Frequency counting + ranking (deterministic tie-breaking)
    STEP 2  Log transformation
    STEP 3  Single overall Zipf fit (slope, exponent, R^2, RMSE)
    STEP 4  Residual analysis
    STEP 5  Piecewise Zipf analysis across rank regions
    STEP 6  Long-tail analysis (hapax stats, coverage by top-K)
    STEP 7  Figures (handled in visualization.py, orchestrated from here)

This module deliberately does NOT conclude "Zipf's Law is disproved" or
"confirmed". It only computes and reports quantitative evidence. Final
interpretation is left to the report author, guided by the neutral
structured summary in `generate_zipf_summary()`.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import r2_score, mean_squared_error

from .utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# STEP 1 — FREQUENCY COUNTING + RANKING
# =============================================================================

def build_rank_frequency_table(counter: Counter) -> pd.DataFrame:
    """
    Build a rank/token/frequency dataframe sorted by descending frequency.

    Tie-breaking is deterministic: tokens with equal frequency are sorted
    alphabetically (by Unicode code point) so that results are fully
    reproducible across runs.
    """
    items = list(counter.items())
    # Sort by (-frequency, token) => descending frequency, then ascending token
    items.sort(key=lambda kv: (-kv[1], kv[0]))

    df = pd.DataFrame(items, columns=["token", "frequency"])
    df.insert(0, "rank", range(1, len(df) + 1))
    return df[["rank", "token", "frequency"]]


# =============================================================================
# STEP 2 — LOG TRANSFORMATION
# =============================================================================

def add_log_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log_rank and log_frequency columns, excluding invalid
    (non-positive) values. Handles empty/tiny corpora gracefully.
    """
    df = df.copy()
    if df.empty:
        df["log_rank"] = pd.Series(dtype=float)
        df["log_frequency"] = pd.Series(dtype=float)
        return df

    valid_mask = (df["rank"] > 0) & (df["frequency"] > 0)
    n_dropped = int((~valid_mask).sum())
    if n_dropped > 0:
        logger.warning(f"Dropping {n_dropped} rows with non-positive rank/frequency.")

    df = df.loc[valid_mask].copy()
    df["log_rank"] = np.log(df["rank"].astype(float))
    df["log_frequency"] = np.log(df["frequency"].astype(float))
    return df


# =============================================================================
# STEP 3 — SINGLE ZIPF MODEL FIT
# =============================================================================

@dataclass
class ZipfFitResult:
    slope: float
    intercept: float
    exponent_s: float
    r_squared: float
    rmse: float
    n_observations: int
    region_label: str = "overall"
    rank_start: int | None = None
    rank_end: int | None = None

    def to_dict(self) -> dict:
        return {
            "region": self.region_label,
            "rank_start": self.rank_start,
            "rank_end": self.rank_end,
            "n_observations": self.n_observations,
            "slope": self.slope,
            "exponent_s": self.exponent_s,
            "intercept": self.intercept,
            "r_squared": self.r_squared,
            "rmse": self.rmse,
        }


def fit_zipf_linear(
    log_rank: np.ndarray,
    log_freq: np.ndarray,
    region_label: str = "overall",
    rank_start: int | None = None,
    rank_end: int | None = None,
    min_observations: int = 5,
) -> ZipfFitResult | None:
    """
    Fit log_frequency = intercept + slope * log_rank via ordinary least
    squares (linear regression). Returns None if there are too few
    observations to fit reliably.
    """
    n = len(log_rank)
    if n < min_observations:
        logger.warning(
            f"Region '{region_label}' has only {n} observations "
            f"(< min_observations={min_observations}); skipping fit."
        )
        return None

    slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(
        log_rank, log_freq
    )
    predicted = intercept + slope * log_rank
    r2 = r2_score(log_freq, predicted)
    rmse = float(np.sqrt(mean_squared_error(log_freq, predicted)))

    return ZipfFitResult(
        slope=float(slope),
        intercept=float(intercept),
        exponent_s=float(-slope),
        r_squared=float(r2),
        rmse=rmse,
        n_observations=n,
        region_label=region_label,
        rank_start=rank_start,
        rank_end=rank_end,
    )


def fit_overall_zipf(df_log: pd.DataFrame, min_observations: int = 5) -> ZipfFitResult | None:
    """Fit the single overall Zipf model across the whole vocabulary."""
    return fit_zipf_linear(
        df_log["log_rank"].values,
        df_log["log_frequency"].values,
        region_label="overall",
        rank_start=int(df_log["rank"].min()) if not df_log.empty else None,
        rank_end=int(df_log["rank"].max()) if not df_log.empty else None,
        min_observations=min_observations,
    )


# =============================================================================
# STEP 4 — RESIDUAL ANALYSIS
# =============================================================================

def compute_residuals(df_log: pd.DataFrame, fit: ZipfFitResult) -> pd.DataFrame:
    """
    Compute residual = observed_log_frequency - predicted_log_frequency
    for every row, using the given (typically overall) fit.
    """
    df = df_log.copy()
    df["predicted_log_frequency"] = fit.intercept + fit.slope * df["log_rank"]
    df["residual"] = df["log_frequency"] - df["predicted_log_frequency"]
    return df[["rank", "token", "frequency", "log_rank", "log_frequency",
               "predicted_log_frequency", "residual"]]


def summarize_residuals(df_residuals: pd.DataFrame) -> dict:
    """
    Numeric summary of residual behavior (no automatic conclusions drawn).
    """
    if df_residuals.empty:
        return {
            "mean_residual": None,
            "std_residual": None,
            "residual_head_mean": None,   # first decile of ranks
            "residual_tail_mean": None,   # last decile of ranks
        }

    res = df_residuals["residual"]
    n = len(df_residuals)
    decile = max(1, n // 10)
    head_mean = float(res.iloc[:decile].mean())
    tail_mean = float(res.iloc[-decile:].mean())

    return {
        "mean_residual": float(res.mean()),
        "std_residual": float(res.std()),
        "residual_head_mean": head_mean,
        "residual_tail_mean": tail_mean,
    }


# =============================================================================
# STEP 5 — PIECEWISE ZIPF ANALYSIS
# =============================================================================

def adapt_rank_regions(
    regions: list[tuple[int, int | None]], vocabulary_size: int
) -> list[tuple[int, int]]:
    """
    Clip configured rank regions to the actual vocabulary size, dropping
    regions that fall entirely outside the available ranks and resolving
    open-ended (None) upper bounds to the vocabulary size.
    """
    adapted: list[tuple[int, int]] = []
    for start, end in regions:
        if start > vocabulary_size:
            continue
        actual_end = vocabulary_size if end is None else min(end, vocabulary_size)
        if actual_end < start:
            continue
        adapted.append((start, actual_end))
    return adapted


def fit_piecewise_zipf(
    df_log: pd.DataFrame,
    regions: list[tuple[int, int | None]],
    min_observations: int = 5,
) -> pd.DataFrame:
    """
    Fit an independent Zipf model within each rank region.

    Returns a dataframe of per-region fit metrics (one row per region;
    regions with too few observations or entirely out of range are
    omitted with a logged warning).
    """
    vocabulary_size = int(df_log["rank"].max()) if not df_log.empty else 0
    adapted_regions = adapt_rank_regions(regions, vocabulary_size)

    rows = []
    for start, end in adapted_regions:
        subset = df_log[(df_log["rank"] >= start) & (df_log["rank"] <= end)]
        label = f"{start}-{end}"
        fit = fit_zipf_linear(
            subset["log_rank"].values,
            subset["log_frequency"].values,
            region_label=label,
            rank_start=start,
            rank_end=end,
            min_observations=min_observations,
        )
        if fit is not None:
            rows.append(fit.to_dict())

    return pd.DataFrame(rows)


# =============================================================================
# STEP 6 — LONG-TAIL ANALYSIS
# =============================================================================

def long_tail_statistics(
    df_rank_freq: pd.DataFrame,
    thresholds: list[int],
    coverage_k_values: list[int],
) -> pd.DataFrame:
    """
    Compute long-tail proportion statistics and top-K coverage.

    - percentage of vocabulary occurring <= threshold times, for each
      threshold in `thresholds` (typically [1, 2, 5, 10])
    - token coverage (% of total tokens) contributed by the top-K most
      frequent words, for each K in `coverage_k_values` that is valid
      for this corpus (K <= vocabulary_size)
    """
    rows = []
    vocabulary_size = len(df_rank_freq)
    total_tokens = int(df_rank_freq["frequency"].sum())

    if vocabulary_size == 0 or total_tokens == 0:
        return pd.DataFrame(rows)

    for t in thresholds:
        n_types = int((df_rank_freq["frequency"] <= t).sum())
        pct_vocab = n_types / vocabulary_size * 100
        rows.append(
            {
                "statistic": f"vocab_pct_freq_leq_{t}",
                "value": round(pct_vocab, 4),
                "description": f"% of vocabulary occurring <= {t} times",
            }
        )

    sorted_df = df_rank_freq.sort_values("rank")
    for k in coverage_k_values:
        if k > vocabulary_size:
            logger.info(f"Coverage K={k} exceeds vocabulary size; skipping.")
            continue
        covered_tokens = int(sorted_df.iloc[:k]["frequency"].sum())
        pct_coverage = covered_tokens / total_tokens * 100
        rows.append(
            {
                "statistic": f"token_coverage_top_{k}",
                "value": round(pct_coverage, 4),
                "description": f"% of total tokens covered by top-{k} most frequent words",
            }
        )

    hapax_count = int((df_rank_freq["frequency"] == 1).sum())
    rows.append(
        {
            "statistic": "hapax_proportion_of_vocab",
            "value": round(hapax_count / vocabulary_size * 100, 4),
            "description": "% of vocabulary that are hapax legomena (freq == 1)",
        }
    )

    return pd.DataFrame(rows)


# =============================================================================
# STRUCTURED SUMMARY (NEUTRAL — NO AUTOMATIC "DISPROVED" CLAIMS)
# =============================================================================

def generate_zipf_summary(
    overall_fit: ZipfFitResult | None,
    piecewise_df: pd.DataFrame,
    long_tail_df: pd.DataFrame,
    residual_summary: dict,
) -> list[str]:
    """
    Produce a list of neutral, factual statements summarizing the Zipf
    analysis results. Deliberately avoids declaring Zipf's Law "disproved"
    or "confirmed" — that judgment is left for the report author.
    """
    lines: list[str] = []

    if overall_fit is None:
        lines.append(
            "The overall Zipf fit could not be computed (insufficient data)."
        )
        return lines

    lines.append(
        f"The single-exponent model shows the following overall fit metrics: "
        f"exponent s = {overall_fit.exponent_s:.4f}, "
        f"R^2 = {overall_fit.r_squared:.4f}, "
        f"RMSE = {overall_fit.rmse:.4f}, "
        f"based on {overall_fit.n_observations} rank observations."
    )

    if not piecewise_df.empty:
        min_row = piecewise_df.loc[piecewise_df["r_squared"].idxmin()]
        max_row = piecewise_df.loc[piecewise_df["r_squared"].idxmax()]
        exp_min = piecewise_df["exponent_s"].min()
        exp_max = piecewise_df["exponent_s"].max()
        lines.append(
            f"The estimated exponent varies across rank regions, ranging from "
            f"{exp_min:.4f} to {exp_max:.4f} across {len(piecewise_df)} regions examined."
        )
        lines.append(
            f"The region '{max_row['region']}' shows the strongest single-power-law "
            f"fit (R^2 = {max_row['r_squared']:.4f}), while region '{min_row['region']}' "
            f"shows the weakest fit (R^2 = {min_row['r_squared']:.4f}) among the "
            f"regions examined."
        )
    else:
        lines.append("Piecewise analysis did not produce enough regions to compare.")

    if residual_summary.get("residual_head_mean") is not None:
        lines.append(
            f"The residual pattern can be inspected for systematic deviations: "
            f"mean residual over the first decile of ranks = "
            f"{residual_summary['residual_head_mean']:.4f}, over the last decile = "
            f"{residual_summary['residual_tail_mean']:.4f} "
            f"(overall residual std = {residual_summary['std_residual']:.4f})."
        )

    if not long_tail_df.empty:
        hapax_row = long_tail_df[long_tail_df["statistic"] == "hapax_proportion_of_vocab"]
        if not hapax_row.empty:
            lines.append(
                f"Long-tail statistics: hapax legomena make up "
                f"{hapax_row['value'].iloc[0]:.2f}% of the observed vocabulary."
            )

    lines.append(
        "These figures are quantitative evidence only; whether a single "
        "power-law model is judged adequate should be decided by the report "
        "author after inspecting the full metrics, residual plot, and "
        "piecewise comparison."
    )

    return lines


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_zipf_analysis(
    counter: Counter,
    output_dir: Path,
    piecewise_regions: list[tuple[int, int | None]],
    coverage_k_values: list[int],
    long_tail_thresholds: list[int],
    min_observations: int = 5,
) -> dict:
    """
    Full Zipf analysis pipeline (Steps 1-6; figures are generated
    separately via visualization.py using the returned dataframes).

    Returns a dict with all intermediate/final dataframes and results so
    that scripts/visualization can reuse them without recomputation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # STEP 1
    rank_freq_df = build_rank_frequency_table(counter)

    # STEP 2
    log_df = add_log_columns(rank_freq_df)

    # STEP 3
    overall_fit = fit_overall_zipf(log_df, min_observations=min_observations)
    if overall_fit is not None:
        pd.DataFrame([overall_fit.to_dict()]).to_csv(
            output_dir / "zipf_metrics.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=["region", "rank_start", "rank_end", "n_observations",
                     "slope", "exponent_s", "intercept", "r_squared", "rmse"]
        ).to_csv(output_dir / "zipf_metrics.csv", index=False)

    # STEP 4
    if overall_fit is not None:
        residuals_df = compute_residuals(log_df, overall_fit)
        residual_summary = summarize_residuals(residuals_df)
    else:
        residuals_df = pd.DataFrame()
        residual_summary = summarize_residuals(residuals_df)
    residuals_df.to_csv(output_dir / "zipf_residuals.csv", index=False)

    # STEP 5
    piecewise_df = fit_piecewise_zipf(
        log_df, piecewise_regions, min_observations=min_observations
    )
    piecewise_df.to_csv(output_dir / "zipf_piecewise_metrics.csv", index=False)

    # STEP 6
    long_tail_df = long_tail_statistics(
        rank_freq_df, long_tail_thresholds, coverage_k_values
    )
    long_tail_df.to_csv(output_dir / "zipf_long_tail_statistics.csv", index=False)

    # Structured neutral summary
    summary_lines = generate_zipf_summary(
        overall_fit, piecewise_df, long_tail_df, residual_summary
    )
    with open(output_dir / "zipf_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    for line in summary_lines:
        logger.info(line)

    return {
        "rank_freq_df": rank_freq_df,
        "log_df": log_df,
        "overall_fit": overall_fit,
        "residuals_df": residuals_df,
        "residual_summary": residual_summary,
        "piecewise_df": piecewise_df,
        "long_tail_df": long_tail_df,
        "summary_lines": summary_lines,
    }
