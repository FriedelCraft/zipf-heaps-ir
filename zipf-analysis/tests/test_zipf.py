"""
test_zipf.py

Unit tests for src/zipf_analysis.py using synthetic, known distributions
(clearly TEST data — never real corpus statistics).
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.zipf_analysis import (
    build_rank_frequency_table, add_log_columns, fit_overall_zipf,
    fit_piecewise_zipf, adapt_rank_regions, long_tail_statistics,
    compute_residuals, summarize_residuals,
)


# =============================================================================
# STEP 1 — Ranking / frequency counting
# =============================================================================

def test_build_rank_frequency_table_sorts_descending():
    counter = Counter({"a": 5, "b": 10, "c": 1})
    df = build_rank_frequency_table(counter)
    assert list(df["token"]) == ["b", "a", "c"]
    assert list(df["rank"]) == [1, 2, 3]
    assert list(df["frequency"]) == [10, 5, 1]


def test_build_rank_frequency_table_deterministic_ties():
    counter = Counter({"zebra": 3, "apple": 3, "mango": 3})
    df = build_rank_frequency_table(counter)
    # equal frequency -> alphabetical order
    assert list(df["token"]) == ["apple", "mango", "zebra"]


def test_build_rank_frequency_table_empty():
    df = build_rank_frequency_table(Counter())
    assert df.empty
    assert list(df.columns) == ["rank", "token", "frequency"]


# =============================================================================
# STEP 2 — Log transformation
# =============================================================================

def test_add_log_columns_basic():
    counter = Counter({"a": 100, "b": 50, "c": 25})
    df = build_rank_frequency_table(counter)
    log_df = add_log_columns(df)
    assert len(log_df) == 3
    assert np.isclose(log_df.iloc[0]["log_rank"], np.log(1))
    assert np.isclose(log_df.iloc[0]["log_frequency"], np.log(100))


def test_add_log_columns_empty_corpus_graceful():
    df = build_rank_frequency_table(Counter())
    log_df = add_log_columns(df)
    assert log_df.empty
    assert "log_rank" in log_df.columns
    assert "log_frequency" in log_df.columns


# =============================================================================
# STEP 3 — Known synthetic Zipf distribution fit
# =============================================================================

def test_fit_known_zipf_distribution_recovers_exponent():
    """
    Construct a PERFECT synthetic Zipf distribution with a known exponent
    and verify the fit recovers it almost exactly (this is TEST data only,
    not a real corpus).
    """
    true_s = 1.0
    C = 1000
    ranks = np.arange(1, 501)
    freqs = (C / (ranks ** true_s)).round().astype(int)
    freqs = np.maximum(freqs, 1)  # avoid zero frequencies

    counter = Counter({f"tok_{r}": int(f) for r, f in zip(ranks, freqs)})
    df = build_rank_frequency_table(counter)
    log_df = add_log_columns(df)
    fit = fit_overall_zipf(log_df, min_observations=5)

    assert fit is not None
    assert fit.exponent_s == pytest.approx(true_s, abs=0.05)
    assert fit.r_squared > 0.95


def test_fit_overall_zipf_insufficient_observations_returns_none():
    counter = Counter({"a": 5, "b": 3})
    df = build_rank_frequency_table(counter)
    log_df = add_log_columns(df)
    fit = fit_overall_zipf(log_df, min_observations=5)
    assert fit is None


# =============================================================================
# STEP 4 — Residual analysis
# =============================================================================

def test_residuals_zero_for_perfect_fit():
    true_s = 1.2
    C = 500
    ranks = np.arange(1, 201)
    freqs = C / (ranks ** true_s)  # exact, no rounding -> perfect log-linear fit

    counter_like_df_rank = ranks
    import pandas as pd
    df = pd.DataFrame({"rank": ranks, "token": [f"t{r}" for r in ranks], "frequency": freqs})
    log_df = add_log_columns(df)
    fit = fit_overall_zipf(log_df, min_observations=5)
    residuals_df = compute_residuals(log_df, fit)

    assert np.allclose(residuals_df["residual"], 0, atol=1e-8)
    summary = summarize_residuals(residuals_df)
    assert summary["mean_residual"] == pytest.approx(0, abs=1e-8)


# =============================================================================
# STEP 5 — Piecewise analysis + region adaptation
# =============================================================================

def test_adapt_rank_regions_clips_to_vocab_size():
    regions = [(1, 100), (101, 1000), (1001, None)]
    adapted = adapt_rank_regions(regions, vocabulary_size=500)
    assert adapted == [(1, 100), (101, 500)]


def test_adapt_rank_regions_drops_out_of_range():
    regions = [(1, 100), (101, 1000)]
    adapted = adapt_rank_regions(regions, vocabulary_size=50)
    assert adapted == [(1, 50)]


def test_fit_piecewise_zipf_produces_region_rows():
    true_s = 1.0
    C = 1000
    ranks = np.arange(1, 301)
    freqs = np.maximum((C / (ranks ** true_s)).round().astype(int), 1)
    counter = Counter({f"t{r}": int(f) for r, f in zip(ranks, freqs)})
    df = build_rank_frequency_table(counter)
    log_df = add_log_columns(df)

    regions = [(1, 100), (101, 300)]
    piecewise_df = fit_piecewise_zipf(log_df, regions, min_observations=5)
    assert len(piecewise_df) == 2
    assert set(piecewise_df["region"]) == {"1-100", "101-300"}


# =============================================================================
# STEP 6 — Long-tail statistics
# =============================================================================

def test_long_tail_statistics_hapax_and_coverage():
    counter = Counter({"a": 100, "b": 1, "c": 1, "d": 2})
    df = build_rank_frequency_table(counter)
    result = long_tail_statistics(df, thresholds=[1, 2], coverage_k_values=[1, 2])

    hapax_row = result[result["statistic"] == "vocab_pct_freq_leq_1"]
    assert hapax_row["value"].iloc[0] == pytest.approx(50.0)  # 2 of 4 tokens have freq==1

    coverage_row = result[result["statistic"] == "token_coverage_top_1"]
    total = 104
    assert coverage_row["value"].iloc[0] == pytest.approx(100 / total * 100)


def test_long_tail_statistics_skips_k_larger_than_vocab():
    counter = Counter({"a": 5, "b": 3})
    df = build_rank_frequency_table(counter)
    result = long_tail_statistics(df, thresholds=[1], coverage_k_values=[100])
    assert "token_coverage_top_100" not in set(result["statistic"])
