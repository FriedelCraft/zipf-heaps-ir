"""
test_heaps_analysis.py

Tests for the standalone Heaps' Law project: checkpoint scheduling
(src/heaps_analysis.py), memory-safe set-based streaming
(src/corpus_streaming.py), and regression fitting / residuals / summary
generation.

Focus areas (matching the design requirements):
    1. Small synthetic corpus where vocabulary growth can be verified by hand.
    2. Token boundaries across chunks -- including cases where MULTIPLE
       checkpoints fall inside a single chunk, and where a token itself
       is split mid-chunk right at a checkpoint boundary (both English
       and Bengali/UTF-8 text).
    3. Known, hand-computable vocabulary growth behavior.
    4. Heaps regression recovers sensible K/beta/R^2 from a synthetic
       power-law dataset.
    5. run_heaps_analysis end-to-end on a small checkpoint table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus_streaming import stream_corpus_for_heaps, _utf8_safe_split_point
from src.heaps_analysis import (
    generate_checkpoint_targets, fit_heaps_law, compute_heaps_residuals,
    generate_heaps_summary, run_heaps_analysis, build_cost_project_compatible_row,
    HeapsFitResult,
)


def _write(tmp_path: Path, text: str, filename: str = "corpus.txt") -> Path:
    path = tmp_path / filename
    path.write_bytes(text.encode("utf-8"))
    return path


# =============================================================================
# CHECKPOINT SCHEDULE
# =============================================================================

def test_generate_checkpoint_targets_starts_at_first_checkpoint():
    gen = generate_checkpoint_targets(first_checkpoint_tokens=1000, checkpoints_per_decade=2)
    assert next(gen) == 1000


def test_generate_checkpoint_targets_is_strictly_ascending():
    gen = generate_checkpoint_targets(first_checkpoint_tokens=100, checkpoints_per_decade=3)
    values = [next(gen) for _ in range(20)]
    assert values == sorted(values)
    assert len(values) == len(set(values))


def test_generate_checkpoint_targets_density_roughly_matches_config():
    gen = generate_checkpoint_targets(first_checkpoint_tokens=10_000, checkpoints_per_decade=4)
    values = [next(gen) for _ in range(9)]
    assert values[0] == 10_000
    assert values[4] == pytest.approx(100_000, rel=0.05)
    assert values[8] == pytest.approx(1_000_000, rel=0.05)


def test_generate_checkpoint_targets_respects_max_checkpoints():
    gen = generate_checkpoint_targets(first_checkpoint_tokens=10, checkpoints_per_decade=10, max_checkpoints=15)
    assert len(list(gen)) == 15


def test_generate_checkpoint_targets_invalid_params_raise():
    with pytest.raises(ValueError):
        next(generate_checkpoint_targets(first_checkpoint_tokens=0, checkpoints_per_decade=4))
    with pytest.raises(ValueError):
        next(generate_checkpoint_targets(first_checkpoint_tokens=100, checkpoints_per_decade=0))


# =============================================================================
# UTF-8 SAFE SPLIT POINT (direct unit tests on the boundary helper)
# =============================================================================

def test_utf8_safe_split_point_pure_ascii():
    chunk = "hello".encode("utf-8")
    assert _utf8_safe_split_point(chunk) == len(chunk)


def test_utf8_safe_split_point_truncated_multibyte_char():
    full = "ম".encode("utf-8")  # 3-byte Bengali character
    for cut in range(1, 3):
        assert _utf8_safe_split_point(full[:cut]) == 0


def test_utf8_safe_split_point_empty():
    assert _utf8_safe_split_point(b"") == 0


# =============================================================================
# SMALL SYNTHETIC CORPUS -- vocabulary growth verified BY HAND
# =============================================================================

def test_vocabulary_growth_hand_verifiable(tmp_path):
    """
    20 tokens: 5 distinct words, each repeated 4 times, round-robin order.
    Vocabulary size after N tokens is hand-computable: after the first 5
    tokens all 5 words have appeared, so vocabulary_size == 5 for every
    checkpoint from N=5 onward.
    """
    words = ["a", "b", "c", "d", "e"]
    text = " ".join(words * 4)  # 20 tokens
    path = _write(tmp_path, text)

    targets = iter([1, 2, 3, 4, 5, 10, 15, 20])
    checkpoints = stream_corpus_for_heaps(path, targets, chunk_size_bytes=3)

    by_n = {c["token_count"]: c["vocabulary_size"] for c in checkpoints}
    assert by_n[1] == 1
    assert by_n[2] == 2
    assert by_n[3] == 3
    assert by_n[4] == 4
    assert by_n[5] == 5
    assert by_n[10] == 5
    assert by_n[15] == 5
    assert by_n[20] == 5
    assert checkpoints[-1]["token_count"] == 20


def test_final_checkpoint_always_equals_true_total(tmp_path):
    text = " ".join(["word"] * 37)  # not a "clean" number
    path = _write(tmp_path, text)

    targets = iter([10, 20, 30])  # schedule stops well short of 37
    checkpoints = stream_corpus_for_heaps(path, targets, chunk_size_bytes=4)

    assert checkpoints[-1]["token_count"] == 37
    assert checkpoints[-1]["vocabulary_size"] == 1


def test_no_checkpoints_reached_still_records_final_total(tmp_path):
    text = "only three tokens"
    path = _write(tmp_path, text)

    targets = iter([1_000_000])
    checkpoints = stream_corpus_for_heaps(path, targets, chunk_size_bytes=5)

    assert len(checkpoints) == 1
    assert checkpoints[0]["token_count"] == 3
    assert checkpoints[0]["vocabulary_size"] == 3


# =============================================================================
# TOKEN / CHUNK BOUNDARY CORRECTNESS
# =============================================================================

def test_multiple_checkpoints_within_a_single_chunk(tmp_path):
    """
    Whole file fits in one chunk, but several checkpoint targets are
    smaller than the file -- the slicing logic must still record each
    checkpoint at the exact right vocabulary size, not just once total.
    """
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    path = _write(tmp_path, text)

    targets = iter([10, 20, 30, 40, 50])
    checkpoints = stream_corpus_for_heaps(path, targets, chunk_size_bytes=10_000)

    by_n = {c["token_count"]: c["vocabulary_size"] for c in checkpoints}
    assert by_n[10] == 10
    assert by_n[20] == 20
    assert by_n[30] == 30
    assert by_n[40] == 40
    assert by_n[50] == 50


def test_bengali_token_split_across_chunk_at_checkpoint_boundary(tmp_path):
    """
    Deliberately tiny chunk size so a Bengali (multi-byte UTF-8) token is
    split across a chunk boundary right around where a checkpoint should
    fire. Verifies vocabulary size is still counted correctly.
    """
    words = ["আমি", "বাংলায়", "গান", "গাই", "প্রতিদিন"]  # 5 distinct Bengali words
    text = " ".join(words * 3)  # 15 tokens
    path = _write(tmp_path, text, filename="bn.txt")

    for chunk_size in [1, 2, 3, 4, 5, 7]:
        targets = iter([3, 5, 10, 15])
        checkpoints = stream_corpus_for_heaps(path, targets, chunk_size_bytes=chunk_size)
        by_n = {c["token_count"]: c["vocabulary_size"] for c in checkpoints}
        assert by_n[3] == 3, f"failed at chunk_size={chunk_size}"
        assert by_n[5] == 5, f"failed at chunk_size={chunk_size}"
        assert by_n[10] == 5, f"failed at chunk_size={chunk_size}"
        assert by_n[15] == 5, f"failed at chunk_size={chunk_size}"


def test_english_token_split_across_chunk_boundary(tmp_path):
    text = "elephant elephant giraffe giraffe giraffe zebra"  # 3 distinct words, 6 tokens
    path = _write(tmp_path, text)

    for chunk_size in range(1, 12):
        targets = iter([2, 4, 6])
        checkpoints = stream_corpus_for_heaps(path, targets, chunk_size_bytes=chunk_size)
        final = checkpoints[-1]
        assert final["token_count"] == 6, f"failed at chunk_size={chunk_size}"
        assert final["vocabulary_size"] == 3, f"failed at chunk_size={chunk_size}"


def test_repeated_word_split_across_boundary_counts_correctly(tmp_path):
    text = "banana banana banana apple"  # 2 distinct words, 4 tokens
    path = _write(tmp_path, text)

    for chunk_size in [1, 2, 3, 4, 5, 6]:
        targets = iter([1, 2, 3, 4])
        checkpoints = stream_corpus_for_heaps(path, targets, chunk_size_bytes=chunk_size)
        by_n = {c["token_count"]: c["vocabulary_size"] for c in checkpoints}
        assert by_n[1] == 1, f"failed at chunk_size={chunk_size}"  # "banana"
        assert by_n[2] == 1, f"failed at chunk_size={chunk_size}"  # "banana" again
        assert by_n[3] == 1, f"failed at chunk_size={chunk_size}"  # "banana" again
        assert by_n[4] == 2, f"failed at chunk_size={chunk_size}"  # "apple" is new


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        stream_corpus_for_heaps(tmp_path / "does_not_exist.txt", iter([100]))


def test_empty_file_returns_single_zero_checkpoint(tmp_path):
    path = _write(tmp_path, "")
    checkpoints = stream_corpus_for_heaps(path, iter([100]), chunk_size_bytes=8)
    assert len(checkpoints) == 1
    assert checkpoints[0]["token_count"] == 0
    assert checkpoints[0]["vocabulary_size"] == 0


# =============================================================================
# REGRESSION FIT -- recovers known K, beta from synthetic power-law data
# =============================================================================

def test_fit_heaps_law_recovers_known_parameters():
    true_K = 5.0
    true_beta = 0.6
    token_counts = np.array([1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000], dtype=float)
    vocabulary_sizes = true_K * (token_counts ** true_beta)  # exact, no noise

    fit = fit_heaps_law(token_counts, vocabulary_sizes, min_observations=5)

    assert fit is not None
    assert fit.beta == pytest.approx(true_beta, abs=0.01)
    assert fit.K == pytest.approx(true_K, rel=0.05)
    assert fit.r_squared > 0.999


def test_fit_heaps_law_insufficient_observations_returns_none():
    fit = fit_heaps_law(np.array([100.0, 200.0]), np.array([10.0, 15.0]), min_observations=5)
    assert fit is None


def test_fit_heaps_law_beta_typically_between_0_and_1():
    token_counts = np.array([1e4, 1e5, 1e6, 1e7], dtype=float)
    vocabulary_sizes = np.array([2000, 8000, 30000, 100000], dtype=float)  # sublinear-ish
    fit = fit_heaps_law(token_counts, vocabulary_sizes, min_observations=3)
    assert fit is not None
    assert 0 < fit.beta < 1


# =============================================================================
# RESIDUALS
# =============================================================================

def test_residuals_zero_for_perfect_fit():
    true_K, true_beta = 2.0, 0.7
    token_counts = np.array([1e3, 1e4, 1e5, 1e6, 1e7])
    vocab = true_K * (token_counts ** true_beta)
    growth_df = pd.DataFrame({"token_count": token_counts, "vocabulary_size": vocab})

    fit = fit_heaps_law(token_counts, vocab, min_observations=3)
    residuals_df = compute_heaps_residuals(growth_df, fit)

    assert np.allclose(residuals_df["residual"], 0, atol=1e-9)


def test_residuals_empty_growth_df_graceful():
    result = compute_heaps_residuals(pd.DataFrame(columns=["token_count", "vocabulary_size"]), None)
    assert result.empty
    assert "residual" in result.columns


# =============================================================================
# SUMMARY -- EMPIRICAL vs FITTED vs QUALITATIVE sections all present
# =============================================================================

def test_summary_distinguishes_empirical_fitted_and_qualitative_sections():
    growth_df = pd.DataFrame({
        "token_count": [1000, 10000, 100000],
        "vocabulary_size": [200, 900, 3000],
    })
    fit = HeapsFitResult(K=10.0, beta=0.55, intercept_log_k=np.log(10.0),
                         r_squared=0.98, rmse=0.05, n_observations=3)

    lines = generate_heaps_summary(fit, growth_df)
    text = "\n".join(lines)

    assert "EMPIRICAL RESULT:" in text
    assert "FITTED MODEL VALUES:" in text
    assert "QUALITATIVE IMPLICATION:" in text
    assert "SUBLINEAR" in text
    assert "NOT DONE HERE" in text
    assert "cost" in text.lower()
    assert "multiplier" in text.lower()


def test_summary_handles_missing_fit_gracefully():
    growth_df = pd.DataFrame({"token_count": [100], "vocabulary_size": [50]})
    lines = generate_heaps_summary(None, growth_df)
    assert "could not be computed" in "\n".join(lines)


# =============================================================================
# ORCHESTRATION -- run_heaps_analysis end-to-end on a small checkpoint table
# =============================================================================

def test_run_heaps_analysis_end_to_end(tmp_path):
    true_K, true_beta = 3.0, 0.65
    token_counts = [10_000, 31_623, 100_000, 316_228, 1_000_000, 3_162_278, 10_000_000]
    checkpoints = [
        {"token_count": n, "vocabulary_size": true_K * (n ** true_beta)}
        for n in token_counts
    ]

    result = run_heaps_analysis(
        checkpoints, output_dir=tmp_path,
        min_tokens_for_fit=1, min_observations_for_fit=5,
    )

    assert result["fit"] is not None
    assert result["fit"].beta == pytest.approx(true_beta, abs=0.01)

    assert (tmp_path / "heaps_vocabulary_growth.csv").exists()
    assert (tmp_path / "heaps_metrics.csv").exists()
    assert (tmp_path / "heaps_residuals.csv").exists()
    assert (tmp_path / "heaps_summary.txt").exists()

    metrics_df = pd.read_csv(tmp_path / "heaps_metrics.csv")
    for col in ["K", "beta", "R_squared", "RMSE", "number_of_checkpoints"]:
        assert col in metrics_df.columns
    assert metrics_df["number_of_checkpoints"].iloc[0] == len(token_counts)


def test_run_heaps_analysis_respects_min_tokens_for_fit(tmp_path):
    checkpoints = [
        {"token_count": 100, "vocabulary_size": 50},
        {"token_count": 1_000, "vocabulary_size": 300},
        {"token_count": 200_000, "vocabulary_size": 5000},
        {"token_count": 500_000, "vocabulary_size": 9000},
        {"token_count": 1_000_000, "vocabulary_size": 15000},
        {"token_count": 5_000_000, "vocabulary_size": 40000},
    ]
    result = run_heaps_analysis(
        checkpoints, output_dir=tmp_path,
        min_tokens_for_fit=100_000, min_observations_for_fit=3,
    )
    assert len(result["growth_df"]) == 6
    assert len(result["fit_df"]) == 4
    assert result["fit"] is not None
    assert result["fit"].n_observations == 4


def test_run_heaps_analysis_empty_checkpoints_graceful(tmp_path):
    result = run_heaps_analysis([], output_dir=tmp_path, min_observations_for_fit=5)
    assert result["fit"] is None
    assert result["growth_df"].empty
    assert (tmp_path / "heaps_summary.txt").exists()


# =============================================================================
# COST-PROJECT COMPATIBILITY EXPORT
# =============================================================================

def test_build_cost_project_compatible_row_schema():
    growth_df = pd.DataFrame({"token_count": [1000, 100000], "vocabulary_size": [200, 3000]})
    fit = HeapsFitResult(K=10.0, beta=0.55, intercept_log_k=np.log(10.0),
                         r_squared=0.98, rmse=0.05, n_observations=2)

    row = build_cost_project_compatible_row(fit, growth_df, corpus_version="test_v1")

    expected_cols = {"corpus_version", "token_count", "vocabulary_size", "K", "beta", "r_squared", "rmse"}
    assert expected_cols.issubset(set(row.columns))
    assert row["corpus_version"].iloc[0] == "test_v1"
    assert row["token_count"].iloc[0] == 100000  # final checkpoint
    assert row["vocabulary_size"].iloc[0] == 3000
    assert row["beta"].iloc[0] == pytest.approx(0.55)


def test_build_cost_project_compatible_row_handles_no_fit():
    growth_df = pd.DataFrame({"token_count": [100], "vocabulary_size": [50]})
    row = build_cost_project_compatible_row(None, growth_df, corpus_version="test_v1")
    assert pd.isna(row["K"].iloc[0])
    assert pd.isna(row["beta"].iloc[0])


# =============================================================================
# INTEGRATION -- streaming feeds directly into the fitting pipeline
# =============================================================================

def test_stream_then_fit_end_to_end(tmp_path):
    """
    A larger synthetic corpus (Zipfian sample from an open vocabulary
    pool) streamed with tiny chunks, then fit -- confirms the full
    stream -> fit -> summary chain works together, not just each piece
    in isolation.
    """
    import random
    random.seed(0)
    vocab = [f"tok{i}" for i in range(500)]
    weights = [1.0 / (i + 1) for i in range(len(vocab))]
    text = " ".join(random.choices(vocab, weights=weights, k=20_000))
    path = _write(tmp_path, text)

    targets = generate_checkpoint_targets(first_checkpoint_tokens=100, checkpoints_per_decade=4)
    checkpoints = stream_corpus_for_heaps(path, targets, chunk_size_bytes=512)

    result = run_heaps_analysis(
        checkpoints, output_dir=tmp_path / "out",
        min_tokens_for_fit=100, min_observations_for_fit=3,
    )
    assert result["fit"] is not None
    assert 0 < result["fit"].beta <= 1.2  # sanity range, not an exact value
