#!/usr/bin/env python3
"""
run_heaps.py

Single entry point for the standalone Heaps' Law Analysis project.

Streams the corpus ONCE, in fixed-size byte chunks, maintaining a plain
set() of unique tokens seen so far and recording vocabulary-size
checkpoints at logarithmically-spaced token counts (see
src/corpus_streaming.py). Fits V(N) = K * N^beta via log-log OLS
regression, computes residuals, and generates all figures/tables/metrics.

This project is fully independent of the companion zipf_analysis_project
and cost_infrastructure_project — it does its own single streaming pass
over the corpus, memory-safe on its own, using the same fixed-size
byte-chunk architecture (UTF-8 boundary safety, token boundary safety),
adapted to track only a set of unique tokens rather than per-token
frequencies, since that's all Heaps' Law needs.

NOTE ON RUNNING ALONGSIDE ZIPF: if you also want Zipf's Law results, run
zipf_analysis_project separately — that means the corpus gets read once
by each project (two total passes) rather than sharing a single pass.
This is the tradeoff of keeping the two analyses in fully independent,
separately runnable projects. zipf_analysis_project also has an
integrated mode that can produce both Zipf and Heaps results in one pass
if you want the efficiency instead of the independence.

Modes:
    demo   Uses the bundled synthetic corpus (data/demo/demo_bengali_corpus.txt).
           Outputs go to results/demo/, clearly labelled [DEMO].
    full   Uses data/processed/bengali_corpus.txt (place your real, processed
           Bengali corpus there first). Outputs go to results/final/.

Usage:
    python scripts/run_heaps.py --mode demo
    python scripts/run_heaps.py --mode full
    python scripts/run_heaps.py --mode full --corpus path/to/corpus.txt --output results/final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly (adds project root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.corpus_streaming import stream_corpus_for_heaps
from src.heaps_analysis import (
    generate_checkpoint_targets, run_heaps_analysis, build_cost_project_compatible_row,
)
from src.visualization import (
    plot_heaps_loglog, plot_heaps_vocabulary_growth, plot_heaps_residuals,
)
from src.utils import get_logger, print_missing_corpus_message, print_banner

logger = get_logger("run_heaps")


def run_pipeline(mode: str, cfg, corpus_override: Path | None, output_override: Path | None) -> int:
    label_prefix = "[DEMO] " if mode == "demo" else ""

    if mode == "demo":
        corpus_path = cfg.demo_corpus_path
    else:
        corpus_path = corpus_override or cfg.processed_corpus_path

    output_dir = output_override or cfg.results_dir_for_mode(mode)
    tables_dir = output_dir / "tables"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print_banner(f"RUNNING HEAPS' LAW ANALYSIS — mode={mode.upper()}")
    logger.info(f"Corpus: {corpus_path}")
    logger.info(f"Output directory: {output_dir}")

    if not corpus_path.exists():
        if mode == "demo":
            logger.error(
                f"Demo corpus not found at {corpus_path}. "
                f"This should be bundled with the repository — check your checkout."
            )
        else:
            print_missing_corpus_message(corpus_path)
        return 1

    # 1. Stream the corpus exactly once, recording Heaps checkpoints.
    checkpoint_iter = generate_checkpoint_targets(
        first_checkpoint_tokens=cfg.heaps_first_checkpoint_tokens,
        checkpoints_per_decade=cfg.heaps_checkpoints_per_decade,
        max_checkpoints=cfg.heaps_max_checkpoints,
    )
    checkpoints = stream_corpus_for_heaps(
        corpus_path,
        checkpoint_iter,
        encoding=cfg.encoding,
        chunk_size_bytes=cfg.corpus_chunk_size_mb * 1024 * 1024,
        progress_log_interval_bytes=cfg.corpus_progress_log_interval_mb * 1024 * 1024,
    )

    # 2. Fit Heaps' Law, compute residuals, save tables/metrics/summary.
    result = run_heaps_analysis(
        checkpoints=checkpoints,
        output_dir=metrics_dir,
        min_tokens_for_fit=cfg.heaps_min_tokens_for_fit,
        min_observations_for_fit=cfg.heaps_min_observations_for_fit,
    )
    # Growth table also saved under tables/ for convenience.
    result["growth_df"].to_csv(tables_dir / "heaps_vocabulary_growth.csv", index=False)

    # 3. Figures.
    plot_heaps_loglog(
        result["growth_df"], result["fit"], figures_dir / "heaps_loglog.png",
        dpi=cfg.fig_dpi, figsize=cfg.fig_size, label_prefix=label_prefix,
    )
    plot_heaps_vocabulary_growth(
        result["growth_df"], figures_dir / "heaps_vocabulary_growth.png",
        dpi=cfg.fig_dpi, figsize=cfg.fig_size, label_prefix=label_prefix,
    )
    plot_heaps_residuals(
        result["residuals_df"], figures_dir / "heaps_residuals.png",
        dpi=cfg.fig_dpi, figsize=cfg.fig_size, label_prefix=label_prefix,
    )

    # 4. Optional bridge file for cost_infrastructure_project: if you want
    #    this project's results picked up there, copy the output file
    #    below to cost_infrastructure_project/data/external/heaps_results.csv
    #    (see that project's README for its expected schema).
    compat_row = build_cost_project_compatible_row(
        result["fit"], result["growth_df"], corpus_version=cfg.heaps_corpus_version,
    )
    compat_row.to_csv(tables_dir / "heaps_results_for_cost_project.csv", index=False)

    print_banner(f"HEAPS ANALYSIS COMPLETE — mode={mode.upper()} — outputs in {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standalone Heaps' Law analysis pipeline.")
    parser.add_argument("--mode", type=str, default="demo", choices=["demo", "full"],
                         help="Pipeline mode (default: demo)")
    parser.add_argument("--corpus", type=str, default=None,
                         help="Override corpus path (used in full mode; default: config.yaml processed_corpus)")
    parser.add_argument("--output", type=str, default=None,
                         help="Override output directory (default: results/demo or results/final)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    corpus_override = Path(args.corpus) if args.corpus else None
    output_override = Path(args.output) if args.output else None

    return run_pipeline(args.mode, cfg, corpus_override, output_override)


if __name__ == "__main__":
    raise SystemExit(main())
