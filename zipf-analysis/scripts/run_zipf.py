#!/usr/bin/env python3
"""
run_zipf.py

Single entry point for the standalone Zipf Analysis project.

Runs: corpus statistics -> Zipf analysis (Steps 1-7: ranking, log
transform, overall fit, residuals, piecewise analysis, long-tail
statistics, figures).

Modes:
    demo   Uses the bundled synthetic corpus (data/demo/demo_bengali_corpus.txt).
           Outputs go to results/demo/, clearly labelled [DEMO].
    full   Uses data/processed/bengali_corpus.txt (place your real, processed
           Bengali corpus there first). Outputs go to results/final/.

Usage:
    python scripts/run_zipf.py --mode demo
    python scripts/run_zipf.py --mode full
    python scripts/run_zipf.py --mode full --corpus path/to/corpus.txt --output results/final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running this script directly (adds project root to sys.path)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.corpus_stats import run_corpus_stats
from src.zipf_analysis import run_zipf_analysis
from src.visualization import (
    plot_zipf_loglog, plot_zipf_residuals, plot_zipf_piecewise_slopes,
)
from src.utils import get_logger, print_missing_corpus_message, print_banner

logger = get_logger("run_zipf")


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

    print_banner(f"RUNNING ZIPF ANALYSIS — mode={mode.upper()}")
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

    # 1. Corpus statistics
    stats_result = run_corpus_stats(
        corpus_path=corpus_path,
        output_dir=tables_dir,
        encoding=cfg.encoding,
        top_k=cfg.top_k_tokens,
        chunk_size_bytes=cfg.corpus_chunk_size_mb * 1024 * 1024,
        progress_log_interval_bytes=cfg.corpus_progress_log_interval_mb * 1024 * 1024,
    )

    # 2-6. Zipf analysis (ranking, log transform, overall fit, residuals,
    #      piecewise analysis, long-tail statistics)
    zipf_result = run_zipf_analysis(
        counter=stats_result["counter"],
        output_dir=metrics_dir,
        piecewise_regions=cfg.zipf_piecewise_regions,
        coverage_k_values=cfg.zipf_coverage_k_values,
        long_tail_thresholds=cfg.zipf_long_tail_thresholds,
        min_observations=cfg.zipf_min_observations_for_fit,
    )
    # Long-tail / piecewise tables also saved under tables/ for convenience
    zipf_result["long_tail_df"].to_csv(tables_dir / "zipf_long_tail_statistics.csv", index=False)
    zipf_result["piecewise_df"].to_csv(tables_dir / "zipf_piecewise_metrics.csv", index=False)

    # 7. Figures
    plot_zipf_loglog(
        zipf_result["log_df"], zipf_result["overall_fit"],
        figures_dir / "zipf_loglog.png", dpi=cfg.fig_dpi, figsize=cfg.fig_size,
        label_prefix=label_prefix,
    )
    plot_zipf_residuals(
        zipf_result["residuals_df"], figures_dir / "zipf_residuals.png",
        dpi=cfg.fig_dpi, figsize=cfg.fig_size, label_prefix=label_prefix,
    )
    plot_zipf_piecewise_slopes(
        zipf_result["piecewise_df"], figures_dir / "zipf_piecewise_slopes.png",
        dpi=cfg.fig_dpi, figsize=cfg.fig_size, label_prefix=label_prefix,
    )

    print_banner(f"ZIPF ANALYSIS COMPLETE — mode={mode.upper()} — outputs in {output_dir}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the standalone Zipf analysis pipeline.")
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
