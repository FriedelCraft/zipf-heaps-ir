#!/usr/bin/env python3
"""
run_integration.py

Single entry point for the FULL standalone Cost / Infrastructure /
Integration pipeline: runs the assignment-based cost model, the
Search-Oriented vs LLM-Oriented architectural cost model (with LOW/BASE/
HIGH sensitivity analysis), and (if available) integrates a teammate's
Heaps' Law results into vocabulary-growth projections and a combined
summary that explicitly separates FACT / SCENARIO ASSUMPTION /
QUALITATIVE INTERPRETATION.

This is the recommended command to run this project end-to-end. Use
`scripts/run_cost_model.py` instead if you only want the cost/infra tables
and figures without Heaps integration.

Usage:
    python scripts/run_integration.py
    python scripts/run_integration.py --scenario LARGE --output results/final
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config
from src.cost_model import run_cost_model
from src.infrastructure_model import run_infrastructure_model
from src.integration import run_integration
from src.visualization import (
    plot_cost_vs_corpus_size,
    plot_initial_cost_comparison,
    plot_recurring_cost_comparison,
    plot_cost_breakdown_by_component,
    plot_cost_scaling_by_scenario,
    plot_sensitivity_range,
    plot_vocabulary_growth_projection,
)
from src.utils import get_logger, print_banner

logger = get_logger("run_integration")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the full cost/infrastructure/Heaps-integration pipeline."
    )
    parser.add_argument("--scenario", type=str, default=None,
                         choices=["SMALL", "MEDIUM", "LARGE"],
                         help="Corpus size scenario for the headline comparison (default: config.yaml)")
    parser.add_argument("--output", type=str, default=None,
                         help="Output directory (default: results/final)")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    scenario = args.scenario or cfg.infra_default_scenario
    output_dir = Path(args.output) if args.output else cfg.results_dir_for_mode("full")

    tables_dir = output_dir / "tables"
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"

    print_banner(f"RUNNING COST / INFRASTRUCTURE / INTEGRATION PIPELINE — scenario={scenario}")
    logger.info(f"Output directory: {output_dir}")

    # -----------------------------------------------------------------
    # 1. FACT: assignment-based data cost baseline + LM approach comparison
    # -----------------------------------------------------------------
    cost_result = run_cost_model(
        scenarios=cfg.cost_scenarios,
        usd_per_100k_words=cfg.usd_per_100k_words,
        lm_from_scratch_params=cfg.lm_from_scratch,
        lm_adapt_existing_params=cfg.lm_adapt_existing,
        output_dir=tables_dir,
    )
    plot_cost_vs_corpus_size(
        cost_result["scenarios_df"], figures_dir / "cost_vs_corpus_size.png",
        dpi=cfg.fig_dpi, figsize=cfg.fig_size,
    )

    # -----------------------------------------------------------------
    # 2. SCENARIO ASSUMPTION: Search-Oriented vs LLM-Oriented architecture model
    # -----------------------------------------------------------------
    infra_totals_df = None
    try:
        infra_result = run_infrastructure_model(
            assumptions_path=cfg.cost_assumptions_path,
            output_dir=tables_dir,
            scenario=scenario,
            default_uncertainty_pct=cfg.default_uncertainty_pct,
        )
        infra_totals_df = infra_result["totals_df"]

        plot_initial_cost_comparison(
            infra_result["totals_df"], scenario, figures_dir / "initial_cost_comparison.png",
            dpi=cfg.fig_dpi, figsize=cfg.fig_size,
        )
        plot_recurring_cost_comparison(
            infra_result["totals_df"], scenario, figures_dir / "recurring_cost_comparison.png",
            dpi=cfg.fig_dpi, figsize=cfg.fig_size,
        )
        plot_cost_breakdown_by_component(
            infra_result["breakdown_df"], scenario, figures_dir / "cost_breakdown_by_component.png",
            dpi=cfg.fig_dpi,
        )
        plot_cost_scaling_by_scenario(
            infra_result["scaling_df"], figures_dir / "cost_scaling_scenarios.png",
            dpi=cfg.fig_dpi,
        )
        plot_sensitivity_range(
            infra_result["sensitivity_df"], scenario, figures_dir / "sensitivity_range.png",
            dpi=cfg.fig_dpi,
        )
    except FileNotFoundError as e:
        logger.warning(f"Skipping infrastructure model: {e}")

    # -----------------------------------------------------------------
    # 3. QUALITATIVE INTERPRETATION: optional Heaps integration + combined summary
    # -----------------------------------------------------------------
    integration_result = run_integration(
        heaps_path=cfg.heaps_results_path,
        heaps_required_columns=cfg.heaps_required_columns,
        heaps_aliases=cfg.heaps_compatible_aliases,
        heaps_growth_multipliers=cfg.heaps_growth_multipliers,
        cost_scenarios_df=cost_result["scenarios_df"],
        infra_totals_df=infra_totals_df,
        scenario=scenario,
        output_dir=metrics_dir,
    )
    if integration_result["growth_projection_df"] is not None:
        integration_result["growth_projection_df"].to_csv(
            tables_dir / "vocabulary_growth_projection.csv", index=False
        )
        plot_vocabulary_growth_projection(
            integration_result["growth_projection_df"],
            figures_dir / "vocabulary_growth_projection.png",
            dpi=cfg.fig_dpi, figsize=cfg.fig_size,
        )

    print_banner(f"PIPELINE COMPLETE — outputs in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
