#!/usr/bin/env python3
"""
run_cost_model.py

Runs the assignment-based data cost model, LM training scenario
comparison, and the Search-Oriented vs LLM-Oriented architectural cost
model (without Heaps integration — see run_integration.py for the full
pipeline including Heaps vocabulary-growth projections and the combined
FACT / SCENARIO ASSUMPTION / QUALITATIVE INTERPRETATION summary).

Generates:
    tables/cost_scenarios.csv                  FACT: assignment data-cost baseline
    tables/lm_training_comparison.csv            FROM_SCRATCH vs ADAPT_EXISTING_MODEL
    tables/infrastructure_totals.csv               BASE initial+recurring totals, both systems
    tables/cost_breakdown_by_component.csv           per-component detail
    tables/cost_scaling_by_scenario.csv                totals across SMALL/MEDIUM/LARGE
    tables/cost_sensitivity_analysis.csv                 LOW/BASE/HIGH totals, both systems

    figures/cost_vs_corpus_size.png
    figures/initial_cost_comparison.png
    figures/recurring_cost_comparison.png
    figures/cost_breakdown_by_component.png
    figures/cost_scaling_scenarios.png
    figures/sensitivity_range.png

Usage:
    python scripts/run_cost_model.py
    python scripts/run_cost_model.py --scenario LARGE --output results/final
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
from src.visualization import (
    plot_cost_vs_corpus_size,
    plot_initial_cost_comparison,
    plot_recurring_cost_comparison,
    plot_cost_breakdown_by_component,
    plot_cost_scaling_by_scenario,
    plot_sensitivity_range,
)
from src.utils import get_logger, print_banner

logger = get_logger("run_cost_model")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the cost + infrastructure model.")
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
    figures_dir = output_dir / "figures"

    print_banner(f"RUNNING COST / INFRASTRUCTURE MODEL — scenario={scenario}")

    # ---------------------------------------------------------------
    # FACT: assignment-based data cost baseline + LM approach comparison
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # SCENARIO ASSUMPTION: Search-Oriented vs LLM-Oriented architecture model
    # ---------------------------------------------------------------
    try:
        infra_result = run_infrastructure_model(
            assumptions_path=cfg.cost_assumptions_path,
            output_dir=tables_dir,
            scenario=scenario,
            default_uncertainty_pct=cfg.default_uncertainty_pct,
        )
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(
            "Cost assumptions file missing. Data-cost-only results were still "
            "generated; add assumptions/cost_assumptions.csv (see templates/) "
            "to enable the infrastructure comparison."
        )
        return 1

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

    logger.info(f"Cost model complete. Outputs written under: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
