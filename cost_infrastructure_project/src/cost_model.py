"""
cost_model.py

Implements:
    1. ASSIGNMENT-PROVIDED DATA COST BASELINE
       data_cost(words) = words / 100,000 * usd_per_100k_words
       Applied to SMALL / MEDIUM / LARGE corpus scenarios.

    2. LANGUAGE MODEL COST SCENARIOS
       Approach A — TRAIN FROM SCRATCH
       Approach B — ADAPT EXISTING MULTILINGUAL MODEL

All values here are either:
    - directly derived from the assignment's stated assumption (data cost), or
    - explicit, clearly-labelled modelling assumptions (LM training scenarios)

Nothing in this module claims to represent Google's or Sarvam's actual
internal costs.

NOTE ON RELATIONSHIP TO assumptions/cost_assumptions.csv: the Approach B
(ADAPT_EXISTING_MODEL) formula below is the SAME formula used to derive the
LLM_Oriented_System 'indexing_or_training' row values in
assumptions/cost_assumptions.csv (see scripts/generate_cost_assumptions.py
for the scenario-scaled parameters actually used). This function is kept
here as a standalone, directly-runnable comparison of the two LM training
strategies at the config.yaml default parameters; it does not feed the
infrastructure totals at runtime — the infrastructure model reads its
numbers from the CSV so there is a single, auditable source of truth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)


# =============================================================================
# ASSIGNMENT-PROVIDED DATA COST BASELINE
# =============================================================================

def data_cost(words: float, usd_per_100k_words: float = 1000.0) -> float:
    """
    ASSIGNMENT-PROVIDED DATA COST BASELINE.

        data_cost(words) = words / 100,000 * usd_per_100k_words

    This is a direct application of the assignment's stated assumption,
    not an independently derived or researched figure.
    """
    if words < 0:
        raise ValueError("words must be non-negative")
    return (words / 100_000.0) * usd_per_100k_words


def compute_cost_scenarios(
    scenarios: dict[str, int], usd_per_100k_words: float = 1000.0
) -> pd.DataFrame:
    """
    Compute the assignment-based data cost for each corpus size scenario.

    Expected (given the assignment's $1,000 / 100,000-words rule):
        10,000,000 words   -> $100,000
        100,000,000 words  -> $1,000,000
        1,000,000,000 words -> $10,000,000
    """
    rows = []
    for name, words in scenarios.items():
        cost = data_cost(words, usd_per_100k_words)
        rows.append(
            {
                "scenario": name,
                "corpus_size_words": words,
                "data_cost_usd": cost,
                "value_type": "assignment_assumption",
                "source_or_assumption": "Assignment: USD 1,000 per 100,000 words",
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# LANGUAGE MODEL COST SCENARIOS
# =============================================================================

def lm_cost_from_scratch(
    number_of_accelerators: float,
    accelerator_cost_per_hour: float,
    training_hours: float,
    storage_cost: float,
    networking_cost: float,
) -> dict:
    """
    APPROACH A — TRAIN FROM SCRATCH.

        compute_cost = number_of_accelerators * accelerator_cost_per_hour * training_hours
        total_training_cost = compute_cost + storage_cost + networking_cost

    This is a MODELLING ASSUMPTION scenario, not a claim about any
    organization's actual training run.
    """
    compute_cost = number_of_accelerators * accelerator_cost_per_hour * training_hours
    total_cost = compute_cost + storage_cost + networking_cost
    return {
        "approach": "FROM_SCRATCH",
        "compute_cost_usd": compute_cost,
        "storage_cost_usd": storage_cost,
        "networking_cost_usd": networking_cost,
        "total_cost_usd": total_cost,
        "value_type": "modelling_assumption",
    }


def lm_cost_adapt_existing(
    number_of_accelerators: float,
    accelerator_cost_per_hour: float,
    adaptation_hours: float,
    storage_cost: float,
) -> dict:
    """
    APPROACH B — ADAPT EXISTING MULTILINGUAL MODEL.

        adaptation_cost = number_of_accelerators * accelerator_cost_per_hour
                           * adaptation_hours + storage_cost

    This is a MODELLING ASSUMPTION scenario, not a claim about any
    organization's actual adaptation strategy.
    """
    compute_cost = number_of_accelerators * accelerator_cost_per_hour * adaptation_hours
    total_cost = compute_cost + storage_cost
    return {
        "approach": "ADAPT_EXISTING_MODEL",
        "compute_cost_usd": compute_cost,
        "storage_cost_usd": storage_cost,
        "networking_cost_usd": 0.0,
        "total_cost_usd": total_cost,
        "value_type": "modelling_assumption",
    }


def compute_lm_training_comparison(
    from_scratch_params: dict, adapt_existing_params: dict
) -> pd.DataFrame:
    """
    Compare FROM_SCRATCH vs ADAPT_EXISTING_MODEL scenarios side by side.

    NOTE: these are scenario analyses for the course project, not claims
    about strategies actually used by Sarvam or any other organization.
    """
    scratch = lm_cost_from_scratch(**from_scratch_params)
    adapt = lm_cost_adapt_existing(**adapt_existing_params)
    df = pd.DataFrame([scratch, adapt])
    return df


# =============================================================================
# ORCHESTRATION
# =============================================================================

def run_cost_model(
    scenarios: dict[str, int],
    usd_per_100k_words: float,
    lm_from_scratch_params: dict,
    lm_adapt_existing_params: dict,
    output_dir: Path,
) -> dict:
    """Run the full cost-model pipeline and save CSV + return dataframes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios_df = compute_cost_scenarios(scenarios, usd_per_100k_words)
    scenarios_df.to_csv(output_dir / "cost_scenarios.csv", index=False)

    lm_comparison_df = compute_lm_training_comparison(
        lm_from_scratch_params, lm_adapt_existing_params
    )
    lm_comparison_df.to_csv(output_dir / "lm_training_comparison.csv", index=False)

    logger.info("Cost scenarios (ASSIGNMENT-PROVIDED DATA COST BASELINE):")
    for _, row in scenarios_df.iterrows():
        logger.info(
            f"  {row['scenario']}: {row['corpus_size_words']:,} words "
            f"-> ${row['data_cost_usd']:,.2f}"
        )

    return {"scenarios_df": scenarios_df, "lm_comparison_df": lm_comparison_df}
