#!/usr/bin/env python3
"""
generate_cost_assumptions.py

Deterministically (re)generates assumptions/cost_assumptions.csv.

This script exists purely for TRANSPARENCY AND REPRODUCIBILITY: every
scenario-assumption number in cost_assumptions.csv is derived here from an
explicit, documented formula or scaling rule rather than typed in by hand.
Running this script should reproduce the checked-in CSV byte-for-byte.

It is NOT invoked automatically by the main pipeline (scripts/run_cost_model.py,
scripts/run_integration.py) — those read assumptions/cost_assumptions.csv as
a static input, same as before. This script is a documentation/audit tool:
if you want to change a scaling rule, change it here, re-run, and diff.

IMPORTANT — WHAT THIS IS AND ISN'T:
    - Every non-assignment number below is a SCENARIO ASSUMPTION /
      MODELLING ASSUMPTION for a course project, not a researched or
      sourced real-world figure for Google, Sarvam, or any other company.
    - The two "systems" (Search_Oriented_System, LLM_Oriented_System) are
      architectural archetypes, not specific companies.
    - Where a number is directly derived from a formula already used
      elsewhere in this codebase (the assignment's data-cost rule, or the
      LM adaptation-cost formula in src/cost_model.py), that is noted
      explicitly via assumption_type='assignment_input' or
      assumption_type='formula_derived'. Everything else is
      assumption_type='modelling_assumption': a reasonable, explicitly
      documented scenario input chosen for this project, nothing more.

Usage:
    python scripts/generate_cost_assumptions.py
    python scripts/generate_cost_assumptions.py --check   # verify CSV matches, exit 1 if not
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "assumptions" / "cost_assumptions.csv"

COLUMNS = [
    "component", "system", "scenario", "cost_type", "cost_usd", "unit",
    "assumption_type", "justification", "source", "confidence_or_uncertainty",
]

# ---------------------------------------------------------------------------
# ASSIGNMENT-PROVIDED INPUT (fixed by the course brief, not a modelling choice)
# ---------------------------------------------------------------------------
USD_PER_100K_WORDS = 1000.0
SCENARIO_WORDS = {"SMALL": 10_000_000, "MEDIUM": 100_000_000, "LARGE": 1_000_000_000}


def data_cost(words: float) -> float:
    """Same formula as src/cost_model.py:data_cost() — the assignment rule."""
    return (words / 100_000.0) * USD_PER_100K_WORDS


# ---------------------------------------------------------------------------
# FORMULA-DERIVED: LLM adaptation cost (Approach B in src/cost_model.py),
# scaled per scenario. Accelerator count and adaptation hours are scaled
# illustratively with corpus size — this scaling rule itself is a modelling
# choice, but the cost FORMULA is the same one used elsewhere in this project.
# ---------------------------------------------------------------------------
LLM_ADAPT_PARAMS = {
    "SMALL":  {"accelerators": 8,  "cost_per_hour": 2.5, "hours": 120, "storage": 4_000},
    "MEDIUM": {"accelerators": 16, "cost_per_hour": 2.5, "hours": 240, "storage": 8_000},
    "LARGE":  {"accelerators": 32, "cost_per_hour": 2.5, "hours": 480, "storage": 16_000},
}


def llm_adaptation_cost(scenario: str) -> float:
    p = LLM_ADAPT_PARAMS[scenario]
    compute_cost = p["accelerators"] * p["cost_per_hour"] * p["hours"]
    return compute_cost + p["storage"]


# ---------------------------------------------------------------------------
# MODELLING ASSUMPTIONS: search-index build cost.
# Illustrative $/1,000-tokens indexing throughput assumption.
# ---------------------------------------------------------------------------
SEARCH_INDEX_USD_PER_1K_TOKENS = 0.50


def search_indexing_cost(words: float) -> float:
    return (words / 1_000.0) * SEARCH_INDEX_USD_PER_1K_TOKENS


# ---------------------------------------------------------------------------
# ROW BUILDER
# ---------------------------------------------------------------------------

def row(component, system, scenario, cost_type, cost_usd, assumption_type,
        justification, source, confidence_pct):
    return {
        "component": component,
        "system": system,
        "scenario": scenario,
        "cost_type": cost_type,
        "cost_usd": round(cost_usd, 2),
        "unit": "USD",
        "assumption_type": assumption_type,
        "justification": justification,
        "source": source,
        "confidence_or_uncertainty": confidence_pct,
    }


def build_rows() -> list[dict]:
    rows: list[dict] = []
    systems = ["Search_Oriented_System", "LLM_Oriented_System"]

    for scenario, words in SCENARIO_WORDS.items():

        # ---------------- INITIAL / ONE-TIME ----------------

        # 1. data_acquisition — ASSIGNMENT INPUT, identical for both systems
        #    (both need the same acquired/cleaned Bengali corpus).
        dc = data_cost(words)
        for system in systems:
            rows.append(row(
                "data_acquisition", system, scenario, "one_time", dc,
                "assignment_input",
                "Assignment-provided rule: USD 1,000 per 100,000 words. "
                "Applied identically to both architectures since both require "
                "the same acquired Bengali corpus.",
                "Assignment brief", 0,
            ))

        # 2. preprocessing_setup — modelling assumption: fixed pipeline setup
        #    + variable cleaning cost that scales sub-linearly with corpus size.
        #    LLM system costs slightly more due to tokenizer-training/model-
        #    specific data prep on top of shared cleaning steps.
        preprocessing_search = {"SMALL": 15_000, "MEDIUM": 35_000, "LARGE": 80_000}
        preprocessing_llm = {"SMALL": 18_000, "MEDIUM": 42_000, "LARGE": 95_000}
        rows.append(row(
            "preprocessing_setup", "Search_Oriented_System", scenario, "one_time",
            preprocessing_search[scenario], "modelling_assumption",
            "Illustrative one-time cost: fixed pipeline setup plus cleaning/"
            "normalization/tokenization effort that scales sub-linearly with "
            "corpus size due to automation economies of scale.",
            "Scenario assumption — not sourced from real vendor pricing", 30,
        ))
        rows.append(row(
            "preprocessing_setup", "LLM_Oriented_System", scenario, "one_time",
            preprocessing_llm[scenario], "modelling_assumption",
            "Same base cleaning pipeline as Search_Oriented_System plus "
            "additional tokenizer-training / model-specific data preparation "
            "effort, scaled sub-linearly with corpus size.",
            "Scenario assumption — not sourced from real vendor pricing", 30,
        ))

        # 3. indexing_or_training
        idx_cost = search_indexing_cost(words)
        rows.append(row(
            "indexing_or_training", "Search_Oriented_System", scenario, "one_time",
            idx_cost, "modelling_assumption",
            f"Illustrative inverted-index build cost of "
            f"${SEARCH_INDEX_USD_PER_1K_TOKENS:.2f} per 1,000 tokens processed, "
            f"an order-of-magnitude assumption for distributed indexing "
            f"throughput, not sourced from real vendor pricing.",
            "Scenario assumption — not sourced from real vendor pricing", 25,
        ))
        llm_cost = llm_adaptation_cost(scenario)
        p = LLM_ADAPT_PARAMS[scenario]
        rows.append(row(
            "indexing_or_training", "LLM_Oriented_System", scenario, "one_time",
            llm_cost, "formula_derived",
            f"Derived from the Approach B (adapt existing multilingual model) "
            f"formula in src/cost_model.py: accelerators({p['accelerators']}) x "
            f"cost_per_hour(${p['cost_per_hour']}) x adaptation_hours({p['hours']}) "
            f"+ one-time storage provisioning(${p['storage']:,}). Accelerator "
            f"count and adaptation hours are scaled illustratively with the "
            f"corpus-size scenario.",
            "Derived from src/cost_model.py Approach B formula", 20,
        ))

        # 4. engineering_setup — one-time integration + initial evaluation/
        #    benchmark construction + QA.
        eng_search = {"SMALL": 20_000, "MEDIUM": 45_000, "LARGE": 90_000}
        eng_llm = {"SMALL": 25_000, "MEDIUM": 55_000, "LARGE": 110_000}
        rows.append(row(
            "engineering_setup", "Search_Oriented_System", scenario, "one_time",
            eng_search[scenario], "modelling_assumption",
            "Illustrative one-time engineering integration cost covering system "
            "integration, initial evaluation/benchmark set construction, and QA, "
            "assumed to scale sub-linearly with system size.",
            "Scenario assumption — not sourced from real vendor pricing", 35,
        ))
        rows.append(row(
            "engineering_setup", "LLM_Oriented_System", scenario, "one_time",
            eng_llm[scenario], "modelling_assumption",
            "Illustrative one-time engineering integration cost covering model "
            "integration, initial evaluation/benchmark set construction, and QA, "
            "assumed to scale sub-linearly with system size. Slightly higher "
            "than Search_Oriented_System to reflect model-serving integration "
            "overhead.",
            "Scenario assumption — not sourced from real vendor pricing", 35,
        ))

        # ---------------- RECURRING / ANNUAL ----------------

        # 5. storage — annual storage of corpus + index/model artifacts.
        storage_search = {"SMALL": 2_000, "MEDIUM": 12_000, "LARGE": 90_000}
        storage_llm = {"SMALL": 1_500, "MEDIUM": 9_000, "LARGE": 70_000}
        rows.append(row(
            "storage", "Search_Oriented_System", scenario, "recurring_annual",
            storage_search[scenario], "modelling_assumption",
            "Illustrative annual storage cost for corpus plus inverted-index "
            "artifacts, assuming typical cloud object-storage pricing order of "
            "magnitude, scaled with corpus/index size.",
            "Scenario assumption — not sourced from real vendor pricing", 25,
        ))
        rows.append(row(
            "storage", "LLM_Oriented_System", scenario, "recurring_annual",
            storage_llm[scenario], "modelling_assumption",
            "Illustrative annual storage cost for corpus plus model checkpoint "
            "artifacts, assuming typical cloud object-storage pricing order of "
            "magnitude, scaled with corpus/model artifact size.",
            "Scenario assumption — not sourced from real vendor pricing", 25,
        ))

        # 6. compute — recurring background compute (periodic reindexing /
        #    incremental model refresh), distinct from query-time serving.
        compute_search = {"SMALL": 3_000, "MEDIUM": 20_000, "LARGE": 150_000}
        compute_llm = {"SMALL": 5_000, "MEDIUM": 30_000, "LARGE": 220_000}
        rows.append(row(
            "compute", "Search_Oriented_System", scenario, "recurring_annual",
            compute_search[scenario], "modelling_assumption",
            "Illustrative annual compute cost for periodic reindexing cycles as "
            "new data arrives, scaled with corpus size assuming reprocessing "
            "volume proportional to corpus growth.",
            "Scenario assumption — not sourced from real vendor pricing", 30,
        ))
        rows.append(row(
            "compute", "LLM_Oriented_System", scenario, "recurring_annual",
            compute_llm[scenario], "modelling_assumption",
            "Illustrative annual compute cost for incremental model refresh / "
            "periodic re-adaptation cycles, scaled with corpus size. Assumed "
            "higher than Search_Oriented_System's reindexing cost since model "
            "refresh is more compute-intensive per unit of new data.",
            "Scenario assumption — not sourced from real vendor pricing", 30,
        ))

        # 7. serving_inference — recurring query-time / generation-time serving.
        serving_search = {"SMALL": 4_000, "MEDIUM": 25_000, "LARGE": 180_000}
        serving_llm = {"SMALL": 6_000, "MEDIUM": 40_000, "LARGE": 300_000}
        rows.append(row(
            "serving_inference", "Search_Oriented_System", scenario, "recurring_annual",
            serving_search[scenario], "modelling_assumption",
            "Illustrative annual serving cost assuming a fixed estimated query "
            "volume that scales with corpus/system size; search retrieval "
            "assumed cheaper per-request than LLM generation.",
            "Scenario assumption — not sourced from real vendor pricing", 35,
        ))
        rows.append(row(
            "serving_inference", "LLM_Oriented_System", scenario, "recurring_annual",
            serving_llm[scenario], "modelling_assumption",
            "Illustrative annual inference-serving cost assuming a fixed "
            "estimated request volume that scales with corpus/system size; "
            "LLM generation assumed more compute-intensive per request than "
            "search retrieval.",
            "Scenario assumption — not sourced from real vendor pricing", 35,
        ))

        # 8. maintenance_operations — recurring evaluation, monitoring, ops.
        maint_search = {"SMALL": 5_000, "MEDIUM": 15_000, "LARGE": 40_000}
        maint_llm = {"SMALL": 6_000, "MEDIUM": 18_000, "LARGE": 48_000}
        rows.append(row(
            "maintenance_operations", "Search_Oriented_System", scenario, "recurring_annual",
            maint_search[scenario], "modelling_assumption",
            "Illustrative annual cost covering ongoing relevance evaluation, "
            "monitoring, and operational engineering support, assumed to scale "
            "sub-linearly with system size due to largely fixed headcount "
            "overhead.",
            "Scenario assumption — not sourced from real vendor pricing", 30,
        ))
        rows.append(row(
            "maintenance_operations", "LLM_Oriented_System", scenario, "recurring_annual",
            maint_llm[scenario], "modelling_assumption",
            "Illustrative annual cost covering ongoing output-quality "
            "evaluation, monitoring, and operational engineering support, "
            "assumed to scale sub-linearly with system size due to largely "
            "fixed headcount overhead.",
            "Scenario assumption — not sourced from real vendor pricing", 30,
        ))

    return rows


def write_csv(rows: list[dict], path: Path) -> str:
    lines = [",".join(COLUMNS)]
    for r in rows:
        vals = []
        for c in COLUMNS:
            v = r[c]
            if isinstance(v, str) and ("," in v or '"' in v):
                v = '"' + v.replace('"', '""') + '"'
            vals.append(str(v))
        lines.append(",".join(vals))
    content = "\n".join(lines) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                         help="Verify the checked-in CSV matches the generated output; exit 1 if not.")
    args = parser.parse_args()

    rows = build_rows()

    if args.check:
        if not OUTPUT_PATH.exists():
            print(f"MISMATCH: {OUTPUT_PATH} does not exist.")
            return 1
        existing = OUTPUT_PATH.read_text(encoding="utf-8")
        # Build to a temp string for comparison
        tmp_lines = [",".join(COLUMNS)]
        for r in rows:
            vals = []
            for c in COLUMNS:
                v = r[c]
                if isinstance(v, str) and ("," in v or '"' in v):
                    v = '"' + v.replace('"', '""') + '"'
                vals.append(str(v))
            tmp_lines.append(",".join(vals))
        generated = "\n".join(tmp_lines) + "\n"
        if existing == generated:
            print(f"OK: {OUTPUT_PATH} matches generator output ({len(rows)} rows).")
            return 0
        else:
            print(f"MISMATCH: {OUTPUT_PATH} does not match generator output.")
            return 1

    content = write_csv(rows, OUTPUT_PATH)
    print(f"Wrote {len(rows)} rows to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
