"""
infrastructure_model.py

Loads and validates the cost-assumptions system (assumptions/cost_assumptions.csv)
and builds a SCENARIO-BASED ARCHITECTURAL cost comparison between two neutral
system archetypes:

    Search_Oriented_System   — crawl/index/serve architecture
    LLM_Oriented_System        — train-or-adapt/fine-tune/serve architecture

These are NOT claims about Google's or Sarvam's actual internal costs.
They are transparent, reproducible scenario models: every row in
assumptions/cost_assumptions.csv carries an explicit assumption_type,
justification, source, and confidence_or_uncertainty so a reader can see
exactly what is fact (assignment input), formula-derived, or a modelling
assumption — and how much any given number could plausibly vary.

Responsibilities:
    - Load assumptions
    - Validate them (reject/flag incomplete rows and unexpected values)
    - Calculate INITIAL (one_time) and RECURRING ANNUAL totals per system
      x scenario, at a chosen sensitivity level (LOW / BASE / HIGH)
    - Build a component-level breakdown comparing the two systems
    - Build a SMALL/MEDIUM/LARGE cost-scaling table
    - Build a LOW/BASE/HIGH sensitivity-range table
    - Never silently treat a missing cost_usd as zero or as a real cost

Components modelled identically for BOTH systems (so the comparison is
apples-to-apples across architectures):

    INITIAL / ONE-TIME:
        data_acquisition, preprocessing_setup, indexing_or_training,
        engineering_setup

    RECURRING ANNUAL:
        storage, compute, serving_inference, maintenance_operations

Components deliberately NOT modelled (see README "Limitations" and
integration_summary.txt for the explicit rationale): things like
marketing, legal/regulatory compliance, and customer support were
considered but excluded rather than represented as a placeholder or a
fabricated number, because no defensible scenario assumption could be
made for a course project.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

REQUIRED_COLUMNS = [
    "component", "system", "scenario", "cost_type", "cost_usd", "unit",
    "assumption_type", "justification", "source", "confidence_or_uncertainty",
]

VALID_ASSUMPTION_TYPES = {
    "assignment_input", "formula_derived", "modelling_assumption", "public_source",
}
VALID_COST_TYPES = {"one_time", "recurring_annual"}
VALID_SYSTEMS = {"Search_Oriented_System", "LLM_Oriented_System"}
VALID_SCENARIOS = {"SMALL", "MEDIUM", "LARGE"}
VALID_SENSITIVITY_LEVELS = ("LOW", "BASE", "HIGH")

# The 8 components every (system, scenario) combination is expected to have,
# split by cost_type. Used to detect genuinely incomplete scenarios (as
# opposed to components intentionally excluded from the schema entirely).
EXPECTED_INITIAL_COMPONENTS = {
    "data_acquisition", "preprocessing_setup", "indexing_or_training", "engineering_setup",
}
EXPECTED_RECURRING_COMPONENTS = {
    "storage", "compute", "serving_inference", "maintenance_operations",
}


def load_cost_assumptions(path: Path) -> pd.DataFrame:
    """
    Load assumptions/cost_assumptions.csv.

    Raises FileNotFoundError if missing, ValueError if required columns
    are absent.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cost assumptions file not found: {path}")

    df = pd.read_csv(path)
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"cost_assumptions.csv is missing required columns: {missing_cols}"
        )
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce")
    df["confidence_or_uncertainty"] = pd.to_numeric(
        df["confidence_or_uncertainty"], errors="coerce"
    )
    return df


def validate_cost_assumptions(df: pd.DataFrame) -> list[str]:
    """
    Validate the assumptions dataframe. Returns a list of human-readable
    warning strings. Structural/vocabulary problems are warned about (not
    fatal, so a partially-edited CSV can still be inspected), but missing
    numeric values are flagged clearly and will be excluded from totals —
    never silently treated as zero or as a real cost.
    """
    warnings: list[str] = []

    bad_systems = set(df["system"].dropna().unique()) - VALID_SYSTEMS
    if bad_systems:
        warnings.append(f"Unexpected system values found: {bad_systems}")

    bad_scenarios = set(df["scenario"].dropna().unique()) - VALID_SCENARIOS
    if bad_scenarios:
        warnings.append(f"Unexpected scenario values found: {bad_scenarios}")

    bad_cost_types = set(df["cost_type"].dropna().unique()) - VALID_COST_TYPES
    if bad_cost_types:
        warnings.append(f"Unexpected cost_type values found: {bad_cost_types}")

    bad_assumption_types = set(df["assumption_type"].dropna().unique()) - VALID_ASSUMPTION_TYPES
    if bad_assumption_types:
        warnings.append(f"Unexpected assumption_type values found: {bad_assumption_types}")

    n_missing_cost = int(df["cost_usd"].isna().sum())
    if n_missing_cost > 0:
        warnings.append(
            f"{n_missing_cost} row(s) have a missing/non-numeric cost_usd — "
            f"these will be EXCLUDED from totals, and the affected "
            f"(system, scenario) combination will be flagged INCOMPLETE."
        )

    n_missing_justification = int(
        df["justification"].isna().sum() + (df["justification"] == "").sum()
    )
    if n_missing_justification > 0:
        warnings.append(
            f"{n_missing_justification} row(s) have a missing 'justification' — "
            f"numbers without a stated rationale are not academically defensible; "
            f"fill these in before using in a final report."
        )

    # Check every (system, scenario) has the full expected component set
    for system in sorted(set(df["system"].dropna().unique()) & VALID_SYSTEMS):
        for scenario in sorted(set(df["scenario"].dropna().unique()) & VALID_SCENARIOS):
            subset = df[(df["system"] == system) & (df["scenario"] == scenario)]
            initial_present = set(subset[subset["cost_type"] == "one_time"]["component"])
            recurring_present = set(subset[subset["cost_type"] == "recurring_annual"]["component"])
            missing_initial = EXPECTED_INITIAL_COMPONENTS - initial_present
            missing_recurring = EXPECTED_RECURRING_COMPONENTS - recurring_present
            if missing_initial:
                warnings.append(
                    f"{system} / {scenario}: missing INITIAL components {sorted(missing_initial)}"
                )
            if missing_recurring:
                warnings.append(
                    f"{system} / {scenario}: missing RECURRING components {sorted(missing_recurring)}"
                )

    for w in warnings:
        logger.warning(w)

    return warnings


def _filter_real_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only rows that have a non-null numeric cost_usd. This is the
    core guard against silently treating a missing/incomplete row as a
    real cost (equivalent to zero) or dropping it without a trace.
    """
    real = df[df["cost_usd"].notna()].copy()
    return real


def _apply_sensitivity(
    df: pd.DataFrame, level: str, default_uncertainty_pct: float = 25.0
) -> pd.DataFrame:
    """
    Return a copy of df with an 'adjusted_cost_usd' column applying the
    requested sensitivity level:

        LOW  = cost_usd * (1 - uncertainty_pct / 100)
        BASE = cost_usd
        HIGH = cost_usd * (1 + uncertainty_pct / 100)

    uncertainty_pct comes from each row's 'confidence_or_uncertainty'
    column; if missing/non-numeric, `default_uncertainty_pct` is used.
    assignment_input rows (assumption_type == 'assignment_input') are
    never perturbed — they are a fixed rule from the assignment brief,
    not a scenario assumption with uncertainty.
    """
    if level not in VALID_SENSITIVITY_LEVELS:
        raise ValueError(f"level must be one of {VALID_SENSITIVITY_LEVELS}, got {level!r}")

    df = df.copy()
    pct = df["confidence_or_uncertainty"].fillna(default_uncertainty_pct)
    is_fixed = df["assumption_type"] == "assignment_input"

    if level == "BASE":
        df["adjusted_cost_usd"] = df["cost_usd"]
    elif level == "LOW":
        multiplier = 1.0 - (pct / 100.0)
        df["adjusted_cost_usd"] = df["cost_usd"] * multiplier.where(~is_fixed, 1.0)
    else:  # HIGH
        multiplier = 1.0 + (pct / 100.0)
        df["adjusted_cost_usd"] = df["cost_usd"] * multiplier.where(~is_fixed, 1.0)

    return df


def calculate_totals(
    df: pd.DataFrame,
    system: str,
    scenario: str,
    level: str = "BASE",
    default_uncertainty_pct: float = 25.0,
) -> dict:
    """
    Calculate INITIAL (one_time) and RECURRING ANNUAL totals for a given
    system + scenario, at a chosen sensitivity level, using ONLY rows
    with a valid numeric cost_usd.
    """
    subset = df[(df["system"] == system) & (df["scenario"] == scenario)]
    real = _filter_real_values(subset)
    real = _apply_sensitivity(real, level, default_uncertainty_pct)

    initial_total = float(real[real["cost_type"] == "one_time"]["adjusted_cost_usd"].sum())
    recurring_total = float(real[real["cost_type"] == "recurring_annual"]["adjusted_cost_usd"].sum())

    initial_present = set(real[real["cost_type"] == "one_time"]["component"])
    recurring_present = set(real[real["cost_type"] == "recurring_annual"]["component"])
    missing_initial = EXPECTED_INITIAL_COMPONENTS - initial_present
    missing_recurring = EXPECTED_RECURRING_COMPONENTS - recurring_present
    n_incomplete = len(missing_initial) + len(missing_recurring)

    return {
        "system": system,
        "scenario": scenario,
        "sensitivity_level": level,
        "initial_total_usd": initial_total,
        "recurring_annual_total_usd": recurring_total,
        "n_components_configured": len(real),
        "n_components_expected": len(EXPECTED_INITIAL_COMPONENTS) + len(EXPECTED_RECURRING_COMPONENTS),
        "missing_components": sorted(missing_initial | missing_recurring),
        "n_components_incomplete": n_incomplete,
        "is_complete": n_incomplete == 0,
    }


def build_totals_table(
    df: pd.DataFrame, scenario: str, level: str = "BASE",
    default_uncertainty_pct: float = 25.0,
) -> pd.DataFrame:
    """Initial + recurring totals for both systems at one scenario/level."""
    rows = [
        calculate_totals(df, system, scenario, level, default_uncertainty_pct)
        for system in sorted(VALID_SYSTEMS)
    ]
    return pd.DataFrame(rows)


def build_component_breakdown(
    df: pd.DataFrame, scenario: str, level: str = "BASE",
    default_uncertainty_pct: float = 25.0,
) -> pd.DataFrame:
    """
    Per-component comparison of the two systems at a given scenario/level.

    Columns: component, cost_type, search_oriented_usd, llm_oriented_usd,
    assumption_type, confidence_or_uncertainty_pct.
    """
    subset = df[df["scenario"] == scenario]
    real = _filter_real_values(subset)
    real = _apply_sensitivity(real, level, default_uncertainty_pct)

    components = sorted(
        set(real["component"]),
        key=lambda c: (c not in EXPECTED_INITIAL_COMPONENTS, c),
    )

    rows = []
    for component in components:
        comp_rows = real[real["component"] == component]
        cost_type = comp_rows.iloc[0]["cost_type"] if not comp_rows.empty else None

        def extract(system: str):
            r = comp_rows[comp_rows["system"] == system]
            if r.empty:
                return None, None, None
            r0 = r.iloc[0]
            return (
                float(r0["adjusted_cost_usd"]),
                r0["assumption_type"],
                r0["confidence_or_uncertainty"],
            )

        search_val, search_type, search_conf = extract("Search_Oriented_System")
        llm_val, llm_type, llm_conf = extract("LLM_Oriented_System")

        rows.append(
            {
                "component": component,
                "cost_type": cost_type,
                "search_oriented_usd": search_val,
                "llm_oriented_usd": llm_val,
                "search_assumption_type": search_type,
                "llm_assumption_type": llm_type,
                "search_confidence_pct": search_conf,
                "llm_confidence_pct": llm_conf,
            }
        )

    return pd.DataFrame(rows)


def build_scaling_table(
    df: pd.DataFrame, level: str = "BASE", default_uncertainty_pct: float = 25.0,
) -> pd.DataFrame:
    """
    Initial + recurring totals for both systems ACROSS all three scenarios
    (SMALL/MEDIUM/LARGE), for the SMALL->LARGE cost-scaling figure.
    """
    rows = []
    for scenario in ["SMALL", "MEDIUM", "LARGE"]:
        for system in sorted(VALID_SYSTEMS):
            rows.append(calculate_totals(df, system, scenario, level, default_uncertainty_pct))
    return pd.DataFrame(rows)


def build_sensitivity_table(
    df: pd.DataFrame, system: str, scenario: str, default_uncertainty_pct: float = 25.0,
) -> pd.DataFrame:
    """
    LOW / BASE / HIGH initial + recurring totals for one system/scenario,
    for the sensitivity-range figure. Makes explicit that these are
    scenario assumptions with a stated uncertainty range, not point-precise
    figures.
    """
    rows = []
    for level in VALID_SENSITIVITY_LEVELS:
        rows.append(calculate_totals(df, system, scenario, level, default_uncertainty_pct))
    return pd.DataFrame(rows)


def run_infrastructure_model(
    assumptions_path: Path,
    output_dir: Path,
    scenario: str,
    default_uncertainty_pct: float = 25.0,
) -> dict:
    """
    Full infrastructure-model pipeline:
        1. Load + validate assumptions
        2. Calculate BASE totals for both systems at the given scenario
        3. Build the component-level breakdown table
        4. Build the SMALL/MEDIUM/LARGE scaling table
        5. Build LOW/BASE/HIGH sensitivity tables for both systems
        6. Save all CSV outputs
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_cost_assumptions(assumptions_path)
    warnings = validate_cost_assumptions(df)

    totals_df = build_totals_table(df, scenario, level="BASE",
                                    default_uncertainty_pct=default_uncertainty_pct)
    totals_df.insert(0, "value_type_label", "SCENARIO-BASED ARCHITECTURAL ESTIMATE")
    totals_df.to_csv(output_dir / "infrastructure_totals.csv", index=False)

    breakdown_df = build_component_breakdown(df, scenario, level="BASE",
                                              default_uncertainty_pct=default_uncertainty_pct)
    breakdown_df.to_csv(output_dir / "cost_breakdown_by_component.csv", index=False)

    scaling_df = build_scaling_table(df, level="BASE",
                                      default_uncertainty_pct=default_uncertainty_pct)
    scaling_df.to_csv(output_dir / "cost_scaling_by_scenario.csv", index=False)

    sensitivity_rows = []
    for system in sorted(VALID_SYSTEMS):
        sens_df = build_sensitivity_table(df, system, scenario, default_uncertainty_pct)
        sensitivity_rows.append(sens_df)
    sensitivity_df = pd.concat(sensitivity_rows, ignore_index=True)
    sensitivity_df.to_csv(output_dir / "cost_sensitivity_analysis.csv", index=False)

    logger.info(f"SCENARIO-BASED ARCHITECTURAL ESTIMATE for scenario={scenario} (BASE level):")
    for _, row_ in totals_df.iterrows():
        completeness = "COMPLETE" if row_["is_complete"] else \
            f"INCOMPLETE (missing: {row_['missing_components']})"
        logger.info(
            f"  {row_['system']}: initial=${row_['initial_total_usd']:,.2f}, "
            f"recurring_annual=${row_['recurring_annual_total_usd']:,.2f} [{completeness}]"
        )

    return {
        "assumptions_df": df,
        "validation_warnings": warnings,
        "totals_df": totals_df,
        "breakdown_df": breakdown_df,
        "scaling_df": scaling_df,
        "sensitivity_df": sensitivity_df,
    }
