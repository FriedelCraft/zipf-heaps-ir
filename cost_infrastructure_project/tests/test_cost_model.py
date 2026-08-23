"""
test_cost_model.py

Unit tests for src/cost_model.py (assignment-based data cost formula and
LM training scenario formulas) and src/infrastructure_model.py (assumption
loading/validation, LOW/BASE/HIGH sensitivity, and completeness checking
for the Search_Oriented_System / LLM_Oriented_System architectural model).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cost_model import (
    data_cost, compute_cost_scenarios, lm_cost_from_scratch, lm_cost_adapt_existing,
)
from src.infrastructure_model import (
    validate_cost_assumptions, calculate_totals, _filter_real_values,
    _apply_sensitivity, build_component_breakdown,
)


# =============================================================================
# ASSIGNMENT-PROVIDED DATA COST BASELINE
# =============================================================================

def test_data_cost_100k_words_is_1000_usd():
    assert data_cost(100_000) == pytest.approx(1000.0)


def test_data_cost_1m_words_is_10000_usd():
    assert data_cost(1_000_000) == pytest.approx(10_000.0)


def test_data_cost_10m_words_is_100000_usd():
    assert data_cost(10_000_000) == pytest.approx(100_000.0)


def test_data_cost_zero_words_is_zero():
    assert data_cost(0) == 0.0


def test_data_cost_negative_words_raises():
    with pytest.raises(ValueError):
        data_cost(-100)


def test_data_cost_respects_custom_rate():
    # If the per-100k rate is changed to 500, 100k words should cost 500
    assert data_cost(100_000, usd_per_100k_words=500) == pytest.approx(500.0)


def test_compute_cost_scenarios_matches_assignment_expectation():
    scenarios = {"SMALL": 10_000_000, "MEDIUM": 100_000_000, "LARGE": 1_000_000_000}
    df = compute_cost_scenarios(scenarios, usd_per_100k_words=1000)

    small = df[df["scenario"] == "SMALL"]["data_cost_usd"].iloc[0]
    medium = df[df["scenario"] == "MEDIUM"]["data_cost_usd"].iloc[0]
    large = df[df["scenario"] == "LARGE"]["data_cost_usd"].iloc[0]

    assert small == pytest.approx(100_000.0)
    assert medium == pytest.approx(1_000_000.0)
    assert large == pytest.approx(10_000_000.0)
    assert (df["value_type"] == "assignment_assumption").all()


# =============================================================================
# LM TRAINING COST SCENARIOS
# =============================================================================

def test_lm_cost_from_scratch_formula():
    result = lm_cost_from_scratch(
        number_of_accelerators=10,
        accelerator_cost_per_hour=2.0,
        training_hours=100,
        storage_cost=500,
        networking_cost=100,
    )
    # compute_cost = 10 * 2.0 * 100 = 2000
    # total = 2000 + 500 + 100 = 2600
    assert result["compute_cost_usd"] == pytest.approx(2000.0)
    assert result["total_cost_usd"] == pytest.approx(2600.0)
    assert result["value_type"] == "modelling_assumption"


def test_lm_cost_adapt_existing_formula():
    result = lm_cost_adapt_existing(
        number_of_accelerators=5,
        accelerator_cost_per_hour=2.0,
        adaptation_hours=50,
        storage_cost=200,
    )
    # compute_cost = 5 * 2.0 * 50 = 500
    # total = 500 + 200 = 700
    assert result["compute_cost_usd"] == pytest.approx(500.0)
    assert result["total_cost_usd"] == pytest.approx(700.0)


# =============================================================================
# COST ASSUMPTIONS SYSTEM — new schema: system / cost_usd / assumption_type /
# confidence_or_uncertainty, with LOW/BASE/HIGH sensitivity and completeness
# checking against the 8 expected components (4 initial + 4 recurring).
# =============================================================================

def _sample_assumptions_df() -> pd.DataFrame:
    """
    A deliberately INCOMPLETE sample (missing 'storage') to test that
    completeness checking and total-exclusion still work under the new
    schema — mirrors the old placeholder tests but with real column names.
    """
    return pd.DataFrame([
        {"component": "data_acquisition", "system": "Search_Oriented_System", "scenario": "SMALL",
         "cost_type": "one_time", "cost_usd": 100000, "unit": "USD",
         "assumption_type": "assignment_input", "justification": "Assignment rule",
         "source": "Assignment brief", "confidence_or_uncertainty": 0},
        {"component": "preprocessing_setup", "system": "Search_Oriented_System", "scenario": "SMALL",
         "cost_type": "one_time", "cost_usd": 5000, "unit": "USD",
         "assumption_type": "modelling_assumption", "justification": "Illustrative setup cost",
         "source": "Scenario assumption", "confidence_or_uncertainty": 30},
        {"component": "indexing_or_training", "system": "Search_Oriented_System", "scenario": "SMALL",
         "cost_type": "one_time", "cost_usd": 2000, "unit": "USD",
         "assumption_type": "modelling_assumption", "justification": "Illustrative index cost",
         "source": "Scenario assumption", "confidence_or_uncertainty": 25},
        {"component": "engineering_setup", "system": "Search_Oriented_System", "scenario": "SMALL",
         "cost_type": "one_time", "cost_usd": 3000, "unit": "USD",
         "assumption_type": "modelling_assumption", "justification": "Illustrative eng cost",
         "source": "Scenario assumption", "confidence_or_uncertainty": 35},
        # 'storage' (recurring_annual) intentionally MISSING to test incompleteness
        {"component": "compute", "system": "Search_Oriented_System", "scenario": "SMALL",
         "cost_type": "recurring_annual", "cost_usd": None, "unit": "USD",
         "assumption_type": "modelling_assumption", "justification": "",
         "source": "TBD", "confidence_or_uncertainty": None},
        {"component": "serving_inference", "system": "Search_Oriented_System", "scenario": "SMALL",
         "cost_type": "recurring_annual", "cost_usd": 1000, "unit": "USD",
         "assumption_type": "modelling_assumption", "justification": "Illustrative serving cost",
         "source": "Scenario assumption", "confidence_or_uncertainty": 35},
        {"component": "maintenance_operations", "system": "Search_Oriented_System", "scenario": "SMALL",
         "cost_type": "recurring_annual", "cost_usd": 500, "unit": "USD",
         "assumption_type": "modelling_assumption", "justification": "Illustrative maintenance cost",
         "source": "Scenario assumption", "confidence_or_uncertainty": 30},
    ])


def _complete_sample_assumptions_df() -> pd.DataFrame:
    """Same as above but with the missing 'storage' row filled in — COMPLETE."""
    df = _sample_assumptions_df()
    storage_row = pd.DataFrame([{
        "component": "storage", "system": "Search_Oriented_System", "scenario": "SMALL",
        "cost_type": "recurring_annual", "cost_usd": 400, "unit": "USD",
        "assumption_type": "modelling_assumption", "justification": "Illustrative storage cost",
        "source": "Scenario assumption", "confidence_or_uncertainty": 25,
    }])
    df = df[df["cost_usd"].notna()]  # drop the missing 'compute' row too
    compute_row = pd.DataFrame([{
        "component": "compute", "system": "Search_Oriented_System", "scenario": "SMALL",
        "cost_type": "recurring_annual", "cost_usd": 600, "unit": "USD",
        "assumption_type": "modelling_assumption", "justification": "Illustrative compute cost",
        "source": "Scenario assumption", "confidence_or_uncertainty": 30,
    }])
    return pd.concat([df, storage_row, compute_row], ignore_index=True)


def test_filter_real_values_excludes_missing_cost():
    df = _sample_assumptions_df()
    real = _filter_real_values(df)
    assert "compute" not in set(real["component"])  # cost_usd was None
    assert len(real) == 6  # 7 rows minus the 1 missing-cost row


def test_calculate_totals_excludes_missing_and_flags_incomplete():
    df = _sample_assumptions_df()
    totals = calculate_totals(df, "Search_Oriented_System", "SMALL")
    # one_time total = 100000 + 5000 + 2000 + 3000 = 110000
    assert totals["initial_total_usd"] == pytest.approx(110000.0)
    # recurring total = 1000 (serving) + 500 (maintenance) = 1500
    # ('storage' missing entirely, 'compute' has no numeric cost_usd)
    assert totals["recurring_annual_total_usd"] == pytest.approx(1500.0)
    assert totals["is_complete"] is False
    assert "storage" in totals["missing_components"]
    assert "compute" in totals["missing_components"]


def test_calculate_totals_complete_when_all_8_components_present():
    df = _complete_sample_assumptions_df()
    totals = calculate_totals(df, "Search_Oriented_System", "SMALL")
    assert totals["is_complete"] is True
    assert totals["n_components_incomplete"] == 0
    assert totals["n_components_configured"] == 8


def test_validate_cost_assumptions_flags_missing_cost_and_missing_components():
    df = _sample_assumptions_df()
    warnings = validate_cost_assumptions(df)
    assert any("missing/non-numeric cost_usd" in w for w in warnings)
    assert any("missing RECURRING components" in w for w in warnings)


def test_validate_cost_assumptions_clean_on_complete_data():
    df = _complete_sample_assumptions_df()
    warnings = validate_cost_assumptions(df)
    # No missing-cost or missing-component warnings for Search_Oriented_System/SMALL
    assert not any("missing/non-numeric cost_usd" in w for w in warnings)
    assert not any("SMALL: missing" in w for w in warnings)


# =============================================================================
# SENSITIVITY ANALYSIS (LOW / BASE / HIGH)
# =============================================================================

def test_sensitivity_base_equals_raw_cost():
    df = _complete_sample_assumptions_df()
    adjusted = _apply_sensitivity(df, "BASE")
    assert (adjusted["adjusted_cost_usd"] == adjusted["cost_usd"]).all()


def test_sensitivity_low_is_below_base_for_modelling_assumptions():
    df = _complete_sample_assumptions_df()
    row = df[df["component"] == "preprocessing_setup"].iloc[0]
    low = _apply_sensitivity(df, "LOW")
    low_val = low[low["component"] == "preprocessing_setup"]["adjusted_cost_usd"].iloc[0]
    # confidence_or_uncertainty = 30 -> LOW = cost * (1 - 0.30)
    assert low_val == pytest.approx(row["cost_usd"] * 0.70)


def test_sensitivity_high_is_above_base_for_modelling_assumptions():
    df = _complete_sample_assumptions_df()
    row = df[df["component"] == "preprocessing_setup"].iloc[0]
    high = _apply_sensitivity(df, "HIGH")
    high_val = high[high["component"] == "preprocessing_setup"]["adjusted_cost_usd"].iloc[0]
    assert high_val == pytest.approx(row["cost_usd"] * 1.30)


def test_sensitivity_never_perturbs_assignment_input_rows():
    df = _complete_sample_assumptions_df()
    low = _apply_sensitivity(df, "LOW")
    high = _apply_sensitivity(df, "HIGH")
    dc_low = low[low["component"] == "data_acquisition"]["adjusted_cost_usd"].iloc[0]
    dc_high = high[high["component"] == "data_acquisition"]["adjusted_cost_usd"].iloc[0]
    dc_base = df[df["component"] == "data_acquisition"]["cost_usd"].iloc[0]
    # assignment_input rows are fixed facts — LOW/HIGH must equal BASE exactly
    assert dc_low == pytest.approx(dc_base)
    assert dc_high == pytest.approx(dc_base)


def test_sensitivity_low_less_than_base_less_than_high_for_totals():
    df = _complete_sample_assumptions_df()
    low = calculate_totals(df, "Search_Oriented_System", "SMALL", level="LOW")
    base = calculate_totals(df, "Search_Oriented_System", "SMALL", level="BASE")
    high = calculate_totals(df, "Search_Oriented_System", "SMALL", level="HIGH")
    assert low["recurring_annual_total_usd"] < base["recurring_annual_total_usd"] < high["recurring_annual_total_usd"]


# =============================================================================
# COMPONENT BREAKDOWN
# =============================================================================

def test_build_component_breakdown_includes_all_present_components():
    df = _complete_sample_assumptions_df()
    breakdown = build_component_breakdown(df, "SMALL", level="BASE")
    assert set(breakdown["component"]) == {
        "data_acquisition", "preprocessing_setup", "indexing_or_training",
        "engineering_setup", "storage", "compute", "serving_inference",
        "maintenance_operations",
    }
    # Only Search_Oriented_System was populated in this fixture
    dc_row = breakdown[breakdown["component"] == "data_acquisition"].iloc[0]
    assert dc_row["search_oriented_usd"] == pytest.approx(100000.0)
    assert dc_row["llm_oriented_usd"] is None
