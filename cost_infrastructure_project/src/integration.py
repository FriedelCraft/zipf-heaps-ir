"""
integration.py

Connects the pieces owned by this project into a coherent summary that
explicitly separates three different kinds of claim:

    FACT / ASSIGNMENT INPUT     — corpus size scenarios, the assignment's
                                    data-acquisition cost rule. Not modelled,
                                    not uncertain: given by the brief.
    SCENARIO ASSUMPTION          — infrastructure cost estimates for the two
                                    architectural archetypes (Search-Oriented,
                                    LLM-Oriented), each with a stated LOW/
                                    BASE/HIGH range. Transparent, reproducible,
                                    NOT real company financial data.
    QUALITATIVE INTERPRETATION    — how Heaps vocabulary-growth characteristics
                                    relate to the general case for continued
                                    data acquisition. Never turned into a
                                    number; the Heaps exponent is NEVER used
                                    as a multiplier on any cost figure.

NOTE ON SCOPE: in the original combined project, this module also folded in
Zipf's Law results (head/tail vocabulary structure) to build a three-part
Zipf + Heaps + Cost narrative. Since Zipf analysis is now a separate,
independently-runnable project (`zipf_analysis_project`), that section has
been removed here rather than kept as a dangling dependency. If you want the
full three-part narrative, run both projects and combine their summary
outputs manually, or re-integrate the two summaries in a downstream report.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import get_logger, print_missing_heaps_message

logger = get_logger(__name__)


# =============================================================================
# HEAPS RESULTS LOADER
# =============================================================================

def _try_alias_rename(df: pd.DataFrame, required_columns: list[str],
                       aliases: dict[str, list[str]]) -> pd.DataFrame:
    """
    Rename columns in df to canonical required_columns names using the
    configured alias map, matching case-insensitively.
    """
    df = df.copy()
    lower_map = {c.lower(): c for c in df.columns}

    rename_dict = {}
    for canonical in required_columns:
        if canonical in df.columns:
            continue  # already canonical
        if canonical.lower() in lower_map:
            rename_dict[lower_map[canonical.lower()]] = canonical
            continue
        for alias in aliases.get(canonical, []):
            if alias.lower() in lower_map:
                rename_dict[lower_map[alias.lower()]] = canonical
                break

    return df.rename(columns=rename_dict)


def load_heaps_results(
    path: Path,
    required_columns: list[str],
    aliases: dict[str, list[str]],
) -> pd.DataFrame | None:
    """
    Load Heaps results from CSV if present, auto-detecting compatible
    column-name variations. Returns None (and prints the standard message)
    if the file does not exist. Does not raise on missing file — this is
    an expected, valid state for the pipeline.
    """
    path = Path(path)
    if not path.exists():
        print_missing_heaps_message()
        return None

    df = pd.read_csv(path)
    df = _try_alias_rename(df, required_columns, aliases)

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        logger.warning(
            f"Heaps results file found but missing expected columns: {missing}. "
            f"Proceeding with whatever columns are available; downstream "
            f"calculations that need missing columns will be skipped."
        )

    numeric_cols = [c for c in ["token_count", "vocabulary_size", "K", "beta",
                                 "r_squared", "rmse"] if c in df.columns]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


def validate_heaps_dataframe(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """Return a list of warning strings about the Heaps dataframe's validity."""
    warnings: list[str] = []
    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        warnings.append(f"Missing columns in Heaps results: {missing}")

    for c in [c for c in ["beta", "K"] if c in df.columns]:
        n_nan = int(df[c].isna().sum())
        if n_nan > 0:
            warnings.append(f"{n_nan} row(s) have non-numeric/missing '{c}' values")

    for w in warnings:
        logger.warning(w)
    return warnings


# =============================================================================
# VOCABULARY GROWTH PROJECTION
# =============================================================================

def vocabulary_growth_factor(multiplier: float, beta: float) -> float:
    """
    Heaps' Law growth factor for a given corpus-size multiplier.

        growth_factor = multiplier ^ beta

    This describes how much the *vocabulary* is expected to grow (as a
    multiplicative factor) if the corpus size grows by `multiplier`x,
    under the Heaps' Law assumption V = K * N^beta.
    """
    if beta < 0 or beta > 1:
        logger.warning(
            f"beta={beta} is outside the typical Heaps' Law range (0, 1); "
            f"proceeding anyway since this may be a valid edge case."
        )
    return float(multiplier ** beta)


def project_vocabulary_growth(
    current_vocabulary_size: float,
    beta: float,
    multipliers: list[float],
) -> pd.DataFrame:
    """
    Project vocabulary size at various corpus-size multipliers using the
    Heaps' Law growth factor. Also reports the marginal (diminishing)
    growth between successive multipliers as an indicator of long-tail /
    language-coverage implications.
    """
    rows = []
    prev_vocab = current_vocabulary_size
    for m in multipliers:
        factor = vocabulary_growth_factor(m, beta)
        projected = current_vocabulary_size * factor
        marginal_growth = projected - prev_vocab
        rows.append(
            {
                "multiplier": m,
                "growth_factor": round(factor, 6),
                "projected_vocabulary_size": round(projected, 2),
                "marginal_growth_vs_previous": round(marginal_growth, 2),
            }
        )
        prev_vocab = projected
    return pd.DataFrame(rows)


# =============================================================================
# INTEGRATED SUMMARY (HEAPS + COST — Zipf lives in the separate project)
# =============================================================================

# Components deliberately NOT modelled anywhere in this project. Rather than
# leaving these as placeholder rows in cost_assumptions.csv, they were
# excluded entirely because no defensible scenario assumption could be made
# for a course project — see infrastructure_model.py module docstring.
EXCLUDED_COMPONENTS_RATIONALE = [
    ("Marketing / user acquisition", "Highly company- and market-specific; no "
     "defensible scaling rule exists relative to corpus size alone."),
    ("Legal / regulatory compliance", "Depends on jurisdiction-specific "
     "requirements (data protection, content regulation) unrelated to corpus "
     "or model size; not something this scenario model can estimate."),
    ("Customer support operations", "Depends on user base size, not corpus or "
     "model size — no reasonable scaling assumption available here."),
]


def build_integration_summary(
    cost_scenarios_df: pd.DataFrame | None,
    infra_totals_df: pd.DataFrame | None,
    scenario: str,
    heaps_df: pd.DataFrame | None,
    growth_projection_df: pd.DataFrame | None,
) -> list[str]:
    """
    Build a structured narrative summary that explicitly separates FACT /
    ASSIGNMENT INPUT, SCENARIO ASSUMPTION, and QUALITATIVE INTERPRETATION.

    Never draws causal claims that Heaps/Zipf exponents directly determine
    cost — the Heaps beta value is never multiplied into any cost figure.
    """
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("INTEGRATED SUMMARY — Cost / Infrastructure Component")
    lines.append("=" * 78)
    lines.append("")
    lines.append(
        "NOTE: Zipf's Law analysis is a separate, independently-runnable "
        "component (zipf_analysis_project) and is not included in this "
        "summary. Combine its zipf_summary.txt output manually if you need "
        "the full three-part narrative."
    )

    # =========================================================================
    # SECTION 1 — FACT / ASSIGNMENT INPUT
    # =========================================================================
    lines.append("")
    lines.append("=" * 78)
    lines.append("SECTION 1 — FACT / ASSIGNMENT INPUT")
    lines.append("(given by the course brief; not modelled, not uncertain)")
    lines.append("=" * 78)
    if cost_scenarios_df is not None and not cost_scenarios_df.empty:
        for _, row in cost_scenarios_df.iterrows():
            lines.append(
                f"  {row['scenario']}: {row['corpus_size_words']:,} words -> "
                f"${row['data_cost_usd']:,.2f} (assignment rule: "
                f"USD 1,000 per 100,000 words)"
            )
    else:
        lines.append("  Cost scenario results not available.")

    # =========================================================================
    # SECTION 2 — SCENARIO ASSUMPTION
    # =========================================================================
    lines.append("")
    lines.append("=" * 78)
    lines.append("SECTION 2 — SCENARIO ASSUMPTION")
    lines.append(
        "(transparent, reproducible architectural cost models; NOT real "
        f"company financial data — see assumptions/cost_assumptions.csv)"
    )
    lines.append("=" * 78)
    lines.append(f"  Scenario shown: {scenario}")
    if infra_totals_df is not None and not infra_totals_df.empty:
        for _, row in infra_totals_df.iterrows():
            completeness = "COMPLETE" if row.get("is_complete") else \
                f"INCOMPLETE (missing: {row.get('missing_components')})"
            lines.append(
                f"  {row['system']} [{row['sensitivity_level']}]: "
                f"initial=${row['initial_total_usd']:,.2f}, "
                f"recurring_annual=${row['recurring_annual_total_usd']:,.2f} "
                f"[{completeness}]"
            )
        lines.append(
            "  See cost_sensitivity_analysis.csv for LOW/BASE/HIGH ranges per "
            "system, and cost_breakdown_by_component.csv for the per-component "
            "detail behind these totals."
        )
    else:
        lines.append("  Infrastructure totals not available.")

    lines.append("")
    lines.append("  Components intentionally NOT modelled (excluded, not fabricated):")
    for name, reason in EXCLUDED_COMPONENTS_RATIONALE:
        lines.append(f"    - {name}: {reason}")

    # =========================================================================
    # SECTION 3 — QUALITATIVE INTERPRETATION
    # =========================================================================
    lines.append("")
    lines.append("=" * 78)
    lines.append("SECTION 3 — QUALITATIVE INTERPRETATION")
    lines.append(
        "(narrative only; Heaps parameters are NEVER multiplied into any "
        "cost figure above)"
    )
    lines.append("=" * 78)
    lines.append("")
    lines.append("[HEAPS] Vocabulary growth as corpus size increases:")
    if heaps_df is not None and not heaps_df.empty:
        row = heaps_df.iloc[0]
        beta = row.get("beta", None)
        k_const = row.get("K", None)
        r2 = row.get("r_squared", None)
        lines.append(
            f"  Heaps' Law parameters (from teammate's results): "
            f"beta={beta}, K={k_const}, R^2={r2}."
        )
        if growth_projection_df is not None and not growth_projection_df.empty:
            lines.append("  Vocabulary growth projections (Heaps' Law, multiplier^beta):")
            for _, gr in growth_projection_df.iterrows():
                lines.append(
                    f"    {gr['multiplier']}x corpus size -> "
                    f"projected vocabulary ~{gr['projected_vocabulary_size']:,.0f} "
                    f"(growth factor {gr['growth_factor']:.4f})"
                )
    else:
        lines.append("  Heaps results not found. Skipping Heaps integration.")

    lines.append("")
    lines.append(
        "  Continued Heaps vocabulary growth (if observed) suggests that "
        "supporting a language may require continued data acquisition to "
        "improve vocabulary coverage. This is a qualitative implication "
        "only — it is not converted into a cost multiplier anywhere in "
        "this project. The SCENARIO ASSUMPTION section above scales with "
        "corpus-size SCENARIO (SMALL/MEDIUM/LARGE) alone, independent of "
        "any Zipf or Heaps statistic."
    )
    lines.append("=" * 78)

    return lines


def run_integration(
    heaps_path: Path,
    heaps_required_columns: list[str],
    heaps_aliases: dict[str, list[str]],
    heaps_growth_multipliers: list[float],
    cost_scenarios_df: pd.DataFrame | None,
    infra_totals_df: pd.DataFrame | None,
    scenario: str,
    output_dir: Path,
) -> dict:
    """Full integration pipeline, saving vocabulary growth CSV + summary text."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    heaps_df = load_heaps_results(heaps_path, heaps_required_columns, heaps_aliases)
    growth_projection_df = None

    if heaps_df is not None and not heaps_df.empty:
        validate_heaps_dataframe(heaps_df, heaps_required_columns)
        if "beta" in heaps_df.columns and "vocabulary_size" in heaps_df.columns:
            row = heaps_df.iloc[0]
            beta = row.get("beta")
            current_vocab = row.get("vocabulary_size")
            if pd.notna(beta) and pd.notna(current_vocab):
                growth_projection_df = project_vocabulary_growth(
                    float(current_vocab), float(beta), heaps_growth_multipliers
                )
                growth_projection_df.to_csv(
                    output_dir / "vocabulary_growth_projection.csv", index=False
                )
            else:
                logger.warning(
                    "Heaps results present but 'beta' or 'vocabulary_size' is "
                    "missing/non-numeric; skipping vocabulary growth projection."
                )
        else:
            logger.warning(
                "Heaps results present but lacks 'beta' and/or 'vocabulary_size' "
                "columns; skipping vocabulary growth projection."
            )

    summary_lines = build_integration_summary(
        cost_scenarios_df, infra_totals_df, scenario, heaps_df, growth_projection_df
    )
    with open(output_dir / "integration_summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines))

    for line in summary_lines:
        print(line)

    return {
        "heaps_df": heaps_df,
        "growth_projection_df": growth_projection_df,
        "summary_lines": summary_lines,
    }
