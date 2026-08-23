# Cost / Infrastructure / Integration Project (Standalone)

A standalone component estimating the infrastructure and investment cost
required to support Bengali under two neutral architectural archetypes —
a **search-oriented system** and an **LLM-oriented system** — and
integrating a teammate's Heaps' Law results into vocabulary-growth
projections.

> This project was split out from a combined Bengali IR project. It
> contains **only** the cost-model, infrastructure-model, and
> Heaps-integration code. Zipf's Law analysis / corpus statistics live in a
> separate `zipf_analysis_project` and are intentionally **not** included
> here.

**This is not an estimate of Google's or Sarvam's actual costs.** Every
non-assignment number is a transparent, reproducible scenario assumption,
explicitly labelled as such, with a stated justification and uncertainty
range. See "What this is and isn't" below before using any figure from
this project in a report.

---

## What this is and isn't

| | |
|---|---|
| ✅ A transparent, reproducible **scenario cost model** for two architectural archetypes | ❌ NOT Google's, Sarvam's, or any other company's real internal costs |
| ✅ Every number traces to a documented formula or explicit assumption | ❌ NOT independently researched/sourced figures (unless marked `public_source`) |
| ✅ Explicit LOW/BASE/HIGH ranges showing how much conclusions depend on assumptions | ❌ NOT a claim of precision — a ±30% "modelling assumption" is not a measurement |
| ✅ The assignment's data-acquisition rule, applied exactly as given | ❌ NOT modified, reinterpreted, or blended with other assumptions |

## 1. What this does

### 1.1 FACT / ASSIGNMENT INPUT — data cost baseline

```
data_cost(words) = (words / 100,000) x 1,000   [USD]
```

Applied to three corpus-size scenarios:

| Scenario | Words | Cost |
|----------|------:|---------------:|
| SMALL  | 10,000,000    | $100,000     |
| MEDIUM | 100,000,000   | $1,000,000   |
| LARGE  | 1,000,000,000 | $10,000,000  |

This is a **direct, unmodified application of the assignment's stated
rule** — not an estimate, not adjustable, never perturbed in the
sensitivity analysis.

### 1.2 SCENARIO ASSUMPTION — architectural cost comparison

Two neutral system archetypes, each with the same 8 cost components so the
comparison is apples-to-apples:

**Initial / one-time:** `data_acquisition`, `preprocessing_setup`,
`indexing_or_training`, `engineering_setup`

**Recurring annual:** `storage`, `compute`, `serving_inference`,
`maintenance_operations`

- **`Search_Oriented_System`** — crawl/index/serve architecture.
  `indexing_or_training` = illustrative inverted-index build cost.
- **`LLM_Oriented_System`** — train-or-adapt/fine-tune/serve architecture.
  `indexing_or_training` = model adaptation cost, derived from the same
  accelerator-hours formula used in `src/cost_model.py` (Approach B).

Every row in `assumptions/cost_assumptions.csv` carries a `justification`,
`source`, and `confidence_or_uncertainty` (a ± percentage). Nothing is a
blank placeholder — see `assumptions/README.md` for the full schema.

### 1.3 Sensitivity analysis (LOW / BASE / HIGH)

Because these are scenario assumptions, not measurements, every total is
also computed at LOW and HIGH bounds using each row's stated uncertainty
percentage:

```
LOW  = cost_usd x (1 - uncertainty% / 100)
HIGH = cost_usd x (1 + uncertainty% / 100)
```

`assignment_input` rows (the data cost baseline) are never perturbed — they
are a fixed fact, not an assumption with a confidence interval.

### 1.4 Heaps' Law integration — QUALITATIVE INTERPRETATION only

If a teammate's `data/external/heaps_results.csv` is available, this
project loads it (auto-detecting common alternate column names), projects
vocabulary growth at 2x/5x/10x/100x corpus size (`multiplier ^ beta`), and
folds it into the integration summary as **narrative context only**. The
Heaps exponent is **never** used as a multiplier on any cost figure — see
`src/integration.py`. If the file is absent, the pipeline prints `"Heaps
results not found. Skipping Heaps integration."` and continues — **it will
not crash.**

## 2. Repository structure

```
cost_infrastructure_project/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── assumptions/
│   ├── cost_assumptions.csv     # single source of truth, 48 fully-populated rows
│   └── README.md                  # full schema documentation
├── templates/
│   └── cost_assumptions_template.csv
├── data/
│   └── external/                  # <-- put heaps_results.csv here
├── src/
│   ├── __init__.py
│   ├── config.py                 # config.yaml loader (incl. sensitivity defaults)
│   ├── utils.py                    # logging / message helpers
│   ├── cost_model.py                 # assignment data-cost formula + LM training scenarios
│   ├── infrastructure_model.py         # load/validate/sensitivity/breakdown/scaling
│   ├── integration.py                    # Heaps loader, FACT/SCENARIO/QUALITATIVE summary
│   └── visualization.py                    # 8 figures (matplotlib only)
├── scripts/
│   ├── generate_cost_assumptions.py    # reproducibly (re)generates the assumptions CSV
│   ├── run_cost_model.py                 # cost + infrastructure only (no Heaps)
│   └── run_integration.py                  # FULL pipeline: cost + infra + Heaps integration
├── results/
│   └── final/                          # figures/tables/metrics
└── tests/
    └── test_cost_model.py
```

## 3. Installation

Requires **Python 3.10+**.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Dependencies

`numpy`, `pandas`, `matplotlib`, `PyYAML`, `pytest`.

## 5. How to run

### Full pipeline (recommended — cost + infrastructure + sensitivity + Heaps)

```bash
python scripts/run_integration.py
```

Outputs go to `results/final/`. Works immediately — `assumptions/cost_assumptions.csv`
ships fully populated, so every total is `COMPLETE` out of the box. If
`data/external/heaps_results.csv` isn't present yet, Heaps integration is
skipped gracefully.

### Cost + infrastructure only (no Heaps)

```bash
python scripts/run_cost_model.py
```

### Choosing a corpus-size scenario for the headline comparison

```bash
python scripts/run_integration.py --scenario LARGE
```

### Regenerating the assumptions CSV (after changing a scaling rule)

```bash
python scripts/generate_cost_assumptions.py          # regenerate
python scripts/generate_cost_assumptions.py --check   # verify it's up to date
```

## 6. Running tests

```bash
pytest tests/ -v
```

20 tests covering:
- the assignment cost formula (100k words -> $1,000, 1M words -> $10,000,
  10M words -> $100,000) — unchanged from the original design
- the LM training/adaptation cost formulas
- that missing `cost_usd` values are excluded from totals and correctly
  flagged incomplete (never silently treated as zero)
- LOW/BASE/HIGH sensitivity bounds compute correctly and
  `assignment_input` rows are never perturbed
- the component-breakdown table

## 7. Explanation of every output

### Tables (`results/final/tables/`)

| File | Contents |
|---|---|
| `cost_scenarios.csv` | FACT: assignment-based data cost per corpus scenario |
| `lm_training_comparison.csv` | FROM_SCRATCH vs ADAPT_EXISTING_MODEL cost comparison |
| `infrastructure_totals.csv` | BASE initial + recurring totals, both systems, one scenario |
| `cost_breakdown_by_component.csv` | Per-component detail, both systems, one scenario |
| `cost_scaling_by_scenario.csv` | Totals across SMALL/MEDIUM/LARGE, both systems |
| `cost_sensitivity_analysis.csv` | LOW/BASE/HIGH totals, both systems, one scenario |
| `vocabulary_growth_projection.csv` | Heaps-based projected vocabulary at 2x/5x/10x/100x (only if Heaps results present) |

### Metrics (`results/final/metrics/`)

| File | Contents |
|---|---|
| `integration_summary.txt` | FACT / SCENARIO ASSUMPTION / QUALITATIVE INTERPRETATION summary |

### Figures (`results/final/figures/`)

| File | Contents |
|---|---|
| `cost_vs_corpus_size.png` | Assignment-based data cost by corpus scenario |
| `initial_cost_comparison.png` | (A) Initial cost, Search-Oriented vs LLM-Oriented |
| `recurring_cost_comparison.png` | (B) Recurring annual cost, Search-Oriented vs LLM-Oriented |
| `cost_breakdown_by_component.png` | (C) All 8 components, both systems, log scale |
| `cost_scaling_scenarios.png` | (D) Initial + recurring cost across SMALL/MEDIUM/LARGE |
| `sensitivity_range.png` | (E) LOW/BASE/HIGH range, both systems, error-bar style |
| `vocabulary_growth_projection.png` | Projected vocabulary vs corpus multiplier (if Heaps present) |

## 8. How to configure cost assumptions

Two ways to change a number:

1. **Edit `scripts/generate_cost_assumptions.py`** (recommended) — every
   scaling rule lives there with an inline explanation; change it and
   re-run `python scripts/generate_cost_assumptions.py`.
2. **Edit `assumptions/cost_assumptions.csv` directly** for quick
   experiments — see `assumptions/README.md` for the full column
   reference, the `assumption_type` categories, and how
   `confidence_or_uncertainty` feeds the sensitivity analysis.

## 9. How to add Heaps results

Ask the teammate responsible for Heaps' Law to place their results at:

```
data/external/heaps_results.csv
```

Preferred columns: `corpus_version, token_count, vocabulary_size, K, beta,
r_squared, rmse`. Common alternate column names (e.g. `heaps_beta`, `b`,
`vocab_size`, `r2`) are auto-detected (see `config.yaml -> heaps`).

## 10. Interpreting the results — FACT vs SCENARIO vs QUALITATIVE

`integration_summary.txt` is structured into three sections, and every
number in this project's output falls into exactly one of them:

- **SECTION 1 — FACT / ASSIGNMENT INPUT**: the corpus-size scenarios and
  the data-acquisition cost baseline. Given directly by the course brief.
  Not modelled, not uncertain, never perturbed.
- **SECTION 2 — SCENARIO ASSUMPTION**: the Search-Oriented vs LLM-Oriented
  architectural cost totals, with LOW/BASE/HIGH ranges. Transparent and
  reproducible, but explicitly **not** real company data.
- **SECTION 3 — QUALITATIVE INTERPRETATION**: how Heaps vocabulary growth
  (if available) relates to the general case for continued data
  acquisition. Narrative only — the Heaps beta is never converted into a
  cost multiplier anywhere in this project.

When writing a report from this project's output, keep these three
categories separate rather than presenting Section 2's numbers with the
same confidence as Section 1's.

## 11. Limitations

- **The infrastructure and cost estimates are scenario-based architectural
  models built under explicit, labelled assumptions. They are NOT Google's,
  Sarvam's, or any other company's actual internal costs**, and should
  never be presented as such in the final report.
- The Heaps exponent (`beta`) is used only as a qualitative indicator of
  vocabulary growth in the integration narrative — it is **not** used as a
  direct multiplier on any cost figure.
- Three components — marketing/user acquisition, legal/regulatory
  compliance, and customer support operations — were considered and
  **deliberately excluded** from the schema rather than represented as
  placeholders or fabricated numbers, because no defensible corpus-size
  scaling assumption could be made for them. See `assumptions/README.md`.
- `confidence_or_uncertainty` ranges are themselves modelling choices
  (typically 20-35%), not statistically derived confidence intervals — they
  represent "how much this particular assumption could plausibly be off
  by," not a rigorous error propagation.
- This project does not compute Zipf's Law statistics or corpus-level
  statistics — see the separate `zipf_analysis_project` for those. Its
  `zipf_summary.txt` output can be combined manually with this project's
  `integration_summary.txt` for a full three-part narrative.
