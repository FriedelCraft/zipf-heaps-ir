# Heaps' Law Analysis Project (Standalone)

A standalone component measuring **vocabulary growth** as corpus size
increases, testing:

```
V(N) = K * N^beta          =>      log(V) = log(K) + beta * log(N)
```

where `N` is the number of tokens processed and `V(N)` is the number of
distinct tokens (vocabulary size) observed so far.

> This project was split out from the combined Bengali IR project as a
> **fully independent** third project, alongside `zipf_analysis_project`
> and `cost_infrastructure_project`. It does not share any code, state,
> or corpus-streaming pass with either of them — see "Relationship to the
> other projects" below for the tradeoff this implies.

---

## 1. What this does

1. Streams the corpus **exactly once**, in fixed-size byte chunks,
   maintaining a plain `set()` of every distinct token seen so far.
2. Records a checkpoint `{token_count, vocabulary_size}` at
   logarithmically-spaced token-count thresholds (see "Checkpoint
   schedule" below) — generated automatically, not hardcoded, and always
   ending with the true final corpus totals.
3. Fits `V(N) = K * N^beta` via log-log ordinary least squares — the same
   methodology (`scipy.stats.linregress` + sklearn R²/RMSE) used by the
   companion `zipf_analysis_project`'s Zipf fit, for consistency.
4. Computes residuals of the log-log fit.
5. Generates three figures and a structured, academically cautious
   summary.

**Important:** results are explicitly split into an EMPIRICAL RESULT
section (the observed checkpoints — measurements, not modelling) and a
FITTED MODEL VALUES section (the regression line — an approximation) in
`heaps_summary.txt`. The exponent `beta` is **never** used as a
multiplier or input into any cost calculation anywhere in this project.

## 2. Repository structure

```
heaps_analysis_project/
├── README.md
├── requirements.txt
├── .gitignore
├── config.yaml
├── data/
│   ├── raw/            # unprocessed source text (not read by pipeline)
│   ├── processed/       # <-- put final bengali_corpus.txt here
│   └── demo/             # synthetic DEMO corpus (bundled)
├── src/
│   ├── __init__.py
│   ├── config.py             # config.yaml loader
│   ├── utils.py                # logging / path / message helpers
│   ├── corpus_streaming.py      # memory-safe streaming: unique-token set + checkpoints
│   ├── heaps_analysis.py         # checkpoint scheduling, fit, residuals, summary
│   └── visualization.py           # the 3 Heaps figures (matplotlib only)
├── scripts/
│   └── run_heaps.py           # single entry point (demo/full modes)
├── results/
│   ├── final/                # real-corpus outputs (figures/tables/metrics)
│   └── demo/                  # synthetic DEMO outputs
└── tests/
    └── test_heaps_analysis.py
```

## 3. Installation

Requires **Python 3.10+**.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 4. Dependencies

`numpy`, `pandas`, `matplotlib`, `scipy` (for the log-log linear
regression), `scikit-learn` (for R²/RMSE), `PyYAML`, `pytest`.

## 5. How to run

### Demo mode (bundled synthetic corpus — verifies the software works)

```bash
python scripts/run_heaps.py --mode demo
```

Outputs go to `results/demo/`, clearly labelled `[DEMO]` in figure titles.
The bundled demo corpus only has 60,000 tokens, so its checkpoints all
fall below the default `min_tokens_for_fit` (100,000) — the growth CSV
and vocabulary-growth figure are still produced, but the regression fit
correctly reports "insufficient checkpoints" rather than fitting on noisy
low-N data. This is expected behavior, not a bug — see `config.yaml` if
you want to lower the threshold to see a fit on the demo corpus.

### Full mode (your real corpus)

1. Place your final, cleaned, UTF-8 Bengali corpus at:
   ```
   data/processed/bengali_corpus.txt
   ```
2. Run:
   ```bash
   python scripts/run_heaps.py --mode full
   ```

No code changes are required once the corpus is in place. You can also
override paths directly:

```bash
python scripts/run_heaps.py --mode full --corpus path/to/corpus.txt --output results/final
```

## 6. Running tests

```bash
pytest tests/ -v
```

30 tests covering: checkpoint schedule generation, UTF-8 boundary safety,
hand-verifiable vocabulary growth on small synthetic corpora, token
boundaries split across chunks (both English and Bengali, at every chunk
size from 1 byte up), multiple checkpoints landing inside a single chunk,
regression fit recovery on synthetic power-law data, residuals, the
structured summary, and full `stream -> fit -> summary` integration.

## 7. Checkpoint schedule

```yaml
heaps:
  first_checkpoint_tokens: 10000
  checkpoints_per_decade: 4     # checkpoint density; total count scales with corpus size
  max_checkpoints: 200          # safety cap
  min_tokens_for_fit: 100000    # checkpoints below this are excluded from the regression
  min_observations_for_fit: 5
```

Checkpoints are **logarithmically spaced and generated on the fly** — the
code never needs to know the total corpus size in advance (which would
require an expensive pre-pass just to count tokens). Starting at
`first_checkpoint_tokens`, each checkpoint is `10**(1/checkpoints_per_decade)`
times the previous one, continuing automatically until the corpus ends.
The *density* of checkpoints (how many per order of magnitude) is fixed
and configurable, while the *total number* scales naturally with however
large the corpus turns out to be — e.g. the default settings produce ~5
checkpoints on a 60,000-token corpus and ~19 on a 440-million-token one.

## 8. Memory safety

The corpus reader in `src/corpus_streaming.py` reads the file in
fixed-size **byte** chunks (default 8 MB, configurable via
`config.yaml -> corpus_streaming.chunk_size_mb`) rather than loading the
whole file or reading line-by-line — a line-based reader's memory usage
is bounded by the length of the *longest line* in the file, which can be
arbitrarily large (or the file may have very few newlines at all).
Fixed-size chunking bounds memory to the chunk size regardless of how the
corpus is laid out.

Three boundary-crossing problems are handled explicitly:
- a multi-byte UTF-8 character (every Bengali character is 3 bytes) split
  across a chunk boundary is detected and the incomplete bytes are
  carried over to the next chunk before decoding;
- a whitespace-delimited word split across a chunk boundary is held back
  and reassembled with the start of the next chunk before being counted;
- an 8 MB chunk can contain hundreds of thousands of tokens and cross
  several checkpoints at once — the chunk's token list is sliced at each
  exact checkpoint position, not just recorded once per chunk.

Unlike `zipf_analysis_project`, this project maintains a plain `set()` of
unique tokens rather than a `collections.Counter` — Heaps' Law only needs
to know whether a token has been seen before, not its frequency, so a set
is the more direct and slightly more memory-/time-efficient choice for a
project with no other use for per-token counts.

Verified on a 15 million-token synthetic corpus with a genuinely large
(500,000-word) open vocabulary (needed for a *meaningful* Heaps test —
a small fixed vocabulary saturates almost immediately and doesn't
exercise real sublinear growth): peak memory ~360 MB, real fit recovered
`beta=0.571, K=47.17, R²=0.979`. This synthetic corpus is not shipped
with the repository and was deleted after verification — see the
project's PR/conversation history for the exact commands used to
regenerate it if you want to reproduce this test yourself.

## 9. Relationship to the other projects

This project is **fully independent** — no shared code, no shared
process, no shared streaming pass with `zipf_analysis_project` or
`cost_infrastructure_project`.

**The tradeoff:** if you want both Zipf and Heaps results, running this
project and `zipf_analysis_project` separately means the corpus gets read
**twice** in total (once per project), rather than sharing a single pass.
For a 7 GB corpus this roughly doubles total I/O time across both runs
compared to `zipf_analysis_project`'s integrated mode (which can produce
both results from one streaming pass, at the cost of the two analyses
no longer being independently deployable). Pick whichever tradeoff suits
your workflow — clean independence (this project + zipf_analysis_project
run separately) vs. shared efficiency (zipf_analysis_project's combined
mode).

**Feeding results to `cost_infrastructure_project`:** that project can
load a `heaps_results.csv` from `data/external/` to build vocabulary
growth projections in its integration summary. This project's
`results/*/tables/heaps_results_for_cost_project.csv` is written in
exactly that expected schema (`corpus_version, token_count,
vocabulary_size, K, beta, r_squared, rmse`) — copy it to
`cost_infrastructure_project/data/external/heaps_results.csv` if you want
this project's results picked up there.

## 10. Statistical interpretation guidance

`heaps_summary.txt` is structured into three parts:

- **EMPIRICAL RESULT** — the observed checkpoints. Direct measurements,
  not modelling.
- **FITTED MODEL VALUES** — `K`, `beta`, `R²`, `RMSE` from the log-log
  regression. An approximation, explicitly labelled as such — never
  presented as an exact description of the data.
- **QUALITATIVE IMPLICATION** — if `beta < 1`, vocabulary growth is
  sublinear (the typical Heaps' Law pattern: new words appear at a
  diminishing but non-zero rate). This is stated as a narrative
  observation about the corpus, not a quantitative prediction of future
  vocabulary size.

`beta` is never used as a multiplier or input into any cost or
infrastructure calculation in this project or (if you use the bridge
file above) in `cost_infrastructure_project` — any connection between
vocabulary growth and cost is qualitative narrative only.

## 11. Limitations

- The tokenizer is a simple whitespace tokenizer; tokenization choices
  upstream will affect vocabulary size and the estimated `beta`.
- `data/demo/demo_bengali_corpus.txt` is synthetic data for software
  verification only and must never be cited as a linguistic finding
  about Bengali.
- The fitted `beta` describes vocabulary growth *within the observed
  corpus range* — extrapolating it far beyond the largest observed
  checkpoint is a much stronger claim than this analysis supports.
- This project does not compute Zipf's Law statistics or cost/
  infrastructure estimates — see the separate `zipf_analysis_project`
  and `cost_infrastructure_project` for those.
