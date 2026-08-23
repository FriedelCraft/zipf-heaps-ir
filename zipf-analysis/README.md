# Zipf Analysis Project (Standalone)

A standalone component investigating whether a **single power-law (Zipf)
model** adequately describes a Bengali (or any UTF-8 text) corpus.

> This project was split out from a combined Bengali IR project. It
> contains **only** the corpus-statistics and Zipf-analysis code. Cost /
> infrastructure / Heaps-integration code lives in a separate
> `cost_infrastructure_project` and is intentionally **not** included here.

---

## 1. What this does

Given a rank-frequency distribution:

```
f(r) = C * r^(-s)          =>      log(f(r)) = log(C) - s * log(r)
```

The pipeline:

1. Counts every token and ranks by descending frequency (deterministic
   alphabetical tie-breaking).
2. Log-transforms rank and frequency.
3. Fits **one** overall linear regression in log-log space → slope,
   exponent `s = -slope`, intercept, R², RMSE.
4. Computes **residuals** (observed − predicted log-frequency) to look for
   systematic deviation between the head and tail of the distribution.
5. Refits the model **independently within rank regions** (default: 1–100,
   101–1,000, 1,001–10,000, 10,001–100,000, 100,001+, auto-clipped to the
   actual vocabulary size) to see whether the exponent is stable or varies
   substantially across regions.
6. Computes **long-tail statistics**: % of vocabulary that are hapax
   legomena / occur ≤2/≤5/≤10 times, and token coverage contributed by the
   top-10/100/1,000/10,000 most frequent words.
7. Generates three figures: `zipf_loglog.png`, `zipf_residuals.png`,
   `zipf_piecewise_slopes.png`.

**Important:** the code never automatically concludes "Zipf's Law is
disproved." It only produces neutral, quantitative statements (see
`zipf_summary.txt`). The final academic conclusion is left to the report
author after inspecting the real results.

## 2. Repository structure

```
zipf_analysis_project/
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
│   ├── config.py           # config.yaml loader
│   ├── utils.py             # logging / path / message helpers
│   ├── corpus_stats.py       # token counting, TTR, hapax stats, top-K tokens
│   ├── zipf_analysis.py       # ranking, log-transform, fit, residuals, piecewise, long-tail
│   └── visualization.py        # the 3 Zipf figures (matplotlib only)
├── scripts/
│   └── run_zipf.py           # single entry point (demo/full modes)
├── results/
│   ├── final/                # real-corpus outputs (figures/tables/metrics)
│   └── demo/                  # synthetic DEMO outputs
└── tests/
    └── test_zipf.py
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
python scripts/run_zipf.py --mode demo
```

Outputs go to `results/demo/` and are clearly labelled `[DEMO]` in figure
titles. **Never treat these as real Bengali findings** — the demo corpus is
synthetic, generated only to exercise the pipeline's Unicode handling and
mechanics.

### Full mode (your real corpus)

1. Place your final, cleaned, UTF-8 Bengali corpus at:
   ```
   data/processed/bengali_corpus.txt
   ```
   See `data/processed/README.md` for format expectations.
2. Run:
   ```bash
   python scripts/run_zipf.py --mode full
   ```

No code changes are required once the corpus is in place. You can also
override paths directly:

```bash
python scripts/run_zipf.py --mode full --corpus path/to/corpus.txt --output results/final
```

## 6. Running tests

```bash
pytest tests/ -v
```

Tests cover ranking/tie-breaking, log transformation, fitting a known
synthetic Zipf distribution (recovering the true exponent), residual
analysis, piecewise region adaptation, and long-tail/coverage statistics.
All test data is synthetic and clearly used for software verification only.

## 7. Explanation of outputs

### `results/*/tables/`

| File | Contents |
|---|---|
| `corpus_statistics.csv` | Total tokens, vocabulary size, TTR, hapax count/% |
| `top_tokens.csv` | Top-K most frequent tokens with rank |
| `zipf_piecewise_metrics.csv` | Per-rank-region fit metrics |
| `zipf_long_tail_statistics.csv` | Hapax %, coverage by top-K, freq-threshold %s |

### `results/*/metrics/`

| File | Contents |
|---|---|
| `zipf_metrics.csv` | Overall single-Zipf fit: slope, exponent, intercept, R², RMSE |
| `zipf_residuals.csv` | Per-token residuals (observed − predicted log-frequency) |
| `zipf_summary.txt` | Neutral, structured plain-text summary of findings |

### `results/*/figures/`

| File | Contents |
|---|---|
| `zipf_loglog.png` | Observed rank-frequency (log-log) + fitted single Zipf line |
| `zipf_residuals.png` | Residuals vs log(rank) |
| `zipf_piecewise_slopes.png` | Estimated exponent per rank region, annotated with R² |

## 8. Large-corpus / memory usage

The corpus reader in `src/corpus_stats.py` is designed to handle
multi-gigabyte corpora on a machine with limited RAM. It reads the file in
fixed-size **byte** chunks (default 8 MB, configurable via
`config.yaml -> corpus_stats.chunk_size_mb`) rather than loading the whole
file or reading line-by-line — this matters because a line-based reader's
memory usage is bounded by the length of the *longest line* in the file,
which can be arbitrarily large (or the file may have very few newlines at
all). Fixed-size chunking bounds memory to the chunk size regardless of
how the corpus is laid out.

Two boundary-crossing problems are handled explicitly so results are
identical to reading the whole file at once:
- a multi-byte UTF-8 character (every Bengali character is 3 bytes) split
  across a chunk boundary is detected and the incomplete bytes are carried
  over to the next chunk before decoding;
- a whitespace-delimited word split across a chunk boundary is held back
  and reassembled with the start of the next chunk before being counted.

Verified on a 1.5 GB synthetic test file (including individual "lines" of
100+ MB with no newlines, deliberately mimicking a pathological real-world
corpus) with peak memory usage around 220 MB. See
`tests/test_corpus_stats.py` for the correctness test suite (chunk sizes
from 1 byte up to 1 MB, Bengali text with boundaries forced through every
byte offset of a multi-byte character, and an integration test confirming
the resulting Counter feeds the existing Zipf pipeline unchanged).

If you need to tune this for your machine, adjust `chunk_size_mb` in
`config.yaml`; larger chunks trade a bit more memory for fewer I/O calls
and slightly faster processing, smaller chunks use less memory.

## 9. Limitations

- The tokenizer is a simple whitespace tokenizer; if a more
  linguistically-informed Bengali tokenizer is applied upstream, this
  codebase just counts whatever tokens are already separated by whitespace
  in the processed file — no code changes needed here, but tokenization
  choices will affect vocabulary size, TTR, and the estimated Zipf exponent.
- `data/demo/demo_bengali_corpus.txt` is synthetic data for software
  verification only and must never be cited as a linguistic finding about
  Bengali.
- Piecewise rank-region boundaries are configurable but are fixed
  boundaries, not automatically detected changepoints.
- This project does not compute cost/infrastructure estimates or Heaps'
  Law statistics — see the separate `cost_infrastructure_project` for those.
