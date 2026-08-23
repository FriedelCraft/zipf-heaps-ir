
## Structure

```
/corpus_tools
/data
  /raw          -- raw corpus + corpus metadata
  /processed    -- cleaned corpus
  /samples      -- progressive samples
  /stats        -- raw vs processed data stats
README.md
requirements.txt
```

---

## Part 1 — Corpus Acquisition & Preprocessing

### Source

- **Dataset:** [AI4Bharat/IndicCorpV2](https://huggingface.co/datasets/ai4bharat/IndicCorpV2)
- **Split used:** `ben_Beng` (Bengali, Bengali script) — config name is fixed
  as `indiccorp_v2`; the language is selected via the `split` argument, not
  `data_dir` (confirmed directly from the dataset's README YAML).
- **Full split size (as published):** ~29,600,000 rows

### Sampling

- **Rows used:** 10,000,000 of ~29,600,000 (**~34% of the full split**),
  capped due to bandwidth/time constraints.
- **Method:** prefix sampling — streamed sequentially and stopped at the
  cap, rather than full reservoir sampling across the entire split. This is
  **not** a strictly uniform random sample; it relies on the underlying
  corpus not being source-sorted in a way that clusters similar text early
  (reasonable for web-scraped text, but should be stated as a limitation,
  not treated as equivalent to true random sampling).
- **Random seed:** N/A for prefix sampling (deterministic — same command
  reproduces the same subset, since HuggingFace streams rows in a fixed
  order).

### Preprocessing Pipeline

1. **Unicode normalization** — NFC, applied before anything else. Bengali
   conjuncts/vowel signs have multiple equivalent encodings; skipping this
   step artificially inflates the vocabulary count.
2. **Cleaning** — strip URLs, emails, digits (Bengali + ASCII), punctuation
   (Bengali danda/double-danda + standard punctuation), and any character
   outside the Bengali Unicode block (U+0980–U+09FF).
3. **Tokenization** — whitespace-based by default; falls back to this
   automatically unless `indic-nlp-library` is installed, in which case it
   uses `indic_tokenize.trivial_tokenize`. **TODO: confirm which one was
   actually active for this run** and state it explicitly in the
   methodology section, since it affects Zipf/Heaps exponents downstream.

### Raw vs. Processed Stats (10M-row sample)

| Metric | Raw | Processed | Change |
|---|---|---|---|
| Total tokens | 452,557,396 | 440,930,241 | −2.6% |
| Unique types | 7,848,604 | 3,012,824 | **−61.6%** |
| Type-token ratio | 0.0173 | 0.0068 | −60.7% |

Token count barely changed, but vocabulary size collapsed by ~62% —
cleaning consolidated spurious duplicate "words" (e.g. `বলেন,` vs `বলেন`)
rather than deleting real content. Full stats incl. top-20 frequency lists:
see `bn_raw_vs_processed_stats.json`.

### Output Files

| File | Contents |
|---|---|
| `bn_corpus_metadata_full.json` | Source, sampling method, row counts |
| `bn_raw_vs_processed_stats.json` | Full raw/processed comparison stats |
| `bn_sample_10000.txt` ... `bn_sample_1000000.txt` | Progressive samples (10K–1M tokens) |
| `bn_sample_5000000.txt`, `bn_sample_10000000.txt`, `bn_sample_20000000.txt` | Progressive samples (5M–20M tokens) |
| `bn_cleaned_corpus_full.txt` | Full cleaned corpus (~440M tokens) |
| `bn_indiccorp_full_raw.txt` | Raw (uncleaned) 10M-row sample |

**External hosting link:** 
Link : https://drive.google.com/drive/folders/1dOMzX_6oJ3vNYoIlNMe1QA_08RErWCxS?usp=drive_link

### Scripts

- `corpus_tools/download__corpus.py` — streams the split via
  `datasets`, resumable (skips already-downloaded rows on re-run), supports
  `--max_rows` cap.
- `corpus_tools/bengali_preprocessing.py` — memory-safe streaming
  preprocessing (never loads the full token list into memory); outputs
  cleaned corpus + progressive samples + raw/processed stats.

## Setup

```bash
pip install -r requirements.txt
```

To regenerate the corpus data from scratch (not required if you're pulling
the already-processed files from external storage):

```bash
python corpus_tools/download_corpus.py --max_rows 10000000
python corpus_tools/bengali_preprocessing.py --input bn_indiccorp_full_raw.txt
```
