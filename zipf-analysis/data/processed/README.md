# data/processed/

## Expected file

```
data/processed/bengali_corpus.txt
```

This is the **final, cleaned, tokenization-ready Bengali corpus** produced by
the corpus-collection/preprocessing teammate.

## Format expectations

- Plain text file, UTF-8 encoded.
- Bengali Unicode text (or a mix, if the corpus is code-switched — the
  tokenizer in `src/corpus_stats.py` uses Unicode-aware whitespace
  tokenization by default and can be swapped for a more advanced tokenizer
  later without changing the rest of the pipeline).
- One document/sentence per line is recommended but not required; the
  pipeline treats the whole file as a token stream.
- No header row, no metadata — pure corpus text only.

## What happens once this file exists

Once `bengali_corpus.txt` is placed here, run:

```bash
python scripts/run_zipf.py
```

or the full pipeline:

```bash
python scripts/run_all.py --mode full
```

No code changes are required. Outputs will be written to `results/final/`.

## If the file is missing

The pipeline will **not crash**. It will print an explanatory message telling
you the expected path and suggesting `--mode demo` to test the software with
synthetic data instead.
