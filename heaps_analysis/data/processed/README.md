# data/processed/

## Expected file

```
data/processed/bengali_corpus.txt
```

This is the **final, cleaned, tokenization-ready Bengali corpus**.

## Format expectations

- Plain text file, UTF-8 encoded.
- Bengali Unicode text (or a mix, if the corpus is code-switched — the
  streaming reader in `src/corpus_streaming.py` uses Unicode-aware
  whitespace tokenization and handles both English and Bengali text).
- No header row, no metadata — pure corpus text only.
- Any line length is fine, including corpora with very few or no
  newlines — the reader processes fixed-size byte chunks, not lines, so
  it never depends on line structure for memory safety.

## What happens once this file exists

Once `bengali_corpus.txt` is placed here, run:

```bash
python scripts/run_heaps.py --mode full
```

No code changes are required. Outputs will be written to `results/final/`.

## If the file is missing

The pipeline will **not crash**. It will print an explanatory message
telling you the expected path and suggesting `--mode demo` to test the
project with synthetic data instead.
