# data/external/

This folder holds results **produced by other parts of the project** (mainly
the teammate responsible for Heaps' Law analysis) that this codebase
integrates but does not itself compute.

## Expected file

```
data/external/heaps_results.csv
```

### Preferred columns

| column           | meaning                                   |
|------------------|--------------------------------------------|
| corpus_version   | identifier/version of the corpus used       |
| token_count      | total tokens the Heaps fit was computed on  |
| vocabulary_size  | observed vocabulary size                    |
| K                | Heaps' Law constant (V = K * N^beta)        |
| beta             | Heaps' Law exponent                         |
| r_squared        | goodness of fit                             |
| rmse             | root mean squared error of the fit          |

### Compatible variations

The loader (`src/integration.py`) will also try to auto-detect common
alternate column names (e.g. `heaps_beta`, `b`, `vocab_size`, `r2`, ...). See
`config.yaml -> heaps.compatible_aliases` for the full alias list.

### If this file is missing

The pipeline will **not crash**. It will print:

```
Heaps results not found. Skipping Heaps integration.
```

and continue with Zipf + cost model results only.

## Other external inputs

Any other externally-supplied CSVs (corpus metadata, alternate cost
assumptions, etc.) can also be placed here and pointed to via `config.yaml`
or CLI flags in `--mode external`.
