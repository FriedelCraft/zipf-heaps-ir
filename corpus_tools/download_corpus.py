"""
Download + sample the Bengali (ben_Beng) subset of AI4Bharat/IndicCorpV2.

Confirmed loading pattern (from the dataset's README YAML):
    load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split="ben_Beng")
-- config_name is fixed at "indiccorp_v2"; the LANGUAGE is chosen via
   the `split` argument, not `data_dir` or a separate config name.
   ben_Beng maps internally to the file data/bn.txt.

Uses streaming mode + reservoir sampling so we never need to hold the
full 29.6M-row dataset in memory or on disk -- just draw a fair random
sample of N rows as we stream through it.

Install first:
    pip install datasets

Usage:
    python download_corpus.py --n_samples 300000 --seed 42 \
        --out_raw bn_indiccorp_raw.txt --out_meta bn_corpus_metadata.json
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone

from datasets import load_dataset

# All valid splits from the dataset's README, in case you want to
# swap languages later without re-checking the YAML.
VALID_SPLITS = {
    "asm_Beng", "ben_Beng", "brx_Deva", "doi_Deva", "gom_Deva", "guj_Gujr",
    "hin_Deva", "kan_Knda", "kas_Arab", "mai_Deva", "mal_Mlym", "mar_Deva",
    "mni_Mtei", "npi_Deva", "ory_Orya", "pan_Guru", "san_Deva", "snd_Deva",
    "tam_Taml", "tel_Telu", "urd_Arab", "khasi", "santhali",
}


def reservoir_sample_stream(dataset_stream, n_samples, seed, text_field="text",
                             progress_every=10_000):
    """Reservoir sampling over a streaming HF dataset.
    Guarantees each row seen has equal probability of ending up in the
    final sample, without needing to know the total row count upfront.
    NOTE: streams through the ENTIRE split (can be slow for large splits)."""
    rng = random.Random(seed)
    reservoir = []
    total_seen = 0
    start = time.time()

    for row in dataset_stream:
        total_seen += 1
        text = row.get(text_field, "")
        if not text:
            continue

        if len(reservoir) < n_samples:
            reservoir.append(text)
        else:
            j = rng.randint(0, total_seen - 1)
            if j < n_samples:
                reservoir[j] = text

        if total_seen % progress_every == 0:
            elapsed = time.time() - start
            rate = total_seen / elapsed if elapsed > 0 else 0
            print(f"  ...streamed {total_seen:,} rows in {elapsed:.0f}s "
                  f"({rate:.0f} rows/s), reservoir filled: {len(reservoir):,}/{n_samples:,}")

    return reservoir, total_seen


def fast_prefix_sample_stream(dataset_stream, n_samples, text_field="text",
                               progress_every=10_000):
    """Fast alternative: take the first n_samples non-empty rows encountered,
    then stop -- no need to stream through the whole split. Slightly less
    statistically rigorous than true reservoir sampling (relies on the
    source file not being pre-sorted in a way that clusters similar text
    early on), but for a class deadline this is a defensible, fast tradeoff.
    Web-scraped corpora like IndicCorp are not typically source-sorted."""
    sample = []
    total_seen = 0
    start = time.time()

    for row in dataset_stream:
        total_seen += 1
        text = row.get(text_field, "")
        if not text:
            continue
        sample.append(text)

        if total_seen % progress_every == 0:
            elapsed = time.time() - start
            rate = total_seen / elapsed if elapsed > 0 else 0
            print(f"  ...streamed {total_seen:,} rows in {elapsed:.0f}s "
                  f"({rate:.0f} rows/s), collected: {len(sample):,}/{n_samples:,}")

        if len(sample) >= n_samples:
            break

    return sample, total_seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_samples', type=int, default=300_000,
                     help='number of rows to sample via reservoir sampling')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--split', default='ben_Beng',
                     help='IndicCorpV2 split = language (ben_Beng = Bengali). '
                          f'Valid: {sorted(VALID_SPLITS)}')
    ap.add_argument('--text_field', default='text')
    ap.add_argument('--out_raw', default='bn_indiccorp_raw.txt')
    ap.add_argument('--out_meta', default='bn_corpus_metadata.json')
    ap.add_argument('--mode', choices=['fast', 'reservoir'], default='fast',
                     help="'fast' = stop after n_samples rows (quick, default). "
                          "'reservoir' = stream the full split for a true uniform "
                          "sample (rigorous, can be slow for large splits).")
    args = ap.parse_args()

    if args.split not in VALID_SPLITS:
        raise ValueError(f"'{args.split}' is not a recognized split. Valid options: {sorted(VALID_SPLITS)}")

    print(f"Opening AI4Bharat/IndicCorpV2 [config='indiccorp_v2', split='{args.split}'] in streaming mode...")
    # encoding="utf-8" is explicit here because on Windows, the underlying
    # text loader otherwise falls back to the system codepage (cp1252),
    # which cannot decode Bengali UTF-8 byte sequences.
    ds = load_dataset("ai4bharat/IndicCorpV2", "indiccorp_v2", split=args.split,
                       streaming=True, encoding="utf-8", encoding_errors="replace")

    start = time.time()
    if args.mode == 'fast':
        print("Mode: fast (stopping after n_samples rows, no full-split streaming)")
        sample, total_seen = fast_prefix_sample_stream(
            ds, n_samples=args.n_samples, text_field=args.text_field
        )
    else:
        print("Mode: reservoir (streaming full split for a true uniform sample)")
        sample, total_seen = reservoir_sample_stream(
            ds, n_samples=args.n_samples, seed=args.seed, text_field=args.text_field
        )
    elapsed = time.time() - start
    print(f"Done. Streamed {total_seen:,} rows total, sampled {len(sample):,} rows in {elapsed:.1f}s")

    with open(args.out_raw, 'w', encoding='utf-8') as f:
        for line in sample:
            f.write(line.replace('\n', ' ').strip() + '\n')
    print(f"Saved raw sample: {args.out_raw}")

    metadata = {
        "dataset": "ai4bharat/IndicCorpV2",
        "config_name": "indiccorp_v2",
        "split_used": args.split,
        "language": "Bengali",
        "script": "Bengali (Beng)",
        "source_file_in_repo": "data/bn.txt",
        "download_date_utc": datetime.now(timezone.utc).isoformat(),
        "total_rows_in_source_streamed_through": total_seen,
        "sample_size": len(sample),
        "sampling_method": (
            "reservoir sampling (streaming full split, uniform random)"
            if args.mode == 'reservoir' else
            "prefix sampling (first N non-empty rows of stream, stopped early -- "
            "not a full-split uniform sample; relies on source not being sorted "
            "in a way that clusters similar text early)"
        ),
        "random_seed": args.seed if args.mode == 'reservoir' else None,
        "text_field_used": args.text_field,
        "note": "total_rows_in_source_streamed_through equals the full split size "
                "only if the stream was exhausted; for large splits this script "
                "streams through everything to guarantee a uniform sample, which "
                "can take a while -- see timing above.",
    }
    with open(args.out_meta, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved metadata: {args.out_meta}")
    print("\n--- Record these in your methodology doc ---")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
