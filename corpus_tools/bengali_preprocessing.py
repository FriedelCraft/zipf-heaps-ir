"""
Memory-safe preprocessing for the FULL Bengali IndicCorpV2 raw file.

This version:
  - streams the raw file line by line
  - tracks raw/processed stats with running Counters (no giant token list)
  - writes progressive sample files incrementally, closing each one the
    moment its token threshold is reached, instead of slicing a full list

Usage:
    python bengali_preprocessing.py --input bn_indiccorp_full_raw.txt
"""

import re
import json
import unicodedata
import argparse
from collections import Counter

try:
    from indicnlp.tokenize import indic_tokenize
    HAVE_INDICNLP = True
except ImportError:
    HAVE_INDICNLP = False

URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
BENGALI_DIGIT_RE = re.compile(r'[০-৯]+')
ASCII_DIGIT_RE = re.compile(r'[0-9]+')
PUNCT_RE = re.compile(r'[।॥,.!?;:\'"()\[\]{}\-–—…‘’“”/\\|@#$%^&*_+=<>~`]')
MULTISPACE_RE = re.compile(r'\s+')
NON_BENGALI_RE = re.compile(r'[^\u0980-\u09FF\s]')


def normalize_unicode(text: str) -> str:
    return unicodedata.normalize('NFC', text)


def clean_text(text: str) -> str:
    text = normalize_unicode(text)
    text = URL_RE.sub(' ', text)
    text = EMAIL_RE.sub(' ', text)
    text = BENGALI_DIGIT_RE.sub(' ', text)
    text = ASCII_DIGIT_RE.sub(' ', text)
    text = PUNCT_RE.sub(' ', text)
    text = NON_BENGALI_RE.sub(' ', text)
    text = MULTISPACE_RE.sub(' ', text).strip()
    return text


def tokenize(text: str):
    if HAVE_INDICNLP:
        return list(indic_tokenize.trivial_tokenize(text, lang='bn'))
    return text.split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='raw IndicCorp Bengali text file')
    ap.add_argument('--out_cleaned', default='bn_cleaned_corpus_full.txt')
    ap.add_argument('--out_prefix', default='bn_sample')
    ap.add_argument('--progress_every_lines', type=int, default=1_000_000)
    args = ap.parse_args()

    # Progressive sample thresholds (nested prefixes of the cleaned token stream)
    sample_sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000,
                     5_000_000, 10_000_000]

    # Open one file handle per sample threshold; close each as soon as it's full
    sample_handles = {size: open(f"{args.out_prefix}_{size}.txt", 'w', encoding='utf-8')
                       for size in sample_sizes}
    sample_counts = {size: 0 for size in sample_sizes}
    sample_done = set()

    raw_counter = Counter()
    processed_counter = Counter()
    raw_token_total = 0
    processed_token_total = 0
    lines_read = 0

    print(f"Streaming {args.input} ...")
    with open(args.input, encoding='utf-8') as fin, \
         open(args.out_cleaned, 'w', encoding='utf-8') as fout_cleaned:

        for line in fin:
            lines_read += 1
            line = line.strip()
            if not line:
                continue

            # raw stats (whitespace split, no cleaning)
            raw_tokens = line.split()
            raw_token_total += len(raw_tokens)
            raw_counter.update(raw_tokens)

            # cleaned + tokenized
            cleaned = clean_text(line)
            tokens = tokenize(cleaned)
            if not tokens:
                continue

            processed_token_total += len(tokens)
            processed_counter.update(tokens)
            fout_cleaned.write(' '.join(tokens) + ' ')

            # write into any still-open progressive sample files
            for size in sample_sizes:
                if size in sample_done:
                    continue
                remaining = size - sample_counts[size]
                take = tokens[:remaining]
                sample_handles[size].write(' '.join(take) + ' ')
                sample_counts[size] += len(take)
                if sample_counts[size] >= size:
                    sample_handles[size].close()
                    sample_done.add(size)
                    print(f"  sample_{size} complete ({size:,} tokens)")

            if lines_read % args.progress_every_lines == 0:
                print(f"  ...{lines_read:,} lines read, "
                      f"{processed_token_total:,} processed tokens so far")

    # close any sample files that never hit their threshold (corpus smaller than target)
    for size in sample_sizes:
        if size not in sample_done:
            sample_handles[size].close()
            print(f"  sample_{size} only reached {sample_counts[size]:,} tokens "
                  f"(corpus exhausted before threshold)")

    raw_stats = {
        'label': 'raw',
        'total_tokens': raw_token_total,
        'unique_types': len(raw_counter),
        'type_token_ratio': round(len(raw_counter) / raw_token_total, 4) if raw_token_total else 0,
        'top20': raw_counter.most_common(20),
    }
    processed_stats = {
        'label': 'processed',
        'total_tokens': processed_token_total,
        'unique_types': len(processed_counter),
        'type_token_ratio': round(len(processed_counter) / processed_token_total, 4) if processed_token_total else 0,
        'top20': processed_counter.most_common(20),
    }

    print(f"\nSaved full cleaned corpus: {args.out_cleaned} ({processed_token_total:,} tokens)")
    print("\n--- Raw vs Processed stats (for your methodology doc) ---")
    print(json.dumps({"raw": raw_stats, "processed": processed_stats}, ensure_ascii=False, indent=2))

    with open('bn_raw_vs_processed_stats.json', 'w', encoding='utf-8') as f:
        json.dump({"raw": raw_stats, "processed": processed_stats}, f, ensure_ascii=False, indent=2)
    print("\nAlso saved to bn_raw_vs_processed_stats.json")


if __name__ == "__main__":
    main()