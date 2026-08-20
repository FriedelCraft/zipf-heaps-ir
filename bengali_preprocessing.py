"""
Bengali Corpus Preprocessing Pipeline
Source corpus: AI4Bharat IndicCorp (Bengali subset)
  https://ai4bharat.iitm.ac.in/corpora  (or via HuggingFace: ai4bharat/IndicCorpV2)

Purpose: produce a cleaned, tokenized Bengali corpus + progressive
size samples for downstream Zipf's Law and Heaps' Law analysis.

Usage:
    python bengali_preprocessing.py --input bn_indiccorp_raw.txt --max_lines 200000
"""

import re
import json
import random
import unicodedata
import argparse
from collections import Counter

# ---------------------------------------------------------------------
# Optional: better tokenization via indic-nlp-library
#   pip install indic-nlp-library
# Falls back to whitespace tokenization if not installed.
# ---------------------------------------------------------------------
try:
    from indicnlp.tokenize import indic_tokenize
    HAVE_INDICNLP = True
except ImportError:
    HAVE_INDICNLP = False


# ---------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------
URL_RE = re.compile(r'https?://\S+|www\.\S+')
EMAIL_RE = re.compile(r'\S+@\S+')
BENGALI_DIGIT_RE = re.compile(r'[০-৯]+')
ASCII_DIGIT_RE = re.compile(r'[0-9]+')
# Bengali + common punctuation (danda '।', double danda '॥', quotes, etc.)
PUNCT_RE = re.compile(r'[।॥,.!?;:\'"()\[\]{}\-–—…‘’“”/\\|@#$%^&*_+=<>~`]')
MULTISPACE_RE = re.compile(r'\s+')
# Bengali Unicode block: U+0980–U+09FF. Keep only these + whitespace
# (this strips stray English words, emoji, other scripts).
NON_BENGALI_RE = re.compile(r'[^\u0980-\u09FF\s]')


def normalize_unicode(text: str) -> str:
    """NFC normalization — merges combining vowel signs/conjuncts consistently.
    Critical for Bengali: without this, visually identical words can hash
    to different token strings and silently inflate the vocabulary count."""
    return unicodedata.normalize('NFC', text)


def clean_text(text: str, remove_numbers=True, keep_bengali_only=True) -> str:
    text = normalize_unicode(text)
    text = URL_RE.sub(' ', text)
    text = EMAIL_RE.sub(' ', text)
    if remove_numbers:
        text = BENGALI_DIGIT_RE.sub(' ', text)
        text = ASCII_DIGIT_RE.sub(' ', text)
    text = PUNCT_RE.sub(' ', text)
    if keep_bengali_only:
        text = NON_BENGALI_RE.sub(' ', text)
    text = MULTISPACE_RE.sub(' ', text).strip()
    return text


def tokenize(text: str):
    if HAVE_INDICNLP:
        return list(indic_tokenize.trivial_tokenize(text, lang='bn'))
    return text.split()


def load_lines(path: str, max_lines=None):
    lines = []
    with open(path, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_lines is not None and i >= max_lines:
                break
            line = line.strip()
            if line:
                lines.append(line)
    return lines


def basic_stats(tokens, label="corpus"):
    counter = Counter(tokens)
    n_tokens = len(tokens)
    n_types = len(counter)
    return {
        'label': label,
        'total_tokens': n_tokens,
        'unique_types': n_types,
        'type_token_ratio': round(n_types / n_tokens, 4) if n_tokens else 0,
        'top20': counter.most_common(20),
    }


def raw_stats(lines, label="raw"):
    """Stats on raw whitespace-split text, no cleaning — the baseline
    for the raw-vs-processed comparison."""
    tokens = [tok for line in lines for tok in line.split()]
    return basic_stats(tokens, label)


def process_corpus(lines):
    tokens = []
    for line in lines:
        cleaned = clean_text(line)
        tokens.extend(tokenize(cleaned))
    return tokens


def save_progressive_samples(tokens, out_prefix, sizes):
    """Write nested prefix samples of increasing size — same underlying
    text, just truncated — so Heaps'/Zipf's curves are directly comparable
    across sample sizes (no confound from re-sampling different text)."""
    for size in sizes:
        sample = tokens[:size]
        fname = f"{out_prefix}_{size}.txt"
        with open(fname, 'w', encoding='utf-8') as f:
            f.write(' '.join(sample))
        print(f"  wrote {fname} ({len(sample)} tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='raw IndicCorp Bengali text file, one doc/line per line')
    ap.add_argument('--max_lines', type=int, default=200000, help='cap raw lines read (corpus is huge; keep this manageable)')
    ap.add_argument('--out_prefix', default='bn_sample')
    args = ap.parse_args()

    print(f"Loading up to {args.max_lines} lines from {args.input} ...")
    lines = load_lines(args.input, max_lines=args.max_lines)
    print(f"Loaded {len(lines)} lines.")

    print("Computing raw stats...")
    rstats = raw_stats(lines)

    print("Cleaning + tokenizing...")
    tokens = process_corpus(lines)
    pstats = basic_stats(tokens, "processed")

    print(json.dumps({"raw": rstats, "processed": pstats}, ensure_ascii=False, indent=2))

    with open("bn_cleaned_corpus.txt", "w", encoding='utf-8') as f:
        f.write(' '.join(tokens))
    print(f"\nSaved full cleaned corpus: bn_cleaned_corpus.txt ({len(tokens)} tokens)")

    print("\nSaving progressive samples for Zipf/Heaps analysis...")
    sizes = [10_000, 50_000, 100_000, 500_000, 1_000_000]
    sizes = [s for s in sizes if s <= len(tokens)] + (
        [len(tokens)] if len(tokens) not in sizes else []
    )
    save_progressive_samples(tokens, args.out_prefix, sizes)


if __name__ == "__main__":
    main()