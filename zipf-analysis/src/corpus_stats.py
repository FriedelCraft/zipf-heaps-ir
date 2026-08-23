"""
corpus_stats.py

Basic corpus statistics for the Bengali corpus (or any UTF-8 text corpus):

- total token count
- vocabulary size
- type-token ratio
- hapax legomena count / percentage
- frequency distribution summary
- top-K most frequent tokens

MEMORY SAFETY
=============
This module is designed to process multi-gigabyte corpora on a machine
with limited RAM. It NEVER holds the full corpus text or the full list of
token occurrences in memory. Instead, `count_corpus_tokens()` reads the
file in fixed-size BYTE chunks (not lines) and builds a `collections.Counter`
incrementally as it goes.

Reading by fixed-size byte chunks — rather than `read_text()` on the whole
file, or line-by-line with `for line in f`/`readline()` — is deliberate:
a line-based reader's peak memory usage is bounded by the length of the
LONGEST LINE in the file, which is unbounded in general (a corpus with
very few or no newlines can turn "line-by-line" into "the whole file at
once"). Fixed-size byte chunks bound peak memory to the chunk size
regardless of how the file is laid out.

Two boundary-crossing problems come from splitting an arbitrary text file
into fixed-size byte chunks, and both are handled explicitly:

1. UTF-8 BYTE BOUNDARY: a multi-byte UTF-8 character (every Bengali
   character is 3 bytes in UTF-8) can be split across a chunk boundary.
   `_utf8_safe_split_point()` finds the last position in a byte chunk
   where it's safe to cut without truncating a character mid-sequence;
   any trailing incomplete bytes are carried over and prepended to the
   next chunk before decoding.

2. TOKEN (WORD) BOUNDARY: a whitespace-delimited token can be split
   across a chunk boundary even after UTF-8 decoding is handled
   correctly. `count_corpus_tokens()` holds back the last (possibly
   partial) token of each decoded chunk and prepends it as a string to
   the next chunk's decoded text before splitting again, so a word is
   never counted as two separate tokens. The held-back token is flushed
   into the Counter only once the file ends (or once whitespace confirms
   it's complete).

The Counter that comes out of this process grows with VOCABULARY SIZE
(number of unique tokens), not corpus size (number of token occurrences),
so even for a multi-gigabyte corpus with a large-but-finite Bengali
vocabulary, the Counter itself stays a modest, bounded size — unlike a
full token list, which would need one list entry per occurrence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .utils import get_logger

logger = get_logger(__name__)

# Defaults, overridable via config.yaml (corpus_stats.chunk_size_mb /
# corpus_stats.progress_log_interval_mb) and threaded through by callers.
DEFAULT_CHUNK_SIZE_BYTES = 8 * 1024 * 1024              # 8 MiB
DEFAULT_PROGRESS_LOG_INTERVAL_BYTES = 250 * 1024 * 1024  # 250 MiB


def _utf8_safe_split_point(chunk: bytes) -> int:
    """
    Given a bytes buffer that may end partway through a multi-byte UTF-8
    character, return the index at which it's safe to cut: chunk[:idx]
    contains only complete UTF-8 characters, and chunk[idx:] is the
    (possibly empty) trailing incomplete sequence to carry over to the
    next chunk.

    Works by scanning backward from the end of the chunk (at most 4
    bytes, the maximum length of a UTF-8 character) looking for the lead
    byte of the trailing character, then checking whether that
    character's full byte sequence actually fits inside the chunk.
    """
    n = len(chunk)
    if n == 0:
        return 0

    max_back = min(4, n)
    for back in range(1, max_back + 1):
        b = chunk[n - back]
        if (b & 0xC0) != 0x80:
            # Not a continuation byte -> this is the lead byte (ASCII or
            # the start of a multi-byte sequence).
            if b < 0x80:
                seq_len = 1
            elif (b & 0xE0) == 0xC0:
                seq_len = 2
            elif (b & 0xF0) == 0xE0:
                seq_len = 3
            elif (b & 0xF8) == 0xF0:
                seq_len = 4
            else:
                seq_len = 1  # invalid lead byte; treat as standalone
            return (n - back) if back < seq_len else n

    # The last `max_back` bytes were all continuation bytes (only possible
    # with malformed UTF-8 or an unusually placed boundary); conservatively
    # cut all of them off and let the next chunk try again.
    return n - max_back


def count_corpus_tokens(
    corpus_path: Path,
    encoding: str = "utf-8",
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    progress_log_interval_bytes: int = DEFAULT_PROGRESS_LOG_INTERVAL_BYTES,
) -> Counter:
    """
    Memory-safe streaming tokenizer for corpora of any size, including
    multi-gigabyte files.

    Reads the file in fixed-size byte chunks, decodes each chunk safely
    across UTF-8 boundaries, splits on whitespace, and builds a Counter
    incrementally — never materializing the full corpus text or the full
    list of token occurrences in memory at once.

    Raises FileNotFoundError if the corpus does not exist (callers should
    catch this and print a friendly message via
    utils.print_missing_corpus_message).
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    counter: Counter = Counter()
    total_bytes = corpus_path.stat().st_size
    total_gb = total_bytes / (1024 ** 3)
    logger.info(
        f"Streaming corpus: {corpus_path.name} ({total_gb:.2f} GB) "
        f"in {chunk_size_bytes / (1024 ** 2):.0f} MB chunks"
    )

    leftover_bytes = b""   # incomplete UTF-8 sequence carried across chunks
    leftover_token = ""    # possibly-partial token carried across chunks
    bytes_read = 0
    next_log_at = progress_log_interval_bytes

    with open(corpus_path, "rb") as f:
        while True:
            raw_chunk = f.read(chunk_size_bytes)
            if not raw_chunk:
                break
            bytes_read += len(raw_chunk)

            if leftover_bytes:
                raw_chunk = leftover_bytes + raw_chunk
                leftover_bytes = b""

            safe_end = _utf8_safe_split_point(raw_chunk)
            if safe_end < len(raw_chunk):
                leftover_bytes = raw_chunk[safe_end:]
                raw_chunk = raw_chunk[:safe_end]

            text = raw_chunk.decode(encoding, errors="ignore")

            if leftover_token:
                text = leftover_token + text
                leftover_token = ""

            if not text:
                continue

            ends_with_whitespace = text[-1].isspace()
            tokens = text.split()

            if tokens and not ends_with_whitespace:
                # The last token in this chunk may continue into the next
                # chunk — hold it back rather than counting it now.
                leftover_token = tokens.pop()

            if tokens:
                counter.update(tokens)

            if bytes_read >= next_log_at:
                pct = (bytes_read / total_bytes * 100) if total_bytes else 100.0
                logger.info(
                    f"  ... {bytes_read / (1024 ** 3):.2f} / {total_gb:.2f} GB "
                    f"processed ({pct:.1f}%), vocabulary so far: {len(counter):,}"
                )
                next_log_at += progress_log_interval_bytes

    # Flush any leftover partial bytes/token once the file has truly ended
    # (a leftover token at this point is complete, since nothing follows it).
    if leftover_bytes:
        leftover_token += leftover_bytes.decode(encoding, errors="ignore")
    if leftover_token:
        counter[leftover_token] += 1

    logger.info(
        f"Finished streaming corpus: {bytes_read / (1024 ** 3):.2f} GB read, "
        f"{sum(counter.values()):,} tokens, {len(counter):,} unique tokens"
    )

    return counter


@dataclass
class CorpusStatistics:
    total_tokens: int
    vocabulary_size: int
    type_token_ratio: float
    hapax_count: int
    hapax_percentage: float
    min_frequency: int
    max_frequency: int
    mean_frequency: float
    median_frequency: float

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"metric": "total_tokens", "value": self.total_tokens},
                {"metric": "vocabulary_size", "value": self.vocabulary_size},
                {"metric": "type_token_ratio", "value": self.type_token_ratio},
                {"metric": "hapax_count", "value": self.hapax_count},
                {"metric": "hapax_percentage", "value": self.hapax_percentage},
                {"metric": "min_frequency", "value": self.min_frequency},
                {"metric": "max_frequency", "value": self.max_frequency},
                {"metric": "mean_frequency", "value": self.mean_frequency},
                {"metric": "median_frequency", "value": self.median_frequency},
            ]
        )


def compute_corpus_statistics(counter: Counter) -> CorpusStatistics:
    """
    Compute summary statistics directly from a token-frequency Counter.

    Takes a Counter (unique token -> frequency) rather than a token list,
    since the Counter is already bounded by vocabulary size regardless of
    how large the source corpus was — this is what keeps statistics
    computation memory-safe for multi-gigabyte corpora.
    """
    if len(counter) == 0:
        logger.warning("Empty counter passed to compute_corpus_statistics().")
        return CorpusStatistics(
            total_tokens=0,
            vocabulary_size=0,
            type_token_ratio=0.0,
            hapax_count=0,
            hapax_percentage=0.0,
            min_frequency=0,
            max_frequency=0,
            mean_frequency=0.0,
            median_frequency=0.0,
        )

    freqs = pd.Series(list(counter.values()))

    total_tokens = int(freqs.sum())
    vocabulary_size = int(len(counter))
    ttr = vocabulary_size / total_tokens if total_tokens > 0 else 0.0
    hapax_count = int((freqs == 1).sum())
    hapax_pct = (hapax_count / vocabulary_size * 100) if vocabulary_size > 0 else 0.0

    return CorpusStatistics(
        total_tokens=total_tokens,
        vocabulary_size=vocabulary_size,
        type_token_ratio=round(ttr, 6),
        hapax_count=hapax_count,
        hapax_percentage=round(hapax_pct, 4),
        min_frequency=int(freqs.min()),
        max_frequency=int(freqs.max()),
        mean_frequency=round(float(freqs.mean()), 4),
        median_frequency=float(freqs.median()),
    )


def top_tokens_dataframe(counter: Counter, top_k: int = 20) -> pd.DataFrame:
    """Return a dataframe of the top-K most frequent tokens with rank."""
    most_common = counter.most_common(top_k)
    df = pd.DataFrame(most_common, columns=["token", "frequency"])
    df.insert(0, "rank", range(1, len(df) + 1))
    return df


def run_corpus_stats(
    corpus_path: Path,
    output_dir: Path,
    encoding: str = "utf-8",
    top_k: int = 20,
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    progress_log_interval_bytes: int = DEFAULT_PROGRESS_LOG_INTERVAL_BYTES,
) -> dict:
    """
    Full corpus-statistics pipeline: stream the corpus, compute stats, save CSVs.

    Returns a dict with keys: 'counter', 'statistics' for reuse by
    downstream modules (e.g. zipf_analysis) without re-reading the file.

    NOTE: unlike the original list-based implementation, this does NOT
    return a 'tokens' key — a full occurrence-level token list is exactly
    what makes multi-gigabyte corpora run out of memory, and nothing in
    this pipeline (run_zipf.py, zipf_analysis.py) ever used that key; both
    only need the Counter and the computed statistics.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    counter = count_corpus_tokens(
        corpus_path,
        encoding=encoding,
        chunk_size_bytes=chunk_size_bytes,
        progress_log_interval_bytes=progress_log_interval_bytes,
    )
    stats = compute_corpus_statistics(counter)

    stats.to_dataframe().to_csv(output_dir / "corpus_statistics.csv", index=False)
    top_tokens_dataframe(counter, top_k=top_k).to_csv(
        output_dir / "top_tokens.csv", index=False
    )

    logger.info(
        f"Corpus stats: {stats.total_tokens} tokens, "
        f"{stats.vocabulary_size} vocabulary, "
        f"TTR={stats.type_token_ratio}"
    )

    return {"counter": counter, "statistics": stats}
