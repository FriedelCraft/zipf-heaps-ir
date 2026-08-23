"""
corpus_streaming.py

Memory-safe streaming reader dedicated to Heaps' Law checkpoint
collection. Reads a (potentially multi-gigabyte) corpus in fixed-size
BYTE chunks, decodes safely across UTF-8 boundaries, splits on
whitespace, and maintains:

    - total_tokens: running count of tokens processed
    - unique_tokens: a plain set() of every distinct token seen so far

recording a checkpoint {"token_count": N, "vocabulary_size": V} whenever
the running token count reaches or passes one of the caller-supplied
checkpoint targets.

This project is intentionally narrower in scope than a Zipf analysis
would be: Heaps' Law only needs to know how many DISTINCT tokens have
been seen at various points, not their frequencies, so this module
maintains a plain `set()` rather than a `collections.Counter` — directly
matching what Heaps' Law needs and nothing more, and keeping this project
fully independent (no shared state, no shared streaming pass with any
other project).

MEMORY SAFETY
=============
Reading by fixed-size byte chunks — rather than `read_text()` on the
whole file, or line-by-line with `for line in f` — is deliberate: a
line-based reader's peak memory usage is bounded by the length of the
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
   correctly. The streamer holds back the last (possibly partial) token
   of each decoded chunk and prepends it as a string to the next chunk's
   decoded text before splitting again, so a word is never counted as
   two separate tokens.

3. CHECKPOINT-WITHIN-A-CHUNK: an 8 MB chunk can contain hundreds of
   thousands of tokens, easily spanning several checkpoints at once. The
   token list for each chunk is sliced at the exact position each
   checkpoint target is crossed, so vocabulary_size is recorded at the
   precise token count, not just once per chunk.

The unique_tokens set grows with VOCABULARY SIZE (number of unique
tokens), not corpus size (number of token occurrences), so even for a
multi-gigabyte corpus with a large-but-finite Bengali vocabulary, it
stays a modest, bounded size — unlike a full token list, which would need
one list entry per occurrence.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from .utils import get_logger

logger = get_logger(__name__)

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


def stream_corpus_for_heaps(
    corpus_path: Path,
    checkpoint_targets: Iterator[int],
    encoding: str = "utf-8",
    chunk_size_bytes: int = DEFAULT_CHUNK_SIZE_BYTES,
    progress_log_interval_bytes: int = DEFAULT_PROGRESS_LOG_INTERVAL_BYTES,
) -> list[dict]:
    """
    Stream the corpus exactly once, recording Heaps' Law checkpoints.

    Args:
        checkpoint_targets: an iterator of ascending token-count
            thresholds at which to record vocabulary size. See
            heaps_analysis.generate_checkpoint_targets().

    Returns:
        A list of {"token_count": int, "vocabulary_size": int} dicts,
        ascending by token_count, always ending with the true final
        corpus totals (even if that doesn't land exactly on the
        log-spaced schedule).

    Does NOT return the token set or any per-token data — only the small
    checkpoint table survives past this function, so memory stays bounded
    by vocabulary size during streaming and by checkpoint count afterward.

    Raises FileNotFoundError if the corpus does not exist (callers should
    catch this and print a friendly message via
    utils.print_missing_corpus_message).
    """
    corpus_path = Path(corpus_path)
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    unique_tokens: set[str] = set()
    checkpoints: list[dict] = []
    total_bytes = corpus_path.stat().st_size
    total_gb = total_bytes / (1024 ** 3)
    logger.info(
        f"Streaming corpus: {corpus_path.name} ({total_gb:.2f} GB) "
        f"in {chunk_size_bytes / (1024 ** 2):.0f} MB chunks (Heaps checkpoints)"
    )

    leftover_bytes = b""   # incomplete UTF-8 sequence carried across chunks
    leftover_token = ""    # possibly-partial token carried across chunks
    bytes_read = 0
    total_tokens = 0
    next_log_at = progress_log_interval_bytes

    next_checkpoint_target = next(checkpoint_targets, None)

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
                if next_checkpoint_target is None:
                    # No more checkpoints pending — fast path, one bulk update.
                    unique_tokens.update(tokens)
                    total_tokens += len(tokens)
                else:
                    # Slice this chunk's token list at each checkpoint
                    # boundary it crosses, so vocabulary_size is recorded
                    # at the EXACT token count, not just once per chunk.
                    idx = 0
                    n = len(tokens)
                    while idx < n:
                        remaining_to_checkpoint = next_checkpoint_target - total_tokens
                        if remaining_to_checkpoint > (n - idx):
                            slice_ = tokens[idx:]
                            unique_tokens.update(slice_)
                            total_tokens += len(slice_)
                            idx = n
                        else:
                            end = idx + max(remaining_to_checkpoint, 0)
                            slice_ = tokens[idx:end]
                            if slice_:
                                unique_tokens.update(slice_)
                                total_tokens += len(slice_)
                            idx = end
                            checkpoints.append({
                                "token_count": total_tokens,
                                "vocabulary_size": len(unique_tokens),
                            })
                            next_checkpoint_target = next(checkpoint_targets, None)
                            if next_checkpoint_target is None:
                                if idx < n:
                                    slice_ = tokens[idx:]
                                    unique_tokens.update(slice_)
                                    total_tokens += len(slice_)
                                    idx = n

            if bytes_read >= next_log_at:
                pct = (bytes_read / total_bytes * 100) if total_bytes else 100.0
                logger.info(
                    f"  ... {bytes_read / (1024 ** 3):.2f} / {total_gb:.2f} GB "
                    f"processed ({pct:.1f}%), vocabulary so far: {len(unique_tokens):,}"
                )
                next_log_at += progress_log_interval_bytes

    # Flush any leftover partial bytes/token once the file has truly ended
    # (a leftover token at this point is complete, since nothing follows it).
    if leftover_bytes:
        leftover_token += leftover_bytes.decode(encoding, errors="ignore")
    if leftover_token:
        unique_tokens.add(leftover_token)
        total_tokens += 1
        if next_checkpoint_target is not None and total_tokens >= next_checkpoint_target:
            checkpoints.append({
                "token_count": total_tokens,
                "vocabulary_size": len(unique_tokens),
            })

    # Always record the TRUE final (token_count, vocabulary_size) as the
    # last checkpoint, even if it doesn't land exactly on the log-spaced
    # schedule — guarantees the full observed range appears in the curve.
    if not checkpoints or checkpoints[-1]["token_count"] != total_tokens:
        checkpoints.append({
            "token_count": total_tokens,
            "vocabulary_size": len(unique_tokens),
        })

    logger.info(
        f"Finished streaming corpus: {bytes_read / (1024 ** 3):.2f} GB read, "
        f"{total_tokens:,} tokens, {len(unique_tokens):,} unique tokens, "
        f"{len(checkpoints)} Heaps checkpoints recorded"
    )

    return checkpoints
