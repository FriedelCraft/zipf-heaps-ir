"""
test_corpus_stats.py

Tests for the memory-safe streaming tokenizer in src/corpus_stats.py.

Focus areas (matching the memory-safety fix requirements):
    1. Normal tokenization matches a straightforward in-memory reference,
       across a range of chunk sizes -- including deliberately tiny ones
       that force nearly every word to cross a chunk boundary.
    2. Bengali UTF-8 text is handled correctly even when chunk boundaries
       fall in the middle of a multi-byte character.
    3. compute_corpus_statistics() produces correct summary statistics
       from a Counter.
    4. The Counter returned by count_corpus_tokens() plugs into the
       existing Zipf pipeline (zipf_analysis.py) exactly as before.

All test files are small, temporary, and created/removed within each test
-- no large files are needed to validate correctness, since chunk-boundary
behavior is exercised by using chunk sizes tiny relative to the test text
rather than by using a huge file.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.corpus_stats import (
    _utf8_safe_split_point,
    count_corpus_tokens,
    compute_corpus_statistics,
    top_tokens_dataframe,
)
from src.zipf_analysis import build_rank_frequency_table, add_log_columns, fit_overall_zipf


# =============================================================================
# HELPERS
# =============================================================================

# A representative range of chunk sizes, including ones smaller than a
# single word/character, to stress-test boundary handling.
CHUNK_SIZES_TO_TEST = [1, 2, 3, 4, 5, 7, 13, 16, 64, 1024, 1_000_000]


def _write_and_count(tmp_path: Path, text: str, chunk_size: int, filename: str = "corpus.txt") -> Counter:
    corpus_path = tmp_path / filename
    corpus_path.write_bytes(text.encode("utf-8"))
    return count_corpus_tokens(corpus_path, encoding="utf-8", chunk_size_bytes=chunk_size)


# =============================================================================
# 1. NORMAL TOKENIZATION — matches reference across many chunk sizes
# =============================================================================

@pytest.mark.parametrize("chunk_size", CHUNK_SIZES_TO_TEST)
def test_ascii_tokenization_matches_reference_across_chunk_sizes(tmp_path, chunk_size):
    text = "the quick brown fox jumps over the lazy dog\n" * 50
    reference = Counter(text.split())

    result = _write_and_count(tmp_path, text, chunk_size)

    assert result == reference
    assert sum(result.values()) == sum(reference.values())


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES_TO_TEST)
def test_tokenization_with_no_trailing_newline(tmp_path, chunk_size):
    # File does NOT end in whitespace -> exercises the final-leftover-token
    # flush path at true EOF.
    text = "alpha beta gamma delta epsilon"
    reference = Counter(text.split())

    result = _write_and_count(tmp_path, text, chunk_size)

    assert result == reference


@pytest.mark.parametrize("chunk_size", CHUNK_SIZES_TO_TEST)
def test_tokenization_with_multiple_newlines_and_blank_lines(tmp_path, chunk_size):
    text = "one two\n\nthree\n   four   five\n\n\nsix"
    reference = Counter(text.split())

    result = _write_and_count(tmp_path, text, chunk_size)

    assert result == reference


def test_empty_file_returns_empty_counter(tmp_path):
    result = _write_and_count(tmp_path, "", chunk_size=8)
    assert result == Counter()
    assert len(result) == 0


def test_single_token_no_whitespace(tmp_path):
    for chunk_size in [1, 2, 3, 1000]:
        result = _write_and_count(tmp_path, "onlyoneword", chunk_size)
        assert result == Counter({"onlyoneword": 1})


def test_missing_file_raises_filenotfounderror(tmp_path):
    with pytest.raises(FileNotFoundError):
        count_corpus_tokens(tmp_path / "does_not_exist.txt")


# =============================================================================
# 2. TOKENS CROSSING CHUNK BOUNDARIES — explicit boundary-position tests
# =============================================================================

def test_word_split_exactly_at_chunk_boundary(tmp_path):
    # "elephant" is 8 bytes; force the chunk boundary to land in the
    # middle of it at every possible split position.
    text = "cat elephant dog"
    reference = Counter(text.split())
    for chunk_size in range(1, len(text.encode("utf-8")) + 1):
        result = _write_and_count(tmp_path, text, chunk_size, filename=f"c{chunk_size}.txt")
        assert result == reference, f"failed at chunk_size={chunk_size}"


def test_repeated_word_split_across_boundary_counts_correctly(tmp_path):
    # If boundary-crossing were handled incorrectly, "banana" appearing 3
    # times could get miscounted as fragments instead of 3 whole words.
    text = "banana banana banana apple"
    reference = Counter({"banana": 3, "apple": 1})
    for chunk_size in [1, 2, 3, 4, 5, 6]:
        result = _write_and_count(tmp_path, text, chunk_size, filename=f"b{chunk_size}.txt")
        assert result == reference, f"failed at chunk_size={chunk_size}"


def test_boundary_exactly_on_whitespace(tmp_path):
    # chunk_size chosen so the split lands exactly on the space character
    text = "aaaa bbbb"
    reference = Counter(text.split())
    result = _write_and_count(tmp_path, text, chunk_size=5)  # "aaaa " | "bbbb"
    assert result == reference


# =============================================================================
# 3. BENGALI UTF-8 TEXT — including chunk boundaries mid-character
# =============================================================================

def test_bengali_tokenization_matches_reference_across_chunk_sizes(tmp_path):
    # Real Bengali words (3-byte-per-character UTF-8), repeated to give
    # every chunk size in CHUNK_SIZES_TO_TEST a chance to split mid-word
    # and mid-character.
    text = "আমি বাংলায় গান গাই " * 30
    reference = Counter(text.split())

    for chunk_size in CHUNK_SIZES_TO_TEST:
        result = _write_and_count(tmp_path, text, chunk_size, filename=f"bn{chunk_size}.txt")
        assert result == reference, f"failed at chunk_size={chunk_size}"
        # Sanity: no mangled/replacement characters snuck into any token
        for token in result:
            assert "\ufffd" not in token, f"replacement character found in token at chunk_size={chunk_size}"


def test_bengali_character_split_at_every_byte_offset(tmp_path):
    # "বাংলা" (Bengali) encodes to multiple 3-byte characters. Force the
    # chunk boundary through every single byte offset within the word to
    # make sure _utf8_safe_split_point() never truncates a character.
    word = "বাংলা"
    text = f"{word} {word} {word}"
    reference = Counter(text.split())
    n_bytes = len(text.encode("utf-8"))
    for chunk_size in range(1, n_bytes + 1):
        result = _write_and_count(tmp_path, text, chunk_size, filename=f"bnbyte{chunk_size}.txt")
        assert result == reference, f"failed at chunk_size={chunk_size}"


def test_mixed_bengali_and_ascii(tmp_path):
    text = "Bengali বাংলা mix 123 সংখ্যা end"
    reference = Counter(text.split())
    for chunk_size in [1, 2, 3, 5, 7, 11]:
        result = _write_and_count(tmp_path, text, chunk_size, filename=f"mix{chunk_size}.txt")
        assert result == reference, f"failed at chunk_size={chunk_size}"


# =============================================================================
# _utf8_safe_split_point — direct unit tests on the boundary-detection helper
# =============================================================================

def test_utf8_safe_split_point_pure_ascii():
    chunk = "hello".encode("utf-8")
    assert _utf8_safe_split_point(chunk) == len(chunk)


def test_utf8_safe_split_point_complete_multibyte_char():
    # A complete Bengali character at the end -> nothing to trim
    chunk = "ম".encode("utf-8")  # 3 bytes, all present
    assert _utf8_safe_split_point(chunk) == len(chunk)


def test_utf8_safe_split_point_truncated_multibyte_char():
    full = "ম".encode("utf-8")  # 3-byte sequence
    for cut in range(1, 3):
        truncated = full[:cut]
        # the lead byte position is 0, so anything less than 3 bytes present
        # should be trimmed back to 0
        assert _utf8_safe_split_point(truncated) == 0


def test_utf8_safe_split_point_empty():
    assert _utf8_safe_split_point(b"") == 0


# =============================================================================
# 4. compute_corpus_statistics — correctness from a Counter
# =============================================================================

def test_compute_corpus_statistics_known_values():
    counter = Counter({"a": 5, "b": 3, "c": 1, "d": 1})
    stats = compute_corpus_statistics(counter)

    assert stats.total_tokens == 10           # 5+3+1+1
    assert stats.vocabulary_size == 4
    assert stats.hapax_count == 2              # 'c' and 'd'
    assert stats.hapax_percentage == pytest.approx(50.0)  # 2/4 * 100
    assert stats.min_frequency == 1
    assert stats.max_frequency == 5
    assert stats.mean_frequency == pytest.approx(2.5)
    assert stats.type_token_ratio == pytest.approx(4 / 10)


def test_compute_corpus_statistics_empty_counter():
    stats = compute_corpus_statistics(Counter())
    assert stats.total_tokens == 0
    assert stats.vocabulary_size == 0
    assert stats.hapax_percentage == 0.0


def test_top_tokens_dataframe_from_streamed_counter(tmp_path):
    text = "x x x y y z"
    counter = _write_and_count(tmp_path, text, chunk_size=2)
    df = top_tokens_dataframe(counter, top_k=2)
    assert list(df["token"]) == ["x", "y"]
    assert list(df["frequency"]) == [3, 2]
    assert list(df["rank"]) == [1, 2]


# =============================================================================
# 5. INTEGRATION — the streamed Counter feeds correctly into zipf_analysis.py
# =============================================================================

def test_streamed_counter_feeds_existing_zipf_pipeline(tmp_path):
    """
    Builds a small synthetic Zipfian-ish corpus, tokenizes it via the new
    memory-safe streaming path with a deliberately tiny chunk size (to
    force boundary-crossing), then confirms the resulting Counter works
    exactly as before with the existing (unmodified) Zipf analysis
    functions: ranking, log transform, and the overall linear fit.
    """
    words = ["one", "two", "three", "four", "five"]
    weights = [50, 25, 12, 8, 5]  # roughly Zipfian
    text_parts = []
    for word, weight in zip(words, weights):
        text_parts.extend([word] * weight)
    import random
    random.seed(42)
    random.shuffle(text_parts)
    text = " ".join(text_parts)

    counter = _write_and_count(tmp_path, text, chunk_size=3)  # tiny chunks

    # Sanity: streaming reproduced the exact expected frequencies
    assert counter["one"] == 50
    assert counter["two"] == 25
    assert counter["five"] == 5

    # Feed directly into the existing (unmodified) Zipf pipeline
    rank_freq_df = build_rank_frequency_table(counter)
    log_df = add_log_columns(rank_freq_df)
    fit = fit_overall_zipf(log_df, min_observations=3)

    assert fit is not None
    assert fit.n_observations == 5
    # A rough Zipfian distribution should fit reasonably well
    assert fit.r_squared > 0.8
