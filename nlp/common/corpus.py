"""Corpus processing utilities with backward compatibility."""

from collections.abc import Iterator
from typing import Iterable

# Import shared tokenizer for backward compatibility
from nlp.common.tokenizer import (
    WordTokenizer,
    TokenizerConfig,
    normalize_for_word_models as _normalize,
    iter_word_tokens as _iter_tokens,
    iter_word_sequences as _iter_seqs,
)


def iter_lines(path: str, *, limit_lines: int | None = None) -> Iterator[str]:
    """Iterate over lines in a file."""
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit_lines is not None and i >= limit_lines:
                break
            yield line.rstrip("\n")


def normalize_for_word_models(text: str) -> str:
    """Normalize text for word-level models - delegates to shared tokenizer."""
    return _normalize(text)


def iter_word_tokens(
    path: str,
    *,
    limit_lines: int | None = None,
    add_sentence_markers: bool = True,
) -> Iterator[str]:
    """Iterate word tokens from file - delegates to shared tokenizer."""
    yield from _iter_tokens(path, limit_lines=limit_lines, add_sentence_markers=add_sentence_markers)


def iter_word_sequences(
    path: str,
    *,
    limit_lines: int | None = None,
    add_sentence_markers: bool = True,
) -> Iterator[list[str]]:
    """Iterate word sequences from file - delegates to shared tokenizer."""
    yield from _iter_seqs(path, limit_lines=limit_lines, add_sentence_markers=add_sentence_markers)

