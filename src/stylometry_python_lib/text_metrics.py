"""Shared deterministic text metric helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache

from stylometry_python_lib.lexicons import (
    VersionedPronunciationResource,
    VersionedSyllableDictionary,
    load_pronunciations,
    load_syllable_dictionary,
)


class SyllableCountSource(StrEnum):
    """Machine-readable source for a syllable count."""

    DICTIONARY = "dictionary"
    HEURISTIC_FALLBACK = "heuristic_fallback"
    NON_WORD = "non_word"


@dataclass(frozen=True)
class SyllableCountResult:
    """A syllable count with source metadata."""

    raw: str
    normalized: str
    count: int
    source: SyllableCountSource


def syllable_count(word: str) -> int:
    """Return a dictionary-backed deterministic syllable count for one token."""
    return syllable_count_result(word).count


def syllable_count_result(word: str) -> SyllableCountResult:
    """Return a deterministic syllable count with source metadata."""
    cleaned = _normalize_syllable_token(word)
    if cleaned == "":
        return SyllableCountResult(raw=word, normalized=cleaned, count=0, source=SyllableCountSource.NON_WORD)
    count_by_token = _default_syllable_dictionary().count_by_token()
    if cleaned in count_by_token:
        return SyllableCountResult(raw=word, normalized=cleaned, count=count_by_token[cleaned], source=SyllableCountSource.DICTIONARY)
    return SyllableCountResult(
        raw=word,
        normalized=cleaned,
        count=_heuristic_syllable_count(cleaned),
        source=SyllableCountSource.HEURISTIC_FALLBACK,
    )


def _normalize_syllable_token(word: str) -> str:
    return re.sub(r"[^a-z]", "", word.lower())


def _heuristic_syllable_count(cleaned: str) -> int:
    groups = re.findall(r"[aeiouy]+", cleaned)
    count = len(groups)
    if cleaned.endswith("e") and count > 1:
        count -= 1
    if count == 0:
        return 1
    return count


@lru_cache(maxsize=1)
def _default_syllable_dictionary() -> VersionedSyllableDictionary:
    return load_syllable_dictionary("syllable_counts")


@dataclass(frozen=True)
class PronunciationRecord:
    """An optional pronunciation sidecar with source metadata."""

    raw: str
    normalized: str
    phonemes: tuple[str, ...]
    source: str


@lru_cache(maxsize=1)
def _default_pronunciation_resource() -> VersionedPronunciationResource:
    return load_pronunciations("pronunciations")


def pronunciation_result(word: str) -> PronunciationRecord:
    """Return a deterministic pronunciation sidecar; absent tokens report empty phonemes."""
    cleaned = _normalize_syllable_token(word)
    if cleaned == "":
        return PronunciationRecord(raw=word, normalized=cleaned, phonemes=(), source="non_word")
    phonemes_by_token = _default_pronunciation_resource().phonemes_by_token()
    if cleaned in phonemes_by_token:
        return PronunciationRecord(raw=word, normalized=cleaned, phonemes=phonemes_by_token[cleaned], source="dictionary")
    return PronunciationRecord(raw=word, normalized=cleaned, phonemes=(), source="absent")
