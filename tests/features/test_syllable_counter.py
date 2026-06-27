"""Tests for dictionary-backed syllable counting."""

from __future__ import annotations

import math

import pandas as pd

from stylometry_python_lib import (
    DistributionStatisticsTransformer,
    SyllableCountSource,
    english_preprocessing_config,
    load_syllable_dictionary,
    syllable_count,
    syllable_count_result,
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_syllable_dictionary_has_required_metadata_order_and_counts() -> None:
    dictionary = load_syllable_dictionary("syllable_counts")

    assert dictionary.name == "syllable_counts"
    assert dictionary.lexicon_id == "syllable_counts_en_v1"
    assert dictionary.language == "en"
    assert dictionary.version == "1.0.0"
    assert dictionary.license_note != ""
    assert dictionary.normalization == "lowercase_alpha_token_match_with_heuristic_fallback"
    assert dictionary.tokens()[:5] == ("people", "queue", "rhythm", "science", "beautiful")
    assert dictionary.count_by_token()["people"] == 2
    assert dictionary.count_by_token()["rhythm"] == 2


def test_syllable_count_reports_dictionary_fallback_and_non_word_sources() -> None:
    dictionary_hit = syllable_count_result("Rhythm")
    fallback = syllable_count_result("brillig")
    non_word = syllable_count_result("123")

    assert dictionary_hit.normalized == "rhythm"
    assert dictionary_hit.count == 2
    assert dictionary_hit.source == SyllableCountSource.DICTIONARY
    assert fallback.normalized == "brillig"
    assert fallback.count == 2
    assert fallback.source == SyllableCountSource.HEURISTIC_FALLBACK
    assert non_word.normalized == ""
    assert non_word.count == 0
    assert non_word.source == SyllableCountSource.NON_WORD
    assert syllable_count("people") == 2
    assert syllable_count("brillig") == 2


def test_syllable_distribution_uses_dictionary_counts_and_heuristic_fallback() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["People queue brillig."]}, index=["doc-syllables"])
    transformer = DistributionStatisticsTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-syllables"]
    assert _cell(result, "text::distribution::syllables_per_word::count") == 3.0
    assert math.isclose(_cell(result, "text::distribution::syllables_per_word::mean"), 5.0 / 3.0)
    assert math.isclose(_cell(result, "text::distribution::syllables_per_word::sample_std"), math.sqrt(1.0 / 3.0))
    assert math.isclose(_cell(result, "text::distribution::syllables_per_word::sample_variance"), 1.0 / 3.0)
    assert _cell(result, "text::distribution::syllables_per_word::min") == 1.0
    assert _cell(result, "text::distribution::syllables_per_word::max") == 2.0
    assert _cell(result, "text::distribution::syllables_per_word::p50") == 2.0
    assert _cell(result, "text::distribution::sentence_syllables::count") == 1.0
    assert _cell(result, "text::distribution::sentence_syllables::mean") == 5.0
    assert _cell(result, "text::distribution::paragraph_syllables::count") == 1.0
    assert _cell(result, "text::distribution::paragraph_syllables::mean") == 5.0
    spec = transformer.registry_.by_name("text::distribution::syllables_per_word::mean")
    paragraph_spec = transformer.registry_.by_name("text::distribution::paragraph_syllables::mean")
    assert "syllable_dictionary=syllable_counts_en_v1" in spec.provenance
    assert "syllable_fallback=heuristic_v1" in spec.provenance
    assert "syllable_dictionary=syllable_counts_en_v1" in paragraph_spec.provenance


def test_pronunciation_sidecar_reports_phonemes_or_absent() -> None:
    from stylometry_python_lib.text_metrics import PronunciationRecord, pronunciation_result

    known = pronunciation_result("rhythm")
    assert isinstance(known, PronunciationRecord)
    assert known.source == "dictionary"
    assert known.phonemes == ("R", "IH", "DH", "AH", "M")

    unknown = pronunciation_result("zzqx")
    assert unknown.source == "absent"
    assert unknown.phonemes == ()

    non_word = pronunciation_result("123")
    assert non_word.source == "non_word"
    assert non_word.phonemes == ()
