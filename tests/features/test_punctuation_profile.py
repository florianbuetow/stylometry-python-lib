"""Tests for dedicated punctuation profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    FeatureExtractor,
    PunctuationProfileSidecar,
    PunctuationProfileTransformer,
    english_preprocessing_config,
    punctuation_profile_feature_names,
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_punctuation_profile_has_golden_mark_class_rate_and_sentence_final_values() -> None:
    config = english_preprocessing_config()
    text = "Hi, “yes” — ok! Wait… No? (fine); cost: €5."
    x = pd.DataFrame({"text": [text]}, index=["doc-punctuation"])
    transformer = PunctuationProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-punctuation"]
    assert result.shape[1] == 97
    assert tuple(result.columns) == punctuation_profile_feature_names()
    assert _cell(result, "text::punctuation_profile::total::count") == 12.0
    assert _cell(result, "text::punctuation_profile::total::per_1000_tokens") == 1500.0
    assert _cell(result, "text::punctuation_profile::total::per_sentence") == 4.0
    assert _cell(result, "text::punctuation_profile::mark=comma::count") == 1.0
    assert _cell(result, "text::punctuation_profile::mark=left_double_quote::count") == 1.0
    assert _cell(result, "text::punctuation_profile::mark=right_double_quote::count") == 1.0
    assert _cell(result, "text::punctuation_profile::mark=em_dash::count") == 1.0
    assert _cell(result, "text::punctuation_profile::mark=ellipsis::count") == 1.0
    assert _cell(result, "text::punctuation_profile::mark=period::per_1000_tokens") == 125.0
    assert _cell(result, "text::punctuation_profile::class=terminal::count") == 4.0
    assert _cell(result, "text::punctuation_profile::class=terminal::per_1000_tokens") == 500.0
    assert _cell(result, "text::punctuation_profile::class=terminal::per_sentence") == 4.0 / 3.0
    assert _cell(result, "text::punctuation_profile::class=semicolon_colon::count") == 2.0
    assert _cell(result, "text::punctuation_profile::class=quote::count") == 2.0
    assert _cell(result, "text::punctuation_profile::class=bracket_parenthesis::count") == 2.0
    assert _cell(result, "text::punctuation_profile::sentence_final=period::count") == 1.0
    assert _cell(result, "text::punctuation_profile::sentence_final=exclamation::count") == 1.0
    assert _cell(result, "text::punctuation_profile::sentence_final=question::count") == 1.0
    assert _cell(result, "text::punctuation_profile::sentence_final=ellipsis::count") == 0.0
    assert _cell(result, "text::punctuation_profile::sentence_final=no_terminal_punctuation::count") == 0.0
    assert _cell(result, "text::punctuation_profile::sentence_final=period::ratio") == 1.0 / 3.0
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, PunctuationProfileSidecar)
    assert sidecar.document_id == "doc-punctuation"
    assert sidecar.schema_version == "punctuation_profile_sidecar_v1"
    assert sidecar.punctuation_count == 12
    assert sidecar.sentence_count == 3
    occurrence_rows = [
        (occurrence.character_index, occurrence.mark_id, occurrence.text, occurrence.class_ids) for occurrence in sidecar.occurrences
    ]
    assert occurrence_rows == [
        (2, "comma", ",", ("comma",)),
        (4, "left_double_quote", "“", ("quote",)),
        (8, "right_double_quote", "”", ("quote",)),
        (10, "em_dash", "—", ("dash",)),
        (14, "exclamation", "!", ("terminal",)),
        (20, "ellipsis", "…", ("terminal", "ellipsis")),
        (24, "question", "?", ("terminal",)),
        (26, "open_parenthesis", "(", ("bracket_parenthesis",)),
        (31, "close_parenthesis", ")", ("bracket_parenthesis",)),
        (32, "semicolon", ";", ("semicolon_colon",)),
        (38, "colon", ":", ("semicolon_colon",)),
        (42, "period", ".", ("terminal",)),
    ]
    assert [(sentence.sentence_index, sentence.final_text, sentence.final_id) for sentence in sidecar.sentence_finals] == [
        (0, "!", "exclamation"),
        (1, "?", "question"),
        (2, ".", "period"),
    ]
    spec = transformer.registry_.by_name("text::punctuation_profile::class=terminal::per_sentence")
    assert spec.topic_dependence.value == "mostly_topic_independent"
    assert "built_in_punctuation_profile_rules:v1" in spec.provenance


def test_punctuation_profile_empty_text_uses_explicit_undefined_rate_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = PunctuationProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::punctuation_profile::total::count") == 0.0
    assert _cell(result, "text::punctuation_profile::mark=period::count") == 0.0
    assert math.isnan(_cell(result, "text::punctuation_profile::total::per_1000_tokens"))
    assert math.isnan(_cell(result, "text::punctuation_profile::total::per_sentence"))
    assert math.isnan(_cell(result, "text::punctuation_profile::sentence_final=period::ratio"))
    assert transformer.last_sidecars_[0].document_id == "0"
    assert transformer.last_sidecars_[0].punctuation_count == 0
    assert transformer.last_sidecars_[0].sentence_count == 0
    assert transformer.last_sidecars_[0].occurrences == ()
    assert transformer.last_sidecars_[0].sentence_finals == ()
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert "zero_tokens" in reasons
    assert "zero_sentences" in reasons


def test_punctuation_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["Hi!", "No punctuation words"]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = PunctuationProfileTransformer(text_column="text", config=config, output="pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(PunctuationProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = PunctuationProfileTransformer(text_column="text", config=config, output="sparse").fit_transform(x, None)
    numpy_result = PunctuationProfileTransformer(text_column="text", config=config, output="numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(PunctuationProfileTransformer("text", config, "pandas"),), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
    assert len(extractor.last_sidecars_) == 1
    assert extractor.last_sidecars_[0].block_name == "PunctuationProfileTransformer"
    sidecars = cast(tuple[PunctuationProfileSidecar, PunctuationProfileSidecar], extractor.last_sidecars_[0].sidecars)
    assert sidecars[0].document_id == "first"
    assert sidecars[0].occurrences[0].mark_id == "exclamation"
    assert sidecars[1].document_id == "second"
    assert sidecars[1].occurrences == ()
