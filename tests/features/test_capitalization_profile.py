"""Tests for capitalization profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    CapitalizationProfileTransformer,
    FeatureExtractor,
    capitalization_profile_feature_names,
    english_preprocessing_config,
)

CAPITALIZATION_TEXT = """Title Case Heading
lowercase heading
camelCase PascalCase NASA API U.S.A. wow.
lowercase sentence starts here. Another Sentence Starts."""


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_capitalization_profile_has_golden_token_line_and_sentence_initial_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [CAPITALIZATION_TEXT]}, index=["doc-capitalization"])
    transformer = CapitalizationProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-capitalization"]
    assert result.shape[1] == 16
    assert tuple(result.columns) == capitalization_profile_feature_names()
    assert _cell(result, "text::capitalization_profile::token_class=all_caps::count") == 2.0
    assert math.isclose(_cell(result, "text::capitalization_profile::token_class=all_caps::per_1000_tokens"), 1000.0 / 9.0)
    assert _cell(result, "text::capitalization_profile::token_class=acronym_like::count") == 3.0
    assert math.isclose(_cell(result, "text::capitalization_profile::token_class=acronym_like::per_1000_tokens"), 500.0 / 3.0)
    assert _cell(result, "text::capitalization_profile::token_class=camel_case::count") == 1.0
    assert math.isclose(_cell(result, "text::capitalization_profile::token_class=camel_case::per_1000_tokens"), 500.0 / 9.0)
    assert _cell(result, "text::capitalization_profile::token_class=pascal_case::count") == 1.0
    assert math.isclose(_cell(result, "text::capitalization_profile::token_class=pascal_case::per_1000_tokens"), 500.0 / 9.0)
    assert _cell(result, "text::capitalization_profile::line_class=titlecase_line::count") == 1.0
    assert _cell(result, "text::capitalization_profile::line_class=titlecase_line::ratio") == 0.25
    assert _cell(result, "text::capitalization_profile::line_class=lowercase_heading_line::count") == 1.0
    assert _cell(result, "text::capitalization_profile::line_class=lowercase_heading_line::ratio") == 0.25
    assert _cell(result, "text::capitalization_profile::sentence_initial=uppercase::count") == 2.0
    assert _cell(result, "text::capitalization_profile::sentence_initial=uppercase::ratio") == 0.5
    assert _cell(result, "text::capitalization_profile::sentence_initial=lowercase::count") == 2.0
    assert _cell(result, "text::capitalization_profile::sentence_initial=lowercase::ratio") == 0.5
    spec = transformer.registry_.by_name("text::capitalization_profile::token_class=camel_case::per_1000_tokens")
    assert spec.topic_dependence.value == "mixed"
    assert "built_in_capitalization_profile_rules:v1" in spec.provenance


def test_capitalization_profile_empty_text_uses_explicit_undefined_rate_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = CapitalizationProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::capitalization_profile::token_class=all_caps::count") == 0.0
    assert math.isnan(_cell(result, "text::capitalization_profile::token_class=all_caps::per_1000_tokens"))
    assert _cell(result, "text::capitalization_profile::line_class=titlecase_line::count") == 0.0
    assert math.isnan(_cell(result, "text::capitalization_profile::line_class=titlecase_line::ratio"))
    assert _cell(result, "text::capitalization_profile::sentence_initial=uppercase::count") == 0.0
    assert math.isnan(_cell(result, "text::capitalization_profile::sentence_initial=uppercase::ratio"))
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert reasons == {"zero_capitalization_tokens", "zero_nonblank_lines", "zero_sentence_initials"}


def test_capitalization_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [CAPITALIZATION_TEXT, "plain words only."]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = CapitalizationProfileTransformer(text_column="text", config=config, output="pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(CapitalizationProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = CapitalizationProfileTransformer("text", config, "sparse").fit_transform(x, None)
    numpy_result = CapitalizationProfileTransformer("text", config, "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(CapitalizationProfileTransformer("text", config, "pandas"),), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
