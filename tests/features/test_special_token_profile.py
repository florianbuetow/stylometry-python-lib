"""Tests for dedicated special-token profile features."""

from __future__ import annotations

import math
import pickle
from dataclasses import replace
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    FeatureExtractor,
    SpecialTokenPolicy,
    SpecialTokenProfileTransformer,
    english_preprocessing_config,
    special_token_profile_feature_names,
)

SPECIAL_TOKEN_TEXT = "Email me@example.com, visit https://example.test, ping @writer about #Style parseHTTPValue snake_case 12.5 😀"


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_special_token_profile_has_golden_counts_rates_and_raw_policy_masking_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [SPECIAL_TOKEN_TEXT]}, index=["doc-special"])
    transformer = SpecialTokenProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-special"]
    assert result.shape[1] == 24
    assert tuple(result.columns) == special_token_profile_feature_names()
    assert _cell(result, "text::special_token_profile::total::count") == 8.0
    assert _cell(result, "text::special_token_profile::total::class_normalized_count") == 0.0
    assert math.isclose(_cell(result, "text::special_token_profile::total::per_1000_orthographic_tokens"), 8000.0 / 13.0)
    assert _cell(result, "text::special_token_profile::kind=number::count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=url::count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=email::count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=hashtag::count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=mention::count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=code_identifier::count") == 2.0
    assert _cell(result, "text::special_token_profile::kind=emoji::count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=code_identifier::class_normalized_count") == 0.0
    assert math.isclose(
        _cell(result, "text::special_token_profile::kind=code_identifier::per_1000_orthographic_tokens"),
        2000.0 / 13.0,
    )
    spec = transformer.registry_.by_name("text::special_token_profile::kind=url::per_1000_orthographic_tokens")
    assert spec.input_layer.value == "orthographic_tokens"
    assert "special_token_policy=raw_text" in spec.provenance


def test_special_token_profile_counts_class_normalized_masking_behavior() -> None:
    config = replace(english_preprocessing_config(), special_token_policy=SpecialTokenPolicy.CLASS_NORMALIZE)
    x = pd.DataFrame({"text": [SPECIAL_TOKEN_TEXT]})
    transformer = SpecialTokenProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::special_token_profile::total::count") == 8.0
    assert _cell(result, "text::special_token_profile::total::class_normalized_count") == 8.0
    assert _cell(result, "text::special_token_profile::kind=number::class_normalized_count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=url::class_normalized_count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=email::class_normalized_count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=hashtag::class_normalized_count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=mention::class_normalized_count") == 1.0
    assert _cell(result, "text::special_token_profile::kind=code_identifier::class_normalized_count") == 2.0
    assert _cell(result, "text::special_token_profile::kind=emoji::class_normalized_count") == 1.0
    assert (
        "special_token_policy=class_normalize"
        in transformer.registry_.by_name("text::special_token_profile::total::class_normalized_count").provenance
    )


def test_special_token_profile_empty_text_uses_explicit_undefined_rate_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = SpecialTokenProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::special_token_profile::total::count") == 0.0
    assert _cell(result, "text::special_token_profile::total::class_normalized_count") == 0.0
    assert math.isnan(_cell(result, "text::special_token_profile::total::per_1000_orthographic_tokens"))
    assert _cell(result, "text::special_token_profile::kind=email::count") == 0.0
    assert math.isnan(_cell(result, "text::special_token_profile::kind=email::per_1000_orthographic_tokens"))
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert reasons == {"zero_orthographic_tokens"}


def test_special_token_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [SPECIAL_TOKEN_TEXT, "Plain words only."]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = SpecialTokenProfileTransformer(text_column="text", config=config, output="pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(SpecialTokenProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = SpecialTokenProfileTransformer("text", config, "sparse").fit_transform(x, None)
    numpy_result = SpecialTokenProfileTransformer("text", config, "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(SpecialTokenProfileTransformer("text", config, "pandas"),), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
