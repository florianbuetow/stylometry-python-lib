"""Tests for hyphenation profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    FeatureExtractor,
    HyphenationProfileTransformer,
    english_preprocessing_config,
    hyphenation_profile_feature_names,
)

HYPHENATION_TEXT = (
    "Use e-mail and email for long-term longterm choices. We re-enter and reenter rooms. "
    "We co-operate and cooperate. A well-known state-of-the-art plan."
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_hyphenation_profile_has_golden_class_and_variant_pair_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [HYPHENATION_TEXT]}, index=["doc-hyphenation"])
    transformer = HyphenationProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-hyphenation"]
    assert result.shape[1] == 22
    assert tuple(result.columns) == hyphenation_profile_feature_names()
    assert _cell(result, "text::hyphenation_profile::total::count") == 6.0
    assert math.isclose(_cell(result, "text::hyphenation_profile::total::per_1000_orthographic_tokens"), 6000.0 / 21.0)
    assert _cell(result, "text::hyphenation_profile::class=single_hyphen::count") == 5.0
    assert math.isclose(_cell(result, "text::hyphenation_profile::class=single_hyphen::per_1000_orthographic_tokens"), 5000.0 / 21.0)
    assert _cell(result, "text::hyphenation_profile::class=multi_hyphen::count") == 1.0
    assert math.isclose(_cell(result, "text::hyphenation_profile::class=multi_hyphen::per_1000_orthographic_tokens"), 1000.0 / 21.0)
    assert _cell(result, "text::hyphenation_profile::class=prefix_hyphen::count") == 2.0
    assert math.isclose(_cell(result, "text::hyphenation_profile::class=prefix_hyphen::per_1000_orthographic_tokens"), 2000.0 / 21.0)
    assert _cell(result, "text::hyphenation_profile::class=non_prefix_compound::count") == 4.0
    assert math.isclose(
        _cell(result, "text::hyphenation_profile::class=non_prefix_compound::per_1000_orthographic_tokens"),
        4000.0 / 21.0,
    )
    for pair_id in ("email_e_mail", "reenter_re_enter", "cooperate_co_operate", "longterm_long_term"):
        assert _cell(result, f"text::hyphenation_profile::variant_pair={pair_id}::hyphenated_count") == 1.0
        assert _cell(result, f"text::hyphenation_profile::variant_pair={pair_id}::solid_count") == 1.0
        assert _cell(result, f"text::hyphenation_profile::variant_pair={pair_id}::hyphenated_share") == 0.5
    spec = transformer.registry_.by_name("text::hyphenation_profile::variant_pair=email_e_mail::hyphenated_share")
    assert spec.input_layer.value == "orthographic_tokens"
    assert "built_in_hyphenation_profile_rules:v1" in spec.provenance


def test_hyphenation_profile_empty_text_uses_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = HyphenationProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::hyphenation_profile::total::count") == 0.0
    assert math.isnan(_cell(result, "text::hyphenation_profile::total::per_1000_orthographic_tokens"))
    assert _cell(result, "text::hyphenation_profile::class=single_hyphen::count") == 0.0
    assert math.isnan(_cell(result, "text::hyphenation_profile::class=single_hyphen::per_1000_orthographic_tokens"))
    assert _cell(result, "text::hyphenation_profile::variant_pair=email_e_mail::hyphenated_count") == 0.0
    assert _cell(result, "text::hyphenation_profile::variant_pair=email_e_mail::solid_count") == 0.0
    assert math.isnan(_cell(result, "text::hyphenation_profile::variant_pair=email_e_mail::hyphenated_share"))
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert reasons == {"zero_orthographic_tokens", "zero_variant_pair_observations"}


def test_hyphenation_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [HYPHENATION_TEXT, "Plain words only."]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = HyphenationProfileTransformer(text_column="text", config=config, output="pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(HyphenationProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = HyphenationProfileTransformer("text", config, "sparse").fit_transform(x, None)
    numpy_result = HyphenationProfileTransformer("text", config, "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(HyphenationProfileTransformer("text", config, "pandas"),), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
