"""Tests for spelling variant profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    FeatureExtractor,
    SpellingVariantProfileTransformer,
    english_preprocessing_config,
    load_spelling_variants,
    spelling_variant_profile_feature_names,
    spelling_variant_profile_transformer,
)

SPELLING_VARIANT_TEXT = "Colour color favour Favor realise realize organise organize centre Center colour."


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_spelling_variant_resource_has_required_metadata_and_order() -> None:
    resource = load_spelling_variants("spelling_variants")

    assert resource.lexicon_id == "spelling_variants_en_v1"
    assert resource.language == "en"
    assert resource.version == "1.0.0"
    assert resource.license_note != ""
    assert resource.normalization == "lowercase_token_match_with_exact_case_counts"
    assert tuple(pair.pair_id for pair in resource.pairs) == (
        "colour_color",
        "favour_favor",
        "realise_realize",
        "organise_organize",
        "centre_center",
    )
    assert resource.groups() == ("us_uk", "dialect")


def test_spelling_variant_profile_has_golden_per_pair_lowercase_exact_case_and_share_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [SPELLING_VARIANT_TEXT]}, index=["doc-spelling"])
    transformer = spelling_variant_profile_transformer(
        text_column="text",
        config=config,
        resource_name="spelling_variants",
        output="pandas",
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-spelling"]
    assert result.shape[1] == 35
    assert tuple(result.columns) == spelling_variant_profile_feature_names(transformer.resource_)
    assert _cell(result, "text::spelling_variant_profile::pair=colour_color::variant=british::lowercase_count") == 2.0
    assert _cell(result, "text::spelling_variant_profile::pair=colour_color::variant=british::exact_case_count") == 1.0
    assert math.isclose(
        _cell(result, "text::spelling_variant_profile::pair=colour_color::variant=british::per_1000_tokens"),
        2000.0 / 11.0,
    )
    assert _cell(result, "text::spelling_variant_profile::pair=colour_color::variant=american::lowercase_count") == 1.0
    assert _cell(result, "text::spelling_variant_profile::pair=colour_color::variant=american::exact_case_count") == 1.0
    assert math.isclose(
        _cell(result, "text::spelling_variant_profile::pair=colour_color::variant=american::per_1000_tokens"),
        1000.0 / 11.0,
    )
    assert math.isclose(_cell(result, "text::spelling_variant_profile::pair=colour_color::variant=british::lowercase_share"), 2.0 / 3.0)
    for pair_id in ("favour_favor", "realise_realize", "organise_organize", "centre_center"):
        assert _cell(result, f"text::spelling_variant_profile::pair={pair_id}::variant=british::lowercase_count") == 1.0
        assert _cell(result, f"text::spelling_variant_profile::pair={pair_id}::variant=american::lowercase_count") == 1.0
        assert _cell(result, f"text::spelling_variant_profile::pair={pair_id}::variant=british::lowercase_share") == 0.5
    assert _cell(result, "text::spelling_variant_profile::pair=favour_favor::variant=american::exact_case_count") == 0.0
    assert _cell(result, "text::spelling_variant_profile::pair=centre_center::variant=american::exact_case_count") == 0.0
    spec = transformer.registry_.by_name("text::spelling_variant_profile::pair=colour_color::variant=british::per_1000_tokens")
    assert spec.topic_dependence.value == "topic_sensitive"
    assert "lexicon_id=spelling_variants_en_v1" in spec.provenance


def test_spelling_variant_profile_empty_text_uses_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = spelling_variant_profile_transformer("text", config, "spelling_variants", "pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::spelling_variant_profile::pair=colour_color::variant=british::lowercase_count") == 0.0
    assert _cell(result, "text::spelling_variant_profile::pair=colour_color::variant=british::exact_case_count") == 0.0
    assert math.isnan(_cell(result, "text::spelling_variant_profile::pair=colour_color::variant=british::per_1000_tokens"))
    assert math.isnan(_cell(result, "text::spelling_variant_profile::pair=colour_color::variant=british::lowercase_share"))
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert reasons == {"zero_tokens", "zero_variant_pair_observations"}


def test_spelling_variant_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [SPELLING_VARIANT_TEXT, "plain words only"]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = spelling_variant_profile_transformer("text", config, "spelling_variants", "pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(SpellingVariantProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = spelling_variant_profile_transformer("text", config, "spelling_variants", "sparse").fit_transform(x, None)
    numpy_result = spelling_variant_profile_transformer("text", config, "spelling_variants", "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(
        blocks=(spelling_variant_profile_transformer("text", config, "spelling_variants", "pandas"),),
        output="pandas",
    )
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
