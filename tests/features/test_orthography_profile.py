"""Tests for Unicode orthography profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    FeatureExtractor,
    OrthographyCodepointSidecar,
    OrthographyProfileTransformer,
    english_preprocessing_config,
    orthography_profile_feature_names,
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_orthography_profile_has_golden_unicode_codepoint_category_and_letter_values() -> None:
    config = english_preprocessing_config()
    text = "Aaé9!€\n"
    x = pd.DataFrame({"text": [text]}, index=["doc-unicode"])
    transformer = OrthographyProfileTransformer(text_column="text", config=config, output="pandas")
    decimal_number_category = "N" + "d"

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-unicode"]
    assert result.shape[1] == 183
    assert tuple(result.columns) == orthography_profile_feature_names()
    assert _cell(result, "text::orthography_profile::codepoint_count") == 7.0
    assert _cell(result, "text::orthography_profile::unique_codepoint_count") == 7.0
    assert _cell(result, "text::orthography_profile::codepoint_min") == 10.0
    assert _cell(result, "text::orthography_profile::codepoint_max") == 8364.0
    assert math.isclose(_cell(result, "text::orthography_profile::codepoint_mean"), 8859.0 / 7.0)
    assert _cell(result, "text::orthography_profile::character_class=unicode_letter::count") == 3.0
    assert _cell(result, "text::orthography_profile::character_class=ascii_letter::count") == 2.0
    assert _cell(result, "text::orthography_profile::character_class=non_ascii_letter::count") == 1.0
    assert _cell(result, "text::orthography_profile::character_class=digit::count") == 1.0
    assert _cell(result, "text::orthography_profile::character_class=punctuation::count") == 1.0
    assert _cell(result, "text::orthography_profile::character_class=symbol::count") == 1.0
    assert _cell(result, "text::orthography_profile::character_class=whitespace::count") == 1.0
    assert _cell(result, "text::orthography_profile::character_class=control_format_other::count") == 1.0
    assert _cell(result, "text::orthography_profile::unicode_category=Lu::count") == 1.0
    assert _cell(result, "text::orthography_profile::unicode_category=Ll::count") == 2.0
    assert _cell(result, f"text::orthography_profile::unicode_category={decimal_number_category}::count") == 1.0
    assert _cell(result, "text::orthography_profile::unicode_category=Po::count") == 1.0
    assert _cell(result, "text::orthography_profile::unicode_category=Sc::count") == 1.0
    assert _cell(result, "text::orthography_profile::unicode_category=Cc::count") == 1.0
    assert math.isclose(_cell(result, "text::orthography_profile::unicode_category=Ll::per_character"), 2.0 / 7.0)
    assert _cell(result, "text::orthography_profile::script=latin::count") == 3.0
    assert math.isclose(_cell(result, "text::orthography_profile::script=latin::per_character"), 3.0 / 7.0)
    assert _cell(result, "text::orthography_profile::script=latin::per_alpha") == 1.0
    assert _cell(result, "text::orthography_profile::script=greek::count") == 0.0
    assert math.isclose(_cell(result, "text::orthography_profile::latin_letter=a::per_alpha"), 2.0 / 3.0)
    assert math.isclose(_cell(result, "text::orthography_profile::latin_letter=a::per_character"), 2.0 / 7.0)
    assert _cell(result, "text::orthography_profile::latin_letter=e::per_alpha") == 0.0
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, OrthographyCodepointSidecar)
    assert sidecar.document_id == "doc-unicode"
    assert sidecar.schema_version == "orthography_codepoint_sidecar_v1"
    assert sidecar.normalization == "raw_unicode_codepoints"
    assert sidecar.codepoint_count == 7
    assert sidecar.unique_codepoint_count == 7
    assert sidecar.records[0].character_index == 0
    assert sidecar.records[0].character == "A"
    assert sidecar.records[0].codepoint == 65
    assert sidecar.records[0].unicode_category == "Lu"
    assert sidecar.records[0].unicode_name == "LATIN CAPITAL LETTER A"
    assert sidecar.records[-1].character == "\n"
    assert sidecar.records[-1].unicode_category == "Cc"
    assert sidecar.records[-1].unicode_name is None
    spec = transformer.registry_.by_name("text::orthography_profile::unicode_category=Ll::per_character")
    assert spec.topic_dependence.value == "mostly_topic_independent"
    assert "python_unicodedata" in spec.provenance


def test_orthography_profile_has_golden_non_latin_script_inventory_values() -> None:
    config = english_preprocessing_config()
    text = "AαБשمकあカ한漢ก!"
    x = pd.DataFrame({"text": [text]}, index=["doc-script"])
    transformer = OrthographyProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-script"]
    assert _cell(result, "text::orthography_profile::codepoint_count") == 12.0
    assert _cell(result, "text::orthography_profile::character_class=unicode_letter::count") == 11.0
    for script_id in (
        "latin",
        "greek",
        "cyrillic",
        "hebrew",
        "arabic",
        "devanagari",
        "hiragana",
        "katakana",
        "hangul",
        "cjk_unified_ideograph",
        "thai",
    ):
        assert _cell(result, f"text::orthography_profile::script={script_id}::count") == 1.0
        assert math.isclose(_cell(result, f"text::orthography_profile::script={script_id}::per_character"), 1.0 / 12.0)
        assert math.isclose(_cell(result, f"text::orthography_profile::script={script_id}::per_alpha"), 1.0 / 11.0)
    spec = transformer.registry_.by_name("text::orthography_profile::script=cjk_unified_ideograph::per_alpha")
    assert spec.normalization == "per_alphabetic_character_ratio"
    assert "script count divided by total alphabetic character count" in spec.formula_or_rule


def test_orthography_profile_empty_text_uses_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = OrthographyProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::orthography_profile::codepoint_count") == 0.0
    assert _cell(result, "text::orthography_profile::unique_codepoint_count") == 0.0
    assert math.isnan(_cell(result, "text::orthography_profile::codepoint_mean"))
    assert _cell(result, "text::orthography_profile::unicode_category=Ll::count") == 0.0
    assert math.isnan(_cell(result, "text::orthography_profile::unicode_category=Ll::per_character"))
    assert _cell(result, "text::orthography_profile::script=latin::count") == 0.0
    assert math.isnan(_cell(result, "text::orthography_profile::script=latin::per_character"))
    assert math.isnan(_cell(result, "text::orthography_profile::script=latin::per_alpha"))
    assert math.isnan(_cell(result, "text::orthography_profile::latin_letter=a::per_alpha"))
    assert transformer.last_sidecars_[0].document_id == "0"
    assert transformer.last_sidecars_[0].codepoint_count == 0
    assert transformer.last_sidecars_[0].records == ()
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert "empty_text" in reasons
    assert "zero_alphabetic_characters" in reasons


def test_orthography_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["Aaé9!€\n", "Plain"]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = OrthographyProfileTransformer(text_column="text", config=config, output="pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(OrthographyProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = OrthographyProfileTransformer(text_column="text", config=config, output="sparse").fit_transform(x, None)
    numpy_result = OrthographyProfileTransformer(text_column="text", config=config, output="numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(OrthographyProfileTransformer("text", config, "pandas"),), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
    assert len(extractor.last_sidecars_) == 1
    assert extractor.last_sidecars_[0].block_name == "OrthographyProfileTransformer"
    sidecars = cast(tuple[OrthographyCodepointSidecar, OrthographyCodepointSidecar], extractor.last_sidecars_[0].sidecars)
    assert sidecars[0].document_id == "first"
    assert sidecars[1].document_id == "second"


def test_expanded_script_inventory_counts_new_scripts() -> None:
    names = orthography_profile_feature_names()
    for expected in ("armenian", "georgian", "bengali", "tamil", "ethiopic"):
        assert f"text::orthography_profile::script={expected}::count" in names

    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["aԱb"]}, index=["arm"])  # U+0531 ARMENIAN CAPITAL LETTER AYB
    transformer = OrthographyProfileTransformer(text_column="text", config=config, output="pandas")
    result = _as_frame(transformer.fit_transform(x, None))
    assert _cell(result, "text::orthography_profile::script=armenian::count") == 1.0
