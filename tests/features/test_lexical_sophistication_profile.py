"""Tests for lexical sophistication frequency-band profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    FeatureExtractor,
    FrequencyBandRecord,
    LexicalSophisticationProfileTransformer,
    LexicalSophisticationSidecar,
    english_preprocessing_config,
    lexical_sophistication_profile_feature_names,
    lexical_sophistication_profile_transformer,
    load_frequency_bands,
)

LEXICAL_SOPHISTICATION_TEXT = "The and people science beautiful quiescent unknown unknown."


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_frequency_band_resource_has_required_metadata_order_and_bands() -> None:
    resource = load_frequency_bands("frequency_bands")

    assert resource.name == "frequency_bands"
    assert resource.lexicon_id == "frequency_bands_en_seed_v1"
    assert resource.language == "en"
    assert resource.version == "1.0.0"
    assert resource.license_note != ""
    assert resource.normalization == "lowercase_token_match"
    assert resource.tokens()[:4] == ("the", "and", "people", "science")
    assert resource.bands() == ("high_frequency", "mid_frequency", "low_frequency", "rare_reference")
    assert resource.entry_by_token()["quiescent"].band == "rare_reference"
    assert resource.entry_by_token()["beautiful"].frequency_per_million == 35.0


def test_lexical_sophistication_profile_has_golden_band_and_out_of_reference_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [LEXICAL_SOPHISTICATION_TEXT]}, index=["doc-lexical"])
    transformer = lexical_sophistication_profile_transformer("text", config, "frequency_bands", "pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-lexical"]
    assert result.shape[1] == 20
    assert tuple(result.columns) == lexical_sophistication_profile_feature_names(transformer.resource_)
    assert _cell(result, "text::lexical_sophistication_profile::band=high_frequency::token_count") == 2.0
    assert _cell(result, "text::lexical_sophistication_profile::band=high_frequency::type_count") == 2.0
    assert _cell(result, "text::lexical_sophistication_profile::band=mid_frequency::token_count") == 2.0
    assert _cell(result, "text::lexical_sophistication_profile::band=mid_frequency::type_count") == 2.0
    assert _cell(result, "text::lexical_sophistication_profile::band=low_frequency::token_count") == 1.0
    assert _cell(result, "text::lexical_sophistication_profile::band=low_frequency::type_count") == 1.0
    assert _cell(result, "text::lexical_sophistication_profile::band=rare_reference::token_count") == 1.0
    assert _cell(result, "text::lexical_sophistication_profile::band=rare_reference::type_count") == 1.0
    assert _cell(result, "text::lexical_sophistication_profile::band=out_of_reference::token_count") == 2.0
    assert _cell(result, "text::lexical_sophistication_profile::band=out_of_reference::type_count") == 1.0
    assert math.isclose(_cell(result, "text::lexical_sophistication_profile::band=high_frequency::token_ratio"), 0.25)
    assert math.isclose(_cell(result, "text::lexical_sophistication_profile::band=high_frequency::type_ratio"), 2.0 / 7.0)
    assert math.isclose(_cell(result, "text::lexical_sophistication_profile::band=out_of_reference::token_ratio"), 0.25)
    assert math.isclose(_cell(result, "text::lexical_sophistication_profile::band=out_of_reference::type_ratio"), 1.0 / 7.0)
    assert transformer.last_diagnostics_[0] == ()
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, LexicalSophisticationSidecar)
    assert sidecar.document_id == "doc-lexical"
    assert sidecar.schema_version == "lexical_sophistication_frequency_bands_v1"
    assert sidecar.resource_id == "frequency_bands_en_seed_v1"
    assert sidecar.language == "en"
    assert sidecar.version == "1.0.0"
    assert sidecar.normalization == "lowercase_token_match"
    assert sidecar.token_count == 8
    assert sidecar.type_count == 7
    assert sidecar.in_reference_token_count == 6
    assert sidecar.out_of_reference_token_count == 2
    assert sidecar.records == (
        FrequencyBandRecord(
            token="and",
            token_count=1,
            band="high_frequency",
            in_reference=True,
            rank=2,
            frequency_per_million=25000.0,
        ),
        FrequencyBandRecord(
            token="beautiful",
            token_count=1,
            band="low_frequency",
            in_reference=True,
            rank=3500,
            frequency_per_million=35.0,
        ),
        FrequencyBandRecord(
            token="people",
            token_count=1,
            band="mid_frequency",
            in_reference=True,
            rank=500,
            frequency_per_million=450.0,
        ),
        FrequencyBandRecord(
            token="quiescent",
            token_count=1,
            band="rare_reference",
            in_reference=True,
            rank=12000,
            frequency_per_million=1.7,
        ),
        FrequencyBandRecord(
            token="science",
            token_count=1,
            band="mid_frequency",
            in_reference=True,
            rank=900,
            frequency_per_million=240.0,
        ),
        FrequencyBandRecord(
            token="the",
            token_count=1,
            band="high_frequency",
            in_reference=True,
            rank=1,
            frequency_per_million=50000.0,
        ),
        FrequencyBandRecord(
            token="unknown",
            token_count=2,
            band="out_of_reference",
            in_reference=False,
            rank=None,
            frequency_per_million=None,
        ),
    )
    spec = transformer.registry_.by_name("text::lexical_sophistication_profile::band=rare_reference::type_ratio")
    assert spec.topic_dependence.value == "topic_sensitive"
    assert "lexicon_id=frequency_bands_en_seed_v1" in spec.provenance
    assert "sidecar_schema=lexical_sophistication_frequency_bands_v1" in spec.provenance


def test_lexical_sophistication_profile_empty_text_uses_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = lexical_sophistication_profile_transformer("text", config, "frequency_bands", "pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::lexical_sophistication_profile::band=high_frequency::token_count") == 0.0
    assert _cell(result, "text::lexical_sophistication_profile::band=high_frequency::type_count") == 0.0
    assert math.isnan(_cell(result, "text::lexical_sophistication_profile::band=high_frequency::token_ratio"))
    assert math.isnan(_cell(result, "text::lexical_sophistication_profile::band=high_frequency::type_ratio"))
    assert transformer.last_sidecars_[0].document_id == "0"
    assert transformer.last_sidecars_[0].token_count == 0
    assert transformer.last_sidecars_[0].type_count == 0
    assert transformer.last_sidecars_[0].records == ()
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert reasons == {"zero_tokens", "zero_types"}


def test_lexical_sophistication_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [LEXICAL_SOPHISTICATION_TEXT, "plain words"]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = lexical_sophistication_profile_transformer("text", config, "frequency_bands", "pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(LexicalSophisticationProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = lexical_sophistication_profile_transformer("text", config, "frequency_bands", "sparse").fit_transform(x, None)
    numpy_result = lexical_sophistication_profile_transformer("text", config, "frequency_bands", "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(
        blocks=(lexical_sophistication_profile_transformer("text", config, "frequency_bands", "pandas"),),
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
    assert len(extractor.last_sidecars_) == 1
    assert extractor.last_sidecars_[0].block_name == "LexicalSophisticationProfileTransformer"
    sidecars = cast(tuple[LexicalSophisticationSidecar, LexicalSophisticationSidecar], extractor.last_sidecars_[0].sidecars)
    assert sidecars[0].document_id == "first"
    assert sidecars[0].out_of_reference_token_count == 2
    assert sidecars[1].document_id == "second"
    assert sidecars[1].records == (
        FrequencyBandRecord(
            token="plain",
            token_count=1,
            band="out_of_reference",
            in_reference=False,
            rank=None,
            frequency_per_million=None,
        ),
        FrequencyBandRecord(
            token="words",
            token_count=1,
            band="out_of_reference",
            in_reference=False,
            rank=None,
            frequency_per_million=None,
        ),
    )
