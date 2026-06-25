"""Tests for lexical richness frequency-spectrum features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    FeatureExtractor,
    FrequencySpectrumBin,
    LexicalRichnessSpectrumTransformer,
    english_preprocessing_config,
    lexical_richness_spectrum_feature_names,
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_lexical_richness_spectrum_has_golden_ratios_bins_and_sidecar_schema() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["alpha alpha alpha beta beta gamma delta"]}, index=["doc-spectrum"])
    transformer = LexicalRichnessSpectrumTransformer(text_column="text", config=config, max_frequency_bin=4, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-spectrum"]
    assert result.shape[1] == 20
    assert tuple(result.columns) == lexical_richness_spectrum_feature_names(4)
    assert _cell(result, "text::lexical_richness_spectrum::hapax::count") == 2.0
    assert _cell(result, "text::lexical_richness_spectrum::hapax::types_ratio") == 0.5
    assert math.isclose(_cell(result, "text::lexical_richness_spectrum::hapax::tokens_ratio"), 2.0 / 7.0)
    assert math.isclose(_cell(result, "text::lexical_richness_spectrum::hapax::per_1000_tokens"), 2000.0 / 7.0)
    assert _cell(result, "text::lexical_richness_spectrum::dis_legomena::count") == 1.0
    assert _cell(result, "text::lexical_richness_spectrum::dis_legomena::types_ratio") == 0.25
    assert math.isclose(_cell(result, "text::lexical_richness_spectrum::dis_legomena::tokens_ratio"), 2.0 / 7.0)
    assert math.isclose(_cell(result, "text::lexical_richness_spectrum::dis_legomena::per_1000_tokens"), 1000.0 / 7.0)
    assert _cell(result, "text::lexical_richness_spectrum::frequency_bin=1::type_count") == 2.0
    assert _cell(result, "text::lexical_richness_spectrum::frequency_bin=1::types_ratio") == 0.5
    assert math.isclose(_cell(result, "text::lexical_richness_spectrum::frequency_bin=1::tokens_ratio"), 2.0 / 7.0)
    assert _cell(result, "text::lexical_richness_spectrum::frequency_bin=2::type_count") == 1.0
    assert _cell(result, "text::lexical_richness_spectrum::frequency_bin=2::types_ratio") == 0.25
    assert math.isclose(_cell(result, "text::lexical_richness_spectrum::frequency_bin=2::tokens_ratio"), 2.0 / 7.0)
    assert _cell(result, "text::lexical_richness_spectrum::frequency_bin=3::type_count") == 1.0
    assert _cell(result, "text::lexical_richness_spectrum::frequency_bin=3::types_ratio") == 0.25
    assert math.isclose(_cell(result, "text::lexical_richness_spectrum::frequency_bin=3::tokens_ratio"), 3.0 / 7.0)
    assert _cell(result, "text::lexical_richness_spectrum::frequency_bin=4::type_count") == 0.0

    sidecar = transformer.last_sidecars_[0]
    assert sidecar.document_id == "doc-spectrum"
    assert sidecar.token_count == 7
    assert sidecar.type_count == 4
    assert sidecar.max_frequency_bin == 4
    assert sidecar.overflow_type_count == 0
    assert sidecar.warnings == ("short_text_unstable",)
    assert sidecar.bins == (
        FrequencySpectrumBin(frequency=1, type_count=2, types=("delta", "gamma")),
        FrequencySpectrumBin(frequency=2, type_count=1, types=("beta",)),
        FrequencySpectrumBin(frequency=3, type_count=1, types=("alpha",)),
    )
    spec = transformer.registry_.by_name("text::lexical_richness_spectrum::frequency_bin=3::tokens_ratio")
    assert spec.topic_dependence.value == "mixed"
    assert "frequency_spectrum_v1" in spec.provenance


def test_lexical_richness_spectrum_empty_text_uses_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = LexicalRichnessSpectrumTransformer(text_column="text", config=config, max_frequency_bin=2, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::lexical_richness_spectrum::hapax::count") == 0.0
    assert _cell(result, "text::lexical_richness_spectrum::frequency_bin=1::type_count") == 0.0
    assert math.isnan(_cell(result, "text::lexical_richness_spectrum::hapax::types_ratio"))
    assert math.isnan(_cell(result, "text::lexical_richness_spectrum::hapax::tokens_ratio"))
    assert math.isnan(_cell(result, "text::lexical_richness_spectrum::frequency_bin=1::types_ratio"))
    assert math.isnan(_cell(result, "text::lexical_richness_spectrum::frequency_bin=1::tokens_ratio"))
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert reasons == {"zero_tokens", "zero_types"}
    assert transformer.last_sidecars_[0].token_count == 0
    assert transformer.last_sidecars_[0].type_count == 0
    assert transformer.last_sidecars_[0].bins == ()


def test_lexical_richness_spectrum_supports_output_modes_serialization_sidecars_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["one two two three three three", "solo"]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = LexicalRichnessSpectrumTransformer(text_column="text", config=config, max_frequency_bin=3, output="pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(LexicalRichnessSpectrumTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = LexicalRichnessSpectrumTransformer("text", config, 3, "sparse").fit_transform(x, None)
    numpy_result = LexicalRichnessSpectrumTransformer("text", config, 3, "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(LexicalRichnessSpectrumTransformer("text", config, 3, "pandas"),), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
    assert len(extractor.last_sidecars_) == 1
    assert extractor.last_sidecars_[0].block_index == 0
    assert extractor.last_sidecars_[0].block_name == "LexicalRichnessSpectrumTransformer"
    assert len(extractor.last_sidecars_[0].sidecars) == 2


def test_lexical_richness_spectrum_rejects_non_positive_bin_configuration() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["alpha"]})
    transformer = LexicalRichnessSpectrumTransformer(text_column="text", config=config, max_frequency_bin=0, output="pandas")

    try:
        transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "max_frequency_bin must be positive"
    else:
        raise AssertionError("Expected max_frequency_bin validation to fail")
