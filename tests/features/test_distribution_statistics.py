"""Tests for deterministic distribution-statistic feature blocks."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import DistributionStatisticsTransformer, FeatureExtractor, english_preprocessing_config


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_distribution_statistics_have_golden_values_for_many_item_fixture() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["A bb ccc!\n\nD eeee."]}, index=["doc-a"])
    transformer = DistributionStatisticsTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-a"]
    assert tuple(result.columns) == tuple(transformer.get_feature_names_out(None).tolist())
    assert _cell(result, "text::distribution::word_characters::count") == 5.0
    assert _cell(result, "text::distribution::word_characters::mean") == 2.2
    assert math.isclose(_cell(result, "text::distribution::word_characters::sample_std"), math.sqrt(1.7))
    assert _cell(result, "text::distribution::word_characters::sample_variance") == 1.7
    assert _cell(result, "text::distribution::word_characters::min") == 1.0
    assert _cell(result, "text::distribution::word_characters::max") == 4.0
    assert _cell(result, "text::distribution::word_characters::p10") == 1.0
    assert _cell(result, "text::distribution::word_characters::p25") == 1.0
    assert _cell(result, "text::distribution::word_characters::p50") == 2.0
    assert _cell(result, "text::distribution::word_characters::p75") == 3.0
    assert _cell(result, "text::distribution::word_characters::p90") == 3.6
    assert math.isclose(_cell(result, "text::distribution::word_characters::skewness"), 0.36317347441943004)
    assert math.isclose(_cell(result, "text::distribution::word_characters::excess_kurtosis"), -1.3719723183391002)
    assert math.isclose(_cell(result, "text::distribution::word_characters::shannon_entropy"), 1.3321790402101223)
    assert _cell(result, "text::distribution::sentence_tokens::count") == 2.0
    assert _cell(result, "text::distribution::sentence_tokens::mean") == 2.5
    assert _cell(result, "text::distribution::sentence_tokens::skewness") == 0.0
    assert _cell(result, "text::distribution::sentence_tokens::excess_kurtosis") == -2.0
    assert math.isclose(_cell(result, "text::distribution::sentence_tokens::shannon_entropy"), math.log(2.0))
    assert _cell(result, "text::distribution::sentence_characters::max") == 9.0
    assert _cell(result, "text::distribution::sentence_syllables::count") == 2.0
    assert _cell(result, "text::distribution::sentence_syllables::mean") == 2.5
    assert math.isclose(_cell(result, "text::distribution::sentence_syllables::sample_std"), math.sqrt(0.5))
    assert _cell(result, "text::distribution::sentence_syllables::max") == 3.0
    assert _cell(result, "text::distribution::paragraph_tokens::mean") == 2.5
    assert _cell(result, "text::distribution::paragraph_characters::count") == 2.0
    assert _cell(result, "text::distribution::paragraph_characters::mean") == 8.0
    assert _cell(result, "text::distribution::paragraph_characters::max") == 9.0
    assert _cell(result, "text::distribution::paragraph_syllables::count") == 2.0
    assert _cell(result, "text::distribution::paragraph_syllables::mean") == 2.5
    assert _cell(result, "text::distribution::paragraph_syllables::skewness") == 0.0
    assert _cell(result, "text::distribution::paragraph_syllables::excess_kurtosis") == -2.0
    assert math.isclose(_cell(result, "text::distribution::paragraph_syllables::shannon_entropy"), math.log(2.0))
    assert _cell(result, "text::distribution::paragraph_sentences::mean") == 1.0
    assert _cell(result, "text::distribution::line_characters::count") == 3.0
    assert _cell(result, "text::distribution::line_characters::min") == 0.0
    assert transformer.registry_.by_name("text::distribution::word_characters::sample_std").topic_dependence.value == "mixed"
    assert "std_type=sample" in transformer.registry_.by_name("text::distribution::word_characters::sample_std").provenance
    assert (
        "moment_statistics=population_central_moments"
        in transformer.registry_.by_name("text::distribution::word_characters::skewness").provenance
    )
    assert transformer.registry_.by_name("text::distribution::word_characters::shannon_entropy").formula_or_rule.endswith(
        "natural logarithms"
    )


def test_distribution_statistics_cover_one_two_empty_and_punctuation_sequence_cases() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["solo", "a bb", "", "Hi!!! Ok??", "aa bb"]})
    transformer = DistributionStatisticsTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::distribution::word_characters::count", row=0) == 1.0
    assert math.isnan(_cell(result, "text::distribution::word_characters::sample_std", row=0))
    assert math.isnan(_cell(result, "text::distribution::word_characters::skewness", row=0))
    assert _cell(result, "text::distribution::word_characters::shannon_entropy", row=0) == 0.0
    assert transformer.last_diagnostics_[0][0].reason == "insufficient_values_for_sample_statistic"
    assert any(diagnostic.reason == "insufficient_values_for_moment_statistic" for diagnostic in transformer.last_diagnostics_[0])
    assert _cell(result, "text::distribution::word_characters::count", row=1) == 2.0
    assert _cell(result, "text::distribution::word_characters::sample_variance", row=1) == 0.5
    assert math.isclose(_cell(result, "text::distribution::word_characters::sample_std", row=1), math.sqrt(0.5))
    assert _cell(result, "text::distribution::word_characters::skewness", row=1) == 0.0
    assert _cell(result, "text::distribution::word_characters::excess_kurtosis", row=1) == -2.0
    assert math.isclose(_cell(result, "text::distribution::word_characters::shannon_entropy", row=1), math.log(2.0))
    assert _cell(result, "text::distribution::word_characters::count", row=2) == 0.0
    assert math.isnan(_cell(result, "text::distribution::word_characters::mean", row=2))
    assert math.isnan(_cell(result, "text::distribution::word_characters::shannon_entropy", row=2))
    assert any(diagnostic.reason == "zero_tokens" for diagnostic in transformer.last_diagnostics_[2])
    assert _cell(result, "text::distribution::sentence_syllables::count", row=2) == 0.0
    assert math.isnan(_cell(result, "text::distribution::sentence_syllables::mean", row=2))
    assert _cell(result, "text::distribution::paragraph_characters::count", row=2) == 0.0
    assert math.isnan(_cell(result, "text::distribution::paragraph_characters::mean", row=2))
    assert _cell(result, "text::distribution::paragraph_syllables::count", row=2) == 0.0
    assert math.isnan(_cell(result, "text::distribution::paragraph_syllables::mean", row=2))
    assert any(diagnostic.reason == "zero_sentences" for diagnostic in transformer.last_diagnostics_[2])
    assert any(diagnostic.reason == "zero_paragraphs" for diagnostic in transformer.last_diagnostics_[2])
    assert _cell(result, "text::distribution::punctuation_sequence_characters::count", row=3) == 2.0
    assert _cell(result, "text::distribution::punctuation_sequence_characters::mean", row=3) == 2.5
    assert _cell(result, "text::distribution::punctuation_sequence_characters::max", row=3) == 3.0
    assert math.isnan(_cell(result, "text::distribution::word_characters::skewness", row=4))
    assert math.isnan(_cell(result, "text::distribution::word_characters::excess_kurtosis", row=4))
    assert _cell(result, "text::distribution::word_characters::shannon_entropy", row=4) == 0.0
    assert any(diagnostic.reason == "zero_variance_distribution" for diagnostic in transformer.last_diagnostics_[4])


def test_distribution_statistics_support_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["A bb ccc!", "D eeee."]}, index=["first", "second"])
    original = x.copy(deep=True)
    pandas_transformer = DistributionStatisticsTransformer(text_column="text", config=config, output="pandas")

    pandas_result = _as_frame(pandas_transformer.fit_transform(x, None))
    loaded = cast(DistributionStatisticsTransformer, pickle.loads(pickle.dumps(pandas_transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = DistributionStatisticsTransformer(text_column="text", config=config, output="sparse").fit_transform(x, None)
    numpy_result = DistributionStatisticsTransformer(text_column="text", config=config, output="numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(DistributionStatisticsTransformer("text", config, "pandas"),), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
