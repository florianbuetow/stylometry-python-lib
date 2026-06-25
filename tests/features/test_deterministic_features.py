"""Tests for deterministic stylometry features."""

from __future__ import annotations

import math
import pickle
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from stylometry_python_lib import (
    DeterministicStylometryTransformer,
    FeatureExtractor,
    MostFrequentWordsTransformer,
    NGramStylometryTransformer,
    OptionalDependencyError,
    english_preprocessing_config,
    function_word_frequency_transformer,
    letter_frequency_transformer,
    llm_annotation_transformer,
    parser_backed_transformer,
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _numeric_cell(frame: pd.DataFrame, column: str) -> float:
    values = frame[column].to_numpy(dtype=float)
    return float(values[0])


def test_deterministic_feature_values_and_metadata_are_stable() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["I can go. I can go."]})
    transformer = DeterministicStylometryTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _numeric_cell(result, "text::counts::token_count") == 6.0
    assert _numeric_cell(result, "text::counts::type_count") == 3.0
    assert _numeric_cell(result, "text::counts::sentence_count") == 2.0
    assert _numeric_cell(result, "text::lexical_richness::ttr") == 0.5
    assert math.isclose(_numeric_cell(result, "text::stance::first_person_ratio"), 2.0 / 6.0)
    assert math.isclose(_numeric_cell(result, "text::stance::modal_verb_ratio"), 2.0 / 6.0)
    assert _numeric_cell(result, "text::orthography::punctuation_count") == 2.0
    names = transformer.get_feature_names_out(None)
    assert list(names) == list(transformer.registry_.names())
    assert len(names) == result.shape[1]
    assert transformer.registry_.by_name("text::lexical_richness::ttr").topic_dependence.value == "mixed"


def test_empty_text_uses_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = DeterministicStylometryTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _numeric_cell(result, "text::counts::token_count") == 0.0
    assert math.isnan(_numeric_cell(result, "text::lexical_richness::ttr"))
    diagnostics = transformer.last_diagnostics_[0]
    diagnostic_names = {diagnostic.feature_name for diagnostic in diagnostics}
    assert "text::lexical_richness::ttr" in diagnostic_names
    reasons = {diagnostic.reason for diagnostic in diagnostics}
    assert "zero_tokens" in reasons


def test_transformer_preserves_rows_and_does_not_mutate_input() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["I can go. I can go.", "You should stay. You should stay."]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = DeterministicStylometryTransformer(text_column="text", config=config, output="numpy")

    result = transformer.fit_transform(x, None)
    pandas_result = _as_frame(DeterministicStylometryTransformer(text_column="text", config=config, output="pandas").fit_transform(x, None))
    sparse_result = DeterministicStylometryTransformer(text_column="text", config=config, output="sparse").fit_transform(x, None)

    pd.testing.assert_frame_equal(x, original)
    assert result.shape[0] == x.shape[0]
    assert result.shape[1] == len(transformer.get_feature_names_out(None))
    assert pandas_result.index.tolist() == ["first", "second"]
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == result.shape


def test_expanded_readability_formulas_have_golden_values_and_threshold_diagnostics() -> None:
    config = english_preprocessing_config()
    sentence = "the cat sat on the mat and dog ran home."
    long_text = " ".join(sentence for _ in range(15))
    x = pd.DataFrame({"text": [long_text]}, index=["readability-long"])
    transformer = DeterministicStylometryTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["readability-long"]
    assert _numeric_cell(result, "text::counts::token_count") == 150.0
    assert _numeric_cell(result, "text::counts::sentence_count") == 15.0
    assert math.isclose(_numeric_cell(result, "text::readability::dale_chall"), 0.496)
    assert math.isclose(_numeric_cell(result, "text::readability::forcast"), 5.0)
    assert math.isclose(_numeric_cell(result, "text::readability::linsear_write"), 4.0)
    assert math.isclose(_numeric_cell(result, "text::readability::lix"), 10.0)
    assert (
        "dale_chall_easy_words=dale_chall_easy_words_en_seed_v1"
        in transformer.registry_.by_name("text::readability::dale_chall").provenance
    )


def test_lexical_richness_formula_variants_have_golden_values_and_true_vocd_metadata() -> None:
    config = english_preprocessing_config()
    base_tokens = [
        "alpha",
        "beta",
        "gamma",
        "delta",
        "epsilon",
        "alpha",
        "beta",
        "gamma",
        "delta",
        "alpha",
        "beta",
        "gamma",
        "alpha",
        "beta",
        "alpha",
        "zeta",
        "eta",
        "theta",
        "iota",
        "kappa",
        "lambda",
        "mu",
        "nu",
        "xi",
        "omicron",
        "pi",
        "rho",
        "sigma",
        "tau",
        "upsilon",
    ]
    text = " ".join(base_tokens * 3)
    x = pd.DataFrame({"text": [text]}, index=["richness-long"])
    transformer = DeterministicStylometryTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["richness-long"]
    assert _numeric_cell(result, "text::counts::token_count") == 90.0
    assert _numeric_cell(result, "text::counts::type_count") == 20.0
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::ttr"), 0.2222222222222222)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::cttr"), 1.4907119849998598)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::herdan_c"), 0.6657464410787219)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::msttr"), 0.45)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::mattr"), 0.4000000000000001)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::mtld"), 20.25)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::hdd"), 0.4195722979914526)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::vocd_d"), 6.341885895578644)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::yules_k"), 666.6666666666666)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::honore_r"), 449.9809670330265)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::guiraud_r"), 2.1081851067789197)
    assert _numeric_cell(result, "text::lexical_richness::sichel_s") == 0.0
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::simpson_d"), 0.06741573033707865)
    assert math.isclose(_numeric_cell(result, "text::lexical_richness::renyi_entropy_alpha_2"), 2.5538995212749516)
    diagnostic_names = {diagnostic.feature_name for diagnostic in transformer.last_diagnostics_[0]}
    assert "text::lexical_richness::vocd_d" not in diagnostic_names
    spec = transformer.registry_.by_name("text::lexical_richness::vocd_d")
    assert spec.formula_or_rule == "vocd-D hash-sampled TTR curve fit over 100 deterministic samples for each size 35 through 50"
    assert "vocd_sampling=v1" in spec.provenance
    assert "sample_sizes=35-50" in spec.provenance
    assert "hash_seed=stylometry_python_lib_vocd_d_v1" in spec.provenance
    assert "fit=golden_section_curve_fit_v1" in spec.provenance


def test_vocd_d_uses_explicit_threshold_and_unique_sample_singularity_diagnostics() -> None:
    config = english_preprocessing_config()
    short_text = "alpha beta gamma alpha beta gamma alpha beta gamma"
    unique_text = " ".join(f"unique{chr(97 + index // 26)}{chr(97 + index % 26)}" for index in range(55))
    x = pd.DataFrame({"text": [short_text, unique_text]}, index=["short", "unique"])
    transformer = DeterministicStylometryTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    vocd_values = result["text::lexical_richness::vocd_d"].to_numpy(dtype=float)
    assert math.isnan(float(vocd_values[0]))
    assert math.isnan(float(vocd_values[1]))
    short_reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    unique_reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[1]}
    assert "below_vocd_50_token_threshold" in short_reasons
    assert "all_samples_unique_vocd_singularity" in unique_reasons


def test_readability_short_text_thresholds_are_explicitly_undefined() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["the cat sat."]})
    transformer = DeterministicStylometryTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert math.isnan(_numeric_cell(result, "text::readability::smog"))
    assert math.isnan(_numeric_cell(result, "text::readability::forcast"))
    assert math.isnan(_numeric_cell(result, "text::readability::linsear_write"))
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert "below_smog_3_sentence_threshold" in reasons
    assert "below_forcast_150_word_threshold" in reasons
    assert "below_linsear_write_100_word_threshold" in reasons


def test_get_feature_names_out_requires_fit() -> None:
    config = english_preprocessing_config()
    transformer = DeterministicStylometryTransformer(text_column="text", config=config, output="pandas")

    with pytest.raises(NotFittedError):
        transformer.get_feature_names_out(None)


def test_sparse_ngram_transformer_has_stable_names_and_sparse_output() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["I can go. I can go.", "You can go too."]})
    transformer = NGramStylometryTransformer(
        text_column="text",
        config=config,
        analyzer="word",
        ngram_range=(1, 2),
        max_features=5,
        output="sparse",
    )

    matrix = transformer.fit_transform(x, None)

    assert sparse.issparse(matrix)
    assert matrix.shape == (2, 5)
    names = transformer.get_feature_names_out(None).tolist()
    assert names == sorted(names, key=names.index)
    assert any(name.startswith("text::word_ngram::gram=can") for name in names)


def test_fixed_and_fitted_frequency_blocks_cover_research_taxonomy() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["The cat and the dog.", "The dog can go."]}, index=["first", "second"])
    original = x.copy(deep=True)
    function_words = function_word_frequency_transformer(text_column="text", config=config, output="pandas")
    letters = letter_frequency_transformer(text_column="text", config=config, output="pandas")
    most_frequent = MostFrequentWordsTransformer(text_column="text", config=config, max_features=3, output="pandas")

    function_frame = _as_frame(function_words.fit_transform(x, None))
    letter_frame = _as_frame(letters.fit_transform(x, None))
    frequent_frame = _as_frame(most_frequent.fit_transform(x, None))
    sparse_function_frame = function_word_frequency_transformer(text_column="text", config=config, output="sparse").fit_transform(x, None)
    sparse_letter_frame = letter_frequency_transformer(text_column="text", config=config, output="sparse").fit_transform(x, None)
    sparse_frequent_frame = MostFrequentWordsTransformer(text_column="text", config=config, max_features=3, output="sparse").fit_transform(
        x,
        None,
    )

    pd.testing.assert_frame_equal(x, original)
    assert function_frame.index.tolist() == ["first", "second"]
    assert letter_frame.index.tolist() == ["first", "second"]
    assert frequent_frame.index.tolist() == ["first", "second"]
    assert math.isclose(_numeric_cell(function_frame, "text::function_word_frequency::token=the"), 2.0 / 5.0)
    assert _numeric_cell(letter_frame, "text::letter_frequency::char=t") > 0.0
    assert frequent_frame.shape == (2, 3)
    assert sparse.issparse(sparse_function_frame)
    assert sparse_function_frame.shape == function_frame.shape
    assert sparse.issparse(sparse_letter_frame)
    assert sparse_letter_frame.shape == letter_frame.shape
    assert sparse.issparse(sparse_frequent_frame)
    assert sparse_frequent_frame.shape == frequent_frame.shape
    assert function_words.registry_.by_name("text::function_word_frequency::token=the").stability_status.value == "deterministic"
    assert most_frequent.registry_.by_name("text::most_frequent_words::token=the").topic_dependence.value == "topic_sensitive"
    assert most_frequent.registry_.by_name("text::most_frequent_words::token=the").stability_status.value == "statistical_fit_dependent"


def test_feature_extractor_combines_dense_sparse_and_serializes() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["I can go. I can go.", "You should stay. You should stay."]})
    extractor = FeatureExtractor(
        blocks=(
            DeterministicStylometryTransformer(text_column="text", config=config, output="pandas"),
            NGramStylometryTransformer(
                text_column="text",
                config=config,
                analyzer="function_word",
                ngram_range=(1, 1),
                max_features=4,
                output="sparse",
            ),
        ),
        output="pandas",
    )

    frame = _as_frame(extractor.fit_transform(x, None))
    loaded = pickle.loads(pickle.dumps(extractor))
    loaded_frame = _as_frame(cast(FeatureExtractor, loaded).transform(x))

    assert isinstance(frame, pd.DataFrame)
    assert frame.shape[0] == 2
    assert list(frame.columns) == loaded.get_feature_names_out(None).tolist()
    pd.testing.assert_frame_equal(frame, loaded_frame)


def test_feature_extractor_supports_sparse_and_pipeline_usage() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["I can go. I can go.", "You should stay. You should stay.", "We can go. We can go."]})
    y = np.asarray([0, 1, 0])
    extractor = FeatureExtractor(
        blocks=(
            NGramStylometryTransformer(
                text_column="text",
                config=config,
                analyzer="word",
                ngram_range=(1, 1),
                max_features=6,
                output="sparse",
            ),
        ),
        output="sparse",
    )
    pipeline: Any = Pipeline([("features", extractor), ("model", LogisticRegression())])

    pipeline.fit(x, y)
    predictions = np.asarray(pipeline.predict(x), dtype=int)

    assert predictions.shape == (3,)
    transformed = extractor.fit_transform(x, y)
    assert sparse.issparse(transformed)


def test_optional_parser_and_llm_features_fail_fast_with_metadata() -> None:
    parser = parser_backed_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=None,
    )
    llm = llm_annotation_transformer(
        provider="example",
        model="judge",
        version="1",
        prompt_version="v1",
        response_schema="tone_schema",
        fake_annotations=None,
    )

    parser_names = {spec.name for spec in parser.feature_specs()}
    llm_names = {spec.name for spec in llm.feature_specs()}
    assert "text::syntax::pos_frequency" in parser_names
    assert "text::syntax::passive_voice_frequency" in parser_names
    assert "text::llm::tone" in llm_names
    assert "text::llm::embedding" in llm_names
    assert parser.feature_specs()[0].provenance == "provider=spacy; model=en_core_web_sm; version=1"
    assert "prompt_version=v1" in llm.feature_specs()[0].formula_or_rule
    with pytest.raises(OptionalDependencyError):
        parser.fit(pd.DataFrame({"text": ["hello"]}), None)
    with pytest.raises(OptionalDependencyError):
        llm.fit(pd.DataFrame({"text": ["hello"]}), None)
