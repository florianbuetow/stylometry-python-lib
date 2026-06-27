"""Exhaustive feature extraction coverage tests."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal

import pandas as pd
import pytest

from stylometry_python_lib import (
    DeterministicStylometryTransformer,
    FeatureImplementationStatus,
    MostFrequentWordsTransformer,
    NGramStylometryTransformer,
    OptionalDependencyError,
    ResearchBucket,
    TopicDependence,
    built_in_research_registry,
    english_preprocessing_config,
    function_word_frequency_transformer,
    letter_frequency_transformer,
    llm_annotation_feature_names,
    llm_annotation_transformer,
    parser_backed_feature_names,
    parser_backed_transformer,
    punctuation_frequency_transformer,
)
from stylometry_python_lib.features.deterministic import deterministic_feature_names, function_words_lexicon
from stylometry_python_lib.specs import InputLayer, StabilityStatus

CoverageAnalyzer = Literal["word", "function_word", "char", "punctuation"]

SCALAR_GOLDEN_TEXT = """# Style Note
Dear Team,

I can't organise color-center data data; however, we should test NASA vs. Dr. Ray.
- "Well," you said, "it is fine!!"
1. On the other hand, they were being careful.
| A | B |
```
CODE
```
Best regards,
"""

EXPECTED_DETERMINISTIC_SCALAR_VALUES = {
    "text::counts::token_count": 39.0,
    "text::counts::type_count": 38.0,
    "text::counts::sentence_count": 6.0,
    "text::counts::paragraph_count": 2.0,
    "text::counts::character_count": 227.0,
    "text::counts::letter_count": 149.0,
    "text::lexical_richness::ttr": 0.9743589743589743,
    "text::lexical_richness::cttr": 4.302652729749464,
    "text::lexical_richness::herdan_c": 0.9929097722620002,
    "text::lexical_richness::msttr": 0.9743589743589743,
    "text::lexical_richness::mattr": 0.9743589743589743,
    "text::lexical_richness::mtld": 425.8799999999997,
    "text::lexical_richness::hdd": 0.9743589743589743,
    "text::lexical_richness::vocd_d": math.nan,
    "text::lexical_richness::vocd_d_fast": math.nan,
    "text::lexical_richness::hapax_count": 37.0,
    "text::lexical_richness::dis_legomena_count": 1.0,
    "text::lexical_richness::yules_k": 13.149243918474689,
    "text::lexical_richness::honore_r": 13921.53425529267,
    "text::lexical_richness::guiraud_r": 6.084869844593311,
    "text::lexical_richness::sichel_s": 0.02631578947368421,
    "text::lexical_richness::simpson_d": 0.001349527665317139,
    "text::lexical_richness::renyi_entropy_alpha_2": 3.613551225554985,
    "text::length::word_mean": 3.871794871794872,
    "text::length::word_median": 4.0,
    "text::length::word_std": 1.742080795021849,
    "text::length::syllables_per_word_mean": 1.3076923076923077,
    "text::length::sentence_tokens_mean": 6.5,
    "text::length::sentence_tokens_std": 5.408326913195984,
    "text::length::paragraph_tokens_mean": 19.5,
    "text::length::line_characters_mean": 19.636363636363637,
    "text::closed_class::function_word_ratio": 0.15384615384615385,
    "text::closed_class::stopword_ratio": 0.3076923076923077,
    "text::stance::pronoun_ratio": 0.1282051282051282,
    "text::stance::first_person_ratio": 0.05128205128205128,
    "text::stance::second_person_ratio": 0.02564102564102564,
    "text::stance::third_person_ratio": 0.05128205128205128,
    "text::stance::modal_verb_ratio": 0.02564102564102564,
    "text::grammar::auxiliary_verb_ratio": 0.07692307692307693,
    "text::register::contraction_count": 1.0,
    "text::lexical_density::content_lexicon_ratio": 0.6410256410256411,
    "text::orthography::punctuation_count": 21.0,
    "text::orthography::punctuation_per_token": 0.5384615384615384,
    "text::orthography::punctuation_sequence_count": 1.0,
    "text::orthography::uppercase_character_ratio": 0.1342281879194631,
    "text::orthography::all_caps_word_count": 2.0,
    "text::orthography::titlecase_line_count": 4.0,
    "text::orthography::spelling_variant_count": 3.0,
    "text::orthography::hyphenated_token_count": 1.0,
    "text::orthography::abbreviation_count": 2.0,
    "text::orthography::acronym_count": 2.0,
    "text::discourse::quote_marker_count": 5.0,
    "text::discourse::dialogue_dash_count": 1.0,
    "text::discourse::discourse_marker_count": 2.0,
    "text::discourse::transition_phrase_count": 1.0,
    "text::layout::heading_line_count": 4.0,
    "text::layout::bullet_line_count": 1.0,
    "text::layout::numbered_line_count": 1.0,
    "text::layout::code_fence_count": 2.0,
    "text::layout::table_line_count": 1.0,
    "text::layout::greeting_count": 1.0,
    "text::layout::signoff_count": 2.0,
    "text::whitespace::blank_line_count": 1.0,
    "text::whitespace::line_break_count": 11.0,
    "text::readability::flesch_reading_ease": 89.6067307692308,
    "text::readability::flesch_kincaid_grade": 2.3757692307692313,
    "text::readability::gunning_fog": 5.676923076923078,
    "text::readability::coleman_liau": 2.4123076923076887,
    "text::readability::smog": 7.168621630094336,
    "text::readability::ari": 0.05615384615384755,
    "text::readability::dale_chall": 8.817361538461538,
    "text::readability::forcast": math.nan,
    "text::readability::linsear_write": math.nan,
    "text::readability::lix": 16.756410256410255,
}

EXPECTED_FUNCTION_WORD_VOCABULARY = (
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "then",
    "there",
    "this",
    "to",
    "was",
    "were",
    "with",
)
EXPECTED_LETTER_CHARACTERS = tuple("abcdefghijklmnopqrstuvwxyz")
EXPECTED_PUNCTUATION_CHARACTERS = (".", ",", ";", ":", "!", "?", "-", "(", ")", "'", '"')
EXPECTED_PARSER_FEATURE_NAMES = (
    "text::syntax::pos_frequency",
    "text::syntax::pos_ngram",
    "text::syntax::pos_skipgram",
    "text::syntax::morphology_frequency",
    "text::syntax::dependency_relation_frequency",
    "text::syntax::head_dependent_pos_pair_frequency",
    "text::syntax::dependency_ngram",
    "text::syntax::dependency_path",
    "text::syntax::dependency_subtree",
    "text::syntax::dependency_dtgram",
    "text::syntax::dependency_distance_mean",
    "text::syntax::root_statistics",
    "text::syntax::parse_depth",
    "text::syntax::syntactic_complexity",
    "text::syntax::clause_count",
    "text::syntax::t_unit_count",
    "text::syntax::subordination_ratio",
    "text::syntax::coordination_ratio",
    "text::syntax::passive_voice_frequency",
    "text::syntax::pos_lexical_density",
    "text::content_control::named_entity_density",
    "text::content_control::content_masking",
    "text::content_control::topic_neutral_distortion",
)
EXPECTED_LLM_FEATURE_NAMES = (
    "text::llm::tone",
    "text::llm::register",
    "text::llm::persona",
    "text::llm::narrative_perspective",
    "text::llm::sentence_intent",
    "text::llm::discourse_function",
    "text::llm::rhetorical_structure",
    "text::llm::argumentation_style",
    "text::llm::cohesion_judgment",
    "text::llm::style_topic_separation",
    "text::llm::stylistic_similarity",
    "text::llm::pairwise_style_comparison",
    "text::llm::style_difference_explanation",
    "text::llm::style_transfer_descriptor",
    "text::llm::authorial_habit_summary",
    "text::llm::prompt_derived_vector",
    "text::llm::embedding",
    "text::llm::style_tuned_embedding",
    "text::llm::same_author_prediction",
    "text::llm::generated_feature_extraction",
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _assert_close(actual: float, expected: float) -> None:
    if math.isnan(expected):
        assert math.isnan(actual)
        return
    assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)


def _assert_single_row(frame: pd.DataFrame, expected: dict[str, float]) -> None:
    assert tuple(frame.columns) == tuple(expected)
    row = frame.iloc[0].to_dict()
    for name, expected_value in expected.items():
        _assert_close(float(row[name]), expected_value)


def test_all_deterministic_scalar_features_have_golden_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [SCALAR_GOLDEN_TEXT]})
    transformer = DeterministicStylometryTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert len(EXPECTED_DETERMINISTIC_SCALAR_VALUES) == 74
    assert tuple(EXPECTED_DETERMINISTIC_SCALAR_VALUES) == deterministic_feature_names()
    assert tuple(transformer.get_feature_names_out(None).tolist()) == deterministic_feature_names()
    _assert_single_row(result, EXPECTED_DETERMINISTIC_SCALAR_VALUES)


def test_research_registry_covers_every_v2_research_row_and_reports_separate_counts() -> None:
    registry = built_in_research_registry()
    bucket_counts = Counter(entry.bucket for entry in registry.entries)
    availability = registry.availability_counts()
    matrix = registry.availability_matrix()

    assert bucket_counts[ResearchBucket.DETERMINISTIC] == 41
    assert bucket_counts[ResearchBucket.OTHER_NON_LLM] == 29
    assert bucket_counts[ResearchBucket.LLM] == 20
    assert availability.planned_research_families == 90
    assert availability.implemented_feature_blocks == 90
    assert availability.emitted_numeric_columns == 1524
    assert availability.sidecar_annotation_types == 56
    assert availability.catalog_only_unavailable_entries == 0
    assert availability.out_of_scope_entries == 0
    assert len(matrix) == 90
    assert matrix[0]["taxonomy_id"] == "det.function_word_frequencies"
    assert matrix[-1]["taxonomy_id"] == "llm.generated_feature_extraction"
    assert all(entry.status in set(FeatureImplementationStatus) for entry in registry.entries)
    assert all(entry.topic_dependence != TopicDependence.UNKNOWN for entry in registry.entries)
    assert all(entry.formula_or_rule != "" for entry in registry.entries)
    assert all(entry.undefined_behavior != "" for entry in registry.entries)
    assert all(entry.provenance_requirements != "" for entry in registry.entries)


def test_research_registry_supports_v2_selection_modes() -> None:
    registry = built_in_research_registry()
    ttr_entry = registry.by_taxonomy_id("det.type_token_ratio")
    selected = registry.select_by_taxonomy_ids(("det.type_token_ratio", "llm.stylistic_similarity_judgments"))
    punctuation_entries = registry.select_by_output_name_regex(r"punctuation")
    owned = registry.select_by_feature_spec(
        DeterministicStylometryTransformer("text", english_preprocessing_config(), "pandas")
        .fit(pd.DataFrame({"text": ["simple text."]}), None)
        .registry_.by_name("text::lexical_richness::ttr")
    )

    assert ttr_entry.block_id == "lexical_richness_ttr"
    assert tuple(entry.taxonomy_id for entry in selected) == ("det.type_token_ratio", "llm.stylistic_similarity_judgments")
    assert {entry.taxonomy_id for entry in punctuation_entries} >= {
        "det.punctuation_frequencies",
        "det.punctuation_sequence_patterns",
    }
    assert owned.taxonomy_id == "det.type_token_ratio"


def test_every_function_word_frequency_feature_has_a_golden_value() -> None:
    config = english_preprocessing_config()
    tokens: tuple[str, ...] = ("the", "and", "the", "to", "if", "because", "we")
    counts: Counter[str] = Counter(tokens)
    x = pd.DataFrame({"text": [" ".join(tokens)]})
    transformer = function_word_frequency_transformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert tuple(sorted(function_words_lexicon())) == EXPECTED_FUNCTION_WORD_VOCABULARY
    expected = {
        f"text::function_word_frequency::token={token}": float(counts[token]) / float(len(tokens))
        for token in EXPECTED_FUNCTION_WORD_VOCABULARY
    }
    _assert_single_row(result, expected)


def test_every_letter_frequency_feature_has_a_golden_value() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["AaBbCc!"]})
    transformer = letter_frequency_transformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    expected = {
        f"text::letter_frequency::char={character}": (2.0 / 6.0 if character in {"a", "b", "c"} else 0.0)
        for character in EXPECTED_LETTER_CHARACTERS
    }
    _assert_single_row(result, expected)


def test_every_punctuation_frequency_feature_has_a_golden_value() -> None:
    config = english_preprocessing_config()
    punctuation_text = "Aa" + "".join(EXPECTED_PUNCTUATION_CHARACTERS)
    x = pd.DataFrame({"text": [punctuation_text]})
    transformer = punctuation_frequency_transformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    expected = {f"text::punctuation_frequency::char={character}": 0.5 for character in EXPECTED_PUNCTUATION_CHARACTERS}
    _assert_single_row(result, expected)


def test_every_most_frequent_word_feature_has_golden_values_for_each_row() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["alpha beta alpha gamma", "beta gamma gamma delta"]})
    transformer = MostFrequentWordsTransformer(text_column="text", config=config, max_features=4, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    expected_rows = pd.DataFrame(
        [
            {
                "text::most_frequent_words::token=gamma": 1.0 / 4.0,
                "text::most_frequent_words::token=alpha": 2.0 / 4.0,
                "text::most_frequent_words::token=beta": 1.0 / 4.0,
                "text::most_frequent_words::token=delta": 0.0,
            },
            {
                "text::most_frequent_words::token=gamma": 2.0 / 4.0,
                "text::most_frequent_words::token=alpha": 0.0,
                "text::most_frequent_words::token=beta": 1.0 / 4.0,
                "text::most_frequent_words::token=delta": 1.0 / 4.0,
            },
        ]
    )
    pd.testing.assert_frame_equal(result, expected_rows)


@pytest.mark.parametrize(
    ("analyzer", "text", "expected"),
    [
        (
            "word",
            "alpha beta alpha",
            {
                "text::word_ngram::gram=alpha": 2.0,
                "text::word_ngram::gram=alpha beta": 1.0,
                "text::word_ngram::gram=beta": 1.0,
                "text::word_ngram::gram=beta alpha": 1.0,
            },
        ),
        (
            "function_word",
            "the and the",
            {
                "text::function_word_ngram::gram=the": 2.0,
                "text::function_word_ngram::gram=and": 1.0,
                "text::function_word_ngram::gram=and the": 1.0,
                "text::function_word_ngram::gram=the and": 1.0,
            },
        ),
        (
            "char",
            "aba",
            {
                "text::char_ngram::gram=a": 2.0,
                "text::char_ngram::gram=ab": 1.0,
                "text::char_ngram::gram=b": 1.0,
                "text::char_ngram::gram=ba": 1.0,
            },
        ),
        (
            "punctuation",
            "?!!",
            {
                "text::punctuation_ngram::gram=!": 2.0,
                "text::punctuation_ngram::gram=!!": 1.0,
                "text::punctuation_ngram::gram=?": 1.0,
                "text::punctuation_ngram::gram=?!": 1.0,
            },
        ),
    ],
)
def test_every_emitted_ngram_feature_has_a_golden_value(analyzer: CoverageAnalyzer, text: str, expected: dict[str, float]) -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [text]}, index=["ngram-doc"])
    transformer = NGramStylometryTransformer(
        text_column="text",
        config=config,
        analyzer=analyzer,
        ngram_range=(1, 2),
        max_features=None,
        output="pandas",
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["ngram-doc"]
    _assert_single_row(result, expected)


def test_parser_backed_catalog_and_fail_fast_extraction_behavior_are_exhaustive() -> None:
    x = pd.DataFrame({"text": ["The quick sentence has one parser-backed request."]})
    parser = parser_backed_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        parsed_documents=None,
    )
    specs = parser.feature_specs()

    assert parser_backed_feature_names() == EXPECTED_PARSER_FEATURE_NAMES
    assert tuple(spec.name for spec in specs) == EXPECTED_PARSER_FEATURE_NAMES
    for spec in specs:
        assert spec.input_layer == InputLayer.NLP
        assert spec.stability_status == StabilityStatus.PARSER_MODEL_DEPENDENT
        assert "provider=spacy" in spec.provenance
        assert "model=en_core_web_sm" in spec.provenance
        assert "version=3.7.0" in spec.provenance
    with pytest.raises(OptionalDependencyError, match=r"provider=spacy.*model=en_core_web_sm.*version=3\.7\.0"):
        parser.fit(x, None)
    with pytest.raises(OptionalDependencyError, match=r"provider=spacy.*model=en_core_web_sm.*version=3\.7\.0"):
        parser.transform(x)


def test_llm_catalog_and_fail_fast_extraction_behavior_are_exhaustive() -> None:
    x = pd.DataFrame({"text": ["The quick sentence has one LLM annotation request."]})
    llm = llm_annotation_transformer(
        provider="openai",
        model="gpt-style-judge",
        version="2026-01-01",
        prompt_version="stylometry-v1",
        response_schema="style_annotation_schema_v1",
        feature_names=llm_annotation_feature_names(),
        fake_annotations=None,
        client=None,
        text_column="text",
    )
    specs = llm.feature_specs()

    assert llm_annotation_feature_names() == EXPECTED_LLM_FEATURE_NAMES
    assert tuple(spec.name for spec in specs) == EXPECTED_LLM_FEATURE_NAMES
    for spec in specs:
        assert spec.input_layer == InputLayer.LLM
        assert spec.stability_status == StabilityStatus.LLM_DEPENDENT
        assert "provider=openai" in spec.provenance
        assert "model=gpt-style-judge" in spec.provenance
        assert "version=2026-01-01" in spec.provenance
        assert "prompt_version=stylometry-v1" in spec.provenance
        assert "response_schema=style_annotation_schema_v1" in spec.formula_or_rule
    registry = built_in_research_registry()
    llm_entries = tuple(entry for entry in registry.entries if entry.bucket == ResearchBucket.LLM)
    assert len(llm_entries) == 20
    for entry in llm_entries:
        assert "configured OpenAI-compatible/LM Studio provider" in entry.dependency_extra
        assert "configured LM Studio row/pair" in entry.test_status
        assert "stylometry_python_lib.llm" in entry.implementation_owner
    with pytest.raises(OptionalDependencyError, match=r"provider=openai.*model=gpt-style-judge.*prompt_version=stylometry-v1"):
        llm.fit(x, None)
    with pytest.raises(OptionalDependencyError, match=r"provider=openai.*model=gpt-style-judge.*prompt_version=stylometry-v1"):
        llm.transform(x)


def test_llm_entries_record_replay_coverage() -> None:
    registry = built_in_research_registry()
    for entry in registry.entries:
        if entry.bucket == ResearchBucket.LLM:
            assert "replay" in entry.test_status
