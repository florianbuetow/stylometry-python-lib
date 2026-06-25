"""Tests for offline fake-provider parser-backed features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse
from sklearn.exceptions import NotFittedError

from stylometry_python_lib import (
    FeatureExtractor,
    OptionalDependencyError,
    ParsedDependencyArc,
    ParsedDocument,
    ParsedMorphologyFeature,
    ParsedNamedEntity,
    ParsedSyntacticCounts,
    ParsedToken,
    ParserBackedTransformer,
    ParserContentMaskingSidecar,
    ParserContentMaskingTransformer,
    ParserDependencyDistanceTransformer,
    ParserDependencyRelationTransformer,
    ParserDependencyStructureTransformer,
    ParserHeadDependentPOSPairTransformer,
    ParserMorphologyTransformer,
    ParserNamedEntityDensityTransformer,
    ParserParseDepthTransformer,
    ParserPassiveVoiceTransformer,
    ParserPOSLexicalDensityTransformer,
    ParserPOSNGramTransformer,
    ParserPOSSkipGramTransformer,
    ParserRootStatisticsTransformer,
    ParserSyntacticComplexityTransformer,
    parser_backed_transformer,
    parser_content_masking_feature_names,
    parser_content_masking_transformer,
    parser_dependency_distance_feature_names,
    parser_dependency_distance_transformer,
    parser_dependency_relation_feature_names,
    parser_dependency_relation_transformer,
    parser_dependency_structure_transformer,
    parser_head_dependent_pos_pair_transformer,
    parser_morphology_transformer,
    parser_named_entity_density_feature_names,
    parser_named_entity_density_transformer,
    parser_parse_depth_feature_names,
    parser_parse_depth_transformer,
    parser_passive_voice_feature_names,
    parser_passive_voice_transformer,
    parser_pos_frequency_feature_names,
    parser_pos_lexical_density_feature_names,
    parser_pos_lexical_density_transformer,
    parser_pos_ngram_transformer,
    parser_pos_skipgram_transformer,
    parser_root_statistics_feature_names,
    parser_root_statistics_transformer,
    parser_syntactic_complexity_feature_names,
    parser_syntactic_complexity_transformer,
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def _fixture_documents() -> tuple[ParsedDocument, ...]:
    return (
        ParsedDocument(
            document_id="doc-a",
            tokens=(
                ParsedToken(text="I", upos="PRON", morphology=()),
                ParsedToken(text="write", upos="VERB", morphology=()),
                ParsedToken(text="tests", upos="NOUN", morphology=()),
                ParsedToken(text=".", upos="PUNCT", morphology=()),
            ),
            dependency_arcs=(),
        ),
        ParsedDocument(
            document_id="doc-b",
            tokens=(
                ParsedToken(text="Careful", upos="ADJ", morphology=()),
                ParsedToken(text="parser", upos="NOUN", morphology=()),
                ParsedToken(text="fixtures", upos="NOUN", morphology=()),
                ParsedToken(text="help", upos="VERB", morphology=()),
            ),
            dependency_arcs=(),
        ),
    )


def _morphology_documents() -> tuple[ParsedDocument, ...]:
    return (
        ParsedDocument(
            document_id="doc-a",
            tokens=(
                ParsedToken(
                    text="wrote",
                    upos="VERB",
                    morphology=(
                        ParsedMorphologyFeature(attribute="Tense", value="Past"),
                        ParsedMorphologyFeature(attribute="Mood", value="Ind"),
                        ParsedMorphologyFeature(attribute="Voice", value="Act"),
                    ),
                ),
                ParsedToken(
                    text="bright",
                    upos="ADJ",
                    morphology=(ParsedMorphologyFeature(attribute="Degree", value="Pos"),),
                ),
                ParsedToken(
                    text="letters",
                    upos="NOUN",
                    morphology=(
                        ParsedMorphologyFeature(attribute="Number", value="Plur"),
                        ParsedMorphologyFeature(attribute="Case", value="Nom"),
                    ),
                ),
            ),
            dependency_arcs=(),
        ),
        ParsedDocument(
            document_id="doc-b",
            tokens=(
                ParsedToken(
                    text="I",
                    upos="PRON",
                    morphology=(
                        ParsedMorphologyFeature(attribute="Person", value="1"),
                        ParsedMorphologyFeature(attribute="Number", value="Sing"),
                    ),
                ),
                ParsedToken(
                    text="her",
                    upos="PRON",
                    morphology=(
                        ParsedMorphologyFeature(attribute="Gender", value="Fem"),
                        ParsedMorphologyFeature(attribute="Case", value="Acc"),
                    ),
                ),
                ParsedToken(
                    text="writing",
                    upos="VERB",
                    morphology=(ParsedMorphologyFeature(attribute="Aspect", value="Prog"),),
                ),
            ),
            dependency_arcs=(),
        ),
    )


def _dependency_documents() -> tuple[ParsedDocument, ...]:
    return (
        ParsedDocument(
            document_id="doc-a",
            tokens=(
                ParsedToken(text="I", upos="PRON", morphology=()),
                ParsedToken(text="write", upos="VERB", morphology=()),
                ParsedToken(text="tests", upos="NOUN", morphology=()),
                ParsedToken(text="daily", upos="ADV", morphology=()),
                ParsedToken(text=".", upos="PUNCT", morphology=()),
            ),
            dependency_arcs=(
                ParsedDependencyArc(head_index=None, dependent_index=1, relation="root"),
                ParsedDependencyArc(head_index=1, dependent_index=0, relation="nsubj"),
                ParsedDependencyArc(head_index=1, dependent_index=2, relation="obj"),
                ParsedDependencyArc(head_index=1, dependent_index=3, relation="advmod"),
                ParsedDependencyArc(head_index=1, dependent_index=4, relation="punct"),
            ),
        ),
        ParsedDocument(
            document_id="doc-b",
            tokens=(
                ParsedToken(text="Careful", upos="ADJ", morphology=()),
                ParsedToken(text="parser", upos="NOUN", morphology=()),
                ParsedToken(text="fixtures", upos="NOUN", morphology=()),
                ParsedToken(text="help", upos="VERB", morphology=()),
                ParsedToken(text="readers", upos="NOUN", morphology=()),
            ),
            dependency_arcs=(
                ParsedDependencyArc(head_index=None, dependent_index=3, relation="root"),
                ParsedDependencyArc(head_index=2, dependent_index=0, relation="amod"),
                ParsedDependencyArc(head_index=2, dependent_index=1, relation="compound"),
                ParsedDependencyArc(head_index=3, dependent_index=2, relation="nsubj"),
                ParsedDependencyArc(head_index=3, dependent_index=4, relation="obj"),
            ),
        ),
    )


def _syntactic_counts() -> tuple[ParsedSyntacticCounts, ...]:
    return (
        ParsedSyntacticCounts(
            document_id="doc-a",
            word_count=20,
            sentence_count=2,
            clause_count=4,
            t_unit_count=3,
            dependent_clause_count=1,
            coordinate_phrase_count=2,
            complex_nominal_count=3,
            verb_phrase_count=4,
        ),
        ParsedSyntacticCounts(
            document_id="doc-b",
            word_count=8,
            sentence_count=1,
            clause_count=1,
            t_unit_count=1,
            dependent_clause_count=0,
            coordinate_phrase_count=0,
            complex_nominal_count=1,
            verb_phrase_count=1,
        ),
    )


def _passive_documents() -> tuple[ParsedDocument, ...]:
    return (
        ParsedDocument(
            document_id="doc-passive",
            tokens=(
                ParsedToken(text="The", upos="DET", morphology=()),
                ParsedToken(text="result", upos="NOUN", morphology=()),
                ParsedToken(text="was", upos="AUX", morphology=()),
                ParsedToken(text="tested", upos="VERB", morphology=(ParsedMorphologyFeature(attribute="Voice", value="Pass"),)),
            ),
            dependency_arcs=(
                ParsedDependencyArc(head_index=None, dependent_index=3, relation="root"),
                ParsedDependencyArc(head_index=1, dependent_index=0, relation="det"),
                ParsedDependencyArc(head_index=3, dependent_index=1, relation="nsubj:pass"),
                ParsedDependencyArc(head_index=3, dependent_index=2, relation="aux:pass"),
            ),
        ),
        ParsedDocument(
            document_id="doc-copular",
            tokens=(
                ParsedToken(text="The", upos="DET", morphology=()),
                ParsedToken(text="result", upos="NOUN", morphology=()),
                ParsedToken(text="was", upos="AUX", morphology=()),
                ParsedToken(text="clear", upos="ADJ", morphology=(ParsedMorphologyFeature(attribute="Degree", value="Pos"),)),
            ),
            dependency_arcs=(
                ParsedDependencyArc(head_index=None, dependent_index=3, relation="root"),
                ParsedDependencyArc(head_index=1, dependent_index=0, relation="det"),
                ParsedDependencyArc(head_index=3, dependent_index=1, relation="nsubj"),
                ParsedDependencyArc(head_index=3, dependent_index=2, relation="cop"),
            ),
        ),
        ParsedDocument(
            document_id="doc-adjectival",
            tokens=(
                ParsedToken(text="The", upos="DET", morphology=()),
                ParsedToken(text="door", upos="NOUN", morphology=()),
                ParsedToken(text="was", upos="AUX", morphology=()),
                ParsedToken(text="closed", upos="ADJ", morphology=(ParsedMorphologyFeature(attribute="Voice", value="Pass"),)),
            ),
            dependency_arcs=(
                ParsedDependencyArc(head_index=None, dependent_index=3, relation="root"),
                ParsedDependencyArc(head_index=1, dependent_index=0, relation="det"),
                ParsedDependencyArc(head_index=3, dependent_index=1, relation="nsubj"),
                ParsedDependencyArc(head_index=3, dependent_index=2, relation="cop"),
            ),
        ),
    )


def _passive_syntactic_counts() -> tuple[ParsedSyntacticCounts, ...]:
    return (
        ParsedSyntacticCounts(
            document_id="doc-passive",
            word_count=4,
            sentence_count=1,
            clause_count=1,
            t_unit_count=1,
            dependent_clause_count=0,
            coordinate_phrase_count=0,
            complex_nominal_count=0,
            verb_phrase_count=1,
        ),
        ParsedSyntacticCounts(
            document_id="doc-copular",
            word_count=4,
            sentence_count=1,
            clause_count=1,
            t_unit_count=1,
            dependent_clause_count=0,
            coordinate_phrase_count=0,
            complex_nominal_count=0,
            verb_phrase_count=1,
        ),
        ParsedSyntacticCounts(
            document_id="doc-adjectival",
            word_count=4,
            sentence_count=1,
            clause_count=1,
            t_unit_count=1,
            dependent_clause_count=0,
            coordinate_phrase_count=0,
            complex_nominal_count=0,
            verb_phrase_count=1,
        ),
    )


def _entity_documents() -> tuple[ParsedDocument, ...]:
    return (
        ParsedDocument(
            document_id="doc-entities",
            tokens=(
                ParsedToken(text="Alice", upos="PROPN", morphology=()),
                ParsedToken(text="joined", upos="VERB", morphology=()),
                ParsedToken(text="OpenAI", upos="PROPN", morphology=()),
                ParsedToken(text="Paris", upos="PROPN", morphology=()),
            ),
            dependency_arcs=(),
        ),
        ParsedDocument(
            document_id="doc-none",
            tokens=(
                ParsedToken(text="Quiet", upos="ADJ", morphology=()),
                ParsedToken(text="prose", upos="NOUN", morphology=()),
            ),
            dependency_arcs=(),
        ),
    )


def _named_entities() -> tuple[ParsedNamedEntity, ...]:
    return (
        ParsedNamedEntity(document_id="doc-entities", text="Alice", label="PERSON", start_token_index=0, end_token_index=1),
        ParsedNamedEntity(document_id="doc-entities", text="OpenAI", label="ORG", start_token_index=2, end_token_index=3),
        ParsedNamedEntity(document_id="doc-entities", text="Paris", label="GPE", start_token_index=3, end_token_index=4),
    )


def test_fake_parser_pos_frequency_has_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_backed_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        parsed_documents=_fixture_documents(),
        output="pandas",
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == parser_pos_frequency_feature_names()
    assert len(result.columns) == 17
    assert _cell(result, "text::syntax::pos_frequency::upos=PRON", 0) == 0.25
    assert _cell(result, "text::syntax::pos_frequency::upos=VERB", 0) == 0.25
    assert _cell(result, "text::syntax::pos_frequency::upos=NOUN", 0) == 0.25
    assert _cell(result, "text::syntax::pos_frequency::upos=PUNCT", 0) == 0.25
    assert _cell(result, "text::syntax::pos_frequency::upos=ADJ", 0) == 0.0
    assert _cell(result, "text::syntax::pos_frequency::upos=ADJ", 1) == 0.25
    assert _cell(result, "text::syntax::pos_frequency::upos=NOUN", 1) == 0.5
    assert _cell(result, "text::syntax::pos_frequency::upos=VERB", 1) == 0.25
    spec = transformer.registry_.by_name("text::syntax::pos_frequency::upos=NOUN")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "count Universal POS tag divided by parsed token count"
    assert spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_parser_tokens"
    assert "provider=fake" in spec.provenance
    assert "model=fixture-pos" in spec.provenance
    assert "tagset=Universal POS" in spec.provenance
    assert transformer.last_diagnostics_ == ((), ())


def test_fake_parser_pos_frequency_supports_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_backed_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        parsed_documents=_fixture_documents(),
        output="pandas",
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserBackedTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_backed_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        parsed_documents=_fixture_documents(),
        output="sparse",
    ).fit_transform(x, None)
    numpy_result = parser_backed_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        parsed_documents=_fixture_documents(),
        output="numpy",
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_pos_lexical_density_has_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_pos_lexical_density_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_fixture_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == parser_pos_lexical_density_feature_names()
    assert _cell(result, "text::syntax::pos_lexical_density", 0) == 0.5
    assert _cell(result, "text::syntax::pos_lexical_density", 1) == 1.0
    assert tuple(transformer.get_feature_names_out(None).tolist()) == parser_pos_lexical_density_feature_names()
    spec = transformer.registry_.by_name("text::syntax::pos_lexical_density")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == ("count parser tokens whose Universal POS tag is ADJ, ADV, NOUN, or VERB divided by parser token count")
    assert spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_parser_tokens"
    assert spec.topic_dependence.value == "mixed"
    assert "content_upos=ADJ,ADV,NOUN,VERB" in spec.provenance
    assert transformer.last_diagnostics_ == ((), ())


def test_fake_parser_pos_lexical_density_supports_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_pos_lexical_density_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_fixture_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserPOSLexicalDensityTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_pos_lexical_density_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        output="sparse",
        parsed_documents=_fixture_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_pos_lexical_density_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        output="numpy",
        parsed_documents=_fixture_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_pos_lexical_density_validates_missing_empty_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_pos_lexical_density_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_fixture_documents()[:1],
    )
    empty_transformer = parser_pos_lexical_density_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=(ParsedDocument(document_id="doc-empty", tokens=(), dependency_arcs=()),),
    )
    not_fitted_transformer = parser_pos_lexical_density_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_fixture_documents(),
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    empty_result = _as_frame(empty_transformer.fit_transform(empty_x, None))
    assert math.isnan(_cell(empty_result, "text::syntax::pos_lexical_density"))
    diagnostics = empty_transformer.last_diagnostics_[0]
    assert len(diagnostics) == 1
    assert diagnostics[0].reason == "zero_parser_tokens"
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_parser_pos_lexical_density_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_pos_lexical_density_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_missing_or_empty_parser_layer_fails_explicitly() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_backed_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        parsed_documents=_fixture_documents()[:1],
        output="pandas",
    )
    empty_transformer = parser_backed_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        parsed_documents=(ParsedDocument(document_id="doc-empty", tokens=(), dependency_arcs=()),),
        output="pandas",
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")

    empty_result = _as_frame(empty_transformer.fit_transform(empty_x, None))
    assert math.isnan(_cell(empty_result, "text::syntax::pos_frequency::upos=NOUN"))
    diagnostics = empty_transformer.last_diagnostics_[0]
    assert len(diagnostics) == len(parser_pos_frequency_feature_names())
    assert {diagnostic.reason for diagnostic in diagnostics} == {"zero_parser_tokens"}


def test_fake_parser_generated_feature_names_require_fit() -> None:
    transformer = parser_backed_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        parsed_documents=_fixture_documents(),
        output="pandas",
    )

    try:
        transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_fake_parser_pos_ngram_has_fitted_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_pos_ngram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features=6,
        output="pandas",
        parsed_documents=_fixture_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    expected_columns = (
        "text::syntax::pos_ngram::gram=NOUN",
        "text::syntax::pos_ngram::gram=VERB",
        "text::syntax::pos_ngram::gram=ADJ",
        "text::syntax::pos_ngram::gram=ADJ NOUN",
        "text::syntax::pos_ngram::gram=NOUN NOUN",
        "text::syntax::pos_ngram::gram=NOUN PUNCT",
    )
    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == expected_columns
    assert tuple(transformer.get_feature_names_out(None).tolist()) == expected_columns
    assert _cell(result, "text::syntax::pos_ngram::gram=NOUN", 0) == 1.0
    assert _cell(result, "text::syntax::pos_ngram::gram=VERB", 0) == 1.0
    assert _cell(result, "text::syntax::pos_ngram::gram=NOUN PUNCT", 0) == 1.0
    assert _cell(result, "text::syntax::pos_ngram::gram=ADJ", 0) == 0.0
    assert _cell(result, "text::syntax::pos_ngram::gram=NOUN", 1) == 2.0
    assert _cell(result, "text::syntax::pos_ngram::gram=ADJ", 1) == 1.0
    assert _cell(result, "text::syntax::pos_ngram::gram=ADJ NOUN", 1) == 1.0
    assert _cell(result, "text::syntax::pos_ngram::gram=NOUN NOUN", 1) == 1.0
    spec = transformer.registry_.by_name("text::syntax::pos_ngram::gram=ADJ NOUN")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "count fitted contiguous Universal POS n-gram in the parser token sequence"
    assert spec.undefined_behavior == "not undefined after fit; absent fitted grams produce valid zero counts"
    assert spec.topic_dependence.value == "mostly_topic_independent"
    assert "provider=fake" in spec.provenance
    assert "ngram_range=1-2" in spec.provenance
    assert "fitted_vocabulary" in spec.provenance


def test_fake_parser_pos_ngram_supports_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_pos_ngram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features=None,
        output="pandas",
        parsed_documents=_fixture_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserPOSNGramTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_pos_ngram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features=None,
        output="sparse",
        parsed_documents=_fixture_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_pos_ngram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features=None,
        output="numpy",
        parsed_documents=_fixture_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_pos_ngram_validates_missing_layers_configuration_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_pos_ngram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features=4,
        output="pandas",
        parsed_documents=_fixture_documents()[:1],
    )
    empty_transformer = parser_pos_ngram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        ngram_range=(2, 2),
        max_features=4,
        output="pandas",
        parsed_documents=(ParsedDocument(document_id="doc-empty", tokens=(), dependency_arcs=()),),
    )
    not_fitted_transformer = parser_pos_ngram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features=4,
        output="pandas",
        parsed_documents=_fixture_documents(),
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    try:
        empty_transformer.fit(empty_x, None)
    except ValueError as error:
        assert str(error) == "No parser POS n-grams found for fitted corpus"
    else:
        raise AssertionError("Expected empty POS n-gram vocabulary validation to fail")
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_parser_pos_ngram_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_pos_ngram_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        ngram_range=(1, 2),
        max_features=4,
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_pos_skipgram_has_fitted_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_pos_skipgram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        skip_distance=1,
        max_features=5,
        output="pandas",
        parsed_documents=_fixture_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    expected_columns = (
        "text::syntax::pos_skipgram::gram=ADJ *1 NOUN",
        "text::syntax::pos_skipgram::gram=NOUN *1 VERB",
        "text::syntax::pos_skipgram::gram=PRON *1 NOUN",
        "text::syntax::pos_skipgram::gram=VERB *1 PUNCT",
    )
    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == expected_columns
    assert tuple(transformer.get_feature_names_out(None).tolist()) == expected_columns
    assert _cell(result, "text::syntax::pos_skipgram::gram=PRON *1 NOUN", 0) == 1.0
    assert _cell(result, "text::syntax::pos_skipgram::gram=VERB *1 PUNCT", 0) == 1.0
    assert _cell(result, "text::syntax::pos_skipgram::gram=ADJ *1 NOUN", 0) == 0.0
    assert _cell(result, "text::syntax::pos_skipgram::gram=ADJ *1 NOUN", 1) == 1.0
    assert _cell(result, "text::syntax::pos_skipgram::gram=NOUN *1 VERB", 1) == 1.0
    spec = transformer.registry_.by_name("text::syntax::pos_skipgram::gram=ADJ *1 NOUN")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "count fitted non-contiguous Universal POS skip-bigram in the parser token sequence"
    assert spec.undefined_behavior == "not undefined after fit; absent fitted skip-grams produce valid zero counts"
    assert spec.topic_dependence.value == "mostly_topic_independent"
    assert "provider=fake" in spec.provenance
    assert "skip_distance=1" in spec.provenance
    assert "fitted_vocabulary" in spec.provenance


def test_fake_parser_pos_skipgram_supports_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_pos_skipgram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        skip_distance=1,
        max_features=None,
        output="pandas",
        parsed_documents=_fixture_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserPOSSkipGramTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_pos_skipgram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        skip_distance=1,
        max_features=None,
        output="sparse",
        parsed_documents=_fixture_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_pos_skipgram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        skip_distance=1,
        max_features=None,
        output="numpy",
        parsed_documents=_fixture_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_pos_skipgram_validates_missing_layers_configuration_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_pos_skipgram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        skip_distance=1,
        max_features=4,
        output="pandas",
        parsed_documents=_fixture_documents()[:1],
    )
    invalid_transformer = parser_pos_skipgram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        skip_distance=0,
        max_features=4,
        output="pandas",
        parsed_documents=_fixture_documents(),
    )
    empty_transformer = parser_pos_skipgram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        skip_distance=1,
        max_features=4,
        output="pandas",
        parsed_documents=(ParsedDocument(document_id="doc-empty", tokens=(), dependency_arcs=()),),
    )
    not_fitted_transformer = parser_pos_skipgram_transformer(
        provider="fake",
        model="fixture-pos",
        version="1",
        text_column="text",
        skip_distance=1,
        max_features=4,
        output="pandas",
        parsed_documents=_fixture_documents(),
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    try:
        invalid_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "skip_distance must be positive"
    else:
        raise AssertionError("Expected skip_distance validation to fail")
    try:
        empty_transformer.fit(empty_x, None)
    except ValueError as error:
        assert str(error) == "No parser POS skip-grams found for fitted corpus"
    else:
        raise AssertionError("Expected empty POS skip-gram vocabulary validation to fail")
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_parser_pos_skipgram_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_pos_skipgram_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        skip_distance=1,
        max_features=4,
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_morphology_has_fitted_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_morphology_transformer(
        provider="fake",
        model="fixture-morph",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_morphology_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    expected_columns = (
        "text::syntax::morphology_frequency::attribute=Aspect::value=Prog",
        "text::syntax::morphology_frequency::attribute=Case::value=Acc",
        "text::syntax::morphology_frequency::attribute=Case::value=Nom",
        "text::syntax::morphology_frequency::attribute=Degree::value=Pos",
        "text::syntax::morphology_frequency::attribute=Gender::value=Fem",
        "text::syntax::morphology_frequency::attribute=Mood::value=Ind",
        "text::syntax::morphology_frequency::attribute=Number::value=Plur",
        "text::syntax::morphology_frequency::attribute=Number::value=Sing",
        "text::syntax::morphology_frequency::attribute=Person::value=1",
        "text::syntax::morphology_frequency::attribute=Tense::value=Past",
        "text::syntax::morphology_frequency::attribute=Voice::value=Act",
    )
    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == expected_columns
    assert tuple(transformer.get_feature_names_out(None).tolist()) == expected_columns
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Tense::value=Past", 0) == 1.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Mood::value=Ind", 0) == 1.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Voice::value=Act", 0) == 1.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Degree::value=Pos", 0) == 1.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Case::value=Nom", 0) == 1.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Person::value=1", 0) == 0.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Person::value=1", 1) == 1.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Number::value=Sing", 1) == 1.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Gender::value=Fem", 1) == 1.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Aspect::value=Prog", 1) == 1.0
    assert _cell(result, "text::syntax::morphology_frequency::attribute=Case::value=Acc", 1) == 1.0
    spec = transformer.registry_.by_name("text::syntax::morphology_frequency::attribute=Tense::value=Past")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "count fitted Universal Dependencies morphology attribute/value observation in the parser token sequence"
    assert spec.undefined_behavior == "not undefined after fit; absent fitted morphology attributes produce valid zero counts"
    assert spec.topic_dependence.value == "mixed"
    assert "provider=fake" in spec.provenance
    assert "model=fixture-morph" in spec.provenance
    assert "morphology_schema=Universal Dependencies" in spec.provenance
    assert "fitted_vocabulary" in spec.provenance


def test_fake_parser_morphology_supports_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_morphology_transformer(
        provider="fake",
        model="fixture-morph",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_morphology_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserMorphologyTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_morphology_transformer(
        provider="fake",
        model="fixture-morph",
        version="1",
        text_column="text",
        output="sparse",
        parsed_documents=_morphology_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_morphology_transformer(
        provider="fake",
        model="fixture-morph",
        version="1",
        text_column="text",
        output="numpy",
        parsed_documents=_morphology_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_morphology_validates_missing_layers_configuration_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_morphology_transformer(
        provider="fake",
        model="fixture-morph",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_morphology_documents()[:1],
    )
    empty_transformer = parser_morphology_transformer(
        provider="fake",
        model="fixture-morph",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=(ParsedDocument(document_id="doc-empty", tokens=(), dependency_arcs=()),),
    )
    not_fitted_transformer = parser_morphology_transformer(
        provider="fake",
        model="fixture-morph",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_morphology_documents(),
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    try:
        empty_transformer.fit(empty_x, None)
    except ValueError as error:
        assert str(error) == "No parser morphology attributes found for fitted corpus"
    else:
        raise AssertionError("Expected empty morphology vocabulary validation to fail")
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_fake_parser_morphology_validates_ud_attribute_fixture_contract() -> None:
    try:
        ParsedMorphologyFeature(attribute="Polarity", value="Neg")
    except ValueError as error:
        assert str(error) == "Unsupported Universal Dependencies morphology attribute: Polarity"
    else:
        raise AssertionError("Expected unsupported morphology attribute validation to fail")
    try:
        ParsedMorphologyFeature(attribute="Tense", value="")
    except ValueError as error:
        assert str(error) == "ParsedMorphologyFeature value must not be empty"
    else:
        raise AssertionError("Expected empty morphology value validation to fail")


def test_parser_morphology_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_morphology_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_dependency_relations_have_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_dependency_relation_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == parser_dependency_relation_feature_names()
    assert len(result.columns) == 37
    assert tuple(transformer.get_feature_names_out(None).tolist()) == parser_dependency_relation_feature_names()
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=root", 0) == 0.2
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=nsubj", 0) == 0.2
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=obj", 0) == 0.2
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=advmod", 0) == 0.2
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=punct", 0) == 0.2
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=amod", 0) == 0.0
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=amod", 1) == 0.2
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=compound", 1) == 0.2
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=nsubj", 1) == 0.2
    assert _cell(result, "text::syntax::dependency_relation_frequency::deprel=obj", 1) == 0.2
    spec = transformer.registry_.by_name("text::syntax::dependency_relation_frequency::deprel=nsubj")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "count Universal Dependencies basic relation label divided by dependency arc count"
    assert spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_dependency_arcs"
    assert spec.topic_dependence.value == "mixed"
    assert "provider=fake" in spec.provenance
    assert "model=fixture-deps" in spec.provenance
    assert "deprel_schema=Universal Dependencies basic relations" in spec.provenance
    assert transformer.last_diagnostics_ == ((), ())


def test_fake_parser_dependency_relations_support_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_dependency_relation_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserDependencyRelationTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_dependency_relation_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="sparse",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_dependency_relation_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="numpy",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_dependency_relations_validate_missing_layers_configuration_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_dependency_relation_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents()[:1],
    )
    empty_transformer = parser_dependency_relation_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=(
            ParsedDocument(
                document_id="doc-empty",
                tokens=(ParsedToken(text="fragment", upos="NOUN", morphology=()),),
                dependency_arcs=(),
            ),
        ),
    )
    not_fitted_transformer = parser_dependency_relation_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    empty_result = _as_frame(empty_transformer.fit_transform(empty_x, None))
    assert math.isnan(_cell(empty_result, "text::syntax::dependency_relation_frequency::deprel=nsubj"))
    diagnostics = empty_transformer.last_diagnostics_[0]
    assert len(diagnostics) == len(parser_dependency_relation_feature_names())
    assert {diagnostic.reason for diagnostic in diagnostics} == {"zero_dependency_arcs"}
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_fake_parser_dependency_relations_validate_ud_arc_fixture_contract() -> None:
    try:
        ParsedDependencyArc(head_index=0, dependent_index=1, relation="madeup")
    except ValueError as error:
        assert str(error) == "Unsupported Universal Dependencies relation label: madeup"
    else:
        raise AssertionError("Expected unsupported dependency relation validation to fail")
    try:
        ParsedDependencyArc(head_index=0, dependent_index=1, relation="root")
    except ValueError as error:
        assert str(error) == "ParsedDependencyArc root relation must use head_index=None"
    else:
        raise AssertionError("Expected root head validation to fail")
    try:
        ParsedDependencyArc(head_index=None, dependent_index=1, relation="nsubj")
    except ValueError as error:
        assert str(error) == "ParsedDependencyArc non-root relation must provide head_index"
    else:
        raise AssertionError("Expected non-root head validation to fail")
    try:
        ParsedDocument(
            document_id="bad-doc",
            tokens=(ParsedToken(text="alone", upos="NOUN", morphology=()),),
            dependency_arcs=(ParsedDependencyArc(head_index=0, dependent_index=1, relation="obj"),),
        )
    except ValueError as error:
        assert str(error) == "Dependency arc dependent_index out of range for document: bad-doc"
    else:
        raise AssertionError("Expected dependency arc index validation to fail")


def test_parser_dependency_relation_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_dependency_relation_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_head_dependent_pos_pairs_have_fitted_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_head_dependent_pos_pair_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    expected_columns = (
        "text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=NOUN",
        "text::syntax::head_dependent_pos_pair_frequency::head_upos=NOUN::dependent_upos=ADJ",
        "text::syntax::head_dependent_pos_pair_frequency::head_upos=NOUN::dependent_upos=NOUN",
        "text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=ADV",
        "text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=PRON",
        "text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=PUNCT",
    )
    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == expected_columns
    assert tuple(transformer.get_feature_names_out(None).tolist()) == expected_columns
    assert _cell(result, "text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=NOUN", 0) == 1.0
    assert _cell(result, "text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=ADV", 0) == 1.0
    assert _cell(result, "text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=PRON", 0) == 1.0
    assert _cell(result, "text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=PUNCT", 0) == 1.0
    assert _cell(result, "text::syntax::head_dependent_pos_pair_frequency::head_upos=NOUN::dependent_upos=ADJ", 0) == 0.0
    assert _cell(result, "text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=NOUN", 1) == 2.0
    assert _cell(result, "text::syntax::head_dependent_pos_pair_frequency::head_upos=NOUN::dependent_upos=ADJ", 1) == 1.0
    assert _cell(result, "text::syntax::head_dependent_pos_pair_frequency::head_upos=NOUN::dependent_upos=NOUN", 1) == 1.0
    spec = transformer.registry_.by_name("text::syntax::head_dependent_pos_pair_frequency::head_upos=VERB::dependent_upos=NOUN")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "count fitted Universal POS head/dependent pair over non-root dependency arcs"
    assert spec.undefined_behavior == "not undefined after fit; absent fitted head-dependent POS pairs produce valid zero counts"
    assert spec.topic_dependence.value == "mixed"
    assert "provider=fake" in spec.provenance
    assert "tagset=Universal POS" in spec.provenance
    assert "dependency_schema=Universal Dependencies" in spec.provenance
    assert "fitted_vocabulary" in spec.provenance


def test_fake_parser_head_dependent_pos_pairs_support_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_head_dependent_pos_pair_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserHeadDependentPOSPairTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_head_dependent_pos_pair_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="sparse",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_head_dependent_pos_pair_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="numpy",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_head_dependent_pos_pairs_validate_missing_layers_configuration_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_head_dependent_pos_pair_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents()[:1],
    )
    root_only_transformer = parser_head_dependent_pos_pair_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=(
            ParsedDocument(
                document_id="doc-root-only",
                tokens=(ParsedToken(text="Root", upos="NOUN", morphology=()),),
                dependency_arcs=(ParsedDependencyArc(head_index=None, dependent_index=0, relation="root"),),
            ),
        ),
    )
    not_fitted_transformer = parser_head_dependent_pos_pair_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )
    root_only_x = pd.DataFrame({"text": ["root"]}, index=["doc-root-only"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    try:
        root_only_transformer.fit(root_only_x, None)
    except ValueError as error:
        assert str(error) == "No parser head-dependent POS pairs found for fitted corpus"
    else:
        raise AssertionError("Expected empty head-dependent POS pair vocabulary validation to fail")
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_parser_head_dependent_pos_pair_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_head_dependent_pos_pair_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_dependency_distances_have_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_dependency_distance_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == parser_dependency_distance_feature_names()
    assert len(result.columns) == 14
    assert tuple(transformer.get_feature_names_out(None).tolist()) == parser_dependency_distance_feature_names()
    assert _cell(result, "text::syntax::dependency_distance_count", 0) == 4.0
    assert _cell(result, "text::syntax::dependency_distance_mean", 0) == 1.75
    assert math.isclose(_cell(result, "text::syntax::dependency_distance_sample_variance", 0), 11.0 / 12.0)
    assert math.isclose(_cell(result, "text::syntax::dependency_distance_sample_std", 0), math.sqrt(11.0 / 12.0))
    assert _cell(result, "text::syntax::dependency_distance_min", 0) == 1.0
    assert _cell(result, "text::syntax::dependency_distance_max", 0) == 3.0
    assert _cell(result, "text::syntax::dependency_distance_p50", 0) == 1.5
    assert _cell(result, "text::syntax::dependency_distance_p90", 0) == 2.7
    assert math.isclose(_cell(result, "text::syntax::dependency_distance_shannon_entropy", 0), 1.0397207708399179)
    assert _cell(result, "text::syntax::dependency_distance_count", 1) == 4.0
    assert _cell(result, "text::syntax::dependency_distance_mean", 1) == 1.25
    assert _cell(result, "text::syntax::dependency_distance_p75", 1) == 1.25
    spec = transformer.registry_.by_name("text::syntax::dependency_distance_mean")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "arithmetic mean over absolute head-dependent token distances"
    assert spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_dependency_distances when the distribution is empty"
    assert "distance=abs(head_index-dependent_index)" in spec.provenance
    assert "std_type=sample" in spec.provenance
    assert transformer.last_diagnostics_ == ((), ())


def test_fake_parser_dependency_distances_support_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_dependency_distance_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserDependencyDistanceTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_dependency_distance_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="sparse",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_dependency_distance_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="numpy",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_dependency_distances_validate_missing_layers_and_empty_distribution() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_dependency_distance_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents()[:1],
    )
    root_only_transformer = parser_dependency_distance_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=(
            ParsedDocument(
                document_id="doc-root-only",
                tokens=(ParsedToken(text="Root", upos="NOUN", morphology=()),),
                dependency_arcs=(ParsedDependencyArc(head_index=None, dependent_index=0, relation="root"),),
            ),
        ),
    )
    empty_x = pd.DataFrame({"text": ["root"]}, index=["doc-root-only"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    empty_result = _as_frame(root_only_transformer.fit_transform(empty_x, None))
    assert _cell(empty_result, "text::syntax::dependency_distance_count") == 0.0
    assert math.isnan(_cell(empty_result, "text::syntax::dependency_distance_mean"))
    diagnostics = root_only_transformer.last_diagnostics_[0]
    assert len(diagnostics) == len(parser_dependency_distance_feature_names()) - 1
    assert {diagnostic.reason for diagnostic in diagnostics} == {"zero_dependency_distances"}


def test_parser_dependency_distance_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_dependency_distance_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_parse_depth_has_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_parse_depth_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == parser_parse_depth_feature_names()
    assert len(result.columns) == 14
    assert tuple(transformer.get_feature_names_out(None).tolist()) == parser_parse_depth_feature_names()
    assert _cell(result, "text::syntax::parse_depth_count", 0) == 5.0
    assert _cell(result, "text::syntax::parse_depth_mean", 0) == 0.8
    assert math.isclose(_cell(result, "text::syntax::parse_depth_sample_variance", 0), 0.2)
    assert math.isclose(_cell(result, "text::syntax::parse_depth_sample_std", 0), math.sqrt(0.2))
    assert _cell(result, "text::syntax::parse_depth_min", 0) == 0.0
    assert _cell(result, "text::syntax::parse_depth_max", 0) == 1.0
    assert _cell(result, "text::syntax::parse_depth_p10", 0) == 0.4
    assert _cell(result, "text::syntax::parse_depth_p50", 0) == 1.0
    assert math.isclose(_cell(result, "text::syntax::parse_depth_skewness", 0), -1.5)
    assert math.isclose(_cell(result, "text::syntax::parse_depth_excess_kurtosis", 0), 0.25)
    assert math.isclose(_cell(result, "text::syntax::parse_depth_shannon_entropy", 0), 0.5004024235381879)
    assert _cell(result, "text::syntax::parse_depth_count", 1) == 5.0
    assert _cell(result, "text::syntax::parse_depth_mean", 1) == 1.2
    assert math.isclose(_cell(result, "text::syntax::parse_depth_sample_variance", 1), 0.7)
    assert _cell(result, "text::syntax::parse_depth_max", 1) == 2.0
    assert _cell(result, "text::syntax::parse_depth_p75", 1) == 2.0
    assert math.isclose(_cell(result, "text::syntax::parse_depth_shannon_entropy", 1), 1.0549201679861442)
    spec = transformer.registry_.by_name("text::syntax::parse_depth_mean")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "arithmetic mean over dependency edge depths from root to token"
    assert spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_parse_depths when the distribution is empty"
    assert spec.topic_dependence.value == "mostly_topic_independent"
    assert "root_depth=0" in spec.provenance
    assert "depth=edge_count_to_root" in spec.provenance
    assert "dependency_schema=Universal Dependencies" in spec.provenance
    assert transformer.last_diagnostics_ == ((), ())


def test_fake_parser_parse_depth_supports_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_parse_depth_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserParseDepthTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_parse_depth_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="sparse",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_parse_depth_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="numpy",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_parse_depth_validates_missing_layers_empty_distribution_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_parse_depth_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents()[:1],
    )
    empty_transformer = parser_parse_depth_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=(ParsedDocument(document_id="doc-empty", tokens=(), dependency_arcs=()),),
    )
    not_fitted_transformer = parser_parse_depth_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    empty_result = _as_frame(empty_transformer.fit_transform(empty_x, None))
    assert _cell(empty_result, "text::syntax::parse_depth_count") == 0.0
    assert math.isnan(_cell(empty_result, "text::syntax::parse_depth_mean"))
    diagnostics = empty_transformer.last_diagnostics_[0]
    assert len(diagnostics) == len(parser_parse_depth_feature_names()) - 1
    assert {diagnostic.reason for diagnostic in diagnostics} == {"zero_parse_depths"}
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_fake_parser_parse_depth_detects_dependency_cycles() -> None:
    x = pd.DataFrame({"text": ["cycle"]}, index=["doc-cycle"])
    transformer = parser_parse_depth_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=(
            ParsedDocument(
                document_id="doc-cycle",
                tokens=(
                    ParsedToken(text="A", upos="NOUN", morphology=()),
                    ParsedToken(text="B", upos="NOUN", morphology=()),
                ),
                dependency_arcs=(
                    ParsedDependencyArc(head_index=1, dependent_index=0, relation="nsubj"),
                    ParsedDependencyArc(head_index=0, dependent_index=1, relation="obj"),
                ),
            ),
        ),
    )
    transformer.fit(x, None)

    try:
        transformer.transform(x)
    except ValueError as error:
        assert str(error) == "Dependency depth contains a cycle"
    else:
        raise AssertionError("Expected dependency cycle validation to fail")


def test_parser_parse_depth_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_parse_depth_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_syntactic_complexity_has_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_syntactic_complexity_transformer(
        provider="fake",
        model="fixture-syntax",
        version="1",
        text_column="text",
        output="pandas",
        syntactic_counts=_syntactic_counts(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == parser_syntactic_complexity_feature_names()
    assert len(result.columns) == 15
    assert tuple(transformer.get_feature_names_out(None).tolist()) == parser_syntactic_complexity_feature_names()
    assert _cell(result, "text::syntax::syntactic_complexity::mls", 0) == 10.0
    assert math.isclose(_cell(result, "text::syntax::syntactic_complexity::mlt", 0), 20.0 / 3.0)
    assert _cell(result, "text::syntax::syntactic_complexity::mlc", 0) == 5.0
    assert _cell(result, "text::syntax::syntactic_complexity::c_per_s", 0) == 2.0
    assert math.isclose(_cell(result, "text::syntax::syntactic_complexity::c_per_t", 0), 4.0 / 3.0)
    assert math.isclose(_cell(result, "text::syntax::syntactic_complexity::cp_per_t", 0), 2.0 / 3.0)
    assert _cell(result, "text::syntax::syntactic_complexity::dc_per_c", 0) == 0.25
    assert _cell(result, "text::syntax::syntactic_complexity::cn_per_c", 0) == 0.75
    assert _cell(result, "text::syntax::syntactic_complexity::cn_per_t", 0) == 1.0
    assert math.isclose(_cell(result, "text::syntax::syntactic_complexity::vp_per_t", 0), 4.0 / 3.0)
    assert _cell(result, "text::syntax::syntactic_complexity::t_per_s", 0) == 1.5
    assert _cell(result, "text::syntax::clause_count", 0) == 4.0
    assert _cell(result, "text::syntax::t_unit_count", 0) == 3.0
    assert _cell(result, "text::syntax::subordination_ratio", 0) == 0.25
    assert math.isclose(_cell(result, "text::syntax::coordination_ratio", 0), 2.0 / 3.0)
    assert _cell(result, "text::syntax::syntactic_complexity::mls", 1) == 8.0
    assert _cell(result, "text::syntax::syntactic_complexity::dc_per_c", 1) == 0.0
    assert _cell(result, "text::syntax::coordination_ratio", 1) == 0.0
    spec = transformer.registry_.by_name("text::syntax::syntactic_complexity::mlt")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "word_count / t_unit_count"
    assert spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_t_unit_count when t_unit_count is zero"
    assert spec.topic_dependence.value == "mixed"
    assert "annotation_source=fake_syntactic_counts" in spec.provenance
    assert "MLS,MLT,MLC" in spec.provenance
    clause_spec = transformer.registry_.by_name("text::syntax::clause_count")
    assert clause_spec.family == "parser_clause_count"
    assert clause_spec.normalization == "raw_count"
    assert transformer.last_diagnostics_ == ((), ())


def test_fake_parser_syntactic_complexity_supports_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_syntactic_complexity_transformer(
        provider="fake",
        model="fixture-syntax",
        version="1",
        text_column="text",
        output="pandas",
        syntactic_counts=_syntactic_counts(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserSyntacticComplexityTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_syntactic_complexity_transformer(
        provider="fake",
        model="fixture-syntax",
        version="1",
        text_column="text",
        output="sparse",
        syntactic_counts=_syntactic_counts(),
    ).fit_transform(x, None)
    numpy_result = parser_syntactic_complexity_transformer(
        provider="fake",
        model="fixture-syntax",
        version="1",
        text_column="text",
        output="numpy",
        syntactic_counts=_syntactic_counts(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_syntactic_complexity_validates_missing_duplicate_and_invalid_counts() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_syntactic_complexity_transformer(
        provider="fake",
        model="fixture-syntax",
        version="1",
        text_column="text",
        output="pandas",
        syntactic_counts=_syntactic_counts()[:1],
    )
    duplicate_transformer = parser_syntactic_complexity_transformer(
        provider="fake",
        model="fixture-syntax",
        version="1",
        text_column="text",
        output="pandas",
        syntactic_counts=(_syntactic_counts()[0], _syntactic_counts()[0]),
    )

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake syntactic counts for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake syntactic counts validation to fail")
    try:
        duplicate_transformer.fit(pd.DataFrame({"text": ["left"]}, index=["doc-a"]), None)
    except ValueError as error:
        assert str(error) == "Duplicate fake syntactic counts document id: doc-a"
    else:
        raise AssertionError("Expected duplicate fake syntactic counts validation to fail")
    try:
        ParsedSyntacticCounts(
            document_id="bad",
            word_count=-1,
            sentence_count=1,
            clause_count=1,
            t_unit_count=1,
            dependent_clause_count=0,
            coordinate_phrase_count=0,
            complex_nominal_count=0,
            verb_phrase_count=0,
        )
    except ValueError as error:
        assert str(error) == "ParsedSyntacticCounts word_count must be non-negative"
    else:
        raise AssertionError("Expected negative syntactic count validation to fail")
    try:
        ParsedSyntacticCounts(
            document_id="bad",
            word_count=1,
            sentence_count=1,
            clause_count=1,
            t_unit_count=1,
            dependent_clause_count=2,
            coordinate_phrase_count=0,
            complex_nominal_count=0,
            verb_phrase_count=0,
        )
    except ValueError as error:
        assert str(error) == "ParsedSyntacticCounts dependent_clause_count must not exceed clause_count"
    else:
        raise AssertionError("Expected dependent-clause bound validation to fail")


def test_fake_parser_syntactic_complexity_reports_undefined_counts_and_fit_state() -> None:
    empty_counts_transformer = parser_syntactic_complexity_transformer(
        provider="fake",
        model="fixture-syntax",
        version="1",
        text_column="text",
        output="pandas",
        syntactic_counts=(
            ParsedSyntacticCounts(
                document_id="doc-empty",
                word_count=0,
                sentence_count=0,
                clause_count=0,
                t_unit_count=0,
                dependent_clause_count=0,
                coordinate_phrase_count=0,
                complex_nominal_count=0,
                verb_phrase_count=0,
            ),
        ),
    )
    not_fitted_transformer = parser_syntactic_complexity_transformer(
        provider="fake",
        model="fixture-syntax",
        version="1",
        text_column="text",
        output="pandas",
        syntactic_counts=_syntactic_counts(),
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    empty_result = _as_frame(empty_counts_transformer.fit_transform(empty_x, None))
    assert _cell(empty_result, "text::syntax::clause_count") == 0.0
    assert _cell(empty_result, "text::syntax::t_unit_count") == 0.0
    assert math.isnan(_cell(empty_result, "text::syntax::syntactic_complexity::mls"))
    assert math.isnan(_cell(empty_result, "text::syntax::subordination_ratio"))
    diagnostics = empty_counts_transformer.last_diagnostics_[0]
    assert len(diagnostics) == len(parser_syntactic_complexity_feature_names()) - 2
    assert {diagnostic.reason for diagnostic in diagnostics} == {
        "zero_sentence_count",
        "zero_clause_count",
        "zero_t_unit_count",
    }
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_parser_syntactic_complexity_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_syntactic_complexity_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        syntactic_counts=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_passive_voice_has_golden_values_and_false_positive_guards() -> None:
    x = pd.DataFrame(
        {"text": ["passive", "copular", "adjectival"]},
        index=["doc-passive", "doc-copular", "doc-adjectival"],
    )
    transformer = parser_passive_voice_transformer(
        provider="fake",
        model="fixture-passive",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_passive_documents(),
        syntactic_counts=_passive_syntactic_counts(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-passive", "doc-copular", "doc-adjectival"]
    assert tuple(result.columns) == parser_passive_voice_feature_names()
    assert _cell(result, "text::syntax::passive_voice_frequency", 0) == 1.0
    assert _cell(result, "text::syntax::passive_voice_frequency", 1) == 0.0
    assert _cell(result, "text::syntax::passive_voice_frequency", 2) == 0.0
    assert tuple(transformer.get_feature_names_out(None).tolist()) == parser_passive_voice_feature_names()
    spec = transformer.registry_.by_name("text::syntax::passive_voice_frequency")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == (
        "count predicate heads with Voice=Pass morphology plus aux:pass and nsubj:pass dependents divided by clause_count"
    )
    assert spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_clause_count"
    assert "rule=Voice=Pass+aux:pass+nsubj:pass" in spec.provenance
    assert "denominator=clause_count" in spec.provenance
    assert transformer.last_diagnostics_ == ((), (), ())


def test_fake_parser_passive_voice_supports_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame(
        {"text": ["passive", "copular", "adjectival"]},
        index=["doc-passive", "doc-copular", "doc-adjectival"],
    )
    original = x.copy(deep=True)
    transformer = parser_passive_voice_transformer(
        provider="fake",
        model="fixture-passive",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_passive_documents(),
        syntactic_counts=_passive_syntactic_counts(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserPassiveVoiceTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_passive_voice_transformer(
        provider="fake",
        model="fixture-passive",
        version="1",
        text_column="text",
        output="sparse",
        parsed_documents=_passive_documents(),
        syntactic_counts=_passive_syntactic_counts(),
    ).fit_transform(x, None)
    numpy_result = parser_passive_voice_transformer(
        provider="fake",
        model="fixture-passive",
        version="1",
        text_column="text",
        output="numpy",
        parsed_documents=_passive_documents(),
        syntactic_counts=_passive_syntactic_counts(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_passive_voice_validates_missing_zero_clause_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["passive", "copular"]}, index=["doc-passive", "doc-copular"])
    missing_document_transformer = parser_passive_voice_transformer(
        provider="fake",
        model="fixture-passive",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_passive_documents()[:1],
        syntactic_counts=_passive_syntactic_counts()[:2],
    )
    missing_counts_transformer = parser_passive_voice_transformer(
        provider="fake",
        model="fixture-passive",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_passive_documents()[:2],
        syntactic_counts=_passive_syntactic_counts()[:1],
    )
    zero_clause_transformer = parser_passive_voice_transformer(
        provider="fake",
        model="fixture-passive",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_passive_documents()[:1],
        syntactic_counts=(
            ParsedSyntacticCounts(
                document_id="doc-passive",
                word_count=4,
                sentence_count=1,
                clause_count=0,
                t_unit_count=1,
                dependent_clause_count=0,
                coordinate_phrase_count=0,
                complex_nominal_count=0,
                verb_phrase_count=1,
            ),
        ),
    )
    not_fitted_transformer = parser_passive_voice_transformer(
        provider="fake",
        model="fixture-passive",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_passive_documents(),
        syntactic_counts=_passive_syntactic_counts(),
    )

    try:
        missing_document_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-copular"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    try:
        missing_counts_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake syntactic counts for row id: doc-copular"
    else:
        raise AssertionError("Expected missing fake syntactic counts validation to fail")
    zero_clause_result = _as_frame(zero_clause_transformer.fit_transform(pd.DataFrame({"text": ["passive"]}, index=["doc-passive"]), None))
    assert math.isnan(_cell(zero_clause_result, "text::syntax::passive_voice_frequency"))
    diagnostics = zero_clause_transformer.last_diagnostics_[0]
    assert len(diagnostics) == 1
    assert diagnostics[0].reason == "zero_clause_count"
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_parser_passive_voice_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_passive_voice_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        parsed_documents=None,
        syntactic_counts=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_named_entity_density_has_golden_values_and_dual_namespaces() -> None:
    x = pd.DataFrame({"text": ["entity-rich", "entity-free"]}, index=["doc-entities", "doc-none"])
    entity_types = ("PERSON", "ORG", "GPE")
    transformer = parser_named_entity_density_transformer(
        provider="fake",
        model="fixture-ner",
        version="1",
        text_column="text",
        entity_types=entity_types,
        output="pandas",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-entities", "doc-none"]
    assert tuple(result.columns) == parser_named_entity_density_feature_names(entity_types)
    assert len(result.columns) == 6
    assert _cell(result, "text::style_adjacent::named_entity_density::entity_type=PERSON", 0) == 0.25
    assert _cell(result, "text::style_adjacent::named_entity_density::entity_type=ORG", 0) == 0.25
    assert _cell(result, "text::style_adjacent::named_entity_density::entity_type=GPE", 0) == 0.25
    assert _cell(result, "text::content_control::named_entity_density::entity_type=PERSON", 0) == 0.25
    assert _cell(result, "text::content_control::named_entity_density::entity_type=ORG", 1) == 0.0
    assert tuple(transformer.get_feature_names_out(None).tolist()) == parser_named_entity_density_feature_names(entity_types)
    style_spec = transformer.registry_.by_name("text::style_adjacent::named_entity_density::entity_type=PERSON")
    topic_spec = transformer.registry_.by_name("text::content_control::named_entity_density::entity_type=PERSON")
    assert style_spec.topic_dependence.value == "mixed"
    assert topic_spec.topic_dependence.value == "topic_sensitive"
    assert style_spec.formula_or_rule == "count configured named-entity spans for the requested label divided by parser token count"
    assert style_spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_parser_tokens"
    assert "entity_types=PERSON,ORG,GPE" in style_spec.provenance
    assert transformer.last_diagnostics_ == ((), ())


def test_fake_parser_named_entity_density_supports_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["entity-rich", "entity-free"]}, index=["doc-entities", "doc-none"])
    original = x.copy(deep=True)
    entity_types = ("PERSON", "ORG", "GPE")
    transformer = parser_named_entity_density_transformer(
        provider="fake",
        model="fixture-ner",
        version="1",
        text_column="text",
        entity_types=entity_types,
        output="pandas",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserNamedEntityDensityTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_named_entity_density_transformer(
        provider="fake",
        model="fixture-ner",
        version="1",
        text_column="text",
        entity_types=entity_types,
        output="sparse",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    ).fit_transform(x, None)
    numpy_result = parser_named_entity_density_transformer(
        provider="fake",
        model="fixture-ner",
        version="1",
        text_column="text",
        entity_types=entity_types,
        output="numpy",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_named_entity_density_validates_missing_layers_labels_and_spans() -> None:
    x = pd.DataFrame({"text": ["entity-rich", "entity-free"]}, index=["doc-entities", "doc-none"])
    missing_document_transformer = parser_named_entity_density_transformer(
        provider="fake",
        model="fixture-ner",
        version="1",
        text_column="text",
        entity_types=("PERSON", "ORG", "GPE"),
        output="pandas",
        parsed_documents=_entity_documents()[:1],
        named_entities=_named_entities(),
    )
    unconfigured_label_transformer = parser_named_entity_density_transformer(
        provider="fake",
        model="fixture-ner",
        version="1",
        text_column="text",
        entity_types=("PERSON", "GPE"),
        output="pandas",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    )
    invalid_span_transformer = parser_named_entity_density_transformer(
        provider="fake",
        model="fixture-ner",
        version="1",
        text_column="text",
        entity_types=("PERSON",),
        output="pandas",
        parsed_documents=_entity_documents()[:1],
        named_entities=(
            ParsedNamedEntity(
                document_id="doc-entities",
                text="Alice OpenAI",
                label="PERSON",
                start_token_index=0,
                end_token_index=5,
            ),
        ),
    )

    try:
        missing_document_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-none"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    try:
        unconfigured_label_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Fake named entity label not configured: ORG"
    else:
        raise AssertionError("Expected unconfigured fake named entity label validation to fail")
    try:
        invalid_span_transformer.fit(pd.DataFrame({"text": ["entity-rich"]}, index=["doc-entities"]), None)
    except ValueError as error:
        assert str(error) == "Named entity span out of range for document: doc-entities"
    else:
        raise AssertionError("Expected invalid fake named entity span validation to fail")


def test_fake_parser_named_entity_density_validates_configuration_empty_docs_and_fit_state() -> None:
    empty_transformer = parser_named_entity_density_transformer(
        provider="fake",
        model="fixture-ner",
        version="1",
        text_column="text",
        entity_types=("PERSON",),
        output="pandas",
        parsed_documents=(ParsedDocument(document_id="doc-empty", tokens=(), dependency_arcs=()),),
        named_entities=(),
    )
    not_fitted_transformer = parser_named_entity_density_transformer(
        provider="fake",
        model="fixture-ner",
        version="1",
        text_column="text",
        entity_types=("PERSON",),
        output="pandas",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities()[:1],
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    try:
        parser_named_entity_density_feature_names(("PERSON", "PERSON"))
    except ValueError as error:
        assert str(error) == "Duplicate entity type: PERSON"
    else:
        raise AssertionError("Expected duplicate entity type validation to fail")
    empty_result = _as_frame(empty_transformer.fit_transform(empty_x, None))
    assert math.isnan(_cell(empty_result, "text::style_adjacent::named_entity_density::entity_type=PERSON"))
    diagnostics = empty_transformer.last_diagnostics_[0]
    assert len(diagnostics) == 2
    assert {diagnostic.reason for diagnostic in diagnostics} == {"zero_parser_tokens"}
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_parser_named_entity_density_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_named_entity_density_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        entity_types=("PERSON",),
        output="pandas",
        parsed_documents=None,
        named_entities=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_content_masking_has_golden_masked_text_and_distortion_sidecars() -> None:
    x = pd.DataFrame({"text": ["entity-rich", "entity-free"]}, index=["doc-entities", "doc-none"])
    transformer = parser_content_masking_transformer(
        provider="fake",
        model="fixture-mask",
        version="1",
        text_column="text",
        mask_upos_tags=("NOUN", "VERB"),
        replacement_token="CONTENT",
        output="pandas",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-entities", "doc-none"]
    assert tuple(result.columns) == parser_content_masking_feature_names()
    assert _cell(result, "text::content_control::topic_neutral_distortion::masked_token_ratio", 0) == 1.0
    assert _cell(result, "text::content_control::topic_neutral_distortion::masked_token_ratio", 1) == 0.5
    assert tuple(transformer.get_feature_names_out(None).tolist()) == parser_content_masking_feature_names()
    first_sidecar = transformer.last_sidecars_[0]
    second_sidecar = transformer.last_sidecars_[1]
    assert isinstance(first_sidecar, ParserContentMaskingSidecar)
    assert first_sidecar.schema_version == "parser_content_masking_v1"
    assert first_sidecar.masked_text == "CONTENT CONTENT CONTENT CONTENT"
    assert first_sidecar.token_count == 4
    assert first_sidecar.masked_token_count == 4
    assert first_sidecar.named_entity_count == 3
    assert first_sidecar.mask_upos_tags == ("NOUN", "VERB")
    assert second_sidecar.masked_text == "Quiet CONTENT"
    assert second_sidecar.masked_token_count == 1
    spec = transformer.registry_.by_name("text::content_control::topic_neutral_distortion::masked_token_ratio")
    assert spec.input_layer.value == "multi"
    assert spec.topic_dependence.value == "topic_control"
    assert spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_parser_tokens"
    assert "sidecar_schema=parser_content_masking_v1" in spec.provenance
    assert transformer.last_diagnostics_ == ((), ())


def test_fake_parser_content_masking_supports_output_modes_serialization_sidecars_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["entity-rich", "entity-free"]}, index=["doc-entities", "doc-none"])
    original = x.copy(deep=True)
    transformer = parser_content_masking_transformer(
        provider="fake",
        model="fixture-mask",
        version="1",
        text_column="text",
        mask_upos_tags=("NOUN", "VERB"),
        replacement_token="CONTENT",
        output="pandas",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserContentMaskingTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_content_masking_transformer(
        provider="fake",
        model="fixture-mask",
        version="1",
        text_column="text",
        mask_upos_tags=("NOUN", "VERB"),
        replacement_token="CONTENT",
        output="sparse",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    ).fit_transform(x, None)
    numpy_result = parser_content_masking_transformer(
        provider="fake",
        model="fixture-mask",
        version="1",
        text_column="text",
        mask_upos_tags=("NOUN", "VERB"),
        replacement_token="CONTENT",
        output="numpy",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert len(extractor.last_sidecars_) == 1
    sidecars = cast(tuple[ParserContentMaskingSidecar, ParserContentMaskingSidecar], extractor.last_sidecars_[0].sidecars)
    assert sidecars[0].masked_text == "CONTENT CONTENT CONTENT CONTENT"
    assert sidecars[1].masked_text == "Quiet CONTENT"


def test_fake_parser_content_masking_validates_missing_config_spans_empty_docs_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["entity-rich", "entity-free"]}, index=["doc-entities", "doc-none"])
    missing_document_transformer = parser_content_masking_transformer(
        provider="fake",
        model="fixture-mask",
        version="1",
        text_column="text",
        mask_upos_tags=("NOUN",),
        replacement_token="CONTENT",
        output="pandas",
        parsed_documents=_entity_documents()[:1],
        named_entities=_named_entities(),
    )
    invalid_span_transformer = parser_content_masking_transformer(
        provider="fake",
        model="fixture-mask",
        version="1",
        text_column="text",
        mask_upos_tags=("NOUN",),
        replacement_token="CONTENT",
        output="pandas",
        parsed_documents=_entity_documents()[:1],
        named_entities=(
            ParsedNamedEntity(
                document_id="doc-entities",
                text="Alice OpenAI",
                label="PERSON",
                start_token_index=0,
                end_token_index=5,
            ),
        ),
    )
    empty_transformer = parser_content_masking_transformer(
        provider="fake",
        model="fixture-mask",
        version="1",
        text_column="text",
        mask_upos_tags=("NOUN",),
        replacement_token="CONTENT",
        output="pandas",
        parsed_documents=(ParsedDocument(document_id="doc-empty", tokens=(), dependency_arcs=()),),
        named_entities=(),
    )
    not_fitted_transformer = parser_content_masking_transformer(
        provider="fake",
        model="fixture-mask",
        version="1",
        text_column="text",
        mask_upos_tags=("NOUN",),
        replacement_token="CONTENT",
        output="pandas",
        parsed_documents=_entity_documents(),
        named_entities=_named_entities(),
    )

    try:
        parser_content_masking_transformer(
            provider="fake",
            model="fixture-mask",
            version="1",
            text_column="text",
            mask_upos_tags=("NOPE",),
            replacement_token="CONTENT",
            output="pandas",
            parsed_documents=_entity_documents(),
            named_entities=_named_entities(),
        ).fit(x, None)
    except ValueError as error:
        assert str(error) == "Unsupported Universal POS tag for content masking: NOPE"
    else:
        raise AssertionError("Expected invalid mask UPOS validation to fail")
    try:
        missing_document_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-none"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    try:
        invalid_span_transformer.fit(pd.DataFrame({"text": ["entity-rich"]}, index=["doc-entities"]), None)
    except ValueError as error:
        assert str(error) == "Named entity span out of range for document: doc-entities"
    else:
        raise AssertionError("Expected invalid fake named entity span validation to fail")
    empty_result = _as_frame(empty_transformer.fit_transform(pd.DataFrame({"text": ["empty"]}, index=["doc-empty"]), None))
    assert math.isnan(_cell(empty_result, "text::content_control::topic_neutral_distortion::masked_token_ratio"))
    assert empty_transformer.last_sidecars_[0].masked_text == ""
    assert empty_transformer.last_diagnostics_[0][0].reason == "zero_parser_tokens"
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_parser_content_masking_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_content_masking_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        mask_upos_tags=("NOUN",),
        replacement_token="CONTENT",
        output="pandas",
        parsed_documents=None,
        named_entities=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_root_statistics_have_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_root_statistics_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == parser_root_statistics_feature_names()
    assert len(result.columns) == 19
    assert tuple(transformer.get_feature_names_out(None).tolist()) == parser_root_statistics_feature_names()
    assert _cell(result, "text::syntax::root_statistics::root_count", 0) == 1.0
    assert _cell(result, "text::syntax::root_statistics::root_per_token", 0) == 0.2
    assert _cell(result, "text::syntax::root_statistics::root_upos_ratio::upos=VERB", 0) == 1.0
    assert _cell(result, "text::syntax::root_statistics::root_upos_ratio::upos=NOUN", 0) == 0.0
    assert _cell(result, "text::syntax::root_statistics::root_count", 1) == 1.0
    assert _cell(result, "text::syntax::root_statistics::root_per_token", 1) == 0.2
    assert _cell(result, "text::syntax::root_statistics::root_upos_ratio::upos=VERB", 1) == 1.0
    spec = transformer.registry_.by_name("text::syntax::root_statistics::root_upos_ratio::upos=VERB")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "count root arcs whose dependent token has the requested Universal POS tag divided by root arc count"
    assert spec.undefined_behavior == "NaN with FeatureDiagnostic reason zero_dependency_roots when no root arcs exist"
    assert "tagset=Universal POS" in spec.provenance
    assert "dependency_schema=Universal Dependencies" in spec.provenance
    assert transformer.last_diagnostics_ == ((), ())


def test_fake_parser_root_statistics_support_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_root_statistics_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserRootStatisticsTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_root_statistics_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="sparse",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_root_statistics_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="numpy",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape


def test_fake_parser_root_statistics_validate_missing_layers_and_empty_roots() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_root_statistics_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=_dependency_documents()[:1],
    )
    empty_transformer = parser_root_statistics_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=(ParsedDocument(document_id="doc-empty", tokens=(), dependency_arcs=()),),
    )
    empty_x = pd.DataFrame({"text": ["empty"]}, index=["doc-empty"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    empty_result = _as_frame(empty_transformer.fit_transform(empty_x, None))
    assert _cell(empty_result, "text::syntax::root_statistics::root_count") == 0.0
    assert math.isnan(_cell(empty_result, "text::syntax::root_statistics::root_per_token"))
    assert math.isnan(_cell(empty_result, "text::syntax::root_statistics::root_upos_ratio::upos=VERB"))
    diagnostics = empty_transformer.last_diagnostics_[0]
    assert len(diagnostics) == len(parser_root_statistics_feature_names()) - 1
    assert {diagnostic.reason for diagnostic in diagnostics} == {"zero_parser_tokens", "zero_dependency_roots"}


def test_parser_root_statistics_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_root_statistics_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")


def test_fake_parser_dependency_structures_have_fitted_golden_values_and_metadata() -> None:
    x = pd.DataFrame({"text": ["ignored by fake provider", "also ignored"]}, index=["doc-a", "doc-b"])
    transformer = parser_dependency_structure_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features_per_kind=None,
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    result = _as_frame(transformer.fit_transform(x, None))

    expected_columns = (
        "text::syntax::dependency_ngram::gram=nsubj",
        "text::syntax::dependency_ngram::gram=nsubj obj",
        "text::syntax::dependency_ngram::gram=obj",
        "text::syntax::dependency_ngram::gram=advmod",
        "text::syntax::dependency_ngram::gram=advmod punct",
        "text::syntax::dependency_ngram::gram=amod",
        "text::syntax::dependency_ngram::gram=amod compound",
        "text::syntax::dependency_ngram::gram=compound",
        "text::syntax::dependency_ngram::gram=compound nsubj",
        "text::syntax::dependency_ngram::gram=obj advmod",
        "text::syntax::dependency_ngram::gram=punct",
        "text::syntax::dependency_path::path=obj:NOUN",
        "text::syntax::dependency_path::path=advmod:ADV",
        "text::syntax::dependency_path::path=nsubj:NOUN",
        "text::syntax::dependency_path::path=nsubj:PRON",
        "text::syntax::dependency_path::path=nsubj>amod:ADJ",
        "text::syntax::dependency_path::path=nsubj>compound:NOUN",
        "text::syntax::dependency_path::path=punct:PUNCT",
        "text::syntax::dependency_subtree::signature=NOUN(amod:ADJ,compound:NOUN)",
        "text::syntax::dependency_subtree::signature=VERB(advmod:ADV,nsubj:PRON,obj:NOUN,punct:PUNCT)",
        "text::syntax::dependency_subtree::signature=VERB(nsubj:NOUN,obj:NOUN)",
        "text::syntax::dependency_dtgram::gram=VERB>obj>NOUN",
        "text::syntax::dependency_dtgram::gram=NOUN>amod>ADJ",
        "text::syntax::dependency_dtgram::gram=NOUN>compound>NOUN",
        "text::syntax::dependency_dtgram::gram=VERB>advmod>ADV",
        "text::syntax::dependency_dtgram::gram=VERB>nsubj>NOUN",
        "text::syntax::dependency_dtgram::gram=VERB>nsubj>PRON",
        "text::syntax::dependency_dtgram::gram=VERB>punct>PUNCT",
    )
    assert result.index.tolist() == ["doc-a", "doc-b"]
    assert tuple(result.columns) == expected_columns
    assert tuple(transformer.get_feature_names_out(None).tolist()) == expected_columns
    assert _cell(result, "text::syntax::dependency_ngram::gram=nsubj", 0) == 1.0
    assert _cell(result, "text::syntax::dependency_ngram::gram=nsubj obj", 0) == 1.0
    assert _cell(result, "text::syntax::dependency_ngram::gram=amod", 0) == 0.0
    assert _cell(result, "text::syntax::dependency_ngram::gram=amod", 1) == 1.0
    assert _cell(result, "text::syntax::dependency_path::path=obj:NOUN", 0) == 1.0
    assert _cell(result, "text::syntax::dependency_path::path=obj:NOUN", 1) == 1.0
    assert _cell(result, "text::syntax::dependency_path::path=nsubj>amod:ADJ", 1) == 1.0
    assert _cell(result, "text::syntax::dependency_subtree::signature=VERB(advmod:ADV,nsubj:PRON,obj:NOUN,punct:PUNCT)", 0) == 1.0
    assert _cell(result, "text::syntax::dependency_subtree::signature=NOUN(amod:ADJ,compound:NOUN)", 1) == 1.0
    assert _cell(result, "text::syntax::dependency_dtgram::gram=VERB>obj>NOUN", 0) == 1.0
    assert _cell(result, "text::syntax::dependency_dtgram::gram=VERB>obj>NOUN", 1) == 1.0
    spec = transformer.registry_.by_name("text::syntax::dependency_path::path=nsubj>amod:ADJ")
    assert spec.input_layer.value == "nlp"
    assert spec.formula_or_rule == "count fitted dependency structure over Universal Dependencies fake-parser tree"
    assert spec.undefined_behavior == "not undefined after fit; absent fitted dependency structures produce valid zero counts"
    assert spec.topic_dependence.value == "mixed"
    assert "provider=fake" in spec.provenance
    assert "dependency_schema=Universal Dependencies" in spec.provenance
    assert "ngram_range=1-2" in spec.provenance
    assert "fitted_vocabulary" in spec.provenance


def test_fake_parser_dependency_structures_support_output_modes_serialization_and_no_mutation() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = parser_dependency_structure_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features_per_kind=None,
        output="pandas",
        parsed_documents=_dependency_documents(),
    )

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ParserDependencyStructureTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = parser_dependency_structure_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features_per_kind=3,
        output="sparse",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    numpy_result = parser_dependency_structure_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features_per_kind=3,
        output="numpy",
        parsed_documents=_dependency_documents(),
    ).fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(transformer,), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(pandas_result, restored)
    pd.testing.assert_frame_equal(pandas_result, extractor_result)
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == (2, 12)
    assert numpy_result.shape == (2, 12)


def test_fake_parser_dependency_structures_validate_missing_layers_configuration_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["left", "right"]}, index=["doc-a", "doc-b"])
    missing_transformer = parser_dependency_structure_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features_per_kind=4,
        output="pandas",
        parsed_documents=_dependency_documents()[:1],
    )
    invalid_transformer = parser_dependency_structure_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        ngram_range=(0, 2),
        max_features_per_kind=4,
        output="pandas",
        parsed_documents=_dependency_documents(),
    )
    root_only_transformer = parser_dependency_structure_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features_per_kind=4,
        output="pandas",
        parsed_documents=(
            ParsedDocument(
                document_id="doc-root-only",
                tokens=(ParsedToken(text="Root", upos="NOUN", morphology=()),),
                dependency_arcs=(ParsedDependencyArc(head_index=None, dependent_index=0, relation="root"),),
            ),
        ),
    )
    not_fitted_transformer = parser_dependency_structure_transformer(
        provider="fake",
        model="fixture-deps",
        version="1",
        text_column="text",
        ngram_range=(1, 2),
        max_features_per_kind=4,
        output="pandas",
        parsed_documents=_dependency_documents(),
    )
    root_only_x = pd.DataFrame({"text": ["root"]}, index=["doc-root-only"])

    try:
        missing_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "Missing fake parser document for row id: doc-b"
    else:
        raise AssertionError("Expected missing fake parser document validation to fail")
    try:
        invalid_transformer.fit(x, None)
    except ValueError as error:
        assert str(error) == "ngram_range lower bound must be positive"
    else:
        raise AssertionError("Expected ngram_range validation to fail")
    try:
        root_only_transformer.fit(root_only_x, None)
    except ValueError as error:
        assert str(error) == "No parser dependency structures found for fitted corpus"
    else:
        raise AssertionError("Expected empty dependency-structure vocabulary validation to fail")
    try:
        not_fitted_transformer.get_feature_names_out(None)
    except NotFittedError as error:
        assert "feature_names_out_" in str(error)
    else:
        raise AssertionError("Expected get_feature_names_out to require fit")


def test_parser_dependency_structures_real_provider_fails_fast_until_adapter_exists() -> None:
    x = pd.DataFrame({"text": ["Parser adapter is not installed."]})
    transformer = parser_dependency_structure_transformer(
        provider="spacy",
        model="en_core_web_sm",
        version="3.7.0",
        text_column="text",
        ngram_range=(1, 2),
        max_features_per_kind=4,
        output="pandas",
        parsed_documents=None,
    )

    try:
        transformer.fit(x, None)
    except OptionalDependencyError as error:
        assert "provider=spacy" in str(error)
        assert "model=en_core_web_sm" in str(error)
    else:
        raise AssertionError("Expected real parser provider to fail fast")
