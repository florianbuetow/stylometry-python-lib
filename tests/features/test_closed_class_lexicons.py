"""Tests for versioned closed-class lexicon feature blocks."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    ClosedClassLexiconTransformer,
    ContractionExpansionSidecar,
    FeatureExtractor,
    auxiliary_lexicon_transformer,
    contraction_lexicon_transformer,
    english_preprocessing_config,
    function_word_lexicon_transformer,
    load_lexicon,
    modal_lexicon_transformer,
    pronoun_lexicon_transformer,
    stopword_lexicon_transformer,
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_versioned_closed_class_lexicons_have_required_metadata_and_order() -> None:
    expected_tokens = {
        "function_words": ("a", "an", "and", "are", "as"),
        "stopwords": ("a", "am", "an", "and", "are"),
        "pronouns": ("he", "her", "hers", "him", "his"),
        "modals": ("can", "could", "may", "might", "must"),
        "auxiliaries": ("am", "are", "be", "been", "being"),
        "contractions": ("ain't", "can't", "couldn't", "he's", "i'm"),
    }

    for name, tokens in expected_tokens.items():
        lexicon = load_lexicon(name)
        assert lexicon.name == name
        assert lexicon.lexicon_id.endswith("_en_v1")
        assert lexicon.language == "en"
        assert lexicon.version == "1.0.0"
        assert lexicon.source != ""
        assert lexicon.license_note != ""
        assert lexicon.normalization != ""
        assert lexicon.tokens()[:5] == tokens
        assert len(set(lexicon.tokens())) == len(lexicon.tokens())
        assert all(len(entry.groups) > 0 for entry in lexicon.entries)

    contractions = load_lexicon("contractions")
    by_token = {entry.token: entry for entry in contractions.entries}
    assert by_token["can't"].expansion == ("can", "not")
    assert by_token["can't"].expansion_alternatives == (("can", "not"),)
    assert by_token["she's"].expansion == ("she", "is")
    assert by_token["she's"].expansion_alternatives == (("she", "is"), ("she", "has"))


def test_pronoun_lexicon_emits_item_and_group_counts_and_rates() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["I you it we they"]}, index=["doc-a"])
    transformer = pronoun_lexicon_transformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-a"]
    assert _cell(result, "text::pronoun::item=i::count") == 1.0
    assert _cell(result, "text::pronoun::item=i::per_1000_tokens") == 200.0
    assert _cell(result, "text::pronoun::group=first_person::count") == 2.0
    assert _cell(result, "text::pronoun::group=first_person::per_1000_tokens") == 400.0
    assert _cell(result, "text::pronoun::group=second_person::count") == 1.0
    assert _cell(result, "text::pronoun::group=third_person::count") == 2.0
    assert "lexicon_id=pronouns_en_v1" in transformer.registry_.by_name("text::pronoun::item=i::count").provenance


def test_function_word_lexicon_emits_item_and_group_counts_and_rates() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["the and for that to it then there"]}, index=["doc-fw"])
    transformer = function_word_lexicon_transformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-fw"]
    assert _cell(result, "text::function_word::item=the::count") == 1.0
    assert _cell(result, "text::function_word::item=the::per_1000_tokens") == 125.0
    assert _cell(result, "text::function_word::group=article::count") == 1.0
    assert _cell(result, "text::function_word::group=coordinating_conjunction::count") == 1.0
    assert _cell(result, "text::function_word::group=subordinator::count") == 2.0
    assert _cell(result, "text::function_word::group=subordinator::per_1000_tokens") == 250.0
    assert _cell(result, "text::function_word::group=preposition::count") == 2.0
    assert _cell(result, "text::function_word::group=pronoun::count") == 1.0
    assert _cell(result, "text::function_word::group=determiner::count") == 1.0
    assert _cell(result, "text::function_word::group=adverbial_connector::count") == 1.0
    assert _cell(result, "text::function_word::group=existential::count") == 1.0
    assert _cell(result, "text::function_word::group=infinitive_marker::count") == 1.0
    spec = transformer.registry_.by_name("text::function_word::group=article::per_1000_tokens")
    assert spec.topic_dependence.value == "mostly_topic_independent"
    assert "lexicon_id=function_words_en_v1" in spec.provenance


def test_stopword_modal_auxiliary_and_contraction_lexicon_golden_values() -> None:
    config = english_preprocessing_config()
    stopword_frame = _as_frame(
        stopword_lexicon_transformer("text", config, "pandas").fit_transform(pd.DataFrame({"text": ["the can we go"]}), None)
    )
    modal_frame = _as_frame(
        modal_lexicon_transformer("text", config, "pandas").fit_transform(pd.DataFrame({"text": ["can must would must"]}), None)
    )
    auxiliary_frame = _as_frame(
        auxiliary_lexicon_transformer("text", config, "pandas").fit_transform(pd.DataFrame({"text": ["am are have did"]}), None)
    )
    contraction_frame = _as_frame(
        contraction_lexicon_transformer("text", config, "pandas").fit_transform(pd.DataFrame({"text": ["can't I'm she's won't"]}), None)
    )

    assert _cell(stopword_frame, "text::stopword::item=the::count") == 1.0
    assert _cell(stopword_frame, "text::stopword::group=function_word::count") == 1.0
    assert _cell(stopword_frame, "text::stopword::group=modal::count") == 1.0
    assert _cell(stopword_frame, "text::stopword::group=pronoun::count") == 1.0
    assert _cell(modal_frame, "text::modal::item=must::count") == 2.0
    assert _cell(modal_frame, "text::modal::group=core_modal::per_1000_tokens") == 1000.0
    assert _cell(modal_frame, "text::modal::group=epistemic::count") == 3.0
    assert _cell(auxiliary_frame, "text::auxiliary::group=be::count") == 2.0
    assert _cell(auxiliary_frame, "text::auxiliary::group=have::count") == 1.0
    assert _cell(auxiliary_frame, "text::auxiliary::group=do::count") == 1.0
    assert _cell(contraction_frame, "text::contraction::item=can't::count") == 1.0
    assert _cell(contraction_frame, "text::contraction::group=negation::count") == 2.0
    assert _cell(contraction_frame, "text::contraction::group=modal::count") == 2.0
    assert _cell(contraction_frame, "text::contraction::group=pronoun_auxiliary::count") == 2.0
    assert _cell(contraction_frame, "text::contraction::group=ambiguous::count") == 1.0


def test_contraction_lexicon_emits_expansion_sidecars_and_ambiguous_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["Can't I'm she's won't."]}, index=["doc-contract"])
    transformer = contraction_lexicon_transformer("text", config, "pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-contract"]
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, ContractionExpansionSidecar)
    assert sidecar.document_id == "doc-contract"
    assert sidecar.lexicon_id == "contractions_en_v1"
    assert sidecar.schema_version == "contraction_expansion_v1"
    assert sidecar.match_count == 4
    assert sidecar.ambiguous_match_count == 1
    assert sidecar.warnings == ("ambiguous_contraction_expansion",)
    by_token = {match.token: match for match in sidecar.matches}
    assert by_token["can't"].orthographic_token_index == 0
    assert by_token["can't"].expansion == ("can", "not")
    assert by_token["can't"].expansion_alternatives == (("can", "not"),)
    assert by_token["she's"].ambiguous
    assert by_token["she's"].expansion == ("she", "is")
    assert by_token["she's"].expansion_alternatives == (("she", "is"), ("she", "has"))
    assert by_token["she's"].warnings == ("ambiguous_contraction_expansion",)
    ambiguous_diagnostics = [
        diagnostic for diagnostic in transformer.last_diagnostics_[0] if diagnostic.reason == "ambiguous_contraction_expansion"
    ]
    assert len(ambiguous_diagnostics) == 1
    assert ambiguous_diagnostics[0].feature_name == "text::contraction::item=she's::count"
    assert ambiguous_diagnostics[0].warnings == ("expansion_alternatives=she is;she has",)


def test_closed_class_lexicon_undefined_rates_sparse_output_and_serialization() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["", "I can go"]})
    transformer = pronoun_lexicon_transformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ClosedClassLexiconTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = pronoun_lexicon_transformer(text_column="text", config=config, output="sparse").fit_transform(x, None)
    numpy_result = pronoun_lexicon_transformer(text_column="text", config=config, output="numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(pronoun_lexicon_transformer("text", config, "pandas"),), output="pandas")
    extractor_frame = _as_frame(extractor.fit_transform(x, None))

    assert _cell(result, "text::pronoun::item=i::count", row=0) == 0.0
    assert math.isnan(_cell(result, "text::pronoun::item=i::per_1000_tokens", row=0))
    assert result["text::pronoun::item=i::per_1000_tokens"].iloc[1] == 1000.0 / 3.0
    assert transformer.last_diagnostics_[0][0].reason == "zero_tokens"
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == result.shape
    assert numpy_result.shape == result.shape
    pd.testing.assert_frame_equal(result, restored)
    assert extractor_frame.shape == result.shape


def test_function_word_lexicon_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["", "the and"]}, index=["empty", "filled"])
    original = x.copy(deep=True)
    transformer = function_word_lexicon_transformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(ClosedClassLexiconTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = function_word_lexicon_transformer(text_column="text", config=config, output="sparse").fit_transform(x, None)
    numpy_result = function_word_lexicon_transformer(text_column="text", config=config, output="numpy").fit_transform(x, None)

    pd.testing.assert_frame_equal(x, original)
    assert _cell(result, "text::function_word::item=the::count", row=0) == 0.0
    assert math.isnan(_cell(result, "text::function_word::item=the::per_1000_tokens", row=0))
    assert result["text::function_word::item=the::per_1000_tokens"].iloc[1] == 500.0
    assert transformer.last_diagnostics_[0][0].reason == "zero_tokens"
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == result.shape
    assert numpy_result.shape == result.shape
    pd.testing.assert_frame_equal(result, restored)


def test_feature_extractor_collects_contraction_sidecars_without_changing_rows() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": ["can't go", "she's here"]}, index=["plain", "ambiguous"])
    extractor = FeatureExtractor(blocks=(contraction_lexicon_transformer("text", config, "pandas"),), output="pandas")

    frame = _as_frame(extractor.fit_transform(x, None))

    assert frame.index.tolist() == ["plain", "ambiguous"]
    assert len(extractor.last_sidecars_) == 1
    assert extractor.last_sidecars_[0].block_name == "ClosedClassLexiconTransformer"
    assert len(extractor.last_sidecars_[0].sidecars) == 2
    sidecars = cast(tuple[ContractionExpansionSidecar, ContractionExpansionSidecar], extractor.last_sidecars_[0].sidecars)
    assert sidecars[0].document_id == "plain"
    assert sidecars[0].ambiguous_match_count == 0
    assert sidecars[1].document_id == "ambiguous"
    assert sidecars[1].ambiguous_match_count == 1
