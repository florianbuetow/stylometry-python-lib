"""Tests for discourse marker and transition phrase profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    DiscourseLexiconProfileTransformer,
    DiscourseLexiconSidecar,
    FeatureExtractor,
    discourse_marker_profile_feature_names,
    discourse_marker_profile_transformer,
    english_preprocessing_config,
    load_lexicon,
    transition_phrase_profile_feature_names,
    transition_phrase_profile_transformer,
)

DISCOURSE_TRANSITION_TEXT = (
    "However, well, we proceed. In fact, it works; therefore, in addition, for example, we revise. "
    "On the other hand, by contrast, we pause."
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_discourse_and_transition_resources_have_required_metadata_and_order() -> None:
    discourse = load_lexicon("discourse_markers")
    transitions = load_lexicon("transition_phrases")

    assert discourse.lexicon_id == "discourse_markers_en_v1"
    assert discourse.language == "en"
    assert discourse.version == "1.0.0"
    assert discourse.normalization == "lowercase_raw_phrase_match_with_boundary_guards"
    assert discourse.tokens() == ("however", "therefore", "moreover", "nevertheless", "anyway", "well", "indeed", "in fact")
    assert discourse.groups() == (
        "contrast",
        "concession",
        "consequence",
        "additive",
        "topic_shift",
        "interactional",
        "emphasis",
        "clarification",
    )
    assert transitions.lexicon_id == "transition_phrases_en_v1"
    assert transitions.language == "en"
    assert transitions.version == "1.0.0"
    assert transitions.tokens() == (
        "on the other hand",
        "in addition",
        "as a result",
        "for example",
        "in conclusion",
        "by contrast",
    )
    assert transitions.groups() == ("contrast", "additive", "consequence", "example", "conclusion")


def test_discourse_marker_profile_has_golden_item_group_rates_and_polyfunctional_warnings() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [DISCOURSE_TRANSITION_TEXT]}, index=["doc-discourse"])
    transformer = discourse_marker_profile_transformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-discourse"]
    assert result.shape[1] == 32
    assert tuple(result.columns) == discourse_marker_profile_feature_names(transformer.lexicon_)
    assert _cell(result, "text::discourse_marker_profile::item=however::count") == 1.0
    assert _cell(result, "text::discourse_marker_profile::item=well::count") == 1.0
    assert _cell(result, "text::discourse_marker_profile::item=in_fact::count") == 1.0
    assert _cell(result, "text::discourse_marker_profile::item=therefore::count") == 1.0
    assert _cell(result, "text::discourse_marker_profile::item=moreover::count") == 0.0
    assert _cell(result, "text::discourse_marker_profile::group=contrast::count") == 1.0
    assert _cell(result, "text::discourse_marker_profile::group=concession::count") == 1.0
    assert _cell(result, "text::discourse_marker_profile::group=topic_shift::count") == 1.0
    assert _cell(result, "text::discourse_marker_profile::group=interactional::count") == 1.0
    assert _cell(result, "text::discourse_marker_profile::group=emphasis::count") == 1.0
    assert _cell(result, "text::discourse_marker_profile::group=clarification::count") == 1.0
    assert math.isclose(_cell(result, "text::discourse_marker_profile::item=however::per_1000_tokens"), 1000.0 / 23.0)
    assert math.isclose(_cell(result, "text::discourse_marker_profile::group=contrast::per_1000_tokens"), 1000.0 / 23.0)
    reasons = [diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]]
    assert reasons == ["polyfunctional_marker", "polyfunctional_marker", "polyfunctional_marker"]
    warnings = {diagnostic.warnings[0] for diagnostic in transformer.last_diagnostics_[0]}
    assert warnings == {"groups=contrast,concession", "groups=interactional,topic_shift", "groups=emphasis,clarification"}
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, DiscourseLexiconSidecar)
    assert sidecar.document_id == "doc-discourse"
    assert sidecar.schema_version == "discourse_marker_profile_matches_v1"
    assert sidecar.lexicon_id == "discourse_markers_en_v1"
    assert sidecar.match_count == 4
    assert [(match.item_id, match.text, match.start, match.end, match.groups, match.polyfunctional) for match in sidecar.matches] == [
        ("however", "However", 0, 7, ("contrast", "concession"), True),
        ("well", "well", 9, 13, ("interactional", "topic_shift"), True),
        ("in_fact", "In fact", 27, 34, ("emphasis", "clarification"), True),
        ("therefore", "therefore", 46, 55, ("consequence",), False),
    ]
    spec = transformer.registry_.by_name("text::discourse_marker_profile::item=however::count")
    assert spec.topic_dependence.value == "mixed"
    assert "lexicon_id=discourse_markers_en_v1" in spec.provenance


def test_transition_phrase_profile_has_golden_item_group_and_rate_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [DISCOURSE_TRANSITION_TEXT]}, index=["doc-transition"])
    transformer = transition_phrase_profile_transformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-transition"]
    assert result.shape[1] == 22
    assert tuple(result.columns) == transition_phrase_profile_feature_names(transformer.lexicon_)
    assert _cell(result, "text::transition_phrase_profile::item=on_the_other_hand::count") == 1.0
    assert _cell(result, "text::transition_phrase_profile::item=in_addition::count") == 1.0
    assert _cell(result, "text::transition_phrase_profile::item=for_example::count") == 1.0
    assert _cell(result, "text::transition_phrase_profile::item=by_contrast::count") == 1.0
    assert _cell(result, "text::transition_phrase_profile::item=as_a_result::count") == 0.0
    assert _cell(result, "text::transition_phrase_profile::group=contrast::count") == 2.0
    assert _cell(result, "text::transition_phrase_profile::group=additive::count") == 1.0
    assert _cell(result, "text::transition_phrase_profile::group=example::count") == 1.0
    assert math.isclose(_cell(result, "text::transition_phrase_profile::item=in_addition::per_1000_tokens"), 1000.0 / 23.0)
    assert math.isclose(_cell(result, "text::transition_phrase_profile::group=contrast::per_1000_tokens"), 2000.0 / 23.0)
    assert transformer.last_diagnostics_[0] == ()
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, DiscourseLexiconSidecar)
    assert sidecar.document_id == "doc-transition"
    assert sidecar.schema_version == "transition_phrase_profile_matches_v1"
    assert sidecar.lexicon_id == "transition_phrases_en_v1"
    assert sidecar.match_count == 4
    assert [(match.item_id, match.text, match.start, match.end, match.groups) for match in sidecar.matches] == [
        ("in_addition", "in addition", 57, 68, ("additive",)),
        ("for_example", "for example", 70, 81, ("example",)),
        ("on_the_other_hand", "On the other hand", 94, 111, ("contrast",)),
        ("by_contrast", "by contrast", 113, 124, ("contrast",)),
    ]
    spec = transformer.registry_.by_name("text::transition_phrase_profile::item=in_addition::count")
    assert spec.topic_dependence.value == "mixed"
    assert "lexicon_id=transition_phrases_en_v1" in spec.provenance


def test_discourse_profiles_empty_text_use_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    discourse = discourse_marker_profile_transformer("text", config, "pandas")
    transitions = transition_phrase_profile_transformer("text", config, "pandas")

    discourse_result = _as_frame(discourse.fit_transform(x, None))
    transition_result = _as_frame(transitions.fit_transform(x, None))

    assert _cell(discourse_result, "text::discourse_marker_profile::item=however::count") == 0.0
    assert math.isnan(_cell(discourse_result, "text::discourse_marker_profile::item=however::per_1000_tokens"))
    assert _cell(transition_result, "text::transition_phrase_profile::item=in_addition::count") == 0.0
    assert math.isnan(_cell(transition_result, "text::transition_phrase_profile::item=in_addition::per_1000_tokens"))
    assert discourse.last_sidecars_[0].document_id == "0"
    assert discourse.last_sidecars_[0].matches == ()
    assert transitions.last_sidecars_[0].document_id == "0"
    assert transitions.last_sidecars_[0].matches == ()
    assert {diagnostic.reason for diagnostic in discourse.last_diagnostics_[0]} == {"zero_tokens"}
    assert {diagnostic.reason for diagnostic in transitions.last_diagnostics_[0]} == {"zero_tokens"}


def test_discourse_profiles_support_output_modes_serialization_no_mutation_and_boundary_guards() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [DISCOURSE_TRANSITION_TEXT, "well-being is not well"]}, index=["first", "second"])
    original = x.copy(deep=True)
    discourse = discourse_marker_profile_transformer("text", config, "pandas")
    transitions = transition_phrase_profile_transformer("text", config, "pandas")

    discourse_result = _as_frame(discourse.fit_transform(x, None))
    transition_result = _as_frame(transitions.fit_transform(x, None))
    loaded = cast(DiscourseLexiconProfileTransformer, pickle.loads(pickle.dumps(discourse)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = transition_phrase_profile_transformer("text", config, "sparse").fit_transform(x, None)
    numpy_result = discourse_marker_profile_transformer("text", config, "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(
        blocks=(
            discourse_marker_profile_transformer("text", config, "pandas"),
            transition_phrase_profile_transformer("text", config, "pandas"),
        ),
        output="pandas",
    )
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    pd.testing.assert_frame_equal(discourse_result, restored)
    assert _cell(discourse_result, "text::discourse_marker_profile::item=well::count", row=1) == 1.0
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == transition_result.shape
    assert numpy_result.shape == discourse_result.shape
    assert extractor_result.shape[0] == 2
    assert extractor_result.shape[1] == discourse_result.shape[1] + transition_result.shape[1]
    assert discourse.last_sidecars_[1].match_count == 1
    assert discourse.last_sidecars_[1].matches[0].text == "well"
    assert discourse.last_sidecars_[1].matches[0].start == len("well-being is not ")
    assert len(extractor.last_sidecars_) == 2
    assert extractor.last_sidecars_[0].block_name == "DiscourseLexiconProfileTransformer"
    discourse_sidecars = cast(tuple[DiscourseLexiconSidecar, DiscourseLexiconSidecar], extractor.last_sidecars_[0].sidecars)
    transition_sidecars = cast(tuple[DiscourseLexiconSidecar, DiscourseLexiconSidecar], extractor.last_sidecars_[1].sidecars)
    assert discourse_sidecars[0].document_id == "first"
    assert discourse_sidecars[1].document_id == "second"
    assert transition_sidecars[0].document_id == "first"
    assert transition_sidecars[1].matches == ()
