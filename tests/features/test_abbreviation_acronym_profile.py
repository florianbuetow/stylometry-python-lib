"""Tests for abbreviation and acronym profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    AbbreviationAcronymProfileTransformer,
    AbbreviationAcronymSidecar,
    FeatureExtractor,
    abbreviation_acronym_profile_feature_names,
    abbreviation_acronym_profile_transformer,
    english_preprocessing_config,
    load_lexicon,
)

ABBREVIATION_ACRONYM_TEXT = "Dr. Ray met Prof. Lin at NASA. E.g. the API and U.S.A. teams used dept. notes vs. ISO9001 drafts."


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_abbreviation_resource_has_required_metadata_and_order() -> None:
    lexicon = load_lexicon("abbreviations")
    acronyms = load_lexicon("acronyms")

    assert lexicon.name == "abbreviations"
    assert lexicon.lexicon_id == "abbreviations_en_v1"
    assert lexicon.language == "en"
    assert lexicon.version == "1.0.0"
    assert lexicon.license_note != ""
    assert lexicon.normalization == "lowercase_raw_exact_match_with_boundary_guards"
    assert lexicon.tokens()[:5] == ("e.g.", "eg", "i.e.", "ie", "etc.")
    assert lexicon.groups() == ("latin", "dotted", "undotted", "comparison", "honorific", "institutional", "numbering")
    assert acronyms.name == "acronyms"
    assert acronyms.lexicon_id == "acronyms_en_v1"
    assert acronyms.language == "en"
    assert acronyms.version == "1.0.0"
    assert acronyms.license_note != ""
    assert acronyms.normalization == "case_sensitive_raw_exact_match_with_boundary_guards"
    assert acronyms.tokens()[:5] == ("NASA", "API", "U.S.A.", "ISO", "FBI")
    assert acronyms.groups() == ("agency", "technology", "geopolitical", "dotted", "standards", "organization", "web")


def test_abbreviation_acronym_profile_has_golden_item_group_and_class_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [ABBREVIATION_ACRONYM_TEXT]}, index=["doc-abbrev"])
    transformer = abbreviation_acronym_profile_transformer(
        text_column="text",
        config=config,
        lexicon_name="abbreviations",
        output="pandas",
    )

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-abbrev"]
    assert result.shape[1] == 80
    assert tuple(result.columns) == abbreviation_acronym_profile_feature_names(transformer.lexicon_)
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_item=dr.::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_item=prof.::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_item=e.g.::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_item=dept.::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_item=vs.::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_item=eg::count") == 0.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_item=ie::count") == 0.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_group=latin::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_group=dotted::count") == 5.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_group=honorific::count") == 2.0
    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_group=institutional::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_item=NASA::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_item=API::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_item=U.S.A.::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_item=ISO::count") == 0.0
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_group=agency::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_group=technology::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_group=geopolitical::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_group=dotted::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_group=standards::count") == 0.0
    assert _cell(result, "text::abbreviation_acronym_profile::acronym_class=all_caps::count") == 2.0
    assert _cell(result, "text::abbreviation_acronym_profile::acronym_class=dotted::count") == 1.0
    assert _cell(result, "text::abbreviation_acronym_profile::acronym_class=mixed_alnum::count") == 1.0
    assert math.isclose(_cell(result, "text::abbreviation_acronym_profile::abbreviation_item=dr.::per_1000_tokens"), 1000.0 / 23.0)
    assert math.isclose(_cell(result, "text::abbreviation_acronym_profile::abbreviation_group=dotted::per_1000_tokens"), 5000.0 / 23.0)
    assert math.isclose(_cell(result, "text::abbreviation_acronym_profile::named_acronym_item=NASA::per_1000_tokens"), 1000.0 / 23.0)
    assert math.isclose(_cell(result, "text::abbreviation_acronym_profile::named_acronym_group=technology::per_1000_tokens"), 1000.0 / 23.0)
    assert math.isclose(_cell(result, "text::abbreviation_acronym_profile::acronym_class=all_caps::per_1000_tokens"), 2000.0 / 23.0)
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, AbbreviationAcronymSidecar)
    assert sidecar.document_id == "doc-abbrev"
    assert sidecar.schema_version == "abbreviation_acronym_matches_v2"
    assert sidecar.lexicon_id == "abbreviations_en_v1"
    assert sidecar.normalization == "lowercase_raw_exact_match_with_boundary_guards"
    assert sidecar.acronym_lexicon_id == "acronyms_en_v1"
    assert sidecar.acronym_normalization == "case_sensitive_raw_exact_match_with_boundary_guards"
    assert sidecar.abbreviation_match_count == 5
    assert sidecar.named_acronym_match_count == 3
    assert sidecar.acronym_match_count == 4
    assert [(match.match_kind, match.match_id, match.text, match.start, match.end) for match in sidecar.matches] == [
        ("abbreviation", "dr.", "Dr.", 0, 3),
        ("abbreviation", "prof.", "Prof.", 12, 17),
        ("acronym", "all_caps", "NASA", 25, 29),
        ("named_acronym", "NASA", "NASA", 25, 29),
        ("abbreviation", "e.g.", "E.g.", 31, 35),
        ("acronym", "all_caps", "API", 40, 43),
        ("named_acronym", "API", "API", 40, 43),
        ("acronym", "dotted", "U.S.A.", 48, 54),
        ("named_acronym", "U.S.A.", "U.S.A.", 48, 54),
        ("abbreviation", "dept.", "dept.", 66, 71),
        ("abbreviation", "vs.", "vs.", 78, 81),
        ("acronym", "mixed_alnum", "ISO9001", 82, 89),
    ]
    assert sidecar.matches[0].groups == ("honorific", "dotted")
    assert sidecar.matches[4].groups == ("latin", "dotted")
    assert sidecar.matches[6].groups == ("technology",)
    spec = transformer.registry_.by_name("text::abbreviation_acronym_profile::abbreviation_item=e.g.::per_1000_tokens")
    assert spec.topic_dependence.value == "mixed"
    assert "lexicon_id=abbreviations_en_v1" in spec.provenance
    assert "acronym_lexicon_id=acronyms_en_v1" in spec.provenance
    assert "built_in_acronym_regex_rules:v1" in spec.provenance


def test_abbreviation_acronym_profile_empty_text_uses_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = abbreviation_acronym_profile_transformer("text", config, "abbreviations", "pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::abbreviation_acronym_profile::abbreviation_item=dr.::count") == 0.0
    assert math.isnan(_cell(result, "text::abbreviation_acronym_profile::abbreviation_item=dr.::per_1000_tokens"))
    assert _cell(result, "text::abbreviation_acronym_profile::named_acronym_item=NASA::count") == 0.0
    assert math.isnan(_cell(result, "text::abbreviation_acronym_profile::named_acronym_item=NASA::per_1000_tokens"))
    assert _cell(result, "text::abbreviation_acronym_profile::acronym_class=all_caps::count") == 0.0
    assert math.isnan(_cell(result, "text::abbreviation_acronym_profile::acronym_class=all_caps::per_1000_tokens"))
    assert transformer.last_sidecars_[0].document_id == "0"
    assert transformer.last_sidecars_[0].abbreviation_match_count == 0
    assert transformer.last_sidecars_[0].named_acronym_match_count == 0
    assert transformer.last_sidecars_[0].acronym_match_count == 0
    assert transformer.last_sidecars_[0].matches == ()
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert reasons == {"zero_tokens"}


def test_abbreviation_acronym_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [ABBREVIATION_ACRONYM_TEXT, "plain words only"]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = abbreviation_acronym_profile_transformer("text", config, "abbreviations", "pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(AbbreviationAcronymProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = abbreviation_acronym_profile_transformer("text", config, "abbreviations", "sparse").fit_transform(x, None)
    numpy_result = abbreviation_acronym_profile_transformer("text", config, "abbreviations", "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(
        blocks=(abbreviation_acronym_profile_transformer("text", config, "abbreviations", "pandas"),),
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
    assert extractor.last_sidecars_[0].block_name == "AbbreviationAcronymProfileTransformer"
    sidecars = cast(tuple[AbbreviationAcronymSidecar, AbbreviationAcronymSidecar], extractor.last_sidecars_[0].sidecars)
    assert sidecars[0].document_id == "first"
    assert sidecars[1].document_id == "second"
    assert sidecars[1].matches == ()
