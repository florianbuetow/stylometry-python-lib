"""Tests for layout and whitespace profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    FeatureExtractor,
    LayoutMarkerSidecar,
    LayoutWhitespaceProfileTransformer,
    english_preprocessing_config,
    layout_whitespace_profile_feature_names,
)

LAYOUT_WHITESPACE_TEXT = "\n".join(
    (
        "# Title",
        "## Details",
        "    indented block",
        "\tTabbed item",
        "- bullet item",
        "1. numbered item",
        "> quoted block",
        "```",
        "code line",
        "```",
        "| A | B |",
        "Subject: Note",
        "This is a deliberately long layout line that should look like it was hard wrapped by a sender",
        "because it continues with lowercase words",
        "trailing spaces   ",
        "Alpha  beta",
    )
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_layout_whitespace_profile_has_golden_marker_depth_spacing_and_wrapping_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [LAYOUT_WHITESPACE_TEXT]}, index=["doc-layout"])
    transformer = LayoutWhitespaceProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-layout"]
    assert result.shape[1] == 34
    assert tuple(result.columns) == layout_whitespace_profile_feature_names()
    assert _cell(result, "text::layout_whitespace_profile::layout_marker=markdown_heading_line::count") == 2.0
    assert _cell(result, "text::layout_whitespace_profile::layout_marker=bullet_line::count") == 1.0
    assert _cell(result, "text::layout_whitespace_profile::layout_marker=numbered_line::count") == 1.0
    assert _cell(result, "text::layout_whitespace_profile::layout_marker=block_quote_line::count") == 1.0
    assert _cell(result, "text::layout_whitespace_profile::layout_marker=code_fence_line::count") == 2.0
    assert _cell(result, "text::layout_whitespace_profile::layout_marker=table_line::count") == 1.0
    assert _cell(result, "text::layout_whitespace_profile::layout_marker=email_header_line::count") == 1.0
    assert math.isclose(
        _cell(result, "text::layout_whitespace_profile::layout_marker=markdown_heading_line::per_100_lines"),
        12.5,
    )
    assert _cell(result, "text::layout_whitespace_profile::section_depth::max_markdown_heading_depth") == 2.0
    assert _cell(result, "text::layout_whitespace_profile::section_depth::mean_markdown_heading_depth") == 1.5
    assert _cell(result, "text::layout_whitespace_profile::whitespace=tab_character::count") == 1.0
    assert _cell(result, "text::layout_whitespace_profile::whitespace=tab_indented_line::count") == 1.0
    assert _cell(result, "text::layout_whitespace_profile::whitespace=space_indented_line::count") == 1.0
    assert _cell(result, "text::layout_whitespace_profile::whitespace=max_indentation_spaces") == 4.0
    assert _cell(result, "text::layout_whitespace_profile::whitespace=mean_indentation_spaces") == 0.25
    assert _cell(result, "text::layout_whitespace_profile::whitespace=repeated_space_run::count") == 3.0
    assert _cell(result, "text::layout_whitespace_profile::whitespace=repeated_space_character::count") == 9.0
    assert _cell(result, "text::layout_whitespace_profile::whitespace=leading_whitespace_line::count") == 2.0
    assert _cell(result, "text::layout_whitespace_profile::whitespace=trailing_whitespace_line::count") == 1.0
    assert _cell(result, "text::layout_whitespace_profile::whitespace=hard_wrap_candidate_line::count") == 1.0
    assert math.isclose(_cell(result, "text::layout_whitespace_profile::whitespace=repeated_space_run::per_100_lines"), 18.75)
    assert transformer.last_diagnostics_[0] == ()
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, LayoutMarkerSidecar)
    assert sidecar.document_id == "doc-layout"
    assert sidecar.schema_version == "layout_marker_sidecar_v1"
    assert sidecar.line_count == 16
    assert len(sidecar.matches) == 9
    assert sidecar.matches[0].line_index == 0
    assert sidecar.matches[0].marker_id == "markdown_heading_line"
    assert sidecar.matches[0].line_text == "# Title"
    assert sidecar.matches[0].markdown_heading_depth == 1
    assert sidecar.matches[1].line_index == 1
    assert sidecar.matches[1].markdown_heading_depth == 2
    assert [match.line_index for match in sidecar.matches if match.marker_id == "code_fence_line"] == [7, 9]
    assert sidecar.matches[-1].marker_id == "email_header_line"
    assert sidecar.matches[-1].line_text == "Subject: Note"
    layout_spec = transformer.registry_.by_name("text::layout_whitespace_profile::layout_marker=markdown_heading_line::count")
    whitespace_spec = transformer.registry_.by_name("text::layout_whitespace_profile::whitespace=tab_character::count")
    assert layout_spec.topic_dependence.value == "mixed"
    assert whitespace_spec.topic_dependence.value == "mostly_topic_independent"
    assert layout_spec.provenance == "built_in_layout_whitespace_profile_rules:v1; preprocessing_config"


def test_layout_whitespace_profile_empty_text_uses_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = LayoutWhitespaceProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::layout_whitespace_profile::layout_marker=markdown_heading_line::count") == 0.0
    assert math.isnan(_cell(result, "text::layout_whitespace_profile::layout_marker=markdown_heading_line::per_100_lines"))
    assert _cell(result, "text::layout_whitespace_profile::section_depth::max_markdown_heading_depth") == 0.0
    assert math.isnan(_cell(result, "text::layout_whitespace_profile::section_depth::mean_markdown_heading_depth"))
    assert _cell(result, "text::layout_whitespace_profile::whitespace=max_indentation_spaces") == 0.0
    assert math.isnan(_cell(result, "text::layout_whitespace_profile::whitespace=mean_indentation_spaces"))
    assert transformer.last_sidecars_[0].document_id == "0"
    assert transformer.last_sidecars_[0].line_count == 0
    assert transformer.last_sidecars_[0].matches == ()
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert reasons == {"zero_lines", "zero_heading_lines", "zero_nonblank_lines"}


def test_layout_whitespace_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [LAYOUT_WHITESPACE_TEXT, "plain text"]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = LayoutWhitespaceProfileTransformer("text", config, "pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(LayoutWhitespaceProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = LayoutWhitespaceProfileTransformer("text", config, "sparse").fit_transform(x, None)
    numpy_result = LayoutWhitespaceProfileTransformer("text", config, "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(LayoutWhitespaceProfileTransformer("text", config, "pandas"),), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert _cell(pandas_result, "text::layout_whitespace_profile::whitespace=hard_wrap_candidate_line::count", row=1) == 0.0
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
    assert len(extractor.last_sidecars_) == 1
    assert extractor.last_sidecars_[0].block_name == "LayoutWhitespaceProfileTransformer"
    sidecars = cast(tuple[LayoutMarkerSidecar, LayoutMarkerSidecar], extractor.last_sidecars_[0].sidecars)
    assert sidecars[0].document_id == "first"
    assert sidecars[1].document_id == "second"
    assert sidecars[1].matches == ()
