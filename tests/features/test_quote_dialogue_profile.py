"""Tests for quote and dialogue profile features."""

from __future__ import annotations

import math
import pickle
from typing import cast

import pandas as pd
from scipy import sparse

from stylometry_python_lib import (
    FeatureExtractor,
    QuoteDialogueProfileTransformer,
    QuoteDialogueSidecar,
    english_preprocessing_config,
    quote_dialogue_profile_feature_names,
)

QUOTE_DIALOGUE_TEXT = (
    "Alice said, \"She called it 'odd'.\"\nBob asked, “Ready?”\n> quoted reply\n—Wait, don't go.\n‘Fine,’ she said. He’s late."
)


def _as_frame(value: object) -> pd.DataFrame:
    assert isinstance(value, pd.DataFrame)
    return value


def _cell(frame: pd.DataFrame, column: str, row: int = 0) -> float:
    return float(frame[column].iloc[row])


def test_quote_dialogue_profile_has_golden_quote_mark_signal_and_rate_values() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [QUOTE_DIALOGUE_TEXT]}, index=["doc-dialogue"])
    transformer = QuoteDialogueProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert result.index.tolist() == ["doc-dialogue"]
    assert result.shape[1] == 20
    assert tuple(result.columns) == quote_dialogue_profile_feature_names()
    assert _cell(result, "text::quote_dialogue_profile::quote_mark=straight_double_quote::count") == 2.0
    assert _cell(result, "text::quote_dialogue_profile::quote_mark=straight_single_quote::count") == 2.0
    assert _cell(result, "text::quote_dialogue_profile::quote_mark=curly_double_quote::count") == 2.0
    assert _cell(result, "text::quote_dialogue_profile::quote_mark=curly_single_quote::count") == 2.0
    assert _cell(result, "text::quote_dialogue_profile::quote_mark=apostrophe::count") == 2.0
    assert _cell(result, "text::quote_dialogue_profile::signal=quoted_span::count") == 4.0
    assert _cell(result, "text::quote_dialogue_profile::signal=nested_quote::count") == 1.0
    assert _cell(result, "text::quote_dialogue_profile::signal=block_quote_line::count") == 1.0
    assert _cell(result, "text::quote_dialogue_profile::signal=dialogue_dash_line::count") == 1.0
    assert _cell(result, "text::quote_dialogue_profile::signal=speaker_tag::count") == 3.0
    assert math.isclose(_cell(result, "text::quote_dialogue_profile::quote_mark=straight_double_quote::per_1000_tokens"), 100.0)
    assert math.isclose(_cell(result, "text::quote_dialogue_profile::signal=quoted_span::per_1000_tokens"), 200.0)
    assert math.isclose(_cell(result, "text::quote_dialogue_profile::signal=speaker_tag::per_1000_tokens"), 150.0)
    assert transformer.last_diagnostics_[0] == ()
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, QuoteDialogueSidecar)
    assert sidecar.document_id == "doc-dialogue"
    assert sidecar.schema_version == "quote_dialogue_attribution_v1"
    assert sidecar.quote_span_count == 4
    assert sidecar.speaker_tag_count == 3
    assert [(span.kind, span.start, span.end) for span in sidecar.quote_spans] == [
        ("straight_double_quote", 12, 34),
        ("straight_single_quote", 27, 32),
        ("curly_double_quote", 46, 54),
        ("curly_single_quote", 87, 94),
    ]
    assert [(tag.pattern_id, tag.speaker, tag.speech_verb, tag.text, tag.start, tag.end) for tag in sidecar.speaker_tags] == [
        ("pre_quote_speaker_verb", "Alice", "said", 'Alice said, "', 0, 13),
        ("pre_quote_speaker_verb", "Bob", "asked", "Bob asked, “", 35, 47),
        ("post_quote_speaker_verb", "she", "said", "’ she said", 93, 103),
    ]
    assert [(line.line_index, line.signal_id, line.line_text) for line in sidecar.line_matches] == [
        (2, "block_quote_line", "> quoted reply"),
        (3, "dialogue_dash_line", "—Wait, don't go."),
    ]
    spec = transformer.registry_.by_name("text::quote_dialogue_profile::signal=speaker_tag::count")
    assert spec.topic_dependence.value == "mixed"
    assert spec.provenance == "built_in_quote_dialogue_profile_rules:v1; preprocessing_config"


def test_quote_dialogue_profile_empty_text_uses_explicit_undefined_diagnostics() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [""]})
    transformer = QuoteDialogueProfileTransformer(text_column="text", config=config, output="pandas")

    result = _as_frame(transformer.fit_transform(x, None))

    assert _cell(result, "text::quote_dialogue_profile::quote_mark=straight_double_quote::count") == 0.0
    assert math.isnan(_cell(result, "text::quote_dialogue_profile::quote_mark=straight_double_quote::per_1000_tokens"))
    assert _cell(result, "text::quote_dialogue_profile::signal=speaker_tag::count") == 0.0
    assert math.isnan(_cell(result, "text::quote_dialogue_profile::signal=speaker_tag::per_1000_tokens"))
    assert transformer.last_sidecars_[0].document_id == "0"
    assert transformer.last_sidecars_[0].quote_span_count == 0
    assert transformer.last_sidecars_[0].speaker_tag_count == 0
    assert transformer.last_sidecars_[0].quote_spans == ()
    assert transformer.last_sidecars_[0].speaker_tags == ()
    assert transformer.last_sidecars_[0].line_matches == ()
    reasons = {diagnostic.reason for diagnostic in transformer.last_diagnostics_[0]}
    assert reasons == {"zero_tokens"}


def test_quote_dialogue_profile_supports_output_modes_serialization_and_no_mutation() -> None:
    config = english_preprocessing_config()
    x = pd.DataFrame({"text": [QUOTE_DIALOGUE_TEXT, "plain don't He’s words"]}, index=["first", "second"])
    original = x.copy(deep=True)
    transformer = QuoteDialogueProfileTransformer("text", config, "pandas")

    pandas_result = _as_frame(transformer.fit_transform(x, None))
    loaded = cast(QuoteDialogueProfileTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = _as_frame(loaded.transform(x))
    sparse_result = QuoteDialogueProfileTransformer("text", config, "sparse").fit_transform(x, None)
    numpy_result = QuoteDialogueProfileTransformer("text", config, "numpy").fit_transform(x, None)
    extractor = FeatureExtractor(blocks=(QuoteDialogueProfileTransformer("text", config, "pandas"),), output="pandas")
    extractor_result = _as_frame(extractor.fit_transform(x, None))

    pd.testing.assert_frame_equal(x, original)
    assert pandas_result.index.tolist() == ["first", "second"]
    pd.testing.assert_frame_equal(pandas_result, restored)
    assert _cell(pandas_result, "text::quote_dialogue_profile::quote_mark=apostrophe::count", row=1) == 2.0
    assert _cell(pandas_result, "text::quote_dialogue_profile::quote_mark=straight_single_quote::count", row=1) == 0.0
    assert _cell(pandas_result, "text::quote_dialogue_profile::quote_mark=curly_single_quote::count", row=1) == 0.0
    assert len(extractor.last_sidecars_) == 1
    assert extractor.last_sidecars_[0].block_name == "QuoteDialogueProfileTransformer"
    sidecars = cast(tuple[QuoteDialogueSidecar, QuoteDialogueSidecar], extractor.last_sidecars_[0].sidecars)
    assert sidecars[0].document_id == "first"
    assert sidecars[0].quote_span_count == 4
    assert sidecars[1].document_id == "second"
    assert sidecars[1].quote_spans == ()
    assert sidecars[1].speaker_tags == ()
    assert sidecars[1].line_matches == ()
    assert sparse.issparse(sparse_result)
    assert sparse_result.shape == pandas_result.shape
    assert numpy_result.shape == pandas_result.shape
    assert extractor_result.shape == pandas_result.shape
