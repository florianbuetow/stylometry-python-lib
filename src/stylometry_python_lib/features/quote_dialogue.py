"""Quote and dialogue profile feature block."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib._tabular import text_series, validate_output_mode
from stylometry_python_lib.document import DocumentView, PreprocessingConfig
from stylometry_python_lib.registry import FeatureRegistry
from stylometry_python_lib.specs import FeatureSpec, InputLayer, StabilityStatus, TopicDependence
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus


@dataclass(frozen=True)
class QuoteSpan:
    """One deterministic quote span."""

    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class QuoteMarkClass:
    """One deterministic quote-mark class."""

    class_id: str
    description: str


@dataclass(frozen=True)
class DialogueSignal:
    """One deterministic dialogue structure signal."""

    signal_id: str
    description: str


@dataclass(frozen=True)
class SpeakerTagMatch:
    """One rule-based speaker attribution match adjacent to a quote mark."""

    pattern_id: str
    speaker: str
    speech_verb: str
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class QuoteLineMatch:
    """One block quote or dialogue dash line match."""

    line_index: int
    signal_id: str
    line_text: str


@dataclass(frozen=True)
class QuoteDialogueSidecar:
    """Quote spans, attribution tags, and dialogue line matches for one document."""

    document_id: str
    schema_version: str
    quote_span_count: int
    speaker_tag_count: int
    quote_spans: tuple[QuoteSpan, ...]
    speaker_tags: tuple[SpeakerTagMatch, ...]
    line_matches: tuple[QuoteLineMatch, ...]


class QuoteDialogueProfileTransformer(BaseEstimator):
    """Sklearn-compatible quote and dialogue profile transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze quote/dialogue feature names."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.feature_names_out_ = np.asarray(quote_dialogue_profile_feature_names(), dtype=object)
        self.registry_ = FeatureRegistry(specs=quote_dialogue_profile_feature_specs())
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute quote and dialogue profile features without changing rows."""
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[QuoteDialogueSidecar] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics, sidecar = _quote_dialogue_row(view)
            rows.append(row)
            diagnostics.append(row_diagnostics)
            sidecars.append(sidecar)
        self.last_diagnostics_ = tuple(diagnostics)
        self.last_sidecars_ = tuple(sidecars)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_, index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable quote/dialogue profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def quote_dialogue_profile_feature_names() -> tuple[str, ...]:
    """Return stable quote/dialogue feature names in output order."""
    names: list[str] = []
    for mark_class in _quote_mark_classes():
        names.extend((_quote_mark_name(mark_class.class_id, "count"), _quote_mark_name(mark_class.class_id, "per_1000_tokens")))
    for signal in _dialogue_signals():
        names.extend((_signal_name(signal.signal_id, "count"), _signal_name(signal.signal_id, "per_1000_tokens")))
    return tuple(names)


def quote_dialogue_profile_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return metadata for quote/dialogue profile features."""
    return tuple(_spec_for_name(name) for name in quote_dialogue_profile_feature_names())


def _quote_dialogue_row(view: DocumentView) -> tuple[list[float], tuple[FeatureDiagnostic, ...], QuoteDialogueSidecar]:
    raw = view.raw
    token_count = len(view.tokens)
    quote_mark_counts = _quote_mark_counts(raw)
    quote_spans = _quote_spans(raw)
    speaker_tags = _speaker_tag_matches(raw)
    line_matches = _quote_line_matches(raw)
    signal_counts = _dialogue_signal_counts(quote_spans, speaker_tags, line_matches)
    diagnostics: list[FeatureDiagnostic] = []
    values: list[float] = []

    for mark_class in _quote_mark_classes():
        count = float(quote_mark_counts[mark_class.class_id])
        values.append(count)
        values.append(_per_1000(_quote_mark_name(mark_class.class_id, "per_1000_tokens"), count, token_count, diagnostics))
    for signal in _dialogue_signals():
        count = float(signal_counts[signal.signal_id])
        values.append(count)
        values.append(_per_1000(_signal_name(signal.signal_id, "per_1000_tokens"), count, token_count, diagnostics))
    sidecar = QuoteDialogueSidecar(
        document_id=view.document_id,
        schema_version="quote_dialogue_attribution_v1",
        quote_span_count=len(quote_spans),
        speaker_tag_count=len(speaker_tags),
        quote_spans=quote_spans,
        speaker_tags=speaker_tags,
        line_matches=line_matches,
    )
    return values, tuple(diagnostics), sidecar


def _quote_mark_counts(raw: str) -> dict[str, int]:
    apostrophes = _apostrophe_count(raw)
    straight_apostrophes = len(tuple(re.finditer(r"(?<=[A-Za-z0-9])'(?=[A-Za-z0-9])", raw)))
    curly_apostrophes = len(tuple(re.finditer(r"(?<=[A-Za-z0-9])’(?=[A-Za-z0-9])", raw)))
    straight_single_total = raw.count("'")
    curly_single_total = raw.count("‘") + raw.count("’")
    return {
        "straight_double_quote": raw.count('"'),
        "straight_single_quote": straight_single_total - straight_apostrophes,
        "curly_double_quote": raw.count("“") + raw.count("”"),
        "curly_single_quote": curly_single_total - curly_apostrophes,
        "apostrophe": apostrophes,
    }


def _apostrophe_count(raw: str) -> int:
    return len(tuple(re.finditer(r"(?<=[A-Za-z0-9])['’](?=[A-Za-z0-9])", raw)))


def _dialogue_signal_counts(
    spans: tuple[QuoteSpan, ...],
    speaker_tags: tuple[SpeakerTagMatch, ...],
    line_matches: tuple[QuoteLineMatch, ...],
) -> dict[str, int]:
    return {
        "quoted_span": len(spans),
        "nested_quote": _nested_quote_count(spans),
        "block_quote_line": sum(1 for line_match in line_matches if line_match.signal_id == "block_quote_line"),
        "dialogue_dash_line": sum(1 for line_match in line_matches if line_match.signal_id == "dialogue_dash_line"),
        "speaker_tag": len(speaker_tags),
    }


def _quote_spans(raw: str) -> tuple[QuoteSpan, ...]:
    spans: list[QuoteSpan] = []
    spans.extend(_paired_spans(raw, '"', '"', "straight_double_quote", skip_word_internal=False))
    spans.extend(_paired_spans(raw, "'", "'", "straight_single_quote", skip_word_internal=True))
    spans.extend(_paired_spans(raw, "“", "”", "curly_double_quote", skip_word_internal=False))
    spans.extend(_paired_spans(raw, "‘", "’", "curly_single_quote", skip_word_internal=True))
    spans.sort(key=lambda span: (span.start, span.end, span.kind))
    return tuple(spans)


def _paired_spans(raw: str, open_quote: str, close_quote: str, kind: str, skip_word_internal: bool) -> tuple[QuoteSpan, ...]:
    positions = tuple(_quote_positions(raw, open_quote, close_quote, skip_word_internal))
    spans: list[QuoteSpan] = []
    index = 0
    while index + 1 < len(positions):
        start = positions[index]
        end = positions[index + 1] + 1
        spans.append(QuoteSpan(kind=kind, start=start, end=end))
        index += 2
    return tuple(spans)


def _quote_positions(raw: str, open_quote: str, close_quote: str, skip_word_internal: bool) -> list[int]:
    positions: list[int] = []
    if open_quote == close_quote:
        for index, character in enumerate(raw):
            if character != open_quote:
                continue
            if skip_word_internal and _is_word_internal_quote(raw, index):
                continue
            positions.append(index)
        return positions
    for index, character in enumerate(raw):
        if character not in {open_quote, close_quote}:
            continue
        if skip_word_internal and _is_word_internal_quote(raw, index):
            continue
        positions.append(index)
    return positions


def _is_word_internal_quote(raw: str, index: int) -> bool:
    if index == 0 or index + 1 >= len(raw):
        return False
    return raw[index - 1].isalnum() and raw[index + 1].isalnum()


def _nested_quote_count(spans: tuple[QuoteSpan, ...]) -> int:
    count = 0
    for candidate in spans:
        if any(outer.start < candidate.start and candidate.end < outer.end for outer in spans):
            count += 1
    return count


def _quote_line_matches(raw: str) -> tuple[QuoteLineMatch, ...]:
    matches: list[QuoteLineMatch] = []
    for line_index, line in enumerate(raw.splitlines()):
        if line.lstrip().startswith(">"):
            matches.append(QuoteLineMatch(line_index=line_index, signal_id="block_quote_line", line_text=line))
        if re.match(r"^\s*[—–―]", line) is not None:
            matches.append(QuoteLineMatch(line_index=line_index, signal_id="dialogue_dash_line", line_text=line))
    return tuple(matches)


def _speaker_tag_matches(raw: str) -> tuple[SpeakerTagMatch, ...]:
    speaker = r"(?P<speaker>I|you|he|she|they|we|it|[A-Z][A-Za-z]+)"
    verb = r"(?P<verb>said|asked|replied|whispered|shouted|murmured|answered)"
    close_quote = r"""["”’]"""
    open_quote = r"""["“‘']"""
    pattern_specs = (
        ("pre_quote_speaker_verb", rf"\b{speaker}[ \t]+{verb}[ \t]*,?[ \t]*{open_quote}"),
        ("post_quote_speaker_verb", rf"{close_quote}[ \t]*,?[ \t]*{speaker}[ \t]+{verb}\b"),
        ("post_quote_verb_speaker", rf"{close_quote}[ \t]*,?[ \t]*{verb}[ \t]+{speaker}\b"),
    )
    matches: list[SpeakerTagMatch] = []
    for pattern_id, pattern in pattern_specs:
        matches.extend(
            (
                SpeakerTagMatch(
                    pattern_id=pattern_id,
                    speaker=match.group("speaker"),
                    speech_verb=match.group("verb"),
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                )
            )
            for match in re.finditer(pattern, raw)
        )
    return tuple(sorted(matches, key=lambda match: (match.start, match.end, match.pattern_id)))


def _per_1000(feature_name: str, count: float, token_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if token_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_tokens"))
        return float("nan")
    return count * 1000.0 / float(token_count)


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _quote_mark_classes() -> tuple[QuoteMarkClass, ...]:
    return (
        QuoteMarkClass("straight_double_quote", "Straight double quotation mark"),
        QuoteMarkClass("straight_single_quote", "Straight single quotation mark excluding word-internal apostrophes"),
        QuoteMarkClass("curly_double_quote", "Curly double quotation mark"),
        QuoteMarkClass("curly_single_quote", "Curly single quotation mark excluding word-internal apostrophes"),
        QuoteMarkClass("apostrophe", "Word-internal straight or curly apostrophe"),
    )


def _dialogue_signals() -> tuple[DialogueSignal, ...]:
    return (
        DialogueSignal("quoted_span", "Paired straight or curly quote span"),
        DialogueSignal("nested_quote", "Quote span nested inside another quote span"),
        DialogueSignal("block_quote_line", "Line beginning with a block-quote marker"),
        DialogueSignal("dialogue_dash_line", "Line beginning with an em dash, en dash, or horizontal bar dialogue marker"),
        DialogueSignal("speaker_tag", "Rule-based speech attribution tag adjacent to a quote mark"),
    )


def _quote_mark_name(class_id: str, measure: str) -> str:
    return f"text::quote_dialogue_profile::quote_mark={class_id}::{measure}"


def _signal_name(signal_id: str, measure: str) -> str:
    return f"text::quote_dialogue_profile::signal={signal_id}::{measure}"


def _spec_for_name(name: str) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the quote/dialogue signal is absent"
    formula_or_rule = "count deterministic quote mark class or dialogue structure signal"
    if name.endswith("per_1000_tokens"):
        normalization = "per_1000_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_tokens when no tokens exist"
        formula_or_rule = "quote/dialogue signal count * 1000 / token count"
    return FeatureSpec(
        name=name,
        family="quote_dialogue_profile",
        description=f"Quote and dialogue profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.RAW,
        topic_dependence=TopicDependence.MIXED,
        text_length_policy="counts are always defined; per-1,000-token rates require token denominator",
        provenance="built_in_quote_dialogue_profile_rules:v1; preprocessing_config",
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
