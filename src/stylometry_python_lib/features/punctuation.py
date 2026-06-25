"""Dedicated punctuation inventory and class profile features."""

from __future__ import annotations

import unicodedata
from collections import Counter
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
class PunctuationMark:
    """One fixed punctuation inventory item."""

    mark_id: str
    text: str
    class_ids: tuple[str, ...]
    final_id: str


@dataclass(frozen=True)
class PunctuationOccurrence:
    """One fixed punctuation mark occurrence."""

    character_index: int
    mark_id: str
    text: str
    class_ids: tuple[str, ...]
    final_id: str


@dataclass(frozen=True)
class SentenceFinalPunctuation:
    """Sentence-final punctuation classification for one detected sentence."""

    sentence_index: int
    final_text: str | None
    final_id: str


@dataclass(frozen=True)
class PunctuationProfileSidecar:
    """Raw punctuation occurrence and sentence-final sidecar."""

    document_id: str
    schema_version: str
    punctuation_count: int
    sentence_count: int
    occurrences: tuple[PunctuationOccurrence, ...]
    sentence_finals: tuple[SentenceFinalPunctuation, ...]


class PunctuationProfileTransformer(BaseEstimator):
    """Sklearn-compatible fixed punctuation profile transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze punctuation profile metadata."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.feature_names_out_ = np.asarray(punctuation_profile_feature_names(), dtype=object)
        self.registry_ = FeatureRegistry(specs=punctuation_profile_feature_specs())
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute punctuation profile features without changing rows."""
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[PunctuationProfileSidecar] = []
        for row_index, text in enumerate(series.tolist()):
            view = DocumentView.from_text(str(text), self.config, document_id=str(series.index[row_index]))
            row, row_diagnostics, sidecar = _punctuation_row(view)
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
        """Return stable punctuation profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def punctuation_profile_feature_names() -> tuple[str, ...]:
    """Return stable punctuation profile feature names in output order."""
    names = [_total_name("count"), _total_name("per_1000_tokens"), _total_name("per_sentence")]
    for mark in _punctuation_marks():
        names.extend(
            (_mark_name(mark.mark_id, "count"), _mark_name(mark.mark_id, "per_1000_tokens"), _mark_name(mark.mark_id, "per_sentence"))
        )
    for class_id in _punctuation_classes():
        names.extend((_class_name(class_id, "count"), _class_name(class_id, "per_1000_tokens"), _class_name(class_id, "per_sentence")))
    for final_id in _sentence_final_classes():
        names.extend((_sentence_final_name(final_id, "count"), _sentence_final_name(final_id, "ratio")))
    return tuple(names)


def punctuation_profile_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return metadata for punctuation profile features."""
    return tuple(_spec_for_name(name) for name in punctuation_profile_feature_names())


def _punctuation_row(view: DocumentView) -> tuple[list[float], tuple[FeatureDiagnostic, ...], PunctuationProfileSidecar]:
    token_count = len(view.tokens)
    sentence_count = len(view.sentences)
    occurrences = _punctuation_occurrences(view.raw)
    sentence_finals = _sentence_final_punctuation(view.sentences)
    mark_counts = _mark_counts(occurrences)
    class_counts = _class_counts(mark_counts)
    final_counts = _sentence_final_counts(sentence_finals)
    total_count = float(sum(mark_counts.values()))
    values: list[float] = []
    diagnostics: list[FeatureDiagnostic] = []

    values.append(total_count)
    values.append(_per_1000(_total_name("per_1000_tokens"), total_count, token_count, diagnostics))
    values.append(_per_sentence(_total_name("per_sentence"), total_count, sentence_count, diagnostics))
    for mark in _punctuation_marks():
        count = float(mark_counts[mark.mark_id])
        values.append(count)
        values.append(_per_1000(_mark_name(mark.mark_id, "per_1000_tokens"), count, token_count, diagnostics))
        values.append(_per_sentence(_mark_name(mark.mark_id, "per_sentence"), count, sentence_count, diagnostics))
    for class_id in _punctuation_classes():
        count = float(class_counts[class_id])
        values.append(count)
        values.append(_per_1000(_class_name(class_id, "per_1000_tokens"), count, token_count, diagnostics))
        values.append(_per_sentence(_class_name(class_id, "per_sentence"), count, sentence_count, diagnostics))
    for final_id in _sentence_final_classes():
        count = float(final_counts[final_id])
        values.append(count)
        values.append(_sentence_ratio(_sentence_final_name(final_id, "ratio"), count, sentence_count, diagnostics))
    sidecar = PunctuationProfileSidecar(
        document_id=view.document_id,
        schema_version="punctuation_profile_sidecar_v1",
        punctuation_count=len(occurrences),
        sentence_count=sentence_count,
        occurrences=occurrences,
        sentence_finals=sentence_finals,
    )
    return values, tuple(diagnostics), sidecar


def _punctuation_occurrences(text: str) -> tuple[PunctuationOccurrence, ...]:
    mark_by_text = {mark.text: mark for mark in _punctuation_marks()}
    occurrences: list[PunctuationOccurrence] = []
    for character_index, character in enumerate(text):
        if character in mark_by_text:
            mark = mark_by_text[character]
            occurrences.append(
                PunctuationOccurrence(
                    character_index=character_index,
                    mark_id=mark.mark_id,
                    text=mark.text,
                    class_ids=mark.class_ids,
                    final_id=mark.final_id,
                )
            )
    return tuple(occurrences)


def _mark_counts(occurrences: tuple[PunctuationOccurrence, ...]) -> Counter[str]:
    counts: Counter[str] = Counter({mark.mark_id: 0 for mark in _punctuation_marks()})
    for occurrence in occurrences:
        counts[occurrence.mark_id] += 1
    return counts


def _sentence_final_punctuation(sentences: tuple[str, ...]) -> tuple[SentenceFinalPunctuation, ...]:
    finals: list[SentenceFinalPunctuation] = []
    final_by_text = {mark.text: mark.final_id for mark in _punctuation_marks() if mark.final_id != "not_sentence_final"}
    for sentence_index, sentence in enumerate(sentences):
        stripped = sentence.rstrip()
        if stripped == "":
            finals.append(SentenceFinalPunctuation(sentence_index=sentence_index, final_text=None, final_id="no_terminal_punctuation"))
            continue
        final_character = stripped[-1]
        if final_character in final_by_text:
            final_id = final_by_text[final_character]
            final_text: str | None = final_character
        else:
            final_id = "no_terminal_punctuation"
            final_text = None
        finals.append(SentenceFinalPunctuation(sentence_index=sentence_index, final_text=final_text, final_id=final_id))
    return tuple(finals)


def _sentence_final_counts(sentence_finals: tuple[SentenceFinalPunctuation, ...]) -> dict[str, int]:
    counts = {class_id: 0 for class_id in _sentence_final_classes()}
    for sentence_final in sentence_finals:
        counts[sentence_final.final_id] += 1
    return counts


def _class_counts(mark_counts: Counter[str]) -> dict[str, int]:
    class_counts = {class_id: 0 for class_id in _punctuation_classes()}
    mark_by_id = {mark.mark_id: mark for mark in _punctuation_marks()}
    for mark_id, count in mark_counts.items():
        for class_id in mark_by_id[mark_id].class_ids:
            class_counts[class_id] += count
    return class_counts


def _per_1000(feature_name: str, count: float, token_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if token_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_tokens"))
        return float("nan")
    return count * 1000.0 / float(token_count)


def _per_sentence(feature_name: str, count: float, sentence_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if sentence_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_sentences"))
        return float("nan")
    return count / float(sentence_count)


def _sentence_ratio(feature_name: str, count: float, sentence_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if sentence_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_sentences"))
        return float("nan")
    return count / float(sentence_count)


def _punctuation_marks() -> tuple[PunctuationMark, ...]:
    return (
        PunctuationMark("period", ".", ("terminal",), "period"),
        PunctuationMark("comma", ",", ("comma",), "not_sentence_final"),
        PunctuationMark("semicolon", ";", ("semicolon_colon",), "not_sentence_final"),
        PunctuationMark("colon", ":", ("semicolon_colon",), "not_sentence_final"),
        PunctuationMark("exclamation", "!", ("terminal",), "exclamation"),
        PunctuationMark("question", "?", ("terminal",), "question"),
        PunctuationMark("hyphen_minus", "-", ("dash",), "not_sentence_final"),
        PunctuationMark("en_dash", "–", ("dash",), "not_sentence_final"),
        PunctuationMark("em_dash", "—", ("dash",), "not_sentence_final"),
        PunctuationMark("ellipsis", "…", ("terminal", "ellipsis"), "ellipsis"),
        PunctuationMark("apostrophe", "'", ("apostrophe",), "not_sentence_final"),
        PunctuationMark("double_quote", '"', ("quote",), "not_sentence_final"),
        PunctuationMark("left_double_quote", "“", ("quote",), "not_sentence_final"),
        PunctuationMark("right_double_quote", "”", ("quote",), "not_sentence_final"),
        PunctuationMark("left_single_quote", "‘", ("quote",), "not_sentence_final"),
        PunctuationMark("right_single_quote", "’", ("quote",), "not_sentence_final"),
        PunctuationMark("open_parenthesis", "(", ("bracket_parenthesis",), "not_sentence_final"),
        PunctuationMark("close_parenthesis", ")", ("bracket_parenthesis",), "not_sentence_final"),
        PunctuationMark("open_bracket", "[", ("bracket_parenthesis",), "not_sentence_final"),
        PunctuationMark("close_bracket", "]", ("bracket_parenthesis",), "not_sentence_final"),
    )


def _punctuation_classes() -> tuple[str, ...]:
    return ("terminal", "comma", "semicolon_colon", "dash", "ellipsis", "quote", "apostrophe", "bracket_parenthesis")


def _sentence_final_classes() -> tuple[str, ...]:
    return ("period", "exclamation", "question", "ellipsis", "no_terminal_punctuation")


def _total_name(measure: str) -> str:
    return f"text::punctuation_profile::total::{measure}"


def _mark_name(mark_id: str, measure: str) -> str:
    return f"text::punctuation_profile::mark={mark_id}::{measure}"


def _class_name(class_id: str, measure: str) -> str:
    return f"text::punctuation_profile::class={class_id}::{measure}"


def _sentence_final_name(final_id: str, measure: str) -> str:
    return f"text::punctuation_profile::sentence_final={final_id}::{measure}"


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _spec_for_name(name: str) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when punctuation phenomenon is absent"
    formula_or_rule = "count exact punctuation mark, punctuation class, or sentence-final punctuation category"
    if name.endswith("per_1000_tokens"):
        normalization = "per_1000_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_tokens when token count is zero"
        formula_or_rule = "punctuation count * 1000 / token count"
    if name.endswith("per_sentence") or name.endswith("ratio"):
        normalization = "per_sentence_ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_sentences when sentence count is zero"
        formula_or_rule = "punctuation count divided by sentence count"
    return FeatureSpec(
        name=name,
        family="punctuation_profile",
        description=f"Fixed punctuation profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.RAW,
        topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
        text_length_policy="counts are always defined; rates require token or sentence denominators",
        provenance=f"built_in_punctuation_profile_rules:v1; unicode_category=python_unicodedata:{unicodedata.unidata_version}",
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
