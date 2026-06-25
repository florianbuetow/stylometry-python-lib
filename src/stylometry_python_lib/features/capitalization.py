"""Capitalization profile feature block."""

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
class CapitalizationTokenClass:
    """One token-level capitalization class."""

    class_id: str
    description: str


class CapitalizationProfileTransformer(BaseEstimator):
    """Sklearn-compatible capitalization habit profile transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze capitalization profile metadata."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.feature_names_out_ = np.asarray(capitalization_profile_feature_names(), dtype=object)
        self.registry_ = FeatureRegistry(specs=capitalization_profile_feature_specs())
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute capitalization profile features without changing rows."""
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics = _capitalization_row(view)
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
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
        """Return stable capitalization profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def capitalization_profile_feature_names() -> tuple[str, ...]:
    """Return stable capitalization profile feature names in output order."""
    names: list[str] = []
    for token_class in _token_classes():
        names.extend((_token_class_name(token_class.class_id, "count"), _token_class_name(token_class.class_id, "per_1000_tokens")))
    for line_class in _line_classes():
        names.extend((_line_class_name(line_class, "count"), _line_class_name(line_class, "ratio")))
    for sentence_initial_class in _sentence_initial_classes():
        names.extend(
            (
                _sentence_initial_name(sentence_initial_class, "count"),
                _sentence_initial_name(sentence_initial_class, "ratio"),
            )
        )
    return tuple(names)


def capitalization_profile_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return metadata for capitalization profile features."""
    return tuple(_spec_for_name(name) for name in capitalization_profile_feature_names())


def _capitalization_row(view: DocumentView) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    token_candidates = _token_candidates(view.raw)
    nonblank_lines = [line for line in view.raw.splitlines() if line.strip() != ""]
    sentence_initials = _sentence_initials(view.sentences)
    diagnostics: list[FeatureDiagnostic] = []
    values: list[float] = []

    for token_class in _token_classes():
        count = float(sum(1 for token in token_candidates if _matches_token_class(token, token_class.class_id)))
        values.append(count)
        values.append(_per_1000(_token_class_name(token_class.class_id, "per_1000_tokens"), count, len(token_candidates), diagnostics))
    for line_class in _line_classes():
        count = float(sum(1 for line in nonblank_lines if _matches_line_class(line, line_class)))
        values.append(count)
        values.append(_ratio(_line_class_name(line_class, "ratio"), count, len(nonblank_lines), "zero_nonblank_lines", diagnostics))
    for sentence_initial_class in _sentence_initial_classes():
        count = float(sum(1 for initial in sentence_initials if _matches_sentence_initial(initial, sentence_initial_class)))
        values.append(count)
        values.append(
            _ratio(
                _sentence_initial_name(sentence_initial_class, "ratio"),
                count,
                len(sentence_initials),
                "zero_sentence_initials",
                diagnostics,
            )
        )
    return values, tuple(diagnostics)


def _token_candidates(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"(?:[A-Za-z]\.){2,}|[A-Za-z][A-Za-z0-9_]*", text))


def _sentence_initials(sentences: tuple[str, ...]) -> tuple[str, ...]:
    initials: list[str] = []
    for sentence in sentences:
        for character in sentence:
            if character.isalpha():
                initials.append(character)
                break
    return tuple(initials)


def _matches_token_class(token: str, class_id: str) -> bool:
    alpha_characters = tuple(character for character in token if character.isalpha())
    if class_id == "all_caps":
        return "." not in token and len(alpha_characters) >= 2 and all(character.isupper() for character in alpha_characters)
    if class_id == "acronym_like":
        return _is_dotted_acronym(token) or (len(alpha_characters) >= 2 and all(character.isupper() for character in alpha_characters))
    if class_id == "camel_case":
        return re.fullmatch(r"[a-z]+(?:[A-Z][A-Za-z0-9]*)+", token) is not None
    if class_id == "pascal_case":
        return re.fullmatch(r"[A-Z][a-z]+(?:[A-Z][A-Za-z0-9]*)+", token) is not None
    raise ValueError(f"Unsupported capitalization token class: {class_id}")


def _matches_line_class(line: str, class_id: str) -> bool:
    words = re.findall(r"[A-Za-z]+", line)
    stripped = line.strip()
    if class_id == "titlecase_line":
        if len(words) == 0:
            return False
        return all(word[0].isupper() for word in words)
    if class_id == "lowercase_heading_line":
        if len(words) < 2:
            return False
        if len(stripped) > 80:
            return False
        if stripped.endswith((".", "!", "?")):
            return False
        return all(word.islower() for word in words)
    raise ValueError(f"Unsupported capitalization line class: {class_id}")


def _matches_sentence_initial(initial: str, class_id: str) -> bool:
    if class_id == "uppercase":
        return initial.isupper()
    if class_id == "lowercase":
        return initial.islower()
    raise ValueError(f"Unsupported sentence-initial class: {class_id}")


def _is_dotted_acronym(token: str) -> bool:
    return re.fullmatch(r"(?:[A-Z]\.){2,}", token) is not None


def _per_1000(feature_name: str, count: float, denominator: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if denominator == 0:
        diagnostics.append(_undefined(feature_name, "zero_capitalization_tokens"))
        return float("nan")
    return count * 1000.0 / float(denominator)


def _ratio(
    feature_name: str,
    count: float,
    denominator: int,
    zero_reason: str,
    diagnostics: list[FeatureDiagnostic],
) -> float:
    if denominator == 0:
        diagnostics.append(_undefined(feature_name, zero_reason))
        return float("nan")
    return count / float(denominator)


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _token_classes() -> tuple[CapitalizationTokenClass, ...]:
    return (
        CapitalizationTokenClass("all_caps", "Alphabetic token with two or more uppercase letters"),
        CapitalizationTokenClass("acronym_like", "All-caps or dotted acronym-like token"),
        CapitalizationTokenClass("camel_case", "Identifier-like token beginning lowercase with an internal uppercase segment"),
        CapitalizationTokenClass("pascal_case", "Identifier-like token beginning uppercase with an internal uppercase segment"),
    )


def _line_classes() -> tuple[str, ...]:
    return ("titlecase_line", "lowercase_heading_line")


def _sentence_initial_classes() -> tuple[str, ...]:
    return ("uppercase", "lowercase")


def _token_class_name(class_id: str, measure: str) -> str:
    return f"text::capitalization_profile::token_class={class_id}::{measure}"


def _line_class_name(class_id: str, measure: str) -> str:
    return f"text::capitalization_profile::line_class={class_id}::{measure}"


def _sentence_initial_name(class_id: str, measure: str) -> str:
    return f"text::capitalization_profile::sentence_initial={class_id}::{measure}"


def _spec_for_name(name: str) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the capitalization phenomenon is absent"
    formula_or_rule = "count raw-text capitalization class matches"
    if name.endswith("per_1000_tokens"):
        normalization = "per_1000_capitalization_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_capitalization_tokens when no token candidates exist"
        formula_or_rule = "capitalization token-class count * 1000 / capitalization token candidate count"
    if name.endswith("ratio"):
        normalization = "ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason matching the missing line or sentence-initial denominator"
        formula_or_rule = "capitalization count divided by matching line or sentence-initial denominator"
    return FeatureSpec(
        name=name,
        family="capitalization_profile",
        description=f"Capitalization profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.RAW,
        topic_dependence=TopicDependence.MIXED,
        text_length_policy="counts are always defined; rates require token, line, or sentence-initial denominators",
        provenance="built_in_capitalization_profile_rules:v1; preprocessing_config",
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
