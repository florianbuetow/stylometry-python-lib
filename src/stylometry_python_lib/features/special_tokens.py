"""Special-token profile feature block."""

from __future__ import annotations

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
class SpecialTokenCategory:
    """One special-token category emitted by the profile block."""

    kind_id: str
    description: str


class SpecialTokenProfileTransformer(BaseEstimator):
    """Sklearn-compatible counts and masking profile for special tokens."""

    def __init__(self, text_column: str, config: PreprocessingConfig, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze special-token profile metadata."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.feature_names_out_ = np.asarray(special_token_profile_feature_names(), dtype=object)
        self.registry_ = FeatureRegistry(specs=special_token_profile_feature_specs(self.config))
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute special-token profile features without changing rows."""
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics = _profile_row(view)
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
        """Return stable special-token profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def special_token_profile_feature_names() -> tuple[str, ...]:
    """Return stable special-token profile feature names in output order."""
    names = [_total_name("count"), _total_name("class_normalized_count"), _total_name("per_1000_orthographic_tokens")]
    for category in _special_token_categories():
        names.extend(
            (
                _kind_name(category.kind_id, "count"),
                _kind_name(category.kind_id, "class_normalized_count"),
                _kind_name(category.kind_id, "per_1000_orthographic_tokens"),
            )
        )
    return tuple(names)


def special_token_profile_feature_specs(config: PreprocessingConfig) -> tuple[FeatureSpec, ...]:
    """Return metadata for special-token profile features under an explicit preprocessing config."""
    return tuple(_spec_for_name(name, config) for name in special_token_profile_feature_names())


def _profile_row(view: DocumentView) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    orthographic_token_count = len(view.orthographic_tokens)
    counts: Counter[str] = Counter(str(token.kind) for token in view.special_tokens)
    class_normalized_counts: Counter[str] = Counter(str(token.kind) for token in view.special_tokens if token.normalized != token.raw)
    diagnostics: list[FeatureDiagnostic] = []
    values: list[float] = []

    total_count = float(len(view.special_tokens))
    total_class_normalized_count = float(sum(class_normalized_counts.values()))
    values.append(total_count)
    values.append(total_class_normalized_count)
    values.append(_rate(_total_name("per_1000_orthographic_tokens"), total_count, orthographic_token_count, diagnostics))
    for category in _special_token_categories():
        count = float(counts[category.kind_id])
        class_normalized_count = float(class_normalized_counts[category.kind_id])
        values.append(count)
        values.append(class_normalized_count)
        values.append(_rate(_kind_name(category.kind_id, "per_1000_orthographic_tokens"), count, orthographic_token_count, diagnostics))
    return values, tuple(diagnostics)


def _rate(feature_name: str, count: float, denominator: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if denominator == 0:
        diagnostics.append(_undefined(feature_name, "zero_orthographic_tokens"))
        return float("nan")
    return count * 1000.0 / float(denominator)


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _special_token_categories() -> tuple[SpecialTokenCategory, ...]:
    return (
        SpecialTokenCategory("number", "Numeric literal retained by preprocessing"),
        SpecialTokenCategory("url", "URL retained by preprocessing"),
        SpecialTokenCategory("email", "Email address retained by preprocessing"),
        SpecialTokenCategory("hashtag", "Hashtag retained by preprocessing"),
        SpecialTokenCategory("mention", "User or account mention retained by preprocessing"),
        SpecialTokenCategory("code_identifier", "CamelCase, PascalCase, or snake_case identifier retained by preprocessing"),
        SpecialTokenCategory("emoji", "Emoji retained by preprocessing"),
    )


def _total_name(measure: str) -> str:
    return f"text::special_token_profile::total::{measure}"


def _kind_name(kind_id: str, measure: str) -> str:
    return f"text::special_token_profile::kind={kind_id}::{measure}"


def _spec_for_name(name: str, config: PreprocessingConfig) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the special-token category is absent"
    formula_or_rule = "count detected special-token spans"
    if name.endswith("class_normalized_count"):
        formula_or_rule = "count detected special-token spans whose normalized token differs from raw text"
    if name.endswith("per_1000_orthographic_tokens"):
        normalization = "per_1000_orthographic_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_orthographic_tokens when no orthographic tokens exist"
        formula_or_rule = "special-token count * 1000 / orthographic token count"
    return FeatureSpec(
        name=name,
        family="special_token_profile",
        description=f"Special-token profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.ORTHOGRAPHIC_TOKENS,
        topic_dependence=TopicDependence.MIXED,
        text_length_policy="counts are always defined; rates require at least one orthographic token",
        provenance=(
            "built_in_special_token_rules:v1; preprocessing_config; "
            f"special_token_policy={config.special_token_policy.value}; retain_urls={config.retain_urls}; "
            f"retain_numbers={config.retain_numbers}"
        ),
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
