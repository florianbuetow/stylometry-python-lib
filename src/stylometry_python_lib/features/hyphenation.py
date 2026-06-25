"""Hyphenation profile feature block."""

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
class HyphenationClass:
    """One deterministic hyphenation class."""

    class_id: str
    description: str


@dataclass(frozen=True)
class HyphenationVariantPair:
    """One fixed hyphenated-vs-solid spelling pair."""

    pair_id: str
    solid_form: str
    hyphenated_form: str


class HyphenationProfileTransformer(BaseEstimator):
    """Sklearn-compatible hyphenation class and variant-pair transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze hyphenation profile metadata."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.feature_names_out_ = np.asarray(hyphenation_profile_feature_names(), dtype=object)
        self.registry_ = FeatureRegistry(specs=hyphenation_profile_feature_specs())
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute hyphenation profile features without changing rows."""
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics = _hyphenation_row(view)
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
        """Return stable hyphenation profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def hyphenation_profile_feature_names() -> tuple[str, ...]:
    """Return stable hyphenation profile feature names in output order."""
    names = [_total_name("count"), _total_name("per_1000_orthographic_tokens")]
    for hyphen_class in _hyphenation_classes():
        names.extend(
            (
                _class_name(hyphen_class.class_id, "count"),
                _class_name(hyphen_class.class_id, "per_1000_orthographic_tokens"),
            )
        )
    for pair in _variant_pairs():
        names.extend(
            (
                _variant_pair_name(pair.pair_id, "hyphenated_count"),
                _variant_pair_name(pair.pair_id, "solid_count"),
                _variant_pair_name(pair.pair_id, "hyphenated_share"),
            )
        )
    return tuple(names)


def hyphenation_profile_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return metadata for hyphenation profile features."""
    return tuple(_spec_for_name(name) for name in hyphenation_profile_feature_names())


def _hyphenation_row(view: DocumentView) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    orthographic_tokens = tuple(token.lower() for token in view.orthographic_tokens)
    hyphenated_tokens = tuple(token.lower() for token in view.hyphenated_tokens)
    token_counts = Counter(orthographic_tokens)
    diagnostics: list[FeatureDiagnostic] = []
    values: list[float] = []

    total_count = float(len(hyphenated_tokens))
    values.append(total_count)
    values.append(_per_1000(_total_name("per_1000_orthographic_tokens"), total_count, len(orthographic_tokens), diagnostics))
    for hyphen_class in _hyphenation_classes():
        count = float(sum(1 for token in hyphenated_tokens if _matches_hyphenation_class(token, hyphen_class.class_id)))
        values.append(count)
        values.append(
            _per_1000(
                _class_name(hyphen_class.class_id, "per_1000_orthographic_tokens"),
                count,
                len(orthographic_tokens),
                diagnostics,
            )
        )
    for pair in _variant_pairs():
        hyphenated_count = float(token_counts[pair.hyphenated_form])
        solid_count = float(token_counts[pair.solid_form])
        values.append(hyphenated_count)
        values.append(solid_count)
        values.append(
            _variant_share(
                _variant_pair_name(pair.pair_id, "hyphenated_share"),
                hyphenated_count,
                solid_count,
                diagnostics,
            )
        )
    return values, tuple(diagnostics)


def _matches_hyphenation_class(token: str, class_id: str) -> bool:
    hyphen_count = token.count("-")
    if class_id == "single_hyphen":
        return hyphen_count == 1
    if class_id == "multi_hyphen":
        return hyphen_count > 1
    if class_id == "prefix_hyphen":
        return token.split("-")[0] in _productive_prefixes()
    if class_id == "non_prefix_compound":
        return token.split("-")[0] not in _productive_prefixes()
    raise ValueError(f"Unsupported hyphenation class: {class_id}")


def _per_1000(feature_name: str, count: float, denominator: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if denominator == 0:
        diagnostics.append(_undefined(feature_name, "zero_orthographic_tokens"))
        return float("nan")
    return count * 1000.0 / float(denominator)


def _variant_share(feature_name: str, hyphenated_count: float, solid_count: float, diagnostics: list[FeatureDiagnostic]) -> float:
    denominator = hyphenated_count + solid_count
    if denominator == 0.0:
        diagnostics.append(_undefined(feature_name, "zero_variant_pair_observations"))
        return float("nan")
    return hyphenated_count / denominator


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _hyphenation_classes() -> tuple[HyphenationClass, ...]:
    return (
        HyphenationClass("single_hyphen", "Token containing exactly one hyphen"),
        HyphenationClass("multi_hyphen", "Token containing more than one hyphen"),
        HyphenationClass("prefix_hyphen", "Hyphenated token beginning with a productive English prefix"),
        HyphenationClass("non_prefix_compound", "Hyphenated token not beginning with a configured productive prefix"),
    )


def _variant_pairs() -> tuple[HyphenationVariantPair, ...]:
    return (
        HyphenationVariantPair("email_e_mail", "email", "e-mail"),
        HyphenationVariantPair("reenter_re_enter", "reenter", "re-enter"),
        HyphenationVariantPair("cooperate_co_operate", "cooperate", "co-operate"),
        HyphenationVariantPair("longterm_long_term", "longterm", "long-term"),
    )


def _productive_prefixes() -> frozenset[str]:
    return frozenset(("anti", "co", "non", "post", "pre", "pro", "re", "self"))


def _total_name(measure: str) -> str:
    return f"text::hyphenation_profile::total::{measure}"


def _class_name(class_id: str, measure: str) -> str:
    return f"text::hyphenation_profile::class={class_id}::{measure}"


def _variant_pair_name(pair_id: str, measure: str) -> str:
    return f"text::hyphenation_profile::variant_pair={pair_id}::{measure}"


def _spec_for_name(name: str) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the hyphenation phenomenon is absent"
    formula_or_rule = "count deterministic hyphenation class or fixed variant-pair observation"
    if name.endswith("per_1000_orthographic_tokens"):
        normalization = "per_1000_orthographic_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_orthographic_tokens when no orthographic tokens exist"
        formula_or_rule = "hyphenation count * 1000 / orthographic token count"
    if name.endswith("hyphenated_share"):
        normalization = "variant_pair_ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_variant_pair_observations when neither variant appears"
        formula_or_rule = "hyphenated variant count divided by hyphenated plus solid variant counts"
    return FeatureSpec(
        name=name,
        family="hyphenation_profile",
        description=f"Hyphenation profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.ORTHOGRAPHIC_TOKENS,
        topic_dependence=TopicDependence.MIXED,
        text_length_policy="counts are always defined; rates require orthographic tokens; variant shares require pair observations",
        provenance="built_in_hyphenation_profile_rules:v1; preprocessing_config",
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
