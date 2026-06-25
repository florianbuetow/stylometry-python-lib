"""Spelling variant profile feature block."""

from __future__ import annotations

import re
from collections import Counter
from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib._tabular import text_series, validate_output_mode
from stylometry_python_lib.document import DocumentView, PreprocessingConfig
from stylometry_python_lib.lexicons import VersionedSpellingVariantResource, load_spelling_variants
from stylometry_python_lib.registry import FeatureRegistry
from stylometry_python_lib.specs import FeatureSpec, InputLayer, StabilityStatus, TopicDependence
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus


class SpellingVariantProfileTransformer(BaseEstimator):
    """Sklearn-compatible resource-backed spelling variant profile transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, resource_name: str, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.resource_name = resource_name
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Load resource metadata and freeze spelling variant feature names."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        resource = load_spelling_variants(self.resource_name)
        self.resource_ = resource
        self.feature_names_out_ = np.asarray(spelling_variant_profile_feature_names(resource), dtype=object)
        self.registry_ = FeatureRegistry(specs=spelling_variant_profile_feature_specs(resource))
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute spelling variant profile features without changing rows."""
        require_fitted(self, "resource_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics = _spelling_variant_row(view, self.resource_)
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
        """Return stable spelling variant profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def spelling_variant_profile_feature_names(resource: VersionedSpellingVariantResource) -> tuple[str, ...]:
    """Return stable spelling variant profile feature names for a resource."""
    names: list[str] = []
    for pair in resource.pairs:
        for form in (pair.variant_a, pair.variant_b):
            names.extend(
                (
                    _variant_name(pair.pair_id, form.label, "lowercase_count"),
                    _variant_name(pair.pair_id, form.label, "exact_case_count"),
                    _variant_name(pair.pair_id, form.label, "per_1000_tokens"),
                )
            )
        names.append(_pair_name(pair.pair_id, pair.variant_a.label, "lowercase_share"))
    return tuple(names)


def spelling_variant_profile_feature_specs(resource: VersionedSpellingVariantResource) -> tuple[FeatureSpec, ...]:
    """Return metadata for spelling variant profile features."""
    return tuple(_spec_for_name(name, resource) for name in spelling_variant_profile_feature_names(resource))


def spelling_variant_profile_transformer(
    text_column: str, config: PreprocessingConfig, resource_name: str, output: str
) -> SpellingVariantProfileTransformer:
    """Build a spelling variant profile transformer for an explicit resource."""
    return SpellingVariantProfileTransformer(text_column=text_column, config=config, resource_name=resource_name, output=output)


def _spelling_variant_row(
    view: DocumentView, resource: VersionedSpellingVariantResource
) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    lowercase_counts = Counter(view.tokens)
    exact_case_counts = Counter(_raw_word_candidates(view.raw))
    token_count = len(view.tokens)
    values: list[float] = []
    diagnostics: list[FeatureDiagnostic] = []
    for pair in resource.pairs:
        for form in (pair.variant_a, pair.variant_b):
            lowercase_count = float(lowercase_counts[form.token])
            exact_case_count = float(exact_case_counts[form.token])
            values.append(lowercase_count)
            values.append(exact_case_count)
            values.append(_per_1000(_variant_name(pair.pair_id, form.label, "per_1000_tokens"), lowercase_count, token_count, diagnostics))
        variant_a_count = float(lowercase_counts[pair.variant_a.token])
        variant_b_count = float(lowercase_counts[pair.variant_b.token])
        values.append(
            _share(
                _pair_name(pair.pair_id, pair.variant_a.label, "lowercase_share"),
                variant_a_count,
                variant_b_count,
                diagnostics,
            )
        )
    return values, tuple(diagnostics)


def _raw_word_candidates(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[A-Za-z]+", text))


def _per_1000(feature_name: str, count: float, token_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if token_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_tokens"))
        return float("nan")
    return count * 1000.0 / float(token_count)


def _share(feature_name: str, variant_a_count: float, variant_b_count: float, diagnostics: list[FeatureDiagnostic]) -> float:
    denominator = variant_a_count + variant_b_count
    if denominator == 0.0:
        diagnostics.append(_undefined(feature_name, "zero_variant_pair_observations"))
        return float("nan")
    return variant_a_count / denominator


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _variant_name(pair_id: str, label: str, measure: str) -> str:
    return f"text::spelling_variant_profile::pair={pair_id}::variant={label}::{measure}"


def _pair_name(pair_id: str, label: str, measure: str) -> str:
    return f"text::spelling_variant_profile::pair={pair_id}::variant={label}::{measure}"


def _spec_for_name(name: str, resource: VersionedSpellingVariantResource) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the spelling variant is absent"
    formula_or_rule = "count resource-defined spelling variant"
    if name.endswith("per_1000_tokens"):
        normalization = "per_1000_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_tokens when no tokens exist"
        formula_or_rule = "lowercase-normalized spelling variant count * 1000 / token count"
    if name.endswith("lowercase_share"):
        normalization = "variant_pair_ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_variant_pair_observations when neither variant appears"
        formula_or_rule = "first resource variant lowercase count divided by both pair variant lowercase counts"
    if name.endswith("exact_case_count"):
        formula_or_rule = "count exact raw-text case matches for the resource spelling variant"
    return FeatureSpec(
        name=name,
        family="spelling_variant_profile",
        description=f"Resource-backed spelling variant profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.TOKENS,
        topic_dependence=TopicDependence.TOPIC_SENSITIVE,
        text_length_policy="counts are always defined; rates require token denominator; pair shares require observed pair variants",
        provenance=(
            f"lexicon_id={resource.lexicon_id}; language={resource.language}; version={resource.version}; "
            f"source={resource.source}; license_note={resource.license_note}; normalization={resource.normalization}"
        ),
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
