"""Lexical-richness spectrum feature blocks."""

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
class FrequencySpectrumBin:
    """One frequency-of-frequencies sidecar bin."""

    frequency: int
    type_count: int
    types: tuple[str, ...]


@dataclass(frozen=True)
class FrequencySpectrumSidecar:
    """Full lexical frequency-spectrum sidecar for one document."""

    document_id: str
    token_count: int
    type_count: int
    max_frequency_bin: int
    bins: tuple[FrequencySpectrumBin, ...]
    overflow_type_count: int
    warnings: tuple[str, ...]


class LexicalRichnessSpectrumTransformer(BaseEstimator):
    """Sklearn-compatible hapax/dis and frequency-spectrum transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, max_frequency_bin: int, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.max_frequency_bin = max_frequency_bin
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze lexical spectrum metadata."""
        del y
        validate_output_mode(self.output)
        if self.max_frequency_bin <= 0:
            raise ValueError("max_frequency_bin must be positive")
        _ = text_series(x, self.text_column)
        self.max_frequency_bin_ = self.max_frequency_bin
        self.feature_names_out_ = np.asarray(lexical_richness_spectrum_feature_names(self.max_frequency_bin_), dtype=object)
        self.registry_ = FeatureRegistry(specs=lexical_richness_spectrum_feature_specs(self.max_frequency_bin_))
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute lexical spectrum features without changing rows."""
        require_fitted(self, "max_frequency_bin_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[FrequencySpectrumSidecar] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics, sidecar = _spectrum_row(view, self.max_frequency_bin_)
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
        """Return stable lexical spectrum feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def lexical_richness_spectrum_feature_names(max_frequency_bin: int) -> tuple[str, ...]:
    """Return lexical richness spectrum feature names in output order."""
    if max_frequency_bin <= 0:
        raise ValueError("max_frequency_bin must be positive")
    names: list[str] = []
    for phenomenon in _count_phenomena():
        names.extend(
            (
                _phenomenon_name(phenomenon, "count"),
                _phenomenon_name(phenomenon, "types_ratio"),
                _phenomenon_name(phenomenon, "tokens_ratio"),
                _phenomenon_name(phenomenon, "per_1000_tokens"),
            )
        )
    for frequency in range(1, max_frequency_bin + 1):
        names.extend(
            (
                _frequency_bin_name(frequency, "type_count"),
                _frequency_bin_name(frequency, "types_ratio"),
                _frequency_bin_name(frequency, "tokens_ratio"),
            )
        )
    return tuple(names)


def lexical_richness_spectrum_feature_specs(max_frequency_bin: int) -> tuple[FeatureSpec, ...]:
    """Return metadata for lexical richness spectrum features."""
    return tuple(_spec_for_name(name) for name in lexical_richness_spectrum_feature_names(max_frequency_bin))


def _spectrum_row(
    view: DocumentView, max_frequency_bin: int
) -> tuple[list[float], tuple[FeatureDiagnostic, ...], FrequencySpectrumSidecar]:
    token_counts = Counter(view.tokens)
    frequency_spectrum = Counter(token_counts.values())
    token_count = len(view.tokens)
    type_count = len(token_counts)
    diagnostics: list[FeatureDiagnostic] = []
    values: list[float] = []

    hapax_type_count = frequency_spectrum[1]
    dis_type_count = frequency_spectrum[2]
    values.extend(_phenomenon_values("hapax", hapax_type_count, 1, token_count, type_count, diagnostics))
    values.extend(_phenomenon_values("dis_legomena", dis_type_count, 2, token_count, type_count, diagnostics))
    for frequency in range(1, max_frequency_bin + 1):
        bin_type_count = frequency_spectrum[frequency]
        values.append(float(bin_type_count))
        values.append(_types_ratio(_frequency_bin_name(frequency, "types_ratio"), bin_type_count, type_count, diagnostics))
        values.append(_tokens_ratio(_frequency_bin_name(frequency, "tokens_ratio"), bin_type_count, frequency, token_count, diagnostics))
    sidecar = _sidecar(view, token_counts, max_frequency_bin)
    return values, tuple(diagnostics), sidecar


def _phenomenon_values(
    phenomenon: str,
    phenomenon_type_count: int,
    token_frequency: int,
    token_count: int,
    type_count: int,
    diagnostics: list[FeatureDiagnostic],
) -> list[float]:
    return [
        float(phenomenon_type_count),
        _types_ratio(_phenomenon_name(phenomenon, "types_ratio"), phenomenon_type_count, type_count, diagnostics),
        _tokens_ratio(_phenomenon_name(phenomenon, "tokens_ratio"), phenomenon_type_count, token_frequency, token_count, diagnostics),
        _per_1000(_phenomenon_name(phenomenon, "per_1000_tokens"), phenomenon_type_count, token_count, diagnostics),
    ]


def _types_ratio(feature_name: str, numerator: int, type_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if type_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_types"))
        return float("nan")
    return float(numerator) / float(type_count)


def _tokens_ratio(
    feature_name: str,
    type_count_at_frequency: int,
    token_frequency: int,
    token_count: int,
    diagnostics: list[FeatureDiagnostic],
) -> float:
    if token_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_tokens"))
        return float("nan")
    return float(type_count_at_frequency * token_frequency) / float(token_count)


def _per_1000(feature_name: str, type_count_at_frequency: int, token_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if token_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_tokens"))
        return float("nan")
    return float(type_count_at_frequency) * 1000.0 / float(token_count)


def _sidecar(view: DocumentView, token_counts: Counter[str], max_frequency_bin: int) -> FrequencySpectrumSidecar:
    frequency_to_types: dict[int, list[str]] = {}
    for token, frequency in token_counts.items():
        if frequency not in frequency_to_types:
            frequency_to_types[frequency] = []
        frequency_to_types[frequency].append(token)
    bins = tuple(
        FrequencySpectrumBin(
            frequency=frequency,
            type_count=len(frequency_to_types[frequency]),
            types=tuple(sorted(frequency_to_types[frequency])),
        )
        for frequency in sorted(frequency_to_types)
    )
    overflow_type_count = sum(bin_item.type_count for bin_item in bins if bin_item.frequency > max_frequency_bin)
    return FrequencySpectrumSidecar(
        document_id=view.document_id,
        token_count=len(view.tokens),
        type_count=len(token_counts),
        max_frequency_bin=max_frequency_bin,
        bins=bins,
        overflow_type_count=overflow_type_count,
        warnings=_warnings(len(view.tokens)),
    )


def _warnings(token_count: int) -> tuple[str, ...]:
    if token_count < 50:
        return ("short_text_unstable",)
    return ()


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _count_phenomena() -> tuple[str, ...]:
    return ("hapax", "dis_legomena")


def _phenomenon_name(phenomenon: str, measure: str) -> str:
    return f"text::lexical_richness_spectrum::{phenomenon}::{measure}"


def _frequency_bin_name(frequency: int, measure: str) -> str:
    return f"text::lexical_richness_spectrum::frequency_bin={frequency}::{measure}"


def _spec_for_name(name: str) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the token layer exists and the phenomenon is absent"
    formula_or_rule = "frequency-of-frequencies type count"
    if name.endswith("types_ratio"):
        normalization = "per_type_ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_types when type count is zero"
        formula_or_rule = "frequency bin type count divided by total type count"
    if name.endswith("tokens_ratio"):
        normalization = "token_mass_ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_tokens when token count is zero"
        formula_or_rule = "frequency bin token mass divided by total token count"
    if name.endswith("per_1000_tokens"):
        normalization = "per_1000_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_tokens when token count is zero"
        formula_or_rule = "frequency bin type count * 1000 / token count"
    return FeatureSpec(
        name=name,
        family="lexical_richness_spectrum",
        description=f"Lexical richness frequency-spectrum feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.TOKENS,
        topic_dependence=TopicDependence.MIXED,
        text_length_policy="counts are always defined; ratios require token or type denominators; short texts are unstable",
        provenance="built_in_frequency_spectrum_rules:v1; preprocessing_config; sidecar_schema=frequency_spectrum_v1",
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
