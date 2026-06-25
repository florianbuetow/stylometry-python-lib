"""Lexical sophistication and frequency-band profile feature block."""

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
from stylometry_python_lib.lexicons import FrequencyBandEntry, VersionedFrequencyBandResource, load_frequency_bands
from stylometry_python_lib.registry import FeatureRegistry
from stylometry_python_lib.specs import FeatureSpec, InputLayer, StabilityStatus, TopicDependence
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus


@dataclass(frozen=True)
class FrequencyBandRecord:
    """One observed normalized token type mapped to reference-frequency metadata."""

    token: str
    token_count: int
    band: str
    in_reference: bool
    rank: int | None
    frequency_per_million: float | None


@dataclass(frozen=True)
class LexicalSophisticationSidecar:
    """Per-document frequency-band token records for lexical sophistication."""

    document_id: str
    schema_version: str
    resource_id: str
    language: str
    version: str
    normalization: str
    token_count: int
    type_count: int
    in_reference_token_count: int
    out_of_reference_token_count: int
    records: tuple[FrequencyBandRecord, ...]


class LexicalSophisticationProfileTransformer(BaseEstimator):
    """Sklearn-compatible lexical frequency-band profile transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, resource_name: str, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.resource_name = resource_name
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Load reference-frequency metadata and freeze output names."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        resource = load_frequency_bands(self.resource_name)
        self.resource_ = resource
        self.feature_names_out_ = np.asarray(lexical_sophistication_profile_feature_names(resource), dtype=object)
        self.registry_ = FeatureRegistry(specs=lexical_sophistication_profile_feature_specs(resource))
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute lexical sophistication profile features without changing rows."""
        require_fitted(self, "resource_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[LexicalSophisticationSidecar] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics, sidecar = _lexical_sophistication_row(view, self.resource_)
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
        """Return stable lexical sophistication feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def lexical_sophistication_profile_transformer(
    text_column: str, config: PreprocessingConfig, resource_name: str, output: str
) -> LexicalSophisticationProfileTransformer:
    """Build a lexical sophistication profile transformer for an explicit resource."""
    return LexicalSophisticationProfileTransformer(text_column=text_column, config=config, resource_name=resource_name, output=output)


def lexical_sophistication_profile_feature_names(resource: VersionedFrequencyBandResource) -> tuple[str, ...]:
    """Return stable lexical sophistication feature names for a resource."""
    names: list[str] = []
    for band in _output_bands(resource):
        names.extend(
            (
                _band_name(band, "token_count"),
                _band_name(band, "type_count"),
                _band_name(band, "token_ratio"),
                _band_name(band, "type_ratio"),
            )
        )
    return tuple(names)


def lexical_sophistication_profile_feature_specs(resource: VersionedFrequencyBandResource) -> tuple[FeatureSpec, ...]:
    """Return metadata for lexical sophistication profile features."""
    return tuple(_spec_for_name(name, resource) for name in lexical_sophistication_profile_feature_names(resource))


def _lexical_sophistication_row(
    view: DocumentView, resource: VersionedFrequencyBandResource
) -> tuple[list[float], tuple[FeatureDiagnostic, ...], LexicalSophisticationSidecar]:
    token_counts = Counter(view.tokens)
    type_tokens = frozenset(view.tokens)
    token_count = len(view.tokens)
    type_count = len(type_tokens)
    diagnostics: list[FeatureDiagnostic] = []
    values: list[float] = []

    for band in _output_bands(resource):
        band_tokens = _tokens_for_band(type_tokens, resource, band)
        band_token_count = float(sum(token_counts[token] for token in band_tokens))
        band_type_count = float(len(band_tokens))
        values.append(band_token_count)
        values.append(band_type_count)
        values.append(_ratio(_band_name(band, "token_ratio"), band_token_count, token_count, "zero_tokens", diagnostics))
        values.append(_ratio(_band_name(band, "type_ratio"), band_type_count, type_count, "zero_types", diagnostics))

    sidecar = _sidecar(view, resource, token_counts)
    return values, tuple(diagnostics), sidecar


def _sidecar(view: DocumentView, resource: VersionedFrequencyBandResource, token_counts: Counter[str]) -> LexicalSophisticationSidecar:
    entry_by_token = resource.entry_by_token()
    records = tuple(_record_for_token(token, token_counts[token], entry_by_token) for token in sorted(token_counts))
    in_reference_token_count = sum(record.token_count for record in records if record.in_reference)
    out_of_reference_token_count = sum(record.token_count for record in records if not record.in_reference)
    return LexicalSophisticationSidecar(
        document_id=view.document_id,
        schema_version="lexical_sophistication_frequency_bands_v1",
        resource_id=resource.lexicon_id,
        language=resource.language,
        version=resource.version,
        normalization=resource.normalization,
        token_count=len(view.tokens),
        type_count=len(token_counts),
        in_reference_token_count=in_reference_token_count,
        out_of_reference_token_count=out_of_reference_token_count,
        records=records,
    )


def _record_for_token(token: str, token_count: int, entry_by_token: dict[str, FrequencyBandEntry]) -> FrequencyBandRecord:
    if token not in entry_by_token:
        return FrequencyBandRecord(
            token=token,
            token_count=token_count,
            band=_out_of_reference_band(),
            in_reference=False,
            rank=None,
            frequency_per_million=None,
        )
    entry = entry_by_token[token]
    return FrequencyBandRecord(
        token=token,
        token_count=token_count,
        band=entry.band,
        in_reference=True,
        rank=entry.rank,
        frequency_per_million=entry.frequency_per_million,
    )


def _tokens_for_band(type_tokens: frozenset[str], resource: VersionedFrequencyBandResource, band: str) -> tuple[str, ...]:
    entry_by_token = resource.entry_by_token()
    if band == _out_of_reference_band():
        return tuple(sorted(token for token in type_tokens if token not in entry_by_token))
    return tuple(sorted(token for token in type_tokens if token in entry_by_token and entry_by_token[token].band == band))


def _output_bands(resource: VersionedFrequencyBandResource) -> tuple[str, ...]:
    return resource.bands() + (_out_of_reference_band(),)


def _out_of_reference_band() -> str:
    return "out_of_reference"


def _ratio(
    feature_name: str,
    numerator: float,
    denominator: int,
    zero_reason: str,
    diagnostics: list[FeatureDiagnostic],
) -> float:
    if denominator == 0:
        diagnostics.append(_undefined(feature_name, zero_reason))
        return float("nan")
    return numerator / float(denominator)


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _band_name(band: str, measure: str) -> str:
    return f"text::lexical_sophistication_profile::band={band}::{measure}"


def _spec_for_name(name: str, resource: VersionedFrequencyBandResource) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when no tokens fall in the frequency band"
    formula_or_rule = "count normalized tokens or unique types assigned to a versioned reference-frequency band"
    if name.endswith("token_ratio"):
        normalization = "token_ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_tokens when no tokens exist"
        formula_or_rule = "frequency-band token count divided by total token count"
    if name.endswith("type_ratio"):
        normalization = "type_ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_types when no types exist"
        formula_or_rule = "frequency-band unique type count divided by total unique type count"
    return FeatureSpec(
        name=name,
        family="lexical_sophistication_profile",
        description=f"Lexical sophistication frequency-band profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.TOKENS,
        topic_dependence=TopicDependence.TOPIC_SENSITIVE,
        text_length_policy="counts are always defined; token ratios require tokens; type ratios require unique types",
        provenance=(
            f"lexicon_id={resource.lexicon_id}; language={resource.language}; version={resource.version}; "
            f"source={resource.source}; license_note={resource.license_note}; normalization={resource.normalization}; "
            "sidecar_schema=lexical_sophistication_frequency_bands_v1"
        ),
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
