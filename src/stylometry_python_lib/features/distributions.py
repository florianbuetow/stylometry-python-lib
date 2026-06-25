"""Distribution-statistic feature blocks for deterministic stylometry."""

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
from stylometry_python_lib.text_metrics import syllable_count
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus

_SAMPLE_STATISTICS = frozenset(("sample_std", "sample_variance"))
_MOMENT_STATISTICS = frozenset(("skewness", "excess_kurtosis"))
_ARRAY_STATISTICS = frozenset(("mean", "min", "max"))
_STATISTIC_FORMULAS = {
    "count": "number of values in the distribution",
    "mean": "arithmetic mean over distribution values",
    "sample_std": "sample standard deviation over distribution values with ddof=1",
    "sample_variance": "sample variance over distribution values with ddof=1",
    "min": "minimum distribution value",
    "max": "maximum distribution value",
    "skewness": "population third central moment divided by population variance to the 3/2 power",
    "excess_kurtosis": "population fourth central moment divided by squared population variance, minus 3",
    "shannon_entropy": "empirical Shannon entropy over distribution value frequencies using natural logarithms",
}


@dataclass(frozen=True)
class DistributionFamily:
    """Metadata for one deterministic distribution family."""

    family_id: str
    description: str
    input_layer: InputLayer
    topic_dependence: TopicDependence
    zero_reason: str


class DistributionStatisticsTransformer(BaseEstimator):
    """Sklearn-compatible deterministic distribution-statistics transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze distribution feature metadata."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.feature_names_out_ = np.asarray(distribution_feature_names(), dtype=object)
        self.registry_ = FeatureRegistry(specs=distribution_feature_specs())
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute distribution statistics for each row without changing row identity."""
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for row_index, text in enumerate(series.tolist()):
            view = DocumentView.from_text(str(text), self.config, document_id=str(row_index))
            row, row_diagnostics = _distribution_row(view)
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
        """Return stable distribution-statistic feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def distribution_feature_names() -> tuple[str, ...]:
    """Return deterministic distribution-statistic feature names in output order."""
    return tuple(
        _feature_name(family.family_id, statistic) for family in _distribution_families() for statistic in _distribution_statistics()
    )


def distribution_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return metadata for deterministic distribution-statistic features."""
    return tuple(_distribution_spec(family, statistic) for family in _distribution_families() for statistic in _distribution_statistics())


def _distribution_row(view: DocumentView) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    values: list[float] = []
    diagnostics: list[FeatureDiagnostic] = []
    distributions = _distribution_values(view)
    families = _distribution_families()
    family_by_id = {family.family_id: family for family in families}
    for family in families:
        family_values = distributions[family.family_id]
        for statistic in _distribution_statistics():
            feature_name = _feature_name(family.family_id, statistic)
            value, diagnostic = _statistic_value(feature_name, statistic, family_values, family.zero_reason)
            values.append(value)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
    if set(distributions) != set(family_by_id):
        raise ValueError("Distribution family definitions and computed values differ")
    return values, tuple(diagnostics)


def _distribution_statistics() -> tuple[str, ...]:
    return (
        "count",
        "mean",
        "sample_std",
        "sample_variance",
        "min",
        "max",
        "p10",
        "p25",
        "p50",
        "p75",
        "p90",
        "skewness",
        "excess_kurtosis",
        "shannon_entropy",
    )


def _distribution_families() -> tuple[DistributionFamily, ...]:
    return (
        DistributionFamily(
            family_id="word_characters",
            description="Character lengths of tokenized words",
            input_layer=InputLayer.TOKENS,
            topic_dependence=TopicDependence.MIXED,
            zero_reason="zero_tokens",
        ),
        DistributionFamily(
            family_id="syllables_per_word",
            description="Hybrid dictionary-backed syllable counts per tokenized word",
            input_layer=InputLayer.TOKENS,
            topic_dependence=TopicDependence.MIXED,
            zero_reason="zero_tokens",
        ),
        DistributionFamily(
            family_id="sentence_tokens",
            description="Token counts per detected sentence",
            input_layer=InputLayer.SENTENCES,
            topic_dependence=TopicDependence.MIXED,
            zero_reason="zero_sentences",
        ),
        DistributionFamily(
            family_id="sentence_characters",
            description="Character counts per detected sentence",
            input_layer=InputLayer.SENTENCES,
            topic_dependence=TopicDependence.MIXED,
            zero_reason="zero_sentences",
        ),
        DistributionFamily(
            family_id="sentence_syllables",
            description="Hybrid dictionary-backed syllable counts per detected sentence",
            input_layer=InputLayer.SENTENCES,
            topic_dependence=TopicDependence.MIXED,
            zero_reason="zero_sentences",
        ),
        DistributionFamily(
            family_id="paragraph_tokens",
            description="Token counts per detected paragraph",
            input_layer=InputLayer.PARAGRAPHS,
            topic_dependence=TopicDependence.MIXED,
            zero_reason="zero_paragraphs",
        ),
        DistributionFamily(
            family_id="paragraph_characters",
            description="Character counts per detected paragraph",
            input_layer=InputLayer.PARAGRAPHS,
            topic_dependence=TopicDependence.MIXED,
            zero_reason="zero_paragraphs",
        ),
        DistributionFamily(
            family_id="paragraph_syllables",
            description="Hybrid dictionary-backed syllable counts per detected paragraph",
            input_layer=InputLayer.PARAGRAPHS,
            topic_dependence=TopicDependence.MIXED,
            zero_reason="zero_paragraphs",
        ),
        DistributionFamily(
            family_id="paragraph_sentences",
            description="Sentence counts per detected paragraph",
            input_layer=InputLayer.PARAGRAPHS,
            topic_dependence=TopicDependence.MIXED,
            zero_reason="zero_paragraphs",
        ),
        DistributionFamily(
            family_id="line_characters",
            description="Character counts per raw line",
            input_layer=InputLayer.RAW,
            topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
            zero_reason="zero_lines",
        ),
        DistributionFamily(
            family_id="punctuation_sequence_characters",
            description="Character lengths of repeated punctuation sequences",
            input_layer=InputLayer.RAW,
            topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
            zero_reason="zero_punctuation_sequences",
        ),
    )


def _distribution_values(view: DocumentView) -> dict[str, list[int]]:
    return {
        "word_characters": [len(token) for token in view.tokens],
        "syllables_per_word": [syllable_count(token) for token in view.tokens],
        "sentence_tokens": [
            _token_count(sentence, view.config, view.document_id, "sentence", index) for index, sentence in enumerate(view.sentences)
        ],
        "sentence_characters": [len(sentence) for sentence in view.sentences],
        "sentence_syllables": [
            _syllable_count(sentence, view.config, view.document_id, "sentence_syllables", index)
            for index, sentence in enumerate(view.sentences)
        ],
        "paragraph_tokens": [
            _token_count(paragraph, view.config, view.document_id, "paragraph", index) for index, paragraph in enumerate(view.paragraphs)
        ],
        "paragraph_characters": [len(paragraph) for paragraph in view.paragraphs],
        "paragraph_syllables": [
            _syllable_count(paragraph, view.config, view.document_id, "paragraph_syllables", index)
            for index, paragraph in enumerate(view.paragraphs)
        ],
        "paragraph_sentences": [
            _sentence_count(paragraph, view.config, view.document_id, index) for index, paragraph in enumerate(view.paragraphs)
        ],
        "line_characters": [len(line) for line in view.raw.splitlines()],
        "punctuation_sequence_characters": [len(match.group(0)) for match in re.finditer(r"[.!?,;:\-]{2,}", view.raw)],
    }


def _token_count(text: str, config: PreprocessingConfig, document_id: str, layer: str, index: int) -> int:
    nested_view = DocumentView.from_text(text, config, document_id=f"{document_id}:{layer}:{index}")
    return len(nested_view.tokens)


def _sentence_count(text: str, config: PreprocessingConfig, document_id: str, index: int) -> int:
    nested_view = DocumentView.from_text(text, config, document_id=f"{document_id}:paragraph_sentence:{index}")
    return len(nested_view.sentences)


def _syllable_count(text: str, config: PreprocessingConfig, document_id: str, layer: str, index: int) -> int:
    nested_view = DocumentView.from_text(text, config, document_id=f"{document_id}:{layer}:{index}")
    return sum(syllable_count(token) for token in nested_view.tokens)


def _statistic_value(feature_name: str, statistic: str, values: list[int], zero_reason: str) -> tuple[float, FeatureDiagnostic | None]:
    if statistic == "count":
        return float(len(values)), None
    if len(values) == 0:
        return float("nan"), _undefined(feature_name, zero_reason)
    array = np.asarray(values, dtype=float)
    if statistic in _SAMPLE_STATISTICS:
        return _sample_statistic(feature_name, statistic, array)
    if statistic in _ARRAY_STATISTICS:
        return _array_statistic(statistic, array), None
    if statistic.startswith("p"):
        percentile = float(statistic.removeprefix("p"))
        return float(np.percentile(array, percentile)), None
    if statistic in _MOMENT_STATISTICS:
        return _moment_statistic(feature_name, statistic, array)
    if statistic == "shannon_entropy":
        return _shannon_entropy(array), None
    raise ValueError(f"Unsupported distribution statistic: {statistic}")


def _sample_statistic(feature_name: str, statistic: str, array: np.ndarray) -> tuple[float, FeatureDiagnostic | None]:
    if array.size < 2:
        return float("nan"), _undefined(feature_name, "insufficient_values_for_sample_statistic")
    if statistic == "sample_std":
        return float(np.std(array, ddof=1)), None
    return float(np.var(array, ddof=1)), None


def _array_statistic(statistic: str, array: np.ndarray) -> float:
    if statistic == "mean":
        return float(np.mean(array))
    if statistic == "min":
        return float(np.min(array))
    if statistic == "max":
        return float(np.max(array))
    raise ValueError(f"Unsupported array statistic: {statistic}")


def _moment_statistic(feature_name: str, statistic: str, array: np.ndarray) -> tuple[float, FeatureDiagnostic | None]:
    if array.size < 2:
        return float("nan"), _undefined(feature_name, "insufficient_values_for_moment_statistic")
    centered = array - float(np.mean(array))
    second_moment = float(np.mean(np.power(centered, 2)))
    if second_moment == 0.0:
        return float("nan"), _undefined(feature_name, "zero_variance_distribution")
    if statistic == "skewness":
        third_moment = float(np.mean(np.power(centered, 3)))
        return third_moment / float(second_moment**1.5), None
    fourth_moment = float(np.mean(np.power(centered, 4)))
    return fourth_moment / float(second_moment**2) - 3.0, None


def _shannon_entropy(array: np.ndarray) -> float:
    unique_values, counts = np.unique(array, return_counts=True)
    del unique_values
    probabilities = counts.astype(float) / float(array.size)
    return float(-np.sum(probabilities * np.log(probabilities)))


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _feature_name(family_id: str, statistic: str) -> str:
    return f"text::distribution::{family_id}::{statistic}"


def _distribution_spec(family: DistributionFamily, statistic: str) -> FeatureSpec:
    undefined_behavior = f"NaN with FeatureDiagnostic reason {family.zero_reason} when the distribution is empty"
    if statistic == "count":
        undefined_behavior = "defined as zero when the distribution is empty"
    if statistic in {"sample_std", "sample_variance"}:
        undefined_behavior = "NaN with FeatureDiagnostic reason insufficient_values_for_sample_statistic when fewer than two values exist"
    if statistic in {"skewness", "excess_kurtosis"}:
        undefined_behavior = (
            "NaN with FeatureDiagnostic reason insufficient_values_for_moment_statistic when fewer than two values exist; "
            "NaN with reason zero_variance_distribution when all values are identical"
        )
    if statistic == "shannon_entropy":
        undefined_behavior = f"NaN with FeatureDiagnostic reason {family.zero_reason} when the distribution is empty"
    return FeatureSpec(
        name=_feature_name(family.family_id, statistic),
        family=f"distribution_{family.family_id}",
        description=f"{family.description}: {statistic}",
        formula_or_rule=_formula_for_statistic(statistic),
        input_layer=family.input_layer,
        topic_dependence=family.topic_dependence,
        text_length_policy=(
            "count is always defined; continuous statistics require at least one value; sample and moment statistics require two values"
        ),
        provenance=_provenance_for_family(family.family_id),
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization="distribution_statistic",
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )


def _formula_for_statistic(statistic: str) -> str:
    if statistic in _STATISTIC_FORMULAS:
        return _STATISTIC_FORMULAS[statistic]
    if statistic.startswith("p"):
        return f"{statistic.removeprefix('p')}th percentile using numpy default linear interpolation"
    raise ValueError(f"Unsupported distribution statistic: {statistic}")


def _provenance_for_family(family_id: str) -> str:
    base = (
        "built_in_distribution_rules:v1; std_type=sample; percentile_method=numpy_default_linear; "
        "moment_statistics=population_central_moments; entropy_log_base=e; preprocessing_config"
    )
    if "syllables" in family_id:
        return f"{base}; syllable_dictionary=syllable_counts_en_v1; syllable_fallback=heuristic_v1"
    return base
