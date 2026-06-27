"""Feature-importance helpers for stylometry evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from stylometry_python_lib.errors import OptionalDependencyError
from stylometry_python_lib.evaluation.distances import as_float_matrix


class ScoringEstimatorProtocol(Protocol):
    """Minimal fitted-estimator protocol required for permutation importance."""

    def score(self, x: object, y: object) -> float:
        """Return a scalar score for matrix and labels."""
        ...


class ProbabilityEstimatorProtocol(Protocol):
    """Fitted-estimator protocol exposing class probabilities for SHAP."""

    def predict_proba(self, x: NDArray[np.float64], /) -> NDArray[Any]:
        """Return class-probability estimates for a feature matrix."""
        ...


@dataclass(frozen=True)
class FeatureImportanceRecord:
    """Permutation-importance summary for one feature."""

    feature_name: str
    mean_importance: float
    std_importance: float
    repeat_importances: tuple[float, ...]


@dataclass(frozen=True)
class FeatureImportanceReport:
    """Permutation-importance report for a fitted estimator."""

    baseline_score: float
    n_repeats: int
    random_state: int
    records: tuple[FeatureImportanceRecord, ...]


def permutation_importance_report(
    estimator: ScoringEstimatorProtocol,
    features: object,
    labels: object,
    feature_names: object,
    n_repeats: int,
    random_state: int,
) -> FeatureImportanceReport:
    """Compute permutation importance from a fitted estimator's score method."""
    matrix = as_float_matrix(features)
    _validate_importance_matrix(matrix)
    label_array = _validate_importance_labels(labels, matrix.shape[0])
    names = _validate_feature_names(feature_names, matrix.shape[1])
    _validate_importance_config(n_repeats)
    baseline_score = float(estimator.score(matrix, label_array))
    rng = np.random.default_rng(random_state)
    records = tuple(
        _feature_importance_record(
            estimator=estimator,
            matrix=matrix,
            labels=label_array,
            feature_name=feature_name,
            feature_index=feature_index,
            baseline_score=baseline_score,
            n_repeats=n_repeats,
            rng=rng,
        )
        for feature_index, feature_name in enumerate(names)
    )
    return FeatureImportanceReport(
        baseline_score=baseline_score,
        n_repeats=n_repeats,
        random_state=random_state,
        records=records,
    )


def _feature_importance_record(
    estimator: ScoringEstimatorProtocol,
    matrix: NDArray[np.float64],
    labels: NDArray[np.object_],
    feature_name: str,
    feature_index: int,
    baseline_score: float,
    n_repeats: int,
    rng: np.random.Generator,
) -> FeatureImportanceRecord:
    repeat_importances: list[float] = []
    for _ in range(n_repeats):
        permuted = matrix.copy()
        permuted[:, feature_index] = rng.permutation(permuted[:, feature_index])
        repeat_importances.append(baseline_score - float(estimator.score(permuted, labels)))
    repeat_array = np.asarray(repeat_importances, dtype=np.float64)
    return FeatureImportanceRecord(
        feature_name=feature_name,
        mean_importance=float(np.mean(repeat_array)),
        std_importance=float(np.std(repeat_array)),
        repeat_importances=tuple(float(value) for value in repeat_array.tolist()),
    )


def _validate_importance_matrix(matrix: NDArray[np.float64]) -> None:
    if not np.all(np.isfinite(matrix)):
        raise ValueError("importance features must be finite")


def _validate_importance_labels(labels: object, sample_count: int) -> NDArray[np.object_]:
    if labels is None:
        raise ValueError("importance labels are required")
    label_array = np.asarray(labels, dtype=object)
    if label_array.ndim != 1:
        raise ValueError("importance labels must be one-dimensional")
    if label_array.shape[0] != sample_count:
        raise ValueError("importance labels length must match feature row count")
    return label_array


def _validate_feature_names(feature_names: object, feature_count: int) -> tuple[str, ...]:
    name_array = np.asarray(feature_names, dtype=object)
    if name_array.ndim != 1:
        raise ValueError("feature_names must be one-dimensional")
    names = tuple(name_array.tolist())
    if len(names) != feature_count:
        raise ValueError("feature_names length must match feature column count")
    seen: set[str] = set()
    for feature_name in names:
        if not isinstance(feature_name, str):
            raise ValueError("feature_names must be strings")
        if len(feature_name) == 0:
            raise ValueError("feature_names must not contain empty values")
        if feature_name in seen:
            raise ValueError(f"Duplicate feature name: {feature_name}")
        seen.add(feature_name)
    return names


def _validate_importance_config(n_repeats: int) -> None:
    if n_repeats <= 0:
        raise ValueError("n_repeats must be positive")


def shap_importance_report(estimator: ProbabilityEstimatorProtocol, features: object, feature_names: object) -> FeatureImportanceReport:
    """Compute mean absolute SHAP importances. Requires the evaluation-shap extra (shap)."""
    try:
        import shap
    except ImportError as exc:
        raise OptionalDependencyError("SHAP feature importance requires the 'evaluation-shap' extra (shap)") from exc

    matrix = as_float_matrix(features)
    _validate_importance_matrix(matrix)
    names = _validate_feature_names(feature_names, matrix.shape[1])
    explainer = shap.Explainer(estimator.predict_proba, matrix)
    values = np.asarray(explainer(matrix).values, dtype=np.float64)
    reduce_axes = tuple(axis for axis in range(values.ndim) if axis != 1)
    mean_abs = np.atleast_1d(np.mean(np.abs(values), axis=reduce_axes))
    records = tuple(
        FeatureImportanceRecord(feature_name=name, mean_importance=float(value), std_importance=0.0, repeat_importances=(float(value),))
        for name, value in zip(names, mean_abs.tolist(), strict=True)
    )
    return FeatureImportanceReport(baseline_score=float("nan"), n_repeats=1, random_state=0, records=records)
