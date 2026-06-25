"""Composite evaluation report objects for style-vs-topic checks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stylometry_python_lib.evaluation.distances import as_float_matrix
from stylometry_python_lib.evaluation.topic import ablation_scores, length_sensitivity, topic_prediction_leakage_score


@dataclass(frozen=True)
class SplitDiagnostics:
    """Simple topic-split diagnostics for report consumers."""

    sample_count: int
    topic_count: int
    min_topic_count: int
    underpowered: bool
    warning: str


@dataclass(frozen=True)
class EvaluationReport:
    """Combined style-vs-topic evaluation report."""

    schema_version: str
    topic_leakage_score: float
    ablation_scores: tuple[tuple[str, float], ...]
    length_sensitivity: tuple[tuple[str, float], ...]
    split_diagnostics: SplitDiagnostics
    family_robustness: tuple[tuple[str, float], ...]


def style_evaluation_report(
    features: object,
    feature_names: object,
    topics: Sequence[str],
    token_counts: Sequence[int],
    features_by_family: dict[str, object],
    labels: Sequence[str],
    scorer: Callable[[NDArray[np.float64], Sequence[str]], float],
    cv_folds: int,
    random_state: int,
) -> EvaluationReport:
    """Build a composite style-vs-topic report from existing evaluation utilities."""
    matrix = as_float_matrix(features)
    names = _validate_report_feature_names(feature_names, matrix.shape[1])
    leakage = topic_prediction_leakage_score(matrix, topics, cv_folds=cv_folds, random_state=random_state)
    ablation = ablation_scores(features_by_family, labels, scorer)
    sensitivity = length_sensitivity(matrix, token_counts)
    ablation_tuple = tuple(sorted((family, float(score)) for family, score in ablation.items()))
    return EvaluationReport(
        schema_version="style_evaluation_report_v1",
        topic_leakage_score=leakage,
        ablation_scores=ablation_tuple,
        length_sensitivity=tuple((feature_name, float(value)) for feature_name, value in zip(names, sensitivity.tolist(), strict=True)),
        split_diagnostics=_split_diagnostics(topics),
        family_robustness=ablation_tuple,
    )


def _split_diagnostics(topics: Sequence[str]) -> SplitDiagnostics:
    if len(topics) == 0:
        raise ValueError("topics must not be empty")
    counts = Counter(topics)
    min_topic_count = min(counts.values())
    underpowered = False
    if len(counts) < 2:
        underpowered = True
    if min_topic_count < 3:
        underpowered = True
    warning = "underpowered_topic_split" if underpowered else "none"
    return SplitDiagnostics(
        sample_count=len(topics),
        topic_count=len(counts),
        min_topic_count=min_topic_count,
        underpowered=underpowered,
        warning=warning,
    )


def _validate_report_feature_names(feature_names: object, feature_count: int) -> tuple[str, ...]:
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
