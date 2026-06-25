"""Supervised classifier wrappers for stylometry evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib.evaluation.distances import as_float_matrix

_CLASSIFIER_NAMES = ("logistic_regression", "linear_svm", "random_forest", "nearest_centroid")


@dataclass(frozen=True)
class ClassifierReport:
    """In-sample classifier diagnostics for an explicit evaluation wrapper."""

    classifier: str
    accuracy: float
    classes: tuple[str, ...]
    predictions: tuple[str, ...]
    true_labels: tuple[str, ...]
    sample_count: int
    feature_count: int


class SupervisedClassifier(BaseEstimator):
    """Sklearn-compatible classifier wrapper with explicit supported algorithms."""

    def __init__(self, classifier: str, random_state: int, max_iter: int, n_estimators: int) -> None:
        self.classifier = classifier
        self.random_state = random_state
        self.max_iter = max_iter
        self.n_estimators = n_estimators

    def fit(self, x: object, y: object) -> Self:
        """Fit the configured classifier against aligned string labels."""
        matrix = as_float_matrix(x)
        labels = _validate_classifier_labels(y, matrix.shape[0])
        _validate_classifier_config(matrix, self.classifier, self.max_iter, self.n_estimators)
        estimator = _build_classifier(self.classifier, self.random_state, self.max_iter, self.n_estimators)
        estimator.fit(matrix, labels)
        self.estimator_ = estimator
        self.classes_ = np.asarray(tuple(str(label) for label in estimator.classes_.tolist()), dtype=object)
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, x: object) -> NDArray[np.object_]:
        """Predict labels with the fitted classifier."""
        require_fitted(self, "estimator_")
        matrix = as_float_matrix(x)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Input feature width changed after fit")
        return np.asarray(self.estimator_.predict(matrix), dtype=object)

    def score(self, x: object, y: object) -> float:
        """Return accuracy for aligned labels."""
        predictions = self.predict(x)
        labels = _validate_classifier_labels(y, predictions.shape[0])
        return float(accuracy_score(labels, predictions))


def classifier_report(
    features: object,
    labels: object,
    classifier: str,
    random_state: int,
    max_iter: int,
    n_estimators: int,
) -> ClassifierReport:
    """Fit a supervised classifier wrapper and return deterministic diagnostics."""
    model = SupervisedClassifier(
        classifier=classifier,
        random_state=random_state,
        max_iter=max_iter,
        n_estimators=n_estimators,
    )
    fitted = model.fit(features, labels)
    matrix = as_float_matrix(features)
    validated_labels = _validate_classifier_labels(labels, matrix.shape[0])
    predictions = fitted.predict(matrix)
    return ClassifierReport(
        classifier=classifier,
        accuracy=float(accuracy_score(validated_labels, predictions)),
        classes=tuple(str(label) for label in fitted.classes_.tolist()),
        predictions=tuple(str(label) for label in predictions.tolist()),
        true_labels=tuple(str(label) for label in validated_labels.tolist()),
        sample_count=matrix.shape[0],
        feature_count=matrix.shape[1],
    )


def _build_classifier(
    classifier: str,
    random_state: int,
    max_iter: int,
    n_estimators: int,
) -> LogisticRegression | LinearSVC | RandomForestClassifier | NearestCentroid:
    if classifier == "logistic_regression":
        return LogisticRegression(max_iter=max_iter, random_state=random_state)
    if classifier == "linear_svm":
        return LinearSVC(max_iter=max_iter, random_state=random_state)
    if classifier == "random_forest":
        return RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    if classifier == "nearest_centroid":
        return NearestCentroid()
    raise ValueError(f"Unsupported classifier: {classifier}")


def _validate_classifier_config(matrix: NDArray[np.float64], classifier: str, max_iter: int, n_estimators: int) -> None:
    if classifier not in _CLASSIFIER_NAMES:
        raise ValueError(f"Unsupported classifier: {classifier}")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("classifier features must be finite")


def _validate_classifier_labels(labels: object, sample_count: int) -> NDArray[np.object_]:
    if labels is None:
        raise ValueError("classifier labels are required")
    label_array = np.asarray(labels, dtype=object)
    if label_array.ndim != 1:
        raise ValueError("classifier labels must be one-dimensional")
    if label_array.shape[0] != sample_count:
        raise ValueError("classifier labels length must match feature row count")
    for label in label_array.tolist():
        if not isinstance(label, str):
            raise ValueError("classifier labels must be strings")
        if len(label) == 0:
            raise ValueError("classifier labels must not be empty")
    if len(np.unique(label_array)) < 2:
        raise ValueError("classifier labels must contain at least two distinct values")
    return label_array
