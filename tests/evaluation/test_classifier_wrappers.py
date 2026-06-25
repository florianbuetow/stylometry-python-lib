"""Tests for supervised classifier evaluation wrappers."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from stylometry_python_lib.evaluation import ClassifierReport, SupervisedClassifier, classifier_report


def _classifier_fixture() -> tuple[np.ndarray, list[str]]:
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [10.0, 10.0],
            [10.1, 10.0],
            [0.0, 0.2],
            [10.0, 10.2],
        ],
        dtype=np.float64,
    )
    return features, ["author_a", "author_a", "author_b", "author_b", "author_a", "author_b"]


@pytest.mark.parametrize("classifier", ["logistic_regression", "linear_svm", "random_forest", "nearest_centroid"])
def test_supervised_classifier_wrappers_fit_predict_score_and_report(classifier: str) -> None:
    features, labels = _classifier_fixture()
    wrapper = SupervisedClassifier(classifier=classifier, random_state=5, max_iter=2000, n_estimators=20)

    fitted = wrapper.fit(features, labels)
    restored = pickle.loads(pickle.dumps(fitted))
    predictions = restored.predict(features)
    report = classifier_report(
        features=features,
        labels=labels,
        classifier=classifier,
        random_state=5,
        max_iter=2000,
        n_estimators=20,
    )

    assert isinstance(report, ClassifierReport)
    assert fitted.classes_.tolist() == ["author_a", "author_b"]
    assert predictions.tolist() == labels
    assert fitted.score(features, labels) == 1.0
    assert report.classifier == classifier
    assert report.accuracy == 1.0
    assert report.classes == ("author_a", "author_b")
    assert report.predictions == tuple(labels)
    assert report.true_labels == tuple(labels)
    assert report.sample_count == 6
    assert report.feature_count == 2


def test_supervised_classifier_validates_config_labels_and_fit_state() -> None:
    features, labels = _classifier_fixture()
    nested_labels = cast(Sequence[str], [["author_a"], ["author_b"], ["author_a"], ["author_b"], ["author_a"], ["author_b"]])
    wrapper = SupervisedClassifier(classifier="logistic_regression", random_state=1, max_iter=1000, n_estimators=10)

    with pytest.raises(NotFittedError):
        wrapper.predict(features)
    with pytest.raises(ValueError, match="Unsupported classifier"):
        classifier_report(features, labels, classifier="xgboost", random_state=1, max_iter=1000, n_estimators=10)
    with pytest.raises(ValueError, match="max_iter must be positive"):
        classifier_report(features, labels, classifier="logistic_regression", random_state=1, max_iter=0, n_estimators=10)
    with pytest.raises(ValueError, match="n_estimators must be positive"):
        classifier_report(features, labels, classifier="random_forest", random_state=1, max_iter=1000, n_estimators=0)
    with pytest.raises(ValueError, match="classifier labels length must match feature row count"):
        classifier_report(features, labels[:1], classifier="logistic_regression", random_state=1, max_iter=1000, n_estimators=10)
    with pytest.raises(ValueError, match="classifier labels must be one-dimensional"):
        classifier_report(features, nested_labels, classifier="logistic_regression", random_state=1, max_iter=1000, n_estimators=10)
    with pytest.raises(ValueError, match="at least two distinct"):
        classifier_report(
            features,
            ["author_a", "author_a", "author_a", "author_a", "author_a", "author_a"],
            classifier="logistic_regression",
            random_state=1,
            max_iter=1000,
            n_estimators=10,
        )
    with pytest.raises(ValueError, match="classifier features must be finite"):
        classifier_report(
            np.asarray([[0.0], [np.nan], [1.0]]),
            ["author_a", "author_b", "author_a"],
            classifier="nearest_centroid",
            random_state=1,
            max_iter=1000,
            n_estimators=10,
        )
