"""Tests for permutation feature-importance helpers."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from stylometry_python_lib.evaluation import (
    FeatureImportanceRecord,
    FeatureImportanceReport,
    SupervisedClassifier,
    permutation_importance_report,
)


def test_permutation_importance_report_scores_informative_feature() -> None:
    features = np.asarray(
        [
            [0.0, 1.0],
            [0.1, 1.0],
            [10.0, 1.0],
            [10.1, 1.0],
            [0.2, 1.0],
            [10.2, 1.0],
        ],
        dtype=np.float64,
    )
    labels = ["author_a", "author_a", "author_b", "author_b", "author_a", "author_b"]
    estimator = SupervisedClassifier(classifier="logistic_regression", random_state=3, max_iter=2000, n_estimators=10).fit(
        features,
        labels,
    )

    report = permutation_importance_report(
        estimator=estimator,
        features=features,
        labels=labels,
        feature_names=("style_signal", "constant_noise"),
        n_repeats=5,
        random_state=9,
    )

    assert isinstance(report, FeatureImportanceReport)
    assert report.baseline_score == 1.0
    assert report.n_repeats == 5
    assert report.random_state == 9
    assert len(report.records) == 2
    assert isinstance(report.records[0], FeatureImportanceRecord)
    assert report.records[0].feature_name == "style_signal"
    assert report.records[1].feature_name == "constant_noise"
    assert report.records[0].mean_importance > report.records[1].mean_importance
    assert report.records[1].mean_importance == 0.0
    assert len(report.records[0].repeat_importances) == 5


def test_permutation_importance_report_validates_inputs_and_fit_state() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0]], dtype=np.float64)
    labels = ["author_a", "author_a", "author_b", "author_b"]
    estimator = SupervisedClassifier(classifier="nearest_centroid", random_state=1, max_iter=1000, n_estimators=10)

    with pytest.raises(NotFittedError):
        permutation_importance_report(estimator, features, labels, ("signal",), n_repeats=3, random_state=1)
    estimator.fit(features, labels)
    with pytest.raises(ValueError, match="feature_names length must match feature column count"):
        permutation_importance_report(estimator, features, labels, ("left", "right"), n_repeats=3, random_state=1)
    with pytest.raises(ValueError, match="feature_names must be one-dimensional"):
        permutation_importance_report(estimator, features, labels, [["signal"]], n_repeats=3, random_state=1)
    with pytest.raises(ValueError, match="Duplicate feature name"):
        permutation_importance_report(
            estimator,
            np.asarray([[0.0, 1.0], [1.0, 1.0], [10.0, 1.0], [11.0, 1.0]]),
            labels,
            ("signal", "signal"),
            n_repeats=3,
            random_state=1,
        )
    with pytest.raises(ValueError, match="n_repeats must be positive"):
        permutation_importance_report(estimator, features, labels, ("signal",), n_repeats=0, random_state=1)
    with pytest.raises(ValueError, match="importance labels length must match feature row count"):
        permutation_importance_report(estimator, features, labels[:1], ("signal",), n_repeats=3, random_state=1)
    with pytest.raises(ValueError, match="importance features must be finite"):
        permutation_importance_report(
            estimator,
            np.asarray([[0.0], [np.nan], [10.0], [11.0]]),
            labels,
            ("signal",),
            n_repeats=3,
            random_state=1,
        )
