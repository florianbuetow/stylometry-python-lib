"""Tests for composite evaluation reports."""

from __future__ import annotations

import numpy as np
import pytest

from stylometry_python_lib.evaluation import EvaluationReport, SplitDiagnostics, style_evaluation_report
from stylometry_python_lib.evaluation.topic import simple_logistic_accuracy


def test_style_evaluation_report_combines_core_metrics() -> None:
    features = np.asarray([[0.0, 1.0], [0.1, 1.1], [3.0, 0.0], [3.1, 0.1], [0.2, 1.2], [3.2, 0.2]])
    topics = ["x", "x", "y", "y", "x", "y"]
    labels = ["a", "a", "b", "b", "a", "b"]

    report = style_evaluation_report(
        features=features,
        feature_names=("style_signal", "topic_signal"),
        topics=topics,
        token_counts=[5, 6, 20, 21, 7, 22],
        features_by_family={"all": features, "style": features[:, :1]},
        labels=labels,
        scorer=simple_logistic_accuracy,
        cv_folds=3,
        random_state=1,
    )

    assert isinstance(report, EvaluationReport)
    assert report.schema_version == "style_evaluation_report_v1"
    assert report.topic_leakage_score >= 0.5
    assert report.ablation_scores == (("all", 1.0), ("style", 1.0))
    assert report.family_robustness == report.ablation_scores
    assert report.length_sensitivity[0][0] == "style_signal"
    assert report.length_sensitivity[1][0] == "topic_signal"
    assert isinstance(report.split_diagnostics, SplitDiagnostics)
    assert report.split_diagnostics.sample_count == 6
    assert report.split_diagnostics.topic_count == 2
    assert report.split_diagnostics.min_topic_count == 3
    assert report.split_diagnostics.underpowered is False
    assert report.split_diagnostics.warning == "none"


def test_style_evaluation_report_validates_feature_names_and_underpowered_splits() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0]])
    labels = ["a", "a", "b", "b"]

    report = style_evaluation_report(
        features=features,
        feature_names=("signal",),
        topics=["x", "x", "y", "y"],
        token_counts=[5, 6, 20, 21],
        features_by_family={"all": features},
        labels=labels,
        scorer=simple_logistic_accuracy,
        cv_folds=2,
        random_state=1,
    )

    assert report.split_diagnostics.underpowered is True
    assert report.split_diagnostics.warning == "underpowered_topic_split"
    with pytest.raises(ValueError, match="feature_names length must match feature column count"):
        style_evaluation_report(
            features, ("left", "right"), ["x", "x", "y", "y"], [5, 6, 20, 21], {"all": features}, labels, simple_logistic_accuracy, 2, 1
        )
    with pytest.raises(ValueError, match="Duplicate feature name"):
        style_evaluation_report(
            np.asarray([[0.0, 1.0], [1.0, 1.0], [10.0, 1.0], [11.0, 1.0]]),
            ("signal", "signal"),
            ["x", "x", "y", "y"],
            [5, 6, 20, 21],
            {"all": features},
            labels,
            simple_logistic_accuracy,
            2,
            1,
        )
