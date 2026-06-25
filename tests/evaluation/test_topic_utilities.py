"""Tests for evaluation and topic-leakage utilities."""

from __future__ import annotations

import pickle
from collections.abc import Sequence
from typing import cast

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from stylometry_python_lib.evaluation import (
    ClusteringResult,
    PCAReducer,
    TopicPredictionControl,
    TopicPredictionReport,
    ZScoreStandardizer,
    ablation_scores,
    burrows_delta,
    cluster_feature_matrix,
    content_mask_text,
    cosine_distance_matrix,
    cross_topic_holdout_indices,
    euclidean_distance_matrix,
    length_sensitivity,
    same_author_topic_shift_pairs,
    same_topic_hard_negative_pairs,
    topic_prediction_control_report,
    topic_prediction_leakage_score,
    two_way_effect_sizes,
)
from stylometry_python_lib.evaluation.topic import simple_logistic_accuracy


def test_standardization_reduction_and_distances() -> None:
    x = np.asarray([[1.0, 2.0], [2.0, 4.0], [4.0, 8.0]])
    standardizer = ZScoreStandardizer(fail_on_zero_variance=True)

    z = standardizer.fit_transform(x, None)
    reduced = PCAReducer(n_components=1, method="pca", random_state=1).fit_transform(z, None)

    assert z.shape == x.shape
    assert reduced.shape == (3, 1)
    assert burrows_delta(x).shape == (3, 3)
    assert cosine_distance_matrix(x).shape == (3, 3)
    assert euclidean_distance_matrix(x)[0, 0] == 0.0


def test_clustering_helper_returns_labels_centers_and_distances() -> None:
    features = np.asarray([[0.0, 0.0], [0.0, 0.2], [10.0, 10.0], [10.2, 10.0]])

    result = cluster_feature_matrix(features, n_clusters=2, method="kmeans", random_state=4)

    assert isinstance(result, ClusteringResult)
    assert result.method == "kmeans"
    assert result.n_clusters == 2
    assert result.sample_count == 4
    assert len(result.labels) == 4
    assert len(result.cluster_centers) == 2
    assert len(result.distance_to_centers) == 4
    assert result.labels[0] == result.labels[1]
    assert result.labels[2] == result.labels[3]
    assert result.labels[0] != result.labels[2]
    for row_index, label in enumerate(result.labels):
        assigned_distance = result.distance_to_centers[row_index][label]
        assert assigned_distance == min(result.distance_to_centers[row_index])


def test_clustering_helper_validates_configuration() -> None:
    features = np.asarray([[0.0], [1.0], [10.0]])

    with pytest.raises(ValueError, match="n_clusters must be at least 2"):
        cluster_feature_matrix(features, n_clusters=1, method="kmeans", random_state=1)
    with pytest.raises(ValueError, match="n_clusters must not exceed feature row count"):
        cluster_feature_matrix(features, n_clusters=4, method="kmeans", random_state=1)
    with pytest.raises(ValueError, match="Unsupported clustering method"):
        cluster_feature_matrix(features, n_clusters=2, method="agglomerative", random_state=1)
    with pytest.raises(ValueError, match="clustering features must be finite"):
        cluster_feature_matrix(np.asarray([[0.0], [np.nan], [1.0]]), n_clusters=2, method="kmeans", random_state=1)


def test_topic_pair_and_masking_utilities() -> None:
    authors = ["a", "b", "a", "c"]
    topics = ["x", "x", "y", "y"]

    train, test = cross_topic_holdout_indices(topics, "y")
    masked = content_mask_text("The cat sat with the dog.", frozenset({"cat", "dog"}), "CONTENT")

    assert train.tolist() == [0, 1]
    assert test.tolist() == [2, 3]
    assert same_topic_hard_negative_pairs(authors, topics) == ((0, 1), (2, 3))
    assert same_author_topic_shift_pairs(authors, topics) == ((0, 2),)
    assert masked == "The CONTENT sat with the CONTENT."


def test_topic_leakage_ablation_length_and_effect_sizes() -> None:
    features = np.asarray([[0.0, 1.0], [0.1, 1.1], [3.0, 0.0], [3.1, 0.1], [0.2, 1.2], [3.2, 0.2]])
    topics = ["x", "x", "y", "y", "x", "y"]
    authors = ["a", "b", "a", "b", "c", "c"]
    leakage = topic_prediction_leakage_score(features, topics, cv_folds=3, random_state=1)
    scores = ablation_scores({"all": features}, topics, simple_logistic_accuracy)
    sensitivity = length_sensitivity(features, [5, 6, 20, 21, 7, 22])
    effects = two_way_effect_sizes([1.0, 2.0, 3.0, 4.0, 2.0, 3.0], authors, topics)

    assert leakage >= 0.5
    assert scores["all"] >= 0.5
    assert sensitivity.shape == (2,)
    assert effects["author_effect"] >= 0.0
    assert effects["topic_effect"] >= 0.0


def test_topic_prediction_control_report_has_known_topic_fixture_and_predictions() -> None:
    features = np.asarray(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [10.0, 10.0],
            [11.0, 10.0],
            [1.0, 0.0],
            [10.0, 11.0],
        ]
    )
    topics = ["x", "x", "y", "y", "x", "y"]
    report = topic_prediction_control_report(features, topics, cv_folds=3, random_state=7, max_iter=1000)
    control = TopicPredictionControl(cv_folds=3, random_state=7, max_iter=1000)

    fitted = control.fit(features, topics)
    restored = pickle.loads(pickle.dumps(fitted))
    predictions = restored.predict(features)

    assert isinstance(report, TopicPredictionReport)
    assert report.classes == ("x", "y")
    assert report.cv_folds == 3
    assert report.random_state == 7
    assert report.classifier_name == "LogisticRegression"
    assert report.test_indices_by_fold == ((0, 3), (1, 4), (2, 5))
    assert report.train_indices_by_fold[0] == (1, 2, 4, 5)
    assert report.true_topics == tuple(topics)
    assert report.predicted_topics == tuple(topics)
    assert report.fold_scores == (1.0, 1.0, 1.0)
    assert report.leakage_score == topic_prediction_leakage_score(features, topics, cv_folds=3, random_state=7)
    assert fitted.report_ == report
    assert fitted.classes_.tolist() == ["x", "y"]
    assert fitted.score(features, topics) == 1.0
    assert predictions.tolist() == topics


def test_topic_prediction_control_validates_label_shapes_and_fit_state() -> None:
    features = np.asarray([[0.0], [1.0], [10.0], [11.0]])
    control = TopicPredictionControl(cv_folds=2, random_state=1, max_iter=1000)
    nested_topics = cast(Sequence[str], [["x"], ["y"], ["x"], ["y"]])

    with pytest.raises(NotFittedError):
        control.predict(features)
    with pytest.raises(ValueError, match="topics length must match feature row count"):
        topic_prediction_control_report(features, ["x"], cv_folds=2, random_state=1, max_iter=1000)
    with pytest.raises(ValueError, match="topic labels must be one-dimensional"):
        topic_prediction_control_report(features, nested_topics, cv_folds=2, random_state=1, max_iter=1000)
    with pytest.raises(ValueError, match="topics must contain at least two distinct labels"):
        topic_prediction_control_report(features, ["x", "x", "x", "x"], cv_folds=2, random_state=1, max_iter=1000)
    with pytest.raises(ValueError, match="cv_folds must not exceed feature row count"):
        topic_prediction_control_report(features, ["x", "y", "x", "y"], cv_folds=5, random_state=1, max_iter=1000)
    with pytest.raises(ValueError, match="at least two train topic labels"):
        topic_prediction_control_report(
            np.asarray([[0.0], [10.0], [11.0]]),
            ["x", "y", "y"],
            cv_folds=2,
            random_state=1,
            max_iter=1000,
        )


def test_undefined_evaluation_inputs_fail_fast() -> None:
    with pytest.raises(ValueError):
        cosine_distance_matrix(np.asarray([[0.0, 0.0], [1.0, 0.0]]))
    with pytest.raises(ValueError):
        content_mask_text("text", frozenset({"text"}), "")
