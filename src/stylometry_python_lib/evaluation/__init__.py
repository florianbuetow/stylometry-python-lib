"""Evaluation utilities for style-vs-topic claims."""

from stylometry_python_lib.evaluation.classification import ClassifierReport, SupervisedClassifier, classifier_report
from stylometry_python_lib.evaluation.clustering import ClusteringResult, cluster_feature_matrix
from stylometry_python_lib.evaluation.distances import burrows_delta, cosine_distance_matrix, euclidean_distance_matrix
from stylometry_python_lib.evaluation.importance import (
    FeatureImportanceRecord,
    FeatureImportanceReport,
    permutation_importance_report,
)
from stylometry_python_lib.evaluation.report import EvaluationReport, SplitDiagnostics, style_evaluation_report
from stylometry_python_lib.evaluation.review import HumanReviewItem, HumanReviewPacket, human_review_packet
from stylometry_python_lib.evaluation.topic import (
    TopicPredictionControl,
    TopicPredictionReport,
    ablation_scores,
    content_mask_text,
    cross_topic_holdout_indices,
    length_sensitivity,
    same_author_topic_shift_pairs,
    same_topic_hard_negative_pairs,
    topic_prediction_control_report,
    topic_prediction_leakage_score,
    two_way_effect_sizes,
)
from stylometry_python_lib.evaluation.transform import PCAReducer, ZScoreStandardizer
from stylometry_python_lib.evaluation.verification import (
    VerificationDecision,
    VerificationReport,
    thresholded_distance_verification,
)

__all__ = [
    "ClassifierReport",
    "PCAReducer",
    "ClusteringResult",
    "FeatureImportanceRecord",
    "FeatureImportanceReport",
    "HumanReviewItem",
    "HumanReviewPacket",
    "EvaluationReport",
    "SupervisedClassifier",
    "SplitDiagnostics",
    "TopicPredictionControl",
    "TopicPredictionReport",
    "VerificationDecision",
    "VerificationReport",
    "ZScoreStandardizer",
    "ablation_scores",
    "classifier_report",
    "burrows_delta",
    "cluster_feature_matrix",
    "content_mask_text",
    "cosine_distance_matrix",
    "cross_topic_holdout_indices",
    "euclidean_distance_matrix",
    "length_sensitivity",
    "permutation_importance_report",
    "human_review_packet",
    "same_author_topic_shift_pairs",
    "same_topic_hard_negative_pairs",
    "style_evaluation_report",
    "thresholded_distance_verification",
    "topic_prediction_control_report",
    "topic_prediction_leakage_score",
    "two_way_effect_sizes",
]
