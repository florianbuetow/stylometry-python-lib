"""Open-world verification helpers for stylometry evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from stylometry_python_lib.evaluation.distances import as_float_matrix, cosine_distance_matrix, euclidean_distance_matrix


@dataclass(frozen=True)
class VerificationDecision:
    """Thresholded pairwise verification decision preserving document identities."""

    left_document_id: str
    right_document_id: str
    distance: float
    threshold: float
    accepted_same_author: bool
    metric: str


@dataclass(frozen=True)
class VerificationReport:
    """Pairwise open-world verification report."""

    metric: str
    threshold: float
    document_ids: tuple[str, ...]
    decisions: tuple[VerificationDecision, ...]


def thresholded_distance_verification(
    features: object,
    document_ids: object,
    pairs: Sequence[tuple[str, str]],
    threshold: float,
    metric: str,
) -> VerificationReport:
    """Apply a thresholded pairwise-distance verification protocol."""
    matrix = as_float_matrix(features)
    ids = _validate_document_ids(document_ids, matrix.shape[0])
    pair_tuple = _validate_verification_pairs(pairs, ids)
    _validate_verification_config(matrix, threshold, metric)
    distance_matrix = _verification_distance_matrix(matrix, metric)
    id_to_index = {document_id: index for index, document_id in enumerate(ids)}
    decisions = tuple(
        _verification_decision(
            left_document_id=left_document_id,
            right_document_id=right_document_id,
            distance=float(distance_matrix[id_to_index[left_document_id], id_to_index[right_document_id]]),
            threshold=threshold,
            metric=metric,
        )
        for left_document_id, right_document_id in pair_tuple
    )
    return VerificationReport(metric=metric, threshold=threshold, document_ids=ids, decisions=decisions)


def _verification_decision(
    left_document_id: str,
    right_document_id: str,
    distance: float,
    threshold: float,
    metric: str,
) -> VerificationDecision:
    return VerificationDecision(
        left_document_id=left_document_id,
        right_document_id=right_document_id,
        distance=distance,
        threshold=threshold,
        accepted_same_author=distance <= threshold,
        metric=metric,
    )


def _validate_document_ids(document_ids: object, sample_count: int) -> tuple[str, ...]:
    id_array = np.asarray(document_ids, dtype=object)
    if id_array.ndim != 1:
        raise ValueError("document_ids must be one-dimensional")
    ids = tuple(id_array.tolist())
    if len(ids) != sample_count:
        raise ValueError("document_ids length must match feature row count")
    seen: set[str] = set()
    for document_id in ids:
        if not isinstance(document_id, str):
            raise ValueError("document_ids must be strings")
        if len(document_id) == 0:
            raise ValueError("document_ids must not contain empty values")
        if document_id in seen:
            raise ValueError(f"Duplicate document id: {document_id}")
        seen.add(document_id)
    return ids


def _validate_verification_pairs(
    pairs: Sequence[tuple[str, str]],
    document_ids: tuple[str, ...],
) -> tuple[tuple[str, str], ...]:
    pair_tuple = tuple(pairs)
    if len(pair_tuple) == 0:
        raise ValueError("verification pairs must not be empty")
    known_ids = set(document_ids)
    for left_document_id, right_document_id in pair_tuple:
        if left_document_id == right_document_id:
            raise ValueError("verification pairs must compare distinct documents")
        if left_document_id not in known_ids:
            raise ValueError(f"Unknown verification document id: {left_document_id}")
        if right_document_id not in known_ids:
            raise ValueError(f"Unknown verification document id: {right_document_id}")
    return pair_tuple


def _validate_verification_config(matrix: NDArray[np.float64], threshold: float, metric: str) -> None:
    if threshold < 0.0:
        raise ValueError("verification threshold must be non-negative")
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"Unsupported verification metric: {metric}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("verification features must be finite")


def _verification_distance_matrix(matrix: NDArray[np.float64], metric: str) -> NDArray[np.float64]:
    if metric == "euclidean":
        return euclidean_distance_matrix(matrix)
    if metric == "cosine":
        return cosine_distance_matrix(matrix)
    raise ValueError(f"Unsupported verification metric: {metric}")


def _verification_index_map(document_ids: Sequence[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for position, document_id in enumerate(document_ids):
        if document_id in mapping:
            raise ValueError(f"Duplicate document id in verification input: {document_id}")
        mapping[document_id] = position
    return mapping


def one_class_verification(
    features: object,
    document_ids: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    nu: float,
    random_state: int,
) -> VerificationReport:
    """One-class SVM novelty verification preserving document identity."""
    from sklearn.svm import OneClassSVM

    matrix = as_float_matrix(features)
    if not 0.0 < nu <= 1.0:
        raise ValueError("nu must be in (0, 1]")
    if matrix.shape[0] != len(document_ids):
        raise ValueError("features row count must match document_ids length")
    index = _verification_index_map(document_ids)
    model = OneClassSVM(nu=nu)
    model.fit(matrix)
    scores = np.asarray(model.decision_function(matrix), dtype=np.float64)
    predictions = np.asarray(model.predict(matrix), dtype=np.int64)
    decisions: list[VerificationDecision] = []
    for left_document_id, right_document_id in pairs:
        if left_document_id not in index or right_document_id not in index:
            raise ValueError(f"Unknown verification document id in pair: {left_document_id}, {right_document_id}")
        distance = float(abs(scores[index[left_document_id]] - scores[index[right_document_id]]))
        accepted = bool(predictions[index[left_document_id]] > 0 and predictions[index[right_document_id]] > 0)
        decisions.append(
            VerificationDecision(
                left_document_id=left_document_id,
                right_document_id=right_document_id,
                distance=distance,
                threshold=0.0,
                accepted_same_author=accepted,
                metric="one_class_svm",
            )
        )
    return VerificationReport(metric="one_class_svm", threshold=0.0, document_ids=tuple(document_ids), decisions=tuple(decisions))


def calibrated_binary_verification(
    train_features: object,
    train_labels: Sequence[str],
    features: object,
    document_ids: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    classifier: str,
    random_state: int,
) -> VerificationReport:
    """Calibrated binary same-author verification preserving document identity.

    Same-author acceptance uses the L1 distance between the two documents'
    calibrated class-probability vectors; smaller distance means more similar
    author profiles. Probabilities come from a logistic model, whose logistic
    link yields calibrated estimates without a separate post-hoc calibrator.
    """
    from collections import Counter

    from sklearn.linear_model import LogisticRegression

    if classifier != "logistic_regression":
        raise ValueError("calibrated binary verification supports classifier='logistic_regression'")
    matrix = as_float_matrix(features)
    train_matrix = as_float_matrix(train_features)
    if matrix.shape[0] != len(document_ids):
        raise ValueError("features row count must match document_ids length")
    if train_matrix.shape[0] != len(train_labels):
        raise ValueError("train_features row count must match train_labels length")
    label_list = [str(label) for label in train_labels]
    if min(Counter(label_list).values()) < 2:
        raise ValueError("calibrated binary verification requires at least 2 training samples per class")
    threshold = 0.5
    index = _verification_index_map(document_ids)
    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(train_matrix, label_list)
    probabilities = np.asarray(model.predict_proba(matrix), dtype=np.float64)
    decisions: list[VerificationDecision] = []
    for left_document_id, right_document_id in pairs:
        if left_document_id not in index or right_document_id not in index:
            raise ValueError(f"Unknown verification document id in pair: {left_document_id}, {right_document_id}")
        distance = float(np.abs(probabilities[index[left_document_id]] - probabilities[index[right_document_id]]).sum())
        decisions.append(
            VerificationDecision(
                left_document_id=left_document_id,
                right_document_id=right_document_id,
                distance=distance,
                threshold=threshold,
                accepted_same_author=distance < threshold,
                metric="calibrated_binary",
            )
        )
    return VerificationReport(metric="calibrated_binary", threshold=threshold, document_ids=tuple(document_ids), decisions=tuple(decisions))
