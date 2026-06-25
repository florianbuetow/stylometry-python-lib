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
