"""Distance metrics for stylometry feature matrices."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def as_float_matrix(matrix: object) -> NDArray[np.float64]:
    """Convert a matrix-like object to a two-dimensional float array."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("Expected a two-dimensional matrix")
    return array


def burrows_delta(matrix: object) -> NDArray[np.float64]:
    """Compute pairwise Burrows' Delta over already comparable feature columns."""
    array = as_float_matrix(matrix)
    standardized = _zscore_array(array)
    sample_count = standardized.shape[0]
    distances = np.zeros((sample_count, sample_count), dtype=np.float64)
    for left in range(sample_count):
        for right in range(sample_count):
            absolute_delta = np.asarray(np.abs(standardized[left] - standardized[right]), dtype=np.float64)
            distances[left, right] = float(np.sum(absolute_delta)) / float(absolute_delta.shape[0])
    return distances


def cosine_distance_matrix(matrix: object) -> NDArray[np.float64]:
    """Compute pairwise cosine distances."""
    array = as_float_matrix(matrix)
    norms = np.linalg.norm(array, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("Cosine distance is undefined for zero-norm rows")
    normalized = array / norms[:, np.newaxis]
    similarity = normalized @ normalized.T
    return np.asarray(1.0 - similarity, dtype=np.float64)


def euclidean_distance_matrix(matrix: object) -> NDArray[np.float64]:
    """Compute pairwise Euclidean distances."""
    array = as_float_matrix(matrix)
    sample_count = array.shape[0]
    distances = np.zeros((sample_count, sample_count), dtype=np.float64)
    for left in range(sample_count):
        deltas = array[left] - array
        distances[left] = np.sqrt(np.sum(deltas * deltas, axis=1))
    return distances


def _zscore_array(array: NDArray[np.float64]) -> NDArray[np.float64]:
    means = np.nanmean(array, axis=0)
    stds = np.nanstd(array, axis=0)
    if np.any(stds == 0.0):
        raise ValueError("Z-score standardization is undefined for zero-variance columns")
    return np.asarray((array - means) / stds, dtype=np.float64)
