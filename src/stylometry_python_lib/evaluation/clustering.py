"""Core clustering helpers for stylometry feature matrices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import KMeans

from stylometry_python_lib.evaluation.distances import as_float_matrix


@dataclass(frozen=True)
class ClusteringResult:
    """Cluster labels and distance diagnostics for one feature matrix."""

    labels: tuple[int, ...]
    distance_to_centers: tuple[tuple[float, ...], ...]
    cluster_centers: tuple[tuple[float, ...], ...]
    method: str
    n_clusters: int
    sample_count: int


def cluster_feature_matrix(features: object, n_clusters: int, method: str, random_state: int) -> ClusteringResult:
    """Cluster feature rows and return labels plus distances to cluster centers."""
    matrix = as_float_matrix(features)
    _validate_clustering_config(matrix, n_clusters, method)
    if method != "kmeans":
        raise ValueError(f"Unsupported clustering method: {method}")
    model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = model.fit_predict(matrix)
    centers = np.asarray(model.cluster_centers_, dtype=np.float64)
    distances = _euclidean_distance_to_centers(matrix, centers)
    return ClusteringResult(
        labels=tuple(int(label) for label in labels.tolist()),
        distance_to_centers=_matrix_to_tuple(distances),
        cluster_centers=_matrix_to_tuple(centers),
        method=method,
        n_clusters=n_clusters,
        sample_count=matrix.shape[0],
    )


def _validate_clustering_config(matrix: NDArray[np.float64], n_clusters: int, method: str) -> None:
    if method == "":
        raise ValueError("clustering method must not be empty")
    if n_clusters < 2:
        raise ValueError("n_clusters must be at least 2")
    if n_clusters > matrix.shape[0]:
        raise ValueError("n_clusters must not exceed feature row count")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("clustering features must be finite")


def _euclidean_distance_to_centers(matrix: NDArray[np.float64], centers: NDArray[np.float64]) -> NDArray[np.float64]:
    distances = np.zeros((matrix.shape[0], centers.shape[0]), dtype=np.float64)
    for row_index, row in enumerate(matrix):
        deltas = centers - row
        distances[row_index] = np.sqrt(np.sum(deltas * deltas, axis=1))
    return distances


def _matrix_to_tuple(matrix: NDArray[np.float64]) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(float(value) for value in row) for row in matrix.tolist())
