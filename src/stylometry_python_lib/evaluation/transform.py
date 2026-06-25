"""Sklearn-compatible evaluation transforms."""

from __future__ import annotations

from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA, TruncatedSVD

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib.evaluation.distances import as_float_matrix


class ZScoreStandardizer(BaseEstimator):
    """Fit corpus z-score statistics and apply them consistently."""

    def __init__(self, fail_on_zero_variance: bool) -> None:
        self.fail_on_zero_variance = fail_on_zero_variance

    def fit(self, x: object, y: object) -> Self:
        """Fit column means and standard deviations."""
        del y
        matrix = as_float_matrix(x)
        self.mean_ = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        if np.any(std == 0.0) and self.fail_on_zero_variance:
            raise ValueError("Cannot fit ZScoreStandardizer with zero-variance columns")
        adjusted_std = std.copy()
        adjusted_std[adjusted_std == 0.0] = 1.0
        self.scale_ = adjusted_std
        self.n_features_in_ = matrix.shape[1]
        return self

    def transform(self, x: object) -> NDArray[np.float64]:
        """Apply fitted z-score transformation."""
        require_fitted(self, "mean_")
        matrix = as_float_matrix(x)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Input feature width changed after fit")
        return np.asarray((matrix - self.mean_) / self.scale_, dtype=np.float64)

    def fit_transform(self, x: object, y: object) -> NDArray[np.float64]:
        """Fit, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)


class PCAReducer(BaseEstimator):
    """Fit PCA or SVD-style dimensionality reduction for feature matrices."""

    def __init__(self, n_components: int, method: str, random_state: int) -> None:
        self.n_components = n_components
        self.method = method
        self.random_state = random_state

    def fit(self, x: object, y: object) -> Self:
        """Fit the configured reducer."""
        del y
        matrix = as_float_matrix(x)
        reducer: PCA | TruncatedSVD
        if self.method == "pca":
            reducer = PCA(n_components=self.n_components, random_state=self.random_state)
        elif self.method == "svd":
            reducer = TruncatedSVD(n_components=self.n_components, random_state=self.random_state)
        else:
            raise ValueError(f"Unsupported reduction method: {self.method}")
        self.reducer_ = reducer.fit(matrix)
        self.n_features_in_ = matrix.shape[1]
        return self

    def transform(self, x: object) -> NDArray[np.float64]:
        """Apply fitted reduction."""
        require_fitted(self, "reducer_")
        matrix = as_float_matrix(x)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Input feature width changed after fit")
        transformed = self.reducer_.transform(matrix)
        return np.asarray(transformed, dtype=np.float64)

    def fit_transform(self, x: object, y: object) -> NDArray[np.float64]:
        """Fit, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)
