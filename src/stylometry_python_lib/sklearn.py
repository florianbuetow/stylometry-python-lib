"""Sklearn adapter layer for stylometry feature blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Self, cast

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, clone

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib._tabular import to_frame, validate_output_mode


class FeatureBlockProtocol(Protocol):
    """Protocol for feature blocks accepted by FeatureExtractor."""

    def fit(self, x: object, y: object) -> object:
        """Fit a block."""

    def transform(self, x: object) -> object:
        """Transform data with a block."""

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return block feature names."""
        ...


class SidecarBlockProtocol(Protocol):
    """Protocol for feature blocks that expose sidecar outputs."""

    last_sidecars_: tuple[object, ...]


class SparseLikeProtocol(Protocol):
    """Small protocol for sparse matrices returned by SciPy."""

    def toarray(self) -> np.ndarray:
        """Return a dense representation."""
        ...


@dataclass(frozen=True)
class FeatureBlockSidecars:
    """Sidecars emitted by one fitted feature block."""

    block_index: int
    block_name: str
    sidecars: tuple[object, ...]


class FeatureExtractor(BaseEstimator):
    """Combine independent stylometry feature blocks without changing rows."""

    def __init__(self, blocks: tuple[BaseEstimator, ...], output: str) -> None:
        self.blocks = blocks
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Fit each feature block and freeze output feature names."""
        validate_output_mode(self.output)
        frame = to_frame(x)
        self.n_features_in_ = frame.shape[1]
        self.feature_names_in_ = np.asarray(frame.columns, dtype=object)
        fitted_blocks: list[FeatureBlockProtocol] = []
        feature_names: list[str] = []
        for block in self.blocks:
            fitted = cast(FeatureBlockProtocol, clone(block))
            fitted.fit(frame, y)
            fitted_blocks.append(fitted)
            names = fitted.get_feature_names_out(self.feature_names_in_)
            feature_names.extend(str(name) for name in names.tolist())
        self.blocks_ = tuple(fitted_blocks)
        self.feature_names_out_ = np.asarray(feature_names, dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Transform data with all blocks and concatenate outputs."""
        require_fitted(self, "blocks_")
        frame = to_frame(x)
        if frame.shape[0] == 0:
            raise ValueError("FeatureExtractor requires at least one row")
        outputs = [block.transform(frame) for block in self.blocks_]
        self.last_sidecars_ = _collect_sidecars(self.blocks_)
        combined = _combine_outputs(outputs, self.feature_names_out_)
        if self.output == "pandas":
            if sparse.issparse(combined):
                return pd.DataFrame(combined.toarray(), columns=self.feature_names_out_, index=frame.index)
            dataframe = pd.DataFrame(combined, columns=self.feature_names_out_)
            dataframe.index = frame.index
            return dataframe
        if self.output == "sparse":
            if sparse.issparse(combined):
                return combined.tocsr()
            return sparse.csr_matrix(combined)
        if sparse.issparse(combined):
            return combined.toarray()
        return np.asarray(combined, dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit all blocks, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return combined fitted feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def _combine_outputs(outputs: list[object], feature_names: np.ndarray) -> np.ndarray | sparse.csr_matrix:
    if len(outputs) == 0:
        raise ValueError("FeatureExtractor requires at least one block")
    first_matrix = _as_matrix(outputs[0])
    matrices: list[np.ndarray | sparse.csr_matrix] = [first_matrix]
    any_sparse = sparse.issparse(first_matrix)
    row_count = first_matrix.shape[0]
    for output in outputs[1:]:
        matrix = _as_matrix(output)
        if sparse.issparse(matrix):
            any_sparse = True
            current_rows = matrix.shape[0]
        else:
            current_rows = matrix.shape[0]
        if row_count != current_rows:
            raise ValueError(f"Feature block changed row count: expected {row_count}, got {current_rows}")
        matrices.append(matrix)
    if any_sparse:
        sparse_matrices = [_to_sparse(matrix) for matrix in matrices]
        combined_sparse = sparse.hstack(sparse_matrices, format="csr")
        if combined_sparse.shape[1] != len(feature_names):
            raise ValueError("Combined sparse output width does not match feature names")
        return sparse.csr_matrix(combined_sparse)
    arrays = [np.asarray(matrix, dtype=float) for matrix in matrices]
    combined = np.hstack(arrays)
    if combined.shape[1] != len(feature_names):
        raise ValueError("Combined dense output width does not match feature names")
    return combined


def _collect_sidecars(blocks: tuple[FeatureBlockProtocol, ...]) -> tuple[FeatureBlockSidecars, ...]:
    sidecar_blocks: list[FeatureBlockSidecars] = []
    for block_index, block in enumerate(blocks):
        if hasattr(block, "last_sidecars_"):
            sidecar_block = cast(SidecarBlockProtocol, block)
            sidecar_blocks.append(
                FeatureBlockSidecars(
                    block_index=block_index,
                    block_name=block.__class__.__name__,
                    sidecars=tuple(sidecar_block.last_sidecars_),
                )
            )
    return tuple(sidecar_blocks)


def _as_matrix(output: object) -> np.ndarray | sparse.csr_matrix:
    if sparse.issparse(output):
        sparse_like = cast(SparseLikeProtocol, output)
        return sparse.csr_matrix(np.asarray(sparse_like.toarray(), dtype=float))
    if isinstance(output, pd.DataFrame):
        return output.to_numpy(dtype=float)
    array = np.asarray(output, dtype=float)
    if array.ndim != 2:
        raise ValueError("Feature block output must be two-dimensional")
    return array


def _to_sparse(matrix: np.ndarray | sparse.csr_matrix) -> sparse.csr_matrix:
    if sparse.issparse(matrix):
        sparse_like = cast(SparseLikeProtocol, matrix)
        return sparse.csr_matrix(np.asarray(sparse_like.toarray(), dtype=float))
    return sparse.csr_matrix(matrix)
