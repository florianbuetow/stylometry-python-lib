"""Direct SVD reducer coverage for the PCA/SVD transform block."""

import pickle

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from stylometry_python_lib.evaluation.transform import PCAReducer


def _matrix() -> np.ndarray:
    return np.array([[0.0, 1.0, 2.0], [2.0, 0.0, 1.0], [1.0, 2.0, 0.0], [3.0, 1.0, 2.0]], dtype=np.float64)


def test_svd_reducer_fits_transforms_and_serializes() -> None:
    reducer = PCAReducer(n_components=2, method="svd", random_state=7)
    reduced = reducer.fit_transform(_matrix(), None)
    assert reduced.shape == (4, 2)
    restored = pickle.loads(pickle.dumps(reducer))
    np.testing.assert_allclose(restored.transform(_matrix()), reduced)


def test_svd_reducer_requires_fit_before_transform() -> None:
    with pytest.raises(NotFittedError):
        PCAReducer(n_components=2, method="svd", random_state=7).transform(_matrix())
