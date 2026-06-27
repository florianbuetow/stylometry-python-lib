"""Low-dimensional projection tests (t-SNE core, UMAP optional extra)."""

import importlib.util

import numpy as np
import pytest

from stylometry_python_lib.errors import OptionalDependencyError
from stylometry_python_lib.evaluation.visualization import tsne_embedding, umap_embedding

_HAS_UMAP = importlib.util.find_spec("umap") is not None


def test_tsne_embedding_is_core_and_shaped() -> None:
    rng = np.random.default_rng(0)
    coords = tsne_embedding(features=rng.normal(size=(12, 5)), n_components=2, random_state=0, perplexity=3.0)
    assert coords.shape == (12, 2)


@pytest.mark.skipif(_HAS_UMAP, reason="gate test only meaningful without umap-learn")
def test_umap_fails_fast_without_extra() -> None:
    with pytest.raises(OptionalDependencyError, match="umap"):
        umap_embedding(features=np.zeros((5, 3)), n_components=2, random_state=0)


@pytest.mark.skipif(not _HAS_UMAP, reason="requires evaluation-plotting extra")
def test_umap_embedding_is_shaped() -> None:
    rng = np.random.default_rng(0)
    coords = umap_embedding(features=rng.normal(size=(20, 5)), n_components=2, random_state=0)
    assert coords.shape == (20, 2)
