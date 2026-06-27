"""Low-dimensional projections for evaluation visualization.

The projections are returned as plain arrays so callers can render them with
whatever plotting tool they prefer; no plotting library is bundled.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from stylometry_python_lib.errors import OptionalDependencyError
from stylometry_python_lib.evaluation.distances import as_float_matrix


def tsne_embedding(features: object, n_components: int, random_state: int, perplexity: float) -> NDArray[np.float64]:
    """Return a t-SNE projection using core scikit-learn."""
    from sklearn.manifold import TSNE

    matrix = as_float_matrix(features)
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    if perplexity <= 0.0:
        raise ValueError("perplexity must be positive")
    model = TSNE(n_components=n_components, random_state=random_state, perplexity=perplexity, init="pca")
    return np.asarray(model.fit_transform(matrix), dtype=np.float64)


def umap_embedding(features: object, n_components: int, random_state: int) -> NDArray[np.float64]:
    """Return a UMAP projection. Requires the evaluation-plotting extra (umap-learn)."""
    try:
        import umap
    except ImportError as exc:
        raise OptionalDependencyError("UMAP projection requires the 'evaluation-plotting' extra (umap-learn)") from exc

    matrix = as_float_matrix(features)
    if n_components < 1:
        raise ValueError("n_components must be >= 1")
    reducer = umap.UMAP(n_components=n_components, random_state=random_state)
    return np.asarray(reducer.fit_transform(matrix), dtype=np.float64)
