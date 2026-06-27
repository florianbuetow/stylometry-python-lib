from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

class _Explanation:
    values: NDArray[np.float64]

class Explainer:
    def __init__(self, model: Callable[..., Any], masker: NDArray[np.float64]) -> None: ...
    def __call__(self, x: NDArray[np.float64]) -> _Explanation: ...
