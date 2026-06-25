"""Tabular input helpers for sklearn-style transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_frame(data: object) -> pd.DataFrame:
    """Convert user input to a copied DataFrame."""
    if isinstance(data, pd.DataFrame):
        return data.copy(deep=True)
    if isinstance(data, pd.Series):
        return data.to_frame()
    array = np.asarray(data, dtype=object)
    if array.ndim == 1:
        frame = pd.DataFrame({"text": array})
        return frame
    if array.ndim == 2:
        return pd.DataFrame(array)
    raise ValueError("Input must be a pandas object or one/two-dimensional array-like data")


def text_series(data: object, text_column: str) -> pd.Series:
    """Return the configured text column as strings without mutating input."""
    frame = to_frame(data)
    if text_column not in frame.columns:
        raise ValueError(f"Missing required text column: {text_column}")
    series = frame[text_column]
    if series.isna().any():
        raise ValueError(f"Text column {text_column} contains missing values")
    return series.astype(str)


def validate_output_mode(output: str) -> None:
    """Validate a transformer output mode."""
    allowed = {"pandas", "numpy", "sparse"}
    if output not in allowed:
        raise ValueError(f"Unsupported output mode: {output}")
