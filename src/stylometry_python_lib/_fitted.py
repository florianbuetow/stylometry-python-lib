"""Fitted-state checks for sklearn-compatible estimators."""

from __future__ import annotations

from sklearn.exceptions import NotFittedError


def require_fitted(instance: object, attribute: str) -> None:
    """Raise sklearn's NotFittedError when a fitted attribute is missing."""
    if not hasattr(instance, attribute):
        raise NotFittedError(f"{instance.__class__.__name__} is not fitted; missing {attribute}")
