"""Custom exceptions for stylometry extraction."""

from __future__ import annotations


class OptionalDependencyError(ImportError):
    """Raised when a requested optional feature dependency is unavailable."""
