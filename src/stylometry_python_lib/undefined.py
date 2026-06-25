"""Undefined-result diagnostics for stylometry features."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FeatureStatus(StrEnum):
    """Machine-readable status for a computed feature value."""

    DEFINED = "defined"
    UNDEFINED = "undefined"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class FeatureDiagnostic:
    """Diagnostic metadata explaining a feature computation result."""

    feature_name: str
    status: FeatureStatus
    reason: str
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class FeatureValue:
    """A feature value with explicit undefined/warning status."""

    name: str
    value: float
    status: FeatureStatus
    reason: str
    warnings: tuple[str, ...]


def defined_value(name: str, value: float, warnings: tuple[str, ...]) -> FeatureValue:
    """Create a defined feature value."""
    status = FeatureStatus.DEFINED
    if len(warnings) > 0:
        status = FeatureStatus.WARNING
    return FeatureValue(name=name, value=value, status=status, reason="defined", warnings=warnings)


def undefined_value(name: str, reason: str, warnings: tuple[str, ...]) -> FeatureValue:
    """Create an undefined feature value represented numerically as NaN."""
    return FeatureValue(name=name, value=float("nan"), status=FeatureStatus.UNDEFINED, reason=reason, warnings=warnings)
