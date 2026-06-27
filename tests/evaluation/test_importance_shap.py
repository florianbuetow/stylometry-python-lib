"""SHAP feature importance behind the optional evaluation-shap extra."""

import importlib.util

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from stylometry_python_lib.errors import OptionalDependencyError
from stylometry_python_lib.evaluation.importance import shap_importance_report

_HAS_SHAP = importlib.util.find_spec("shap") is not None


@pytest.mark.skipif(_HAS_SHAP, reason="gate test only meaningful when shap is absent")
def test_shap_importance_fails_fast_without_extra() -> None:
    est = LogisticRegression(max_iter=200).fit(np.array([[0.0], [1.0]]), ["a", "b"])
    with pytest.raises(OptionalDependencyError, match="shap"):
        shap_importance_report(estimator=est, features=np.array([[0.0], [1.0]]), feature_names=("f0",))


@pytest.mark.skipif(not _HAS_SHAP, reason="requires evaluation-shap extra")
def test_shap_importance_returns_one_record_per_feature() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 3))
    y = ["a" if row[0] > 0 else "b" for row in x]
    est = LogisticRegression(max_iter=500).fit(x, y)
    report = shap_importance_report(estimator=est, features=x, feature_names=("f0", "f1", "f2"))
    assert len(report.records) == 3
    assert all(record.mean_importance >= 0.0 for record in report.records)
