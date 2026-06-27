"""One-class and calibrated binary open-world verification protocols."""

import numpy as np

from stylometry_python_lib.evaluation.verification import (
    VerificationReport,
    calibrated_binary_verification,
    one_class_verification,
)


def _features() -> tuple[np.ndarray, tuple[str, ...]]:
    x = np.array([[0.0, 0.0], [0.1, 0.0], [0.0, 0.1], [5.0, 5.0]], dtype=np.float64)
    return x, ("d0", "d1", "d2", "d3")


def test_one_class_verification_flags_outlier_pair() -> None:
    x, ids = _features()
    report = one_class_verification(features=x, document_ids=ids, pairs=[("d0", "d3")], nu=0.25, random_state=0)
    assert isinstance(report, VerificationReport)
    assert report.metric == "one_class_svm"
    assert len(report.decisions) == 1


def test_calibrated_binary_verification_returns_decisions() -> None:
    x = np.array(
        [[0.0, 0.0], [0.1, 0.0], [0.2, 0.1], [5.0, 5.0], [5.1, 5.0], [4.9, 5.2]],
        dtype=np.float64,
    )
    ids = ("d0", "d1", "d2", "d3", "d4", "d5")
    report = calibrated_binary_verification(
        train_features=x,
        train_labels=["a", "a", "a", "b", "b", "b"],
        features=x,
        document_ids=ids,
        pairs=[("d0", "d1")],
        classifier="logistic_regression",
        random_state=0,
    )
    assert report.metric == "calibrated_binary"
    assert len(report.decisions) == 1
