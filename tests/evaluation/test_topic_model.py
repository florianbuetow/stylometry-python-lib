"""Explicit LDA/NMF topic-model protocol tests."""

import numpy as np
import pytest

from stylometry_python_lib.evaluation.topic import TopicModelReport, topic_model_report


def _counts() -> np.ndarray:
    return np.array([[5, 0, 1, 0], [4, 1, 0, 0], [0, 0, 5, 4], [0, 1, 4, 5]], dtype=np.float64)


@pytest.mark.parametrize("method", ["lda", "nmf"])
def test_topic_model_report_shapes(method: str) -> None:
    report = topic_model_report(
        counts=_counts(), feature_names=("w0", "w1", "w2", "w3"), n_topics=2, method=method, random_state=0, top_terms=2
    )
    assert isinstance(report, TopicModelReport)
    assert report.method == method
    assert len(report.document_topic_matrix) == 4
    assert len(report.top_terms_by_topic) == 2
    assert all(len(terms) == 2 for terms in report.top_terms_by_topic)


def test_topic_model_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="method"):
        topic_model_report(
            counts=_counts(), feature_names=("w0", "w1", "w2", "w3"), n_topics=2, method="bogus", random_state=0, top_terms=2
        )
