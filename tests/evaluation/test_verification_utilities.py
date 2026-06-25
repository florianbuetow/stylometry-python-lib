"""Tests for open-world verification helpers."""

from __future__ import annotations

import numpy as np
import pytest

from stylometry_python_lib.evaluation import (
    VerificationDecision,
    VerificationReport,
    thresholded_distance_verification,
)


def test_thresholded_distance_verification_preserves_pair_identities_and_decisions() -> None:
    features = np.asarray([[1.0, 0.0], [1.1, 0.0], [10.0, 10.0]], dtype=np.float64)
    document_ids = ["doc-a-1", "doc-a-2", "doc-unknown"]
    pairs = (("doc-a-1", "doc-a-2"), ("doc-a-1", "doc-unknown"))

    report = thresholded_distance_verification(
        features=features,
        document_ids=document_ids,
        pairs=pairs,
        threshold=0.25,
        metric="euclidean",
    )
    cosine_report = thresholded_distance_verification(
        features=features,
        document_ids=document_ids,
        pairs=(("doc-a-1", "doc-a-2"),),
        threshold=0.01,
        metric="cosine",
    )

    assert isinstance(report, VerificationReport)
    assert report.document_ids == tuple(document_ids)
    assert report.metric == "euclidean"
    assert report.threshold == 0.25
    assert len(report.decisions) == 2
    assert isinstance(report.decisions[0], VerificationDecision)
    assert report.decisions[0].left_document_id == "doc-a-1"
    assert report.decisions[0].right_document_id == "doc-a-2"
    assert report.decisions[0].accepted_same_author is True
    assert report.decisions[1].right_document_id == "doc-unknown"
    assert report.decisions[1].accepted_same_author is False
    assert cosine_report.decisions[0].metric == "cosine"
    assert cosine_report.decisions[0].accepted_same_author is True


def test_thresholded_distance_verification_validates_ids_pairs_and_config() -> None:
    features = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float64)
    document_ids = ["doc-a", "doc-b", "doc-c"]
    nested_ids = [["doc-a"], ["doc-b"], ["doc-c"]]

    with pytest.raises(ValueError, match="document_ids length must match feature row count"):
        thresholded_distance_verification(features, ["doc-a"], (("doc-a", "doc-b"),), threshold=1.0, metric="euclidean")
    with pytest.raises(ValueError, match="document_ids must be one-dimensional"):
        thresholded_distance_verification(features, nested_ids, (("doc-a", "doc-b"),), threshold=1.0, metric="euclidean")
    with pytest.raises(ValueError, match="Duplicate document id"):
        thresholded_distance_verification(features, ["doc-a", "doc-a", "doc-c"], (("doc-a", "doc-c"),), 1.0, "euclidean")
    with pytest.raises(ValueError, match="verification pairs must not be empty"):
        thresholded_distance_verification(features, document_ids, (), threshold=1.0, metric="euclidean")
    with pytest.raises(ValueError, match="distinct documents"):
        thresholded_distance_verification(features, document_ids, (("doc-a", "doc-a"),), threshold=1.0, metric="euclidean")
    with pytest.raises(ValueError, match="Unknown verification document id"):
        thresholded_distance_verification(features, document_ids, (("doc-a", "missing"),), threshold=1.0, metric="euclidean")
    with pytest.raises(ValueError, match="verification threshold must be non-negative"):
        thresholded_distance_verification(features, document_ids, (("doc-a", "doc-b"),), threshold=-1.0, metric="euclidean")
    with pytest.raises(ValueError, match="Unsupported verification metric"):
        thresholded_distance_verification(features, document_ids, (("doc-a", "doc-b"),), threshold=1.0, metric="manhattan")
    with pytest.raises(ValueError, match="verification features must be finite"):
        thresholded_distance_verification(
            np.asarray([[1.0], [np.inf], [3.0]]),
            document_ids,
            (("doc-a", "doc-b"),),
            threshold=1.0,
            metric="euclidean",
        )
