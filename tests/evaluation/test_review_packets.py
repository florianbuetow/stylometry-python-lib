"""Tests for human-review packet helpers."""

from __future__ import annotations

import numpy as np
import pytest

from stylometry_python_lib.evaluation import HumanReviewItem, HumanReviewPacket, human_review_packet


def test_human_review_packet_preserves_ids_excerpts_and_feature_values() -> None:
    packet = human_review_packet(
        document_ids=["doc-a", "doc-b"],
        texts=["abcdef", "ghijkl"],
        features=np.asarray([[1.0, 2.5], [3.0, 4.5]], dtype=np.float64),
        feature_names=("style_a", "style_b"),
        max_text_characters=3,
    )

    assert isinstance(packet, HumanReviewPacket)
    assert packet.schema_version == "human_review_packet_v1"
    assert packet.feature_names == ("style_a", "style_b")
    assert packet.max_text_characters == 3
    assert packet.validation_claim == "manual_review_required_no_automated_expert_validation"
    assert len(packet.items) == 2
    assert isinstance(packet.items[0], HumanReviewItem)
    assert packet.items[0].document_id == "doc-a"
    assert packet.items[0].text_excerpt == "abc"
    assert packet.items[0].feature_values == (("style_a", 1.0), ("style_b", 2.5))
    assert packet.items[1].document_id == "doc-b"
    assert packet.items[1].text_excerpt == "ghi"


def test_human_review_packet_validates_alignment_and_names() -> None:
    features = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)

    with pytest.raises(ValueError, match="document_ids length must match feature row count"):
        human_review_packet(["doc-a"], ["text-a", "text-b"], features, ("a", "b"), max_text_characters=5)
    with pytest.raises(ValueError, match="review texts length must match feature row count"):
        human_review_packet(["doc-a", "doc-b"], ["text-a"], features, ("a", "b"), max_text_characters=5)
    with pytest.raises(ValueError, match="feature_names length must match feature column count"):
        human_review_packet(["doc-a", "doc-b"], ["text-a", "text-b"], features, ("a",), max_text_characters=5)
    with pytest.raises(ValueError, match="Duplicate feature name"):
        human_review_packet(["doc-a", "doc-b"], ["text-a", "text-b"], features, ("a", "a"), max_text_characters=5)
    with pytest.raises(ValueError, match="max_text_characters must be positive"):
        human_review_packet(["doc-a", "doc-b"], ["text-a", "text-b"], features, ("a", "b"), max_text_characters=0)
    with pytest.raises(ValueError, match="human review feature values must be finite"):
        human_review_packet(
            ["doc-a", "doc-b"],
            ["text-a", "text-b"],
            np.asarray([[1.0, np.nan], [3.0, 4.0]]),
            ("a", "b"),
            max_text_characters=5,
        )
