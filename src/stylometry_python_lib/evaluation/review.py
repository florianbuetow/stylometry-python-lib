"""Human-review packet helpers for stylometry evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from stylometry_python_lib.evaluation.distances import as_float_matrix


@dataclass(frozen=True)
class HumanReviewItem:
    """One document entry prepared for human stylistic review."""

    document_id: str
    text_excerpt: str
    feature_values: tuple[tuple[str, float], ...]


@dataclass(frozen=True)
class HumanReviewPacket:
    """Structured packet for external human review, without automated validation claims."""

    schema_version: str
    items: tuple[HumanReviewItem, ...]
    feature_names: tuple[str, ...]
    max_text_characters: int
    validation_claim: str


def human_review_packet(
    document_ids: object,
    texts: object,
    features: object,
    feature_names: object,
    max_text_characters: int,
) -> HumanReviewPacket:
    """Build a structured packet for human review of texts and feature values."""
    matrix = as_float_matrix(features)
    _validate_review_matrix(matrix)
    ids = _validate_review_document_ids(document_ids, matrix.shape[0])
    text_tuple = _validate_review_texts(texts, matrix.shape[0])
    names = _validate_review_feature_names(feature_names, matrix.shape[1])
    if max_text_characters <= 0:
        raise ValueError("max_text_characters must be positive")
    items = tuple(
        HumanReviewItem(
            document_id=document_id,
            text_excerpt=text[:max_text_characters],
            feature_values=tuple((feature_name, float(value)) for feature_name, value in zip(names, row.tolist(), strict=True)),
        )
        for document_id, text, row in zip(ids, text_tuple, matrix, strict=True)
    )
    return HumanReviewPacket(
        schema_version="human_review_packet_v1",
        items=items,
        feature_names=names,
        max_text_characters=max_text_characters,
        validation_claim="manual_review_required_no_automated_expert_validation",
    )


def _validate_review_matrix(matrix: np.ndarray) -> None:
    if not np.all(np.isfinite(matrix)):
        raise ValueError("human review feature values must be finite")


def _validate_review_document_ids(document_ids: object, sample_count: int) -> tuple[str, ...]:
    id_array = np.asarray(document_ids, dtype=object)
    if id_array.ndim != 1:
        raise ValueError("document_ids must be one-dimensional")
    ids = tuple(id_array.tolist())
    if len(ids) != sample_count:
        raise ValueError("document_ids length must match feature row count")
    seen: set[str] = set()
    for document_id in ids:
        if not isinstance(document_id, str):
            raise ValueError("document_ids must be strings")
        if len(document_id) == 0:
            raise ValueError("document_ids must not contain empty values")
        if document_id in seen:
            raise ValueError(f"Duplicate document id: {document_id}")
        seen.add(document_id)
    return ids


def _validate_review_texts(texts: object, sample_count: int) -> tuple[str, ...]:
    text_array = np.asarray(texts, dtype=object)
    if text_array.ndim != 1:
        raise ValueError("review texts must be one-dimensional")
    text_tuple = tuple(text_array.tolist())
    if len(text_tuple) != sample_count:
        raise ValueError("review texts length must match feature row count")
    for text in text_tuple:
        if not isinstance(text, str):
            raise ValueError("review texts must be strings")
    return text_tuple


def _validate_review_feature_names(feature_names: object, feature_count: int) -> tuple[str, ...]:
    name_array = np.asarray(feature_names, dtype=object)
    if name_array.ndim != 1:
        raise ValueError("feature_names must be one-dimensional")
    names = tuple(name_array.tolist())
    if len(names) != feature_count:
        raise ValueError("feature_names length must match feature column count")
    seen: set[str] = set()
    for feature_name in names:
        if not isinstance(feature_name, str):
            raise ValueError("feature_names must be strings")
        if len(feature_name) == 0:
            raise ValueError("feature_names must not contain empty values")
        if feature_name in seen:
            raise ValueError(f"Duplicate feature name: {feature_name}")
        seen.add(feature_name)
    return names
