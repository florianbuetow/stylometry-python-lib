"""Tests for LLM vector-output transformers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stylometry_python_lib import (
    LLMVectorFixture,
    configured_llm_vector_feature_names,
    llm_vector_transformer,
)


def test_llm_vector_transformer_preserves_row_identity_width_and_sidecars() -> None:
    feature_name = "text::llm::embedding"
    x = pd.DataFrame({"text": ["first", "second"]}, index=["doc-a", "doc-b"])
    transformer = llm_vector_transformer(
        feature_name=feature_name,
        provider="fake",
        model="fixture-embedding",
        source="fake_embedding_provider",
        text_column="text",
        vectors=(
            LLMVectorFixture(document_id="doc-a", vector=(0.1, 0.2, 0.3)),
            LLMVectorFixture(document_id="doc-b", vector=(0.4, 0.5, 0.6)),
        ),
    )

    result = transformer.fit_transform(x, None)

    np.testing.assert_allclose(result, np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64))
    assert transformer.vector_width_ == 3
    assert tuple(transformer.get_feature_names_out(None).tolist()) == (
        "text::llm::embedding::dim_0000",
        "text::llm::embedding::dim_0001",
        "text::llm::embedding::dim_0002",
    )
    assert transformer.last_sidecars_[0].document_id == "doc-a"
    assert transformer.last_sidecars_[0].feature_name == feature_name
    assert transformer.last_sidecars_[0].vector_width == 3
    assert transformer.last_sidecars_[0].provider == "fake"
    assert transformer.last_sidecars_[0].source == "fake_embedding_provider"
    assert transformer.last_sidecars_[0].sidecar_schema == "llm_vector_sidecar_v1"
    assert transformer.last_sidecars_[0].config_metadata == {
        "source": "fake_embedding_provider",
        "vector_width": 3,
    }
    assert transformer.last_sidecars_[0].preprocessing_settings == {
        "input_kind": "row_vector_fixture",
        "text_column": "text",
        "normalization": "none",
    }


def test_llm_vector_transformer_accepts_user_provided_style_tuned_embeddings() -> None:
    feature_name = "text::llm::style_tuned_embedding"
    x = pd.DataFrame({"text": ["first"]}, index=["doc-a"])
    transformer = llm_vector_transformer(
        feature_name=feature_name,
        provider="user",
        model="external-style-encoder",
        source="user_provided_embedding_matrix",
        text_column="text",
        vectors=(LLMVectorFixture(document_id="doc-a", vector=(1.0, -1.0)),),
    )

    result = transformer.fit_transform(x, None)

    np.testing.assert_allclose(result, np.asarray([[1.0, -1.0]], dtype=np.float64))
    assert transformer.last_sidecars_[0].source == "user_provided_embedding_matrix"


@pytest.mark.parametrize("feature_name", configured_llm_vector_feature_names())
def test_llm_vector_transformer_accepts_each_vector_feature_name(feature_name: str) -> None:
    x = pd.DataFrame({"text": ["first"]}, index=["doc-a"])
    transformer = llm_vector_transformer(
        feature_name=feature_name,
        provider="fake",
        model="fixture-vector-model",
        source="fake_vector_provider",
        text_column="text",
        vectors=(LLMVectorFixture(document_id="doc-a", vector=(0.25, 0.75)),),
    )

    result = transformer.fit_transform(x, None)

    np.testing.assert_allclose(result, np.asarray([[0.25, 0.75]], dtype=np.float64))
    assert transformer.last_sidecars_[0].feature_name == feature_name


def test_llm_vector_transformer_fails_fast_on_shape_identity_and_feature_errors() -> None:
    x = pd.DataFrame({"text": ["first", "second"]}, index=["doc-a", "doc-b"])
    with pytest.raises(ValueError, match="width mismatch"):
        llm_vector_transformer(
            feature_name="text::llm::embedding",
            provider="fake",
            model="fixture-embedding",
            source="fake_embedding_provider",
            text_column="text",
            vectors=(
                LLMVectorFixture(document_id="doc-a", vector=(0.1, 0.2)),
                LLMVectorFixture(document_id="doc-b", vector=(0.3,)),
            ),
        ).fit(x, None)
    with pytest.raises(ValueError, match="Missing LLM vector fixture"):
        llm_vector_transformer(
            feature_name="text::llm::embedding",
            provider="fake",
            model="fixture-embedding",
            source="fake_embedding_provider",
            text_column="text",
            vectors=(LLMVectorFixture(document_id="doc-a", vector=(0.1, 0.2)),),
        ).fit(x, None)
    with pytest.raises(ValueError, match="not vector-output"):
        llm_vector_transformer(
            feature_name="text::llm::tone",
            provider="fake",
            model="fixture-embedding",
            source="fake_embedding_provider",
            text_column="text",
            vectors=(
                LLMVectorFixture(document_id="doc-a", vector=(0.1, 0.2)),
                LLMVectorFixture(document_id="doc-b", vector=(0.3, 0.4)),
            ),
        ).fit(x, None)


def test_configured_llm_vector_feature_names_are_explicit() -> None:
    assert configured_llm_vector_feature_names() == (
        "text::llm::prompt_derived_vector",
        "text::llm::embedding",
        "text::llm::style_tuned_embedding",
    )
