"""Tests for offline fake-provider LLM annotation contracts."""

from __future__ import annotations

import pickle
from typing import cast

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from stylometry_python_lib import (
    FakeLLMAnnotation,
    LLMAnnotationSidecar,
    LLMAnnotationTransformer,
    llm_annotation_feature_names,
    llm_annotation_transformer,
)


def _fake_values(offset: float) -> tuple[tuple[str, float], ...]:
    return tuple((feature_name, float(index) + offset) for index, feature_name in enumerate(llm_annotation_feature_names()))


def _fake_annotations() -> tuple[FakeLLMAnnotation, ...]:
    return (
        FakeLLMAnnotation(
            document_id="doc-a",
            feature_values=_fake_values(0.0),
            structured_response=(("tone", "dry"), ("confidence", "high")),
        ),
        FakeLLMAnnotation(
            document_id="doc-b",
            feature_values=_fake_values(100.0),
            structured_response=(("tone", "formal"), ("confidence", "medium")),
        ),
    )


def test_fake_llm_annotation_provider_has_golden_values_sidecars_and_metadata() -> None:
    x = pd.DataFrame({"text": ["first", "second"]}, index=["doc-a", "doc-b"])
    transformer = llm_annotation_transformer(
        provider="fake",
        model="fixture-judge",
        version="1",
        prompt_version="prompt-v1",
        response_schema="style_annotation_schema_v1",
        fake_annotations=_fake_annotations(),
    )

    result = transformer.fit_transform(x, None)

    assert result.shape == (2, len(llm_annotation_feature_names()))
    assert result[0, 0] == 0.0
    assert result[0, -1] == 19.0
    assert result[1, 0] == 100.0
    assert result[1, -1] == 119.0
    assert tuple(transformer.get_feature_names_out(None).tolist()) == llm_annotation_feature_names()
    sidecar = transformer.last_sidecars_[0]
    assert isinstance(sidecar, LLMAnnotationSidecar)
    assert sidecar.document_id == "doc-a"
    assert sidecar.schema_version == "fake_llm_annotation_v1"
    assert sidecar.prompt_version == "prompt-v1"
    assert sidecar.response_schema == "style_annotation_schema_v1"
    assert sidecar.structured_response == (("tone", "dry"), ("confidence", "high"))
    spec = transformer.registry_.by_name("text::llm::tone")
    assert spec.input_layer.value == "llm"
    assert spec.stability_status.value == "llm_dependent"
    assert "provider=fake" in spec.provenance
    assert "prompt_version=prompt-v1" in spec.provenance


def test_fake_llm_annotation_provider_supports_serialization_and_no_input_mutation() -> None:
    x = pd.DataFrame({"text": ["first", "second"]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = llm_annotation_transformer(
        provider="fake",
        model="fixture-judge",
        version="1",
        prompt_version="prompt-v1",
        response_schema="style_annotation_schema_v1",
        fake_annotations=_fake_annotations(),
    )

    result = transformer.fit_transform(x, None)
    restored_transformer = cast(LLMAnnotationTransformer, pickle.loads(pickle.dumps(transformer)))
    restored = restored_transformer.transform(x)

    pd.testing.assert_frame_equal(x, original)
    np.testing.assert_allclose(result, restored)
    assert restored_transformer.last_sidecars_[1].document_id == "doc-b"


def test_fake_llm_annotation_provider_validates_fixture_contract_and_fit_state() -> None:
    x = pd.DataFrame({"text": ["first", "second"]}, index=["doc-a", "doc-b"])
    not_fitted = llm_annotation_transformer(
        provider="fake",
        model="fixture-judge",
        version="1",
        prompt_version="prompt-v1",
        response_schema="style_annotation_schema_v1",
        fake_annotations=_fake_annotations(),
    )

    with pytest.raises(NotFittedError):
        not_fitted.get_feature_names_out(None)
    with pytest.raises(ValueError, match="Missing fake LLM annotation for row id: doc-b"):
        llm_annotation_transformer(
            provider="fake",
            model="fixture-judge",
            version="1",
            prompt_version="prompt-v1",
            response_schema="style_annotation_schema_v1",
            fake_annotations=_fake_annotations()[:1],
        ).fit(x, None)
    with pytest.raises(ValueError, match="Fake LLM annotation document id has no input row"):
        llm_annotation_transformer(
            provider="fake",
            model="fixture-judge",
            version="1",
            prompt_version="prompt-v1",
            response_schema="style_annotation_schema_v1",
            fake_annotations=(
                *_fake_annotations(),
                FakeLLMAnnotation(
                    document_id="doc-extra",
                    feature_values=_fake_values(200.0),
                    structured_response=(("tone", "extra"),),
                ),
            ),
        ).fit(x, None)
    with pytest.raises(ValueError, match="Fake LLM annotation missing feature value"):
        llm_annotation_transformer(
            provider="fake",
            model="fixture-judge",
            version="1",
            prompt_version="prompt-v1",
            response_schema="style_annotation_schema_v1",
            fake_annotations=(
                FakeLLMAnnotation(
                    document_id="doc-a",
                    feature_values=_fake_values(0.0)[:-1],
                    structured_response=(("tone", "dry"),),
                ),
                _fake_annotations()[1],
            ),
        ).fit(x, None)
    with pytest.raises(ValueError, match="Fake LLM annotations require pandas input"):
        not_fitted.fit(np.asarray([[1.0], [2.0]]), None)
