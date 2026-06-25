"""Tests for configured row-wise LLM annotation transformers."""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import cast

import numpy as np
import pandas as pd
import pytest

from stylometry_python_lib import (
    ConfiguredLLMAnnotationSidecar,
    ConfiguredLLMAnnotationTransformer,
    LLMRequest,
    LLMResponse,
    OptionalDependencyError,
    configured_llm_annotation_transformer,
    configured_llm_row_feature_names,
    llm_annotation_transformer,
)
from stylometry_python_lib.llm import JsonValue
from stylometry_python_lib.llm_features import LLMPromptTemplate, llm_prompt_template


def _llm_request_calls() -> list[LLMRequest]:
    return []


@dataclass
class FakeConfiguredLLMClient:
    responses: Mapping[str, str]
    calls: list[LLMRequest] = field(default_factory=_llm_request_calls)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        content = self.responses[request.feature_name]
        return LLMResponse(
            content=content,
            raw_response={"id": "fixture-response", "choices": []},
            provider="fake-configured",
            model="fixture-model",
            finish_reason="stop",
            usage=None,
            decoding_settings={
                "max_tokens": 64,
                "temperature": 0.0,
            },
        )

    def list_models(self) -> tuple[str, ...]:
        return ("fixture-model",)


def test_configured_llm_annotation_transformer_projects_all_row_features_and_sidecars() -> None:
    feature_names = configured_llm_row_feature_names()
    responses = {feature_name: _sample_content(llm_prompt_template(feature_name)) for feature_name in feature_names}
    client = FakeConfiguredLLMClient(responses=responses)
    x = pd.DataFrame({"text": ["A clipped and formal note.", "A lush and reflective paragraph."]}, index=["doc-a", "doc-b"])
    original = x.copy(deep=True)
    transformer = configured_llm_annotation_transformer(client=client, text_column="text", feature_names=feature_names)

    result = transformer.fit_transform(x, None)

    assert result.shape == (2, len(feature_names))
    assert result[0, 0] == 0.0
    assert result[0, feature_names.index("text::llm::cohesion_judgment")] == 0.42
    pd.testing.assert_frame_equal(x, original)
    assert tuple(transformer.get_feature_names_out(None).tolist()) == feature_names
    assert len(client.calls) == len(feature_names) * 2
    assert len(transformer.last_sidecars_) == len(feature_names) * 2
    first_sidecar = transformer.last_sidecars_[0]
    assert first_sidecar.document_id == "doc-a"
    assert first_sidecar.feature_name == "text::llm::tone"
    assert first_sidecar.validation_status == "valid"
    assert first_sidecar.prompt_version == "tone_prompt_v1"
    assert first_sidecar.schema_id == "tone_schema_v1"
    assert first_sidecar.prompt_hash != ""
    assert first_sidecar.sidecar_schema == "llm_label_confidence_sidecar_v1"
    assert first_sidecar.resolved_model_id == "fixture-model"
    assert first_sidecar.decoding_settings == {"max_tokens": 64, "temperature": 0.0}
    assert first_sidecar.preprocessing_settings == {
        "input_kind": "row_text",
        "text_column": "text",
        "normalization": "none",
    }
    assert first_sidecar.raw_response == {"id": "fixture-response", "choices": []}
    assert first_sidecar.parsed_response is not None


def test_configured_llm_annotation_transformer_supports_pickle_roundtrip() -> None:
    feature_names = ("text::llm::tone", "text::llm::cohesion_judgment")
    responses = {feature_name: _sample_content(llm_prompt_template(feature_name)) for feature_name in feature_names}
    client = FakeConfiguredLLMClient(responses=responses)
    x = pd.DataFrame({"text": ["A clipped and formal note."]}, index=["doc-a"])
    transformer = configured_llm_annotation_transformer(client=client, text_column="text", feature_names=feature_names)

    result = transformer.fit_transform(x, None)
    restored = cast(ConfiguredLLMAnnotationTransformer, pickle.loads(pickle.dumps(transformer)))
    restored_result = restored.transform(x)

    np.testing.assert_allclose(restored_result, result)
    assert restored.last_sidecars_[0].document_id == "doc-a"


def test_legacy_llm_annotation_transformer_delegates_to_configured_client() -> None:
    feature_names = ("text::llm::tone", "text::llm::cohesion_judgment")
    responses = {feature_name: _sample_content(llm_prompt_template(feature_name)) for feature_name in feature_names}
    client = FakeConfiguredLLMClient(responses=responses)
    x = pd.DataFrame({"text": ["A clipped and formal note."]}, index=["doc-a"])
    transformer = llm_annotation_transformer(
        provider="openai-compatible",
        model="fixture-model",
        version="1",
        prompt_version="configured-row-prompts-v1",
        response_schema="configured-row-schemas-v1",
        feature_names=feature_names,
        fake_annotations=None,
        client=client,
        text_column="text",
    )

    result = transformer.fit_transform(x, None)

    assert result.shape == (1, 2)
    assert np.isfinite(result).all()
    assert len(client.calls) == 2
    assert isinstance(transformer.last_sidecars_[0], ConfiguredLLMAnnotationSidecar)
    assert transformer.last_sidecars_[0].provider == "fake-configured"


def test_configured_llm_annotation_transformer_records_invalid_response_diagnostics() -> None:
    client = FakeConfiguredLLMClient(responses={"text::llm::tone": "not-json"})
    x = pd.DataFrame({"text": ["A clipped and formal note."]}, index=["doc-a"])
    transformer = configured_llm_annotation_transformer(client=client, text_column="text", feature_names=("text::llm::tone",))

    result = transformer.fit_transform(x, None)

    assert np.isnan(result[0, 0])
    sidecar = transformer.last_sidecars_[0]
    assert sidecar.validation_status == "invalid"
    assert sidecar.diagnostics[0].reason == "invalid_json"
    assert sidecar.diagnostics[0].feature_name == "text::llm::tone"
    assert sidecar.llm_diagnostics[0].reason == "invalid_json"
    assert sidecar.llm_diagnostics[0].document_id == "doc-a"
    assert sidecar.llm_diagnostics[0].pair_id is None
    assert sidecar.llm_diagnostics[0].provider == "fake-configured"
    assert sidecar.llm_diagnostics[0].model == "fixture-model"
    assert sidecar.llm_diagnostics[0].prompt_version == "tone_prompt_v1"
    assert sidecar.llm_diagnostics[0].schema_id == "tone_schema_v1"
    assert sidecar.raw_response == {"id": "fixture-response", "choices": []}
    assert sidecar.decoding_settings == {"max_tokens": 64, "temperature": 0.0}

    missing_field_client = FakeConfiguredLLMClient(responses={"text::llm::tone": '{"label":"formal","confidence":0.8}'})
    missing_field_transformer = configured_llm_annotation_transformer(
        client=missing_field_client,
        text_column="text",
        feature_names=("text::llm::tone",),
    )
    missing_field_result = missing_field_transformer.fit_transform(x, None)
    assert np.isnan(missing_field_result[0, 0])
    assert missing_field_transformer.last_sidecars_[0].diagnostics[0].reason == "missing_required_field"

    schema_mismatch_client = FakeConfiguredLLMClient(
        responses={"text::llm::tone": '{"label":"unsupported","confidence":0.8,"evidence":"bad label"}'}
    )
    schema_mismatch_transformer = configured_llm_annotation_transformer(
        client=schema_mismatch_client,
        text_column="text",
        feature_names=("text::llm::tone",),
    )
    schema_mismatch_result = schema_mismatch_transformer.fit_transform(x, None)
    assert np.isnan(schema_mismatch_result[0, 0])
    assert schema_mismatch_transformer.last_sidecars_[0].diagnostics[0].reason == "schema_mismatch"


def test_configured_llm_annotation_transformer_fails_fast_on_missing_provider_and_invalid_feature() -> None:
    x = pd.DataFrame({"text": ["A clipped and formal note."]}, index=["doc-a"])
    with pytest.raises(OptionalDependencyError, match="explicit LLM client"):
        configured_llm_annotation_transformer(client=None, text_column="text", feature_names=("text::llm::tone",)).fit(x, None)
    with pytest.raises(ValueError, match="not row-wise scalar"):
        configured_llm_annotation_transformer(
            client=FakeConfiguredLLMClient(responses={}),
            text_column="text",
            feature_names=("text::llm::stylistic_similarity",),
        ).fit(x, None)


def _sample_content(template: LLMPromptTemplate) -> str:
    return json.dumps(_sample_payload(template))


def _sample_payload(template: LLMPromptTemplate) -> dict[str, JsonValue]:
    if template.output_kind == "label_confidence":
        return {"label": template.response_schema.labels[0], "confidence": 0.8, "evidence": "Stable stylistic cue."}
    if template.output_kind == "score_explanation":
        return {"score": 0.42, "confidence": 0.7, "evidence": "Balanced stylistic evidence."}
    if template.output_kind == "descriptor":
        return {"summary": "Compact descriptive summary.", "confidence": 0.7, "evidence": "Observed stylistic evidence."}
    if template.output_kind == "generated_features":
        return {
            "features": {
                "sentence_count": 1,
                "word_count": 8,
                "average_word_length": 4.1,
                "punctuation_density": 0.1,
                "formality_score": 0.8,
            },
            "confidence": 0.7,
            "evidence": "Generated cues.",
        }
    raise AssertionError(f"Unhandled row-wise output kind: {template.output_kind}")
