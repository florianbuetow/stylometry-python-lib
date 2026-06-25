"""Tests for configured pairwise LLM estimators."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pytest

from stylometry_python_lib import (
    LLMPair,
    LLMRequest,
    LLMResponse,
    OptionalDependencyError,
    configured_llm_pairwise_estimator,
    configured_llm_pairwise_feature_names,
)
from stylometry_python_lib.llm import JsonValue
from stylometry_python_lib.llm_features import LLMPromptTemplate, llm_prompt_template


def _llm_request_calls() -> list[LLMRequest]:
    return []


@dataclass
class FakePairwiseLLMClient:
    responses: Mapping[str, str]
    calls: list[LLMRequest] = field(default_factory=_llm_request_calls)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        content = self.responses[request.feature_name]
        return LLMResponse(
            content=content,
            raw_response={"id": "fixture-pairwise-response", "choices": []},
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


def test_configured_pairwise_estimator_preserves_pair_identity_and_sidecars() -> None:
    feature_names = configured_llm_pairwise_feature_names()
    responses = {feature_name: _sample_content(llm_prompt_template(feature_name)) for feature_name in feature_names}
    client = FakePairwiseLLMClient(responses=responses)
    pairs = (_pair(),)
    estimator = configured_llm_pairwise_estimator(client=client, feature_names=feature_names)

    result = estimator.fit_transform(pairs, None)

    assert result.shape == (1, len(feature_names))
    assert np.isfinite(result).all()
    assert tuple(estimator.get_feature_names_out(None).tolist()) == feature_names
    assert len(client.calls) == len(feature_names)
    assert len(estimator.last_sidecars_) == len(feature_names)
    first_sidecar = estimator.last_sidecars_[0]
    assert first_sidecar.pair_id == "pair-1"
    assert first_sidecar.document_id_a == "doc-a"
    assert first_sidecar.document_id_b == "doc-b"
    assert first_sidecar.prompt_order == ("doc-a", "doc-b")
    assert first_sidecar.validation_status == "valid"
    assert first_sidecar.sidecar_schema == "llm_score_explanation_sidecar_v1"
    assert first_sidecar.resolved_model_id == "fixture-model"
    assert first_sidecar.decoding_settings == {"max_tokens": 64, "temperature": 0.0}
    assert first_sidecar.preprocessing_settings == {
        "input_kind": "explicit_pair_text",
        "pair_order": "A_then_B",
        "normalization": "none",
    }
    assert first_sidecar.raw_response == {"id": "fixture-pairwise-response", "choices": []}


def test_configured_pairwise_estimator_reversed_pair_audit_records_both_directions() -> None:
    feature_names = ("text::llm::stylistic_similarity",)
    responses = {feature_name: _sample_content(llm_prompt_template(feature_name)) for feature_name in feature_names}
    client = FakePairwiseLLMClient(responses=responses)
    estimator = configured_llm_pairwise_estimator(client=client, feature_names=feature_names).fit((_pair(),), None)

    audit = estimator.reversed_pair_audit((_pair(),))

    assert audit.feature_names == feature_names
    assert audit.original_pair_ids == ("pair-1",)
    assert audit.reversed_pair_ids == ("pair-1::reversed",)
    assert audit.sidecars[0].prompt_order == ("doc-a", "doc-b")
    assert audit.sidecars[1].prompt_order == ("doc-b", "doc-a")
    assert audit.original_values == ((0.42,),)
    assert audit.reversed_values == ((0.42,),)


def test_configured_pairwise_estimator_records_invalid_response_diagnostics() -> None:
    client = FakePairwiseLLMClient(responses={"text::llm::stylistic_similarity": "not-json"})
    estimator = configured_llm_pairwise_estimator(client=client, feature_names=("text::llm::stylistic_similarity",))

    result = estimator.fit_transform((_pair(),), None)

    assert np.isnan(result[0, 0])
    sidecar = estimator.last_sidecars_[0]
    assert sidecar.pair_id == "pair-1"
    assert sidecar.validation_status == "invalid"
    assert sidecar.diagnostics[0].reason == "invalid_json"
    assert sidecar.llm_diagnostics[0].reason == "invalid_json"
    assert sidecar.llm_diagnostics[0].document_id is None
    assert sidecar.llm_diagnostics[0].pair_id == "pair-1"
    assert sidecar.llm_diagnostics[0].provider == "fake-configured"
    assert sidecar.llm_diagnostics[0].model == "fixture-model"
    assert sidecar.llm_diagnostics[0].prompt_version == "stylistic_similarity_prompt_v1"
    assert sidecar.llm_diagnostics[0].schema_id == "stylistic_similarity_schema_v1"
    assert sidecar.raw_response == {"id": "fixture-pairwise-response", "choices": []}


def test_configured_pairwise_estimator_fails_fast_on_missing_provider_pair_ids_and_wrong_feature() -> None:
    pairs = (_pair(),)
    with pytest.raises(OptionalDependencyError, match="explicit LLM client"):
        configured_llm_pairwise_estimator(client=None, feature_names=("text::llm::stylistic_similarity",)).fit(pairs, None)
    with pytest.raises(ValueError, match="not pairwise"):
        configured_llm_pairwise_estimator(
            client=FakePairwiseLLMClient(responses={}),
            feature_names=("text::llm::tone",),
        ).fit(pairs, None)
    with pytest.raises(ValueError, match="Duplicate configured pairwise LLM pair id"):
        configured_llm_pairwise_estimator(
            client=FakePairwiseLLMClient(
                responses={"text::llm::stylistic_similarity": _sample_content(llm_prompt_template("text::llm::stylistic_similarity"))}
            ),
            feature_names=("text::llm::stylistic_similarity",),
        ).fit_transform((pairs[0], pairs[0]), None)


def _pair() -> LLMPair:
    return LLMPair(
        pair_id="pair-1",
        document_id_a="doc-a",
        text_a="A clipped formal note.",
        document_id_b="doc-b",
        text_b="An expansive ornate paragraph.",
    )


def _sample_content(template: LLMPromptTemplate) -> str:
    return json.dumps(_sample_payload(template))


def _sample_payload(template: LLMPromptTemplate) -> dict[str, JsonValue]:
    if template.output_kind == "label_confidence":
        return {"label": template.response_schema.labels[0], "confidence": 0.8, "evidence": "Stable pairwise style cue."}
    if template.output_kind == "score_explanation":
        return {"score": 0.42, "confidence": 0.7, "evidence": "Balanced pairwise style evidence."}
    if template.output_kind == "descriptor":
        return {"summary": "Compact pairwise difference summary.", "confidence": 0.7, "evidence": "Observed style evidence."}
    raise AssertionError(f"Unhandled pairwise output kind: {template.output_kind}")
