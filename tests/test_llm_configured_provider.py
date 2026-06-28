"""Configured LM Studio provider smoke tests."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stylometry_python_lib import (
    ConfiguredLLMAnnotationSidecar,
    ConfiguredLLMPairwiseSidecar,
    LLMDiagnosticReason,
    LLMPair,
    LLMUnsupportedCapabilityError,
    configured_llm_annotation_transformer,
    configured_llm_pairwise_estimator,
    configured_llm_pairwise_feature_names,
    configured_llm_row_feature_names,
    llm_annotation_transformer,
    load_llm_config,
    openai_compatible_llm_client,
    render_llm_prompt,
    summarize_llm_prompt_paraphrases,
    summarize_llm_repeated_runs,
    validate_llm_schema_payload,
)
from stylometry_python_lib.llm import JsonObject, JsonValue

pytestmark = pytest.mark.skipif(
    os.environ.get("STYLOMETRY_LIVE_LLM") != "1",
    reason="opt-in live LM Studio test; set STYLOMETRY_LIVE_LLM=1 to run via `just test-live-llm`",
)


def test_configured_lm_studio_endpoint_exposes_configured_model() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if not config_path.exists():
        pytest.skip("llm config missing")

    config = load_llm_config(config_path)
    client = openai_compatible_llm_client(config)

    models = client.list_models()

    assert config.model in models


@pytest.mark.timeout(1800)
def test_configured_lm_studio_row_transformer_generates_valid_sidecars() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if not config_path.exists():
        pytest.skip("llm config missing")

    config = load_llm_config(config_path)
    client = openai_compatible_llm_client(config)
    x = pd.DataFrame(
        {"text": ["Dear colleague, I write to confirm the arrangements in a restrained and formal manner."]},
        index=["live-doc-1"],
    )
    transformer = llm_annotation_transformer(
        provider="openai-compatible",
        model=config.model,
        version="lm-studio-local",
        prompt_version="configured-row-prompts-v1",
        response_schema="configured-row-schemas-v1",
        feature_names=configured_llm_row_feature_names(),
        fake_annotations=None,
        client=client,
        text_column="text",
    )

    with _local_llm_generation_lock():
        result = transformer.fit_transform(x, None)

    assert result.shape == (1, len(configured_llm_row_feature_names()))
    assert np.isfinite(result).all()
    assert len(transformer.last_sidecars_) == len(configured_llm_row_feature_names())
    for sidecar in transformer.last_sidecars_:
        assert isinstance(sidecar, ConfiguredLLMAnnotationSidecar)
        assert sidecar.document_id == "live-doc-1"
        assert sidecar.validation_status == "valid"
        assert sidecar.provider == "openai-compatible"
        assert sidecar.model != ""
        assert sidecar.resolved_model_id == sidecar.model
        assert sidecar.finish_reason == "stop"
        assert sidecar.sidecar_schema != ""
        assert sidecar.decoding_settings["request_timeout_seconds"] == config.request_timeout_seconds
        assert sidecar.preprocessing_settings == {
            "input_kind": "row_text",
            "text_column": "text",
            "normalization": "none",
        }
        assert sidecar.parsed_response is not None
        _assert_completion_usage(sidecar.raw_response)
        _assert_completion_message_contains_model_text(sidecar.raw_response)


@pytest.mark.timeout(1800)
def test_configured_lm_studio_pairwise_estimator_generates_valid_sidecars() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if not config_path.exists():
        pytest.skip("llm config missing")

    config = load_llm_config(config_path)
    client = openai_compatible_llm_client(config)
    pair = LLMPair(
        pair_id="live-pair-1",
        document_id_a="live-doc-a",
        text_a="Dear colleague, I write to confirm the arrangements in a restrained and formal manner.",
        document_id_b="live-doc-b",
        text_b="Hey, just checking that the plans still work; the note is casual and direct.",
    )
    estimator = configured_llm_pairwise_estimator(
        client=client,
        feature_names=configured_llm_pairwise_feature_names(),
    )

    with _local_llm_generation_lock():
        result = estimator.fit_transform((pair,), None)

    assert result.shape == (1, len(configured_llm_pairwise_feature_names()))
    assert np.isfinite(result).all()
    assert len(estimator.last_sidecars_) == len(configured_llm_pairwise_feature_names())
    for sidecar in estimator.last_sidecars_:
        assert isinstance(sidecar, ConfiguredLLMPairwiseSidecar)
        assert sidecar.pair_id == "live-pair-1"
        assert sidecar.prompt_order == ("live-doc-a", "live-doc-b")
        assert sidecar.validation_status == "valid"
        assert sidecar.provider == "openai-compatible"
        assert sidecar.model != ""
        assert sidecar.resolved_model_id == sidecar.model
        assert sidecar.finish_reason == "stop"
        assert sidecar.sidecar_schema != ""
        assert sidecar.decoding_settings["request_timeout_seconds"] == config.request_timeout_seconds
        assert sidecar.preprocessing_settings == {
            "input_kind": "explicit_pair_text",
            "pair_order": "A_then_B",
            "normalization": "none",
        }
        assert sidecar.parsed_response is not None
        _assert_completion_usage(sidecar.raw_response)
        _assert_completion_message_contains_model_text(sidecar.raw_response)


@pytest.mark.timeout(1800)
def test_configured_lm_studio_stability_audits_use_real_responses() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if not config_path.exists():
        pytest.skip("llm config missing")

    config = load_llm_config(config_path)
    client = openai_compatible_llm_client(config)
    repeated_x = pd.DataFrame(
        {"text": ["Dear colleague, I write to confirm the arrangements in a restrained and formal manner."]},
        index=["audit-doc-1"],
    )
    paraphrase_x = pd.DataFrame(
        {"text": ["A formal confirmation note.", "A restrained formal confirmation note."]},
        index=["audit-paraphrase-a", "audit-paraphrase-b"],
    )
    pair = LLMPair(
        pair_id="audit-pair-1",
        document_id_a="audit-doc-a",
        text_a="Dear colleague, I write to confirm the arrangements in a restrained and formal manner.",
        document_id_b="audit-doc-b",
        text_b="Hey, just checking that the plans still work; the note is casual and direct.",
    )
    row_transformer = configured_llm_annotation_transformer(
        client=client,
        text_column="text",
        feature_names=("text::llm::tone",),
    )
    paraphrase_transformer = configured_llm_annotation_transformer(
        client=client,
        text_column="text",
        feature_names=("text::llm::tone",),
    )
    pairwise_estimator = configured_llm_pairwise_estimator(
        client=client,
        feature_names=("text::llm::stylistic_similarity",),
    )

    with _local_llm_generation_lock():
        first_run = row_transformer.fit_transform(repeated_x, None)
        first_sidecar = row_transformer.last_sidecars_[0]
        second_run = row_transformer.transform(repeated_x)
        second_sidecar = row_transformer.last_sidecars_[0]
        paraphrase_values = paraphrase_transformer.fit_transform(paraphrase_x, None)
        reversed_audit = pairwise_estimator.fit((pair,), None).reversed_pair_audit((pair,))

    repeated_audit = summarize_llm_repeated_runs(
        feature_names=("text::llm::tone",),
        run_values=((float(first_run[0, 0]),), (float(second_run[0, 0]),)),
        run_count=2,
        model=config.model,
        prompt_versions=(first_sidecar.prompt_version,),
    )
    paraphrase_prompts = (
        render_llm_prompt("text::llm::tone", document_id="audit-paraphrase-a", text="A formal confirmation note."),
        render_llm_prompt("text::llm::tone", document_id="audit-paraphrase-b", text="A restrained formal confirmation note."),
    )
    paraphrase_audit = summarize_llm_prompt_paraphrases(
        feature_name="text::llm::tone",
        rendered_prompts=paraphrase_prompts,
        values=(float(paraphrase_values[0, 0]), float(paraphrase_values[1, 0])),
        model=config.model,
    )

    assert repeated_audit.run_count == 2
    assert repeated_audit.prompt_versions == (first_sidecar.prompt_version,)
    assert first_sidecar.prompt_hash == second_sidecar.prompt_hash
    assert np.isfinite(np.asarray(repeated_audit.variances, dtype=float)).all()
    assert paraphrase_audit.variant_count == 2
    assert paraphrase_audit.prompt_hashes[0] != paraphrase_audit.prompt_hashes[1]
    assert np.isfinite(paraphrase_audit.variance)
    assert reversed_audit.original_pair_ids == ("audit-pair-1",)
    assert reversed_audit.reversed_pair_ids == ("audit-pair-1::reversed",)
    assert reversed_audit.sidecars[0].prompt_order == ("audit-doc-a", "audit-doc-b")
    assert reversed_audit.sidecars[1].prompt_order == ("audit-doc-b", "audit-doc-a")
    assert np.isfinite(np.asarray(reversed_audit.original_values, dtype=float)).all()
    assert np.isfinite(np.asarray(reversed_audit.reversed_values, dtype=float)).all()


@pytest.mark.timeout(1800)
def test_configured_lm_studio_vector_prompts_generate_valid_payloads() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if not config_path.exists():
        pytest.skip("llm config missing")

    config = load_llm_config(config_path)
    client = openai_compatible_llm_client(config)
    feature_names = ("text::llm::prompt_derived_vector", "text::llm::style_tuned_embedding")

    with _local_llm_generation_lock():
        responses = tuple(
            client.complete(
                render_llm_prompt(
                    feature_name,
                    document_id=f"live-vector-{index}",
                    text="A compact formal confirmation note with restrained syntax.",
                ).to_request()
            )
            for index, feature_name in enumerate(feature_names, start=1)
        )

    for feature_name, response in zip(feature_names, responses, strict=True):
        payload = json.loads(response.content)
        validate_llm_schema_payload(feature_name, payload)
        vector: JsonValue = payload["vector"]
        assert isinstance(vector, list)
        assert len(vector) > 0
        assert np.isfinite(np.asarray(vector, dtype=float)).all()
        assert response.provider == "openai-compatible"
        assert response.model != ""
        assert response.finish_reason == "stop"
        assert response.decoding_settings["request_timeout_seconds"] == config.request_timeout_seconds
        _assert_completion_usage(response.raw_response)
        _assert_completion_message_contains_model_text(response.raw_response)


def test_configured_lm_studio_embedding_endpoint_reports_vectors_or_capability_failure() -> None:
    config_path = Path(__file__).resolve().parents[1] / "config.yaml"
    if not config_path.exists():
        pytest.skip("llm config missing")

    config = load_llm_config(config_path)
    client = openai_compatible_llm_client(config)

    with _local_llm_generation_lock():
        try:
            response = client.embed_texts(("A compact formal note.",))
        except LLMUnsupportedCapabilityError as error:
            assert error.diagnostic.reason == LLMDiagnosticReason.UNSUPPORTED_ENDPOINT_CAPABILITY
            assert error.diagnostic.model == config.model
        else:
            assert len(response.vectors) == 1
            assert len(response.vectors[0]) > 0
            assert np.isfinite(np.asarray(response.vectors[0], dtype=float)).all()


@contextmanager
def _local_llm_generation_lock() -> Generator[None]:
    lock_path = Path(tempfile.gettempdir()) / "stylometry-python-lib-llm-generation.lock"
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _assert_completion_usage(raw_response: JsonObject | None) -> None:
    assert raw_response is not None
    choices: JsonValue = raw_response.get("choices")
    assert isinstance(choices, list)
    assert len(choices) > 0
    usage: JsonValue = raw_response.get("usage")
    assert isinstance(usage, dict)
    total_tokens: JsonValue = usage.get("total_tokens")
    prompt_tokens: JsonValue = usage.get("prompt_tokens")
    completion_tokens: JsonValue = usage.get("completion_tokens")
    assert isinstance(total_tokens, int)
    assert isinstance(prompt_tokens, int)
    assert isinstance(completion_tokens, int)
    assert total_tokens > 0
    assert prompt_tokens > 0
    assert completion_tokens > 0


def _assert_completion_message_contains_model_text(raw_response: JsonObject | None) -> None:
    assert raw_response is not None
    choices: JsonValue = raw_response.get("choices")
    assert isinstance(choices, list)
    assert len(choices) > 0
    first_choice = choices[0]
    assert isinstance(first_choice, dict)
    message = first_choice.get("message")
    assert isinstance(message, dict)
    content = message.get("content")
    reasoning_content = message.get("reasoning_content")
    has_content = isinstance(content, str) and content.strip() != ""
    has_reasoning_content = isinstance(reasoning_content, str) and reasoning_content.strip() != ""
    assert has_content or has_reasoning_content
