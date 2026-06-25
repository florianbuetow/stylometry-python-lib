"""Tests for LLM prompt templates, schemas, and projections."""

from __future__ import annotations

import pytest

from stylometry_python_lib import (
    LLMFeatureMetadata,
    LLMPromptTemplate,
    llm_annotation_feature_names,
    llm_feature_metadata,
    llm_feature_names,
    llm_prompt_templates,
    project_llm_label,
    render_llm_prompt,
    render_pairwise_llm_prompt,
    validate_llm_schema_payload,
)
from stylometry_python_lib.llm import JsonValue


def test_every_llm_family_has_prompt_template_schema_and_metadata() -> None:
    templates = llm_prompt_templates()
    metadata = llm_feature_metadata()

    assert llm_feature_names() == llm_annotation_feature_names()
    assert len(templates) == 20
    assert len(metadata) == 20
    assert len({template.feature_name for template in templates}) == 20
    assert len({template.response_schema.schema_id for template in templates}) == 20
    for template in templates:
        assert isinstance(template, LLMPromptTemplate)
        assert template.prompt_version.endswith("_prompt_v1")
        assert template.response_schema.schema_id.endswith("_schema_v1")
        assert template.response_schema.schema["type"] == "object"
        assert template.response_schema.schema["additionalProperties"] is False
        assert template.numeric_projection_rule != ""
        assert template.sidecar_schema != ""
    for item in metadata:
        assert isinstance(item, LLMFeatureMetadata)
        assert item.numeric_projection_rule != ""
        assert item.sidecar_schema != ""


def test_prompt_rendering_is_deterministic_and_records_provenance() -> None:
    first = render_llm_prompt("text::llm::tone", document_id="doc-1", text="This is a restrained sentence.")
    second = render_llm_prompt("text::llm::tone", document_id="doc-1", text="This is a restrained sentence.")
    changed = render_llm_prompt("text::llm::tone", document_id="doc-2", text="This is a loud sentence!")

    assert first == second
    assert first.prompt_hash != changed.prompt_hash
    assert first.prompt_version == "tone_prompt_v1"
    assert first.schema_id == "tone_schema_v1"
    assert first.messages[1].content.count("doc-1") == 1
    request = first.to_request()
    assert request.feature_name == "text::llm::tone"
    assert request.prompt_version == first.prompt_version
    assert request.schema_id == first.schema_id
    assert request.response_schema == first.response_schema


def test_pairwise_prompt_rendering_preserves_pair_and_document_identity() -> None:
    rendered = render_pairwise_llm_prompt(
        "text::llm::stylistic_similarity",
        pair_id="pair-1",
        document_id_a="doc-a",
        text_a="The first document is clipped and direct.",
        document_id_b="doc-b",
        text_b="The second document is expansive and ornate.",
    )

    assert rendered.feature_name == "text::llm::stylistic_similarity"
    assert rendered.prompt_version == "stylistic_similarity_prompt_v1"
    assert "pair-1" in rendered.messages[1].content
    assert "doc-a" in rendered.messages[1].content
    assert "doc-b" in rendered.messages[1].content
    with pytest.raises(ValueError, match="requires pairwise rendering"):
        render_llm_prompt("text::llm::stylistic_similarity", document_id="doc-a", text="Text")


def test_closed_label_schemas_reject_unknown_labels_and_project_stably() -> None:
    validate_llm_schema_payload(
        "text::llm::tone",
        {"label": "formal", "confidence": 0.8, "evidence": "Consistent elevated diction."},
    )

    assert project_llm_label("text::llm::tone", "formal") == 0.0
    assert project_llm_label("text::llm::tone", "restrained") == 5.0
    with pytest.raises(ValueError, match="closed taxonomy"):
        validate_llm_schema_payload(
            "text::llm::tone",
            {"label": "cheerful", "confidence": 0.8, "evidence": "Unsupported label."},
        )
    with pytest.raises(ValueError, match="Unknown label"):
        project_llm_label("text::llm::tone", "cheerful")


def test_all_llm_schema_shapes_accept_valid_samples_and_reject_extra_fields() -> None:
    for template in llm_prompt_templates():
        payload = _sample_payload(template)
        validate_llm_schema_payload(template.feature_name, payload)
        invalid_payload = dict(payload)
        invalid_payload["unexpected"] = "field"
        with pytest.raises(ValueError, match="unsupported field"):
            validate_llm_schema_payload(template.feature_name, invalid_payload)


def _sample_payload(template: LLMPromptTemplate) -> dict[str, JsonValue]:
    if template.output_kind == "label_confidence":
        if len(template.response_schema.labels) == 0:
            raise AssertionError(f"label schema has no labels: {template.feature_name}")
        return {"label": template.response_schema.labels[0], "confidence": 0.8, "evidence": "Stable stylistic cue."}
    if template.output_kind == "score_explanation":
        return {"score": 0.5, "confidence": 0.7, "evidence": "Balanced stylistic evidence."}
    if template.output_kind == "descriptor":
        return {"summary": "Compact descriptive summary.", "confidence": 0.7, "evidence": "Observed stylistic evidence."}
    if template.output_kind == "vector":
        return {"vector": [0.1, -0.2, 0.3], "confidence": 0.7, "evidence": "Vector derived from style cues."}
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
    raise AssertionError(f"Unhandled output kind: {template.output_kind}")
