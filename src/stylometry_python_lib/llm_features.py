"""Prompt templates, schemas, and validation for LLM feature families."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from stylometry_python_lib.llm import JsonObject, JsonValue, LLMMessage, LLMRequest

type LLMFeatureScope = Literal["row", "pairwise", "vector"]
type LLMOutputKind = Literal["label_confidence", "score_explanation", "descriptor", "vector", "generated_features"]


@dataclass(frozen=True)
class LLMResponseSchema:
    """Feature response schema plus local validation metadata."""

    schema_id: str
    schema: JsonObject
    required_keys: tuple[str, ...]
    allowed_keys: tuple[str, ...]
    label_key: str | None
    labels: tuple[str, ...]
    numeric_keys: tuple[str, ...]
    text_keys: tuple[str, ...]
    vector_key: str | None
    object_keys: tuple[str, ...]


@dataclass(frozen=True)
class LLMPromptTemplate:
    """Versioned prompt template and projection metadata for one LLM feature."""

    feature_name: str
    prompt_version: str
    scope: LLMFeatureScope
    output_kind: LLMOutputKind
    system_prompt: str
    user_template: str
    response_schema: LLMResponseSchema
    numeric_projection: tuple[tuple[str, float], ...]
    numeric_projection_rule: str
    sidecar_schema: str


@dataclass(frozen=True)
class RenderedLLMPrompt:
    """Rendered prompt provenance for a configured LLM request."""

    feature_name: str
    prompt_version: str
    schema_id: str
    prompt_hash: str
    messages: tuple[LLMMessage, ...]
    response_schema: JsonObject

    def to_request(self) -> LLMRequest:
        """Convert rendered prompt provenance into a provider-neutral request."""
        return LLMRequest(
            feature_name=self.feature_name,
            prompt_version=self.prompt_version,
            schema_id=self.schema_id,
            messages=self.messages,
            response_schema=self.response_schema,
        )


@dataclass(frozen=True)
class LLMFeatureMetadata:
    """Discovery metadata for one configured LLM feature family."""

    feature_name: str
    prompt_version: str
    schema_id: str
    scope: LLMFeatureScope
    output_kind: LLMOutputKind
    numeric_projection_rule: str
    sidecar_schema: str


def llm_feature_names() -> tuple[str, ...]:
    """Return all configured LLM feature family names."""
    return tuple(template.feature_name for template in _LLM_PROMPT_TEMPLATES)


def llm_row_feature_names() -> tuple[str, ...]:
    """Return configured row-wise scalar LLM feature family names."""
    return tuple(
        template.feature_name for template in _LLM_PROMPT_TEMPLATES if template.scope == "row" and template.output_kind != "vector"
    )


def llm_pairwise_feature_names() -> tuple[str, ...]:
    """Return configured pairwise LLM feature family names."""
    return tuple(template.feature_name for template in _LLM_PROMPT_TEMPLATES if template.scope == "pairwise")


def llm_vector_feature_names() -> tuple[str, ...]:
    """Return configured vector-output LLM feature family names."""
    return tuple(template.feature_name for template in _LLM_PROMPT_TEMPLATES if template.output_kind == "vector")


def llm_prompt_templates() -> tuple[LLMPromptTemplate, ...]:
    """Return prompt templates for all LLM feature families."""
    return _LLM_PROMPT_TEMPLATES


def llm_prompt_template(feature_name: str) -> LLMPromptTemplate:
    """Return the prompt template for a feature family."""
    for template in _LLM_PROMPT_TEMPLATES:
        if template.feature_name == feature_name:
            return template
    raise ValueError(f"Unknown LLM feature family: {feature_name}")


def llm_feature_metadata() -> tuple[LLMFeatureMetadata, ...]:
    """Return prompt/schema/projection metadata for all LLM feature families."""
    return tuple(
        LLMFeatureMetadata(
            feature_name=template.feature_name,
            prompt_version=template.prompt_version,
            schema_id=template.response_schema.schema_id,
            scope=template.scope,
            output_kind=template.output_kind,
            numeric_projection_rule=template.numeric_projection_rule,
            sidecar_schema=template.sidecar_schema,
        )
        for template in _LLM_PROMPT_TEMPLATES
    )


def render_llm_prompt(feature_name: str, document_id: str, text: str) -> RenderedLLMPrompt:
    """Render a row-wise prompt with deterministic provenance."""
    template = llm_prompt_template(feature_name)
    if template.scope == "pairwise":
        raise ValueError(f"LLM feature requires pairwise rendering: {feature_name}")
    _require_non_empty("document_id", document_id)
    _require_non_empty("text", text)
    user_prompt = template.user_template.format(document_id=document_id, text=text)
    return _render_template(template, user_prompt)


def render_pairwise_llm_prompt(
    feature_name: str,
    pair_id: str,
    document_id_a: str,
    text_a: str,
    document_id_b: str,
    text_b: str,
) -> RenderedLLMPrompt:
    """Render a pairwise prompt with deterministic pair provenance."""
    template = llm_prompt_template(feature_name)
    if template.scope != "pairwise":
        raise ValueError(f"LLM feature is not pairwise: {feature_name}")
    _require_non_empty("pair_id", pair_id)
    _require_non_empty("document_id_a", document_id_a)
    _require_non_empty("document_id_b", document_id_b)
    _require_non_empty("text_a", text_a)
    _require_non_empty("text_b", text_b)
    user_prompt = template.user_template.format(
        pair_id=pair_id,
        document_id_a=document_id_a,
        text_a=text_a,
        document_id_b=document_id_b,
        text_b=text_b,
    )
    return _render_template(template, user_prompt)


def validate_llm_schema_payload(feature_name: str, payload: Mapping[str, JsonValue]) -> None:
    """Validate a parsed provider payload against the feature schema contract."""
    schema = llm_prompt_template(feature_name).response_schema
    _validate_required_and_allowed_keys(feature_name, payload, schema)
    if schema.label_key is not None:
        _validate_label(feature_name, payload, schema)
    for numeric_key in schema.numeric_keys:
        _validate_numeric_key(feature_name, payload, numeric_key)
    for text_key in schema.text_keys:
        _validate_text_key(feature_name, payload, text_key)
    if schema.vector_key is not None:
        _validate_vector_key(feature_name, payload, schema.vector_key)
    for object_key in schema.object_keys:
        _validate_object_key(feature_name, payload, object_key)


def project_llm_label(feature_name: str, label: str) -> float:
    """Project a closed label into the feature's stable numeric value."""
    template = llm_prompt_template(feature_name)
    projection = dict(template.numeric_projection)
    if label not in projection:
        raise ValueError(f"Unknown label for {feature_name}: {label}")
    return projection[label]


def _render_template(template: LLMPromptTemplate, user_prompt: str) -> RenderedLLMPrompt:
    messages = (LLMMessage(role="system", content=template.system_prompt), LLMMessage(role="user", content=user_prompt))
    prompt_hash = _prompt_hash(template, messages)
    return RenderedLLMPrompt(
        feature_name=template.feature_name,
        prompt_version=template.prompt_version,
        schema_id=template.response_schema.schema_id,
        prompt_hash=prompt_hash,
        messages=messages,
        response_schema=template.response_schema.schema,
    )


def _prompt_hash(template: LLMPromptTemplate, messages: tuple[LLMMessage, ...]) -> str:
    payload = {
        "feature_name": template.feature_name,
        "prompt_version": template.prompt_version,
        "schema_id": template.response_schema.schema_id,
        "messages": [{"role": message.role, "content": message.content} for message in messages],
    }
    rendered = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _validate_required_and_allowed_keys(feature_name: str, payload: Mapping[str, JsonValue], schema: LLMResponseSchema) -> None:
    for key in schema.required_keys:
        if key not in payload:
            raise ValueError(f"{feature_name} response missing required field: {key}")
    allowed = set(schema.allowed_keys)
    for key in payload:
        if key not in allowed:
            raise ValueError(f"{feature_name} response contains unsupported field: {key}")


def _validate_label(feature_name: str, payload: Mapping[str, JsonValue], schema: LLMResponseSchema) -> None:
    if schema.label_key is None:
        raise ValueError(f"{feature_name} schema has no label key")
    label = payload[schema.label_key]
    if not isinstance(label, str) or label == "":
        raise ValueError(f"{feature_name} response field must be a non-empty string: {schema.label_key}")
    if label not in schema.labels:
        raise ValueError(f"{feature_name} response label is not in the closed taxonomy: {label}")


def _validate_numeric_key(feature_name: str, payload: Mapping[str, JsonValue], key: str) -> None:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{feature_name} response field must be finite numeric: {key}")
    if key in {"confidence", "score", "probability"} and (float(value) < 0.0 or float(value) > 1.0):
        raise ValueError(f"{feature_name} response field must be between 0 and 1: {key}")


def _validate_text_key(feature_name: str, payload: Mapping[str, JsonValue], key: str) -> None:
    value = payload[key]
    if not isinstance(value, str) or value.strip() == "":
        raise ValueError(f"{feature_name} response field must be non-empty text: {key}")


def _validate_vector_key(feature_name: str, payload: Mapping[str, JsonValue], key: str) -> None:
    value = payload[key]
    if not isinstance(value, list) or len(value) == 0:
        raise ValueError(f"{feature_name} response vector must be a non-empty array: {key}")
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float) or not math.isfinite(float(item)):
            raise ValueError(f"{feature_name} response vector values must be finite numeric: {key}")


def _validate_object_key(feature_name: str, payload: Mapping[str, JsonValue], key: str) -> None:
    value = payload[key]
    if not isinstance(value, dict) or len(value) == 0:
        raise ValueError(f"{feature_name} response field must be a non-empty object: {key}")
    for object_key, object_value in value.items():
        if object_key == "":
            raise ValueError(f"{feature_name} response object keys must be non-empty strings: {key}")
        if isinstance(object_value, bool) or not isinstance(object_value, int | float) or not math.isfinite(float(object_value)):
            raise ValueError(f"{feature_name} response object values must be finite numeric: {key}")


def _require_non_empty(name: str, value: str) -> None:
    if value.strip() == "":
        raise ValueError(f"{name} must not be empty")


def _label_schema(schema_id: str, labels: tuple[str, ...]) -> LLMResponseSchema:
    properties: JsonObject = {
        "label": {"type": "string", "enum": list(labels)},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    }
    return LLMResponseSchema(
        schema_id=schema_id,
        schema=_object_schema(properties, ("label", "confidence", "evidence")),
        required_keys=("label", "confidence", "evidence"),
        allowed_keys=("label", "confidence", "evidence"),
        label_key="label",
        labels=labels,
        numeric_keys=("confidence",),
        text_keys=("evidence",),
        vector_key=None,
        object_keys=(),
    )


def _score_schema(schema_id: str) -> LLMResponseSchema:
    properties: JsonObject = {
        "score": {"type": "number", "minimum": 0, "maximum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    }
    return LLMResponseSchema(
        schema_id=schema_id,
        schema=_object_schema(properties, ("score", "confidence", "evidence")),
        required_keys=("score", "confidence", "evidence"),
        allowed_keys=("score", "confidence", "evidence"),
        label_key=None,
        labels=(),
        numeric_keys=("score", "confidence"),
        text_keys=("evidence",),
        vector_key=None,
        object_keys=(),
    )


def _descriptor_schema(schema_id: str) -> LLMResponseSchema:
    properties: JsonObject = {
        "summary": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    }
    return LLMResponseSchema(
        schema_id=schema_id,
        schema=_object_schema(properties, ("summary", "confidence", "evidence")),
        required_keys=("summary", "confidence", "evidence"),
        allowed_keys=("summary", "confidence", "evidence"),
        label_key=None,
        labels=(),
        numeric_keys=("confidence",),
        text_keys=("summary", "evidence"),
        vector_key=None,
        object_keys=(),
    )


def _vector_schema(schema_id: str) -> LLMResponseSchema:
    properties: JsonObject = {
        "vector": {"type": "array", "items": {"type": "number"}, "minItems": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    }
    return LLMResponseSchema(
        schema_id=schema_id,
        schema=_object_schema(properties, ("vector", "confidence", "evidence")),
        required_keys=("vector", "confidence", "evidence"),
        allowed_keys=("vector", "confidence", "evidence"),
        label_key=None,
        labels=(),
        numeric_keys=("confidence",),
        text_keys=("evidence",),
        vector_key="vector",
        object_keys=(),
    )


def _generated_features_schema(schema_id: str) -> LLMResponseSchema:
    feature_properties: JsonObject = {
        "sentence_count": {"type": "number"},
        "word_count": {"type": "number"},
        "average_word_length": {"type": "number"},
        "punctuation_density": {"type": "number"},
        "formality_score": {"type": "number", "minimum": 0, "maximum": 1},
    }
    properties: JsonObject = {
        "features": {
            "type": "object",
            "properties": feature_properties,
            "required": list(feature_properties.keys()),
            "additionalProperties": False,
        },
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "evidence": {"type": "string"},
    }
    return LLMResponseSchema(
        schema_id=schema_id,
        schema=_object_schema(properties, ("features", "confidence", "evidence")),
        required_keys=("features", "confidence", "evidence"),
        allowed_keys=("features", "confidence", "evidence"),
        label_key=None,
        labels=(),
        numeric_keys=("confidence",),
        text_keys=("evidence",),
        vector_key=None,
        object_keys=("features",),
    )


def _object_schema(properties: JsonObject, required: tuple[str, ...]) -> JsonObject:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


def _label_projection(labels: tuple[str, ...]) -> tuple[tuple[str, float], ...]:
    return tuple((label, float(index)) for index, label in enumerate(labels))


def _label_template(feature_name: str, prompt_slug: str, labels: tuple[str, ...], instruction: str) -> LLMPromptTemplate:
    return LLMPromptTemplate(
        feature_name=feature_name,
        prompt_version=f"{prompt_slug}_prompt_v1",
        scope="row",
        output_kind="label_confidence",
        system_prompt=_SYSTEM_PROMPT,
        user_template=_ROW_USER_TEMPLATE.format(instruction=instruction, expected="Choose exactly one closed label from the schema enum."),
        response_schema=_label_schema(f"{prompt_slug}_schema_v1", labels),
        numeric_projection=_label_projection(labels),
        numeric_projection_rule="Closed taxonomy labels project to their zero-based enum index in declaration order.",
        sidecar_schema="llm_label_confidence_sidecar_v1",
    )


def _pairwise_label_template(feature_name: str, prompt_slug: str, labels: tuple[str, ...], instruction: str) -> LLMPromptTemplate:
    return LLMPromptTemplate(
        feature_name=feature_name,
        prompt_version=f"{prompt_slug}_prompt_v1",
        scope="pairwise",
        output_kind="label_confidence",
        system_prompt=_SYSTEM_PROMPT,
        user_template=_PAIRWISE_USER_TEMPLATE.format(
            instruction=instruction,
            expected="Choose exactly one closed label from the schema enum.",
        ),
        response_schema=_label_schema(f"{prompt_slug}_schema_v1", labels),
        numeric_projection=_label_projection(labels),
        numeric_projection_rule="Closed taxonomy labels project to their zero-based enum index in declaration order.",
        sidecar_schema="llm_pairwise_label_sidecar_v1",
    )


def _score_template(feature_name: str, prompt_slug: str, scope: LLMFeatureScope, instruction: str) -> LLMPromptTemplate:
    user_template = _PAIRWISE_USER_TEMPLATE if scope == "pairwise" else _ROW_USER_TEMPLATE
    return LLMPromptTemplate(
        feature_name=feature_name,
        prompt_version=f"{prompt_slug}_prompt_v1",
        scope=scope,
        output_kind="score_explanation",
        system_prompt=_SYSTEM_PROMPT,
        user_template=user_template.format(instruction=instruction, expected="Return a calibrated score from 0 to 1."),
        response_schema=_score_schema(f"{prompt_slug}_schema_v1"),
        numeric_projection=(),
        numeric_projection_rule="Score fields are already numeric probabilities or calibrated ratings on [0, 1].",
        sidecar_schema="llm_score_explanation_sidecar_v1",
    )


def _descriptor_template(feature_name: str, prompt_slug: str, scope: LLMFeatureScope, instruction: str) -> LLMPromptTemplate:
    user_template = _PAIRWISE_USER_TEMPLATE if scope == "pairwise" else _ROW_USER_TEMPLATE
    return LLMPromptTemplate(
        feature_name=feature_name,
        prompt_version=f"{prompt_slug}_prompt_v1",
        scope=scope,
        output_kind="descriptor",
        system_prompt=_SYSTEM_PROMPT,
        user_template=user_template.format(instruction=instruction, expected="Return concise structured descriptive text."),
        response_schema=_descriptor_schema(f"{prompt_slug}_schema_v1"),
        numeric_projection=(),
        numeric_projection_rule="Confidence is numeric; descriptive fields stay in the sidecar and are not scalarized.",
        sidecar_schema="llm_descriptor_sidecar_v1",
    )


def _vector_template(feature_name: str, prompt_slug: str, instruction: str) -> LLMPromptTemplate:
    return LLMPromptTemplate(
        feature_name=feature_name,
        prompt_version=f"{prompt_slug}_prompt_v1",
        scope="vector",
        output_kind="vector",
        system_prompt=_SYSTEM_PROMPT,
        user_template=_ROW_USER_TEMPLATE.format(
            instruction=instruction,
            expected="Return a stable numeric vector and short evidence note.",
        ),
        response_schema=_vector_schema(f"{prompt_slug}_schema_v1"),
        numeric_projection=(),
        numeric_projection_rule="Vector elements remain ordered numeric dimensions; width is fixed after fit.",
        sidecar_schema="llm_vector_sidecar_v1",
    )


def _generated_features_template() -> LLMPromptTemplate:
    return LLMPromptTemplate(
        feature_name="text::llm::generated_feature_extraction",
        prompt_version="generated_feature_extraction_prompt_v1",
        scope="row",
        output_kind="generated_features",
        system_prompt=_SYSTEM_PROMPT,
        user_template=_ROW_USER_TEMPLATE.format(
            instruction="Extract explicit numeric style features that are not simple topic labels.",
            expected=(
                "Return features with sentence_count, word_count, average_word_length, "
                "punctuation_density, and formality_score numeric keys."
            ),
        ),
        response_schema=_generated_features_schema("generated_feature_extraction_schema_v1"),
        numeric_projection=(),
        numeric_projection_rule="Generated feature object values are finite numeric scalars keyed by model-proposed feature name.",
        sidecar_schema="llm_generated_features_sidecar_v1",
    )


_SYSTEM_PROMPT = (
    "You are a stylometry annotation component. Return only JSON that conforms to the supplied schema. "
    "Base judgments on style, wording, syntax, discourse, and rhetoric rather than topic facts."
)

_ROW_USER_TEMPLATE = "Task: {instruction}\nExpected output: {expected}\nDocument id: {{document_id}}\nText:\n{{text}}\n"

_PAIRWISE_USER_TEMPLATE = (
    "Task: {instruction}\n"
    "Expected output: {expected}\n"
    "Pair id: {{pair_id}}\n"
    "Document A id: {{document_id_a}}\n"
    "Document A text:\n"
    "{{text_a}}\n"
    "Document B id: {{document_id_b}}\n"
    "Document B text:\n"
    "{{text_b}}\n"
)

_TONE_LABELS = ("formal", "informal", "neutral", "ironic", "emotive", "restrained")
_REGISTER_LABELS = ("casual", "standard", "academic", "journalistic", "technical", "literary")
_PERSPECTIVE_LABELS = ("first_person", "second_person", "third_person_limited", "third_person_omniscient", "mixed")
_INTENT_LABELS = ("inform", "argue", "narrate", "describe", "question", "command", "persuade", "reflect")
_DISCOURSE_LABELS = ("claim", "evidence", "elaboration", "contrast", "transition", "conclusion", "setup")
_RHETORICAL_LABELS = ("definition", "analogy", "exemplification", "cause_effect", "comparison", "concession", "appeal")
_ARGUMENT_LABELS = ("deductive", "inductive", "causal", "comparative", "narrative", "dialogic", "unsupported")
_PAIRWISE_STYLE_LABELS = ("same_style", "similar_style", "mixed_style", "different_style")
_SAME_AUTHOR_LABELS = ("same_author", "different_author", "uncertain")

_LLM_PROMPT_TEMPLATES = (
    _label_template(
        "text::llm::tone",
        "tone",
        _TONE_LABELS,
        "Classify the dominant tone of the document.",
    ),
    _label_template(
        "text::llm::register",
        "register",
        _REGISTER_LABELS,
        "Classify the document's register and formality level.",
    ),
    _descriptor_template(
        "text::llm::persona",
        "persona",
        "row",
        "Summarize the implied speaking persona in style-focused terms.",
    ),
    _label_template(
        "text::llm::narrative_perspective",
        "narrative_perspective",
        _PERSPECTIVE_LABELS,
        "Classify the document's narrative perspective.",
    ),
    _label_template(
        "text::llm::sentence_intent",
        "sentence_intent",
        _INTENT_LABELS,
        "Classify the dominant sentence-level intent pattern.",
    ),
    _label_template(
        "text::llm::discourse_function",
        "discourse_function",
        _DISCOURSE_LABELS,
        "Classify the dominant discourse function.",
    ),
    _label_template(
        "text::llm::rhetorical_structure",
        "rhetorical_structure",
        _RHETORICAL_LABELS,
        "Classify the dominant rhetorical structure.",
    ),
    _label_template(
        "text::llm::argumentation_style",
        "argumentation_style",
        _ARGUMENT_LABELS,
        "Classify the dominant argumentation style.",
    ),
    _score_template(
        "text::llm::cohesion_judgment",
        "cohesion_judgment",
        "row",
        "Judge how cohesive the document's style and discourse flow are.",
    ),
    _score_template(
        "text::llm::style_topic_separation",
        "style_topic_separation",
        "pairwise",
        "Score how separable stylistic similarity is from topical similarity across the pair.",
    ),
    _score_template(
        "text::llm::stylistic_similarity",
        "stylistic_similarity",
        "pairwise",
        "Score the stylistic similarity of the two documents.",
    ),
    _pairwise_label_template(
        "text::llm::pairwise_style_comparison",
        "pairwise_style_comparison",
        _PAIRWISE_STYLE_LABELS,
        "Classify the pair's style relationship.",
    ),
    _descriptor_template(
        "text::llm::style_difference_explanation",
        "style_difference_explanation",
        "pairwise",
        "Explain the most important style differences between the two documents.",
    ),
    _descriptor_template(
        "text::llm::style_transfer_descriptor",
        "style_transfer_descriptor",
        "row",
        "Describe the style-transfer cues needed to imitate this document's style.",
    ),
    _descriptor_template(
        "text::llm::authorial_habit_summary",
        "authorial_habit_summary",
        "row",
        "Summarize recurring authorial habits visible in the document.",
    ),
    _vector_template(
        "text::llm::prompt_derived_vector",
        "prompt_derived_vector",
        "Produce a compact prompt-derived style vector.",
    ),
    _vector_template(
        "text::llm::embedding",
        "embedding",
        "Return or adapt a general embedding vector for stylistic comparison.",
    ),
    _vector_template(
        "text::llm::style_tuned_embedding",
        "style_tuned_embedding",
        "Return a style-tuned embedding vector and provenance evidence.",
    ),
    _pairwise_label_template(
        "text::llm::same_author_prediction",
        "same_author_prediction",
        _SAME_AUTHOR_LABELS,
        "Predict whether the pair was written by the same author based on style.",
    ),
    _generated_features_template(),
)
