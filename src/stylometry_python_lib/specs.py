"""Feature specification metadata."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TopicDependence(StrEnum):
    """Topic-dependence labels required for every feature."""

    MOSTLY_TOPIC_INDEPENDENT = "mostly_topic_independent"
    MIXED = "mixed"
    TOPIC_SENSITIVE = "topic_sensitive"
    TOPIC_CONTROL = "topic_control"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class InputLayer(StrEnum):
    """DocumentView layer consumed by a feature."""

    RAW = "raw"
    NORMALIZED = "normalized"
    TOKENS = "tokens"
    ORTHOGRAPHIC_TOKENS = "orthographic_tokens"
    SENTENCES = "sentences"
    PARAGRAPHS = "paragraphs"
    NLP = "nlp"
    LLM = "llm"
    MULTI = "multi"


class StabilityStatus(StrEnum):
    """Reproducibility class for a feature."""

    DETERMINISTIC = "deterministic"
    PARSER_MODEL_DEPENDENT = "parser_model_dependent"
    STATISTICAL_FIT_DEPENDENT = "statistical_fit_dependent"
    LLM_DEPENDENT = "llm_dependent"


class ResearchBucket(StrEnum):
    """Top-level research bucket from docs/RESEARCH.md."""

    DETERMINISTIC = "deterministic"
    OTHER_NON_LLM = "other_non_llm"
    LLM = "llm"


class FeatureImplementationStatus(StrEnum):
    """Implementation status values allowed by the v2 roadmap."""

    IMPLEMENTED = "implemented"
    PARTIAL = "partial"
    CATALOG_ONLY = "catalog_only"
    BLOCKED_BY_PROVIDER = "blocked_by_provider"
    BLOCKED_BY_EXTERNAL_DECISION = "blocked_by_external_decision"
    OUT_OF_SCOPE = "out_of_scope"


class FeatureOutputMode(StrEnum):
    """Registry-level output shape classification."""

    SCALAR = "scalar"
    DENSE_VECTOR = "dense_vector"
    SPARSE_VECTOR = "sparse_vector"
    PANDAS_DIAGNOSTIC = "pandas_only_diagnostic"
    STRUCTURED_SIDECAR = "structured_sidecar"
    PAIRWISE_MATRIX = "pairwise_matrix"


@dataclass(frozen=True)
class FeatureSpec:
    """Machine-readable contract for one feature or generated feature family."""

    name: str
    family: str
    description: str
    formula_or_rule: str
    input_layer: InputLayer
    topic_dependence: TopicDependence
    text_length_policy: str
    provenance: str
    output_dtype: str
    undefined_behavior: str
    normalization: str
    sparsity: str
    stability_status: StabilityStatus


@dataclass(frozen=True)
class ResearchFeatureEntry:
    """Registry row for one research family from docs/RESEARCH.md."""

    taxonomy_id: str
    bucket: ResearchBucket
    family_id: str
    block_id: str
    output_name_pattern: str
    status: FeatureImplementationStatus
    formula_or_rule: str
    input_layer: InputLayer
    topic_dependence: TopicDependence
    text_length_policy: str
    undefined_behavior: str
    provenance_requirements: str
    output_dtype: str
    output_shape: str
    output_mode: FeatureOutputMode
    dependency_extra: str
    implementation_owner: str
    test_status: str
    documentation_link: str
    emitted_numeric_columns: int
    sidecar_schema: str


def validate_feature_spec(spec: FeatureSpec) -> None:
    """Validate required feature metadata."""
    missing: list[str] = []
    for field_name, field_value in spec.__dict__.items():
        if isinstance(field_value, str) and field_value == "":
            missing.append(field_name)
    if len(missing) > 0:
        joined = ", ".join(missing)
        raise ValueError(f"FeatureSpec {spec.name} has empty required fields: {joined}")
    if spec.topic_dependence == TopicDependence.UNKNOWN:
        raise ValueError(f"FeatureSpec {spec.name} must not use unknown topic dependence in released metadata")


def validate_research_feature_entry(entry: ResearchFeatureEntry) -> None:
    """Validate required research registry metadata."""
    missing: list[str] = []
    for field_name, field_value in entry.__dict__.items():
        if isinstance(field_value, str) and field_value == "":
            missing.append(field_name)
    if len(missing) > 0:
        joined = ", ".join(missing)
        raise ValueError(f"ResearchFeatureEntry {entry.taxonomy_id} has empty required fields: {joined}")
    if entry.topic_dependence == TopicDependence.UNKNOWN:
        raise ValueError(f"ResearchFeatureEntry {entry.taxonomy_id} must not use unknown topic dependence")
    if entry.emitted_numeric_columns < 0:
        raise ValueError(f"ResearchFeatureEntry {entry.taxonomy_id} has negative emitted column count")
