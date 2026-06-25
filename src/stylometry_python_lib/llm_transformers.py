"""Configured LLM transformers for optional stylometry feature extraction."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal, Self

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from stylometry_python_lib.errors import OptionalDependencyError
from stylometry_python_lib.llm import JsonObject, JsonValue, LLMClientProtocol, LLMDiagnosticReason, LLMProviderError
from stylometry_python_lib.llm_features import (
    LLMPromptTemplate,
    RenderedLLMPrompt,
    llm_pairwise_feature_names,
    llm_prompt_template,
    llm_row_feature_names,
    llm_vector_feature_names,
    project_llm_label,
    render_llm_prompt,
    render_pairwise_llm_prompt,
    validate_llm_schema_payload,
)
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus

type LLMValidationStatus = Literal["valid", "invalid"]


@dataclass(frozen=True)
class LLMFeatureDiagnostic:
    """LLM-scoped diagnostic metadata for configured-provider failures."""

    feature_name: str
    status: FeatureStatus
    reason: str
    warnings: tuple[str, ...]
    document_id: str | None
    pair_id: str | None
    provider: str
    model: str
    prompt_version: str
    schema_id: str


@dataclass(frozen=True)
class ConfiguredLLMAnnotationSidecar:
    """Structured configured-provider sidecar for one row/feature LLM response."""

    document_id: str
    feature_name: str
    prompt_version: str
    schema_id: str
    prompt_hash: str
    sidecar_schema: str
    validation_status: LLMValidationStatus
    diagnostics: tuple[FeatureDiagnostic, ...]
    llm_diagnostics: tuple[LLMFeatureDiagnostic, ...]
    provider: str
    model: str
    resolved_model_id: str
    finish_reason: str
    decoding_settings: JsonObject
    preprocessing_settings: JsonObject
    raw_response: JsonObject | None
    parsed_response: JsonObject | None


@dataclass(frozen=True)
class LLMPair:
    """Explicit pair input for configured pairwise LLM features."""

    pair_id: str
    document_id_a: str
    text_a: str
    document_id_b: str
    text_b: str


@dataclass(frozen=True)
class ConfiguredLLMPairwiseSidecar:
    """Structured configured-provider sidecar for one pair/feature LLM response."""

    pair_id: str
    document_id_a: str
    document_id_b: str
    feature_name: str
    prompt_version: str
    schema_id: str
    prompt_hash: str
    sidecar_schema: str
    prompt_order: tuple[str, str]
    validation_status: LLMValidationStatus
    diagnostics: tuple[FeatureDiagnostic, ...]
    llm_diagnostics: tuple[LLMFeatureDiagnostic, ...]
    provider: str
    model: str
    resolved_model_id: str
    finish_reason: str
    decoding_settings: JsonObject
    preprocessing_settings: JsonObject
    raw_response: JsonObject | None
    parsed_response: JsonObject | None


@dataclass(frozen=True)
class ConfiguredLLMPairwiseAudit:
    """Reversed-pair audit outputs for configured pairwise LLM features."""

    feature_names: tuple[str, ...]
    original_pair_ids: tuple[str, ...]
    reversed_pair_ids: tuple[str, ...]
    original_values: tuple[tuple[float, ...], ...]
    reversed_values: tuple[tuple[float, ...], ...]
    sidecars: tuple[ConfiguredLLMPairwiseSidecar, ...]


@dataclass(frozen=True)
class LLMVectorSidecar:
    """Structured sidecar for user-provided or fake LLM vector features."""

    document_id: str
    feature_name: str
    vector_width: int
    provider: str
    model: str
    source: str
    sidecar_schema: str
    config_metadata: JsonObject
    preprocessing_settings: JsonObject
    validation_status: LLMValidationStatus
    diagnostics: tuple[FeatureDiagnostic, ...]


@dataclass(frozen=True)
class LLMVectorFixture:
    """Explicit row-keyed vector fixture for LLM vector features."""

    document_id: str
    vector: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate fixture identity and vector values."""
        _require_non_empty("document_id", self.document_id)
        _validate_vector(self.document_id, self.vector)


class ConfiguredLLMAnnotationTransformer(BaseEstimator):
    """Row-wise configured-provider LLM annotation transformer."""

    def __init__(self, client: LLMClientProtocol | None, text_column: str, feature_names: tuple[str, ...]) -> None:
        self.client = client
        self.text_column = text_column
        self.feature_names = feature_names

    def fit(self, x: object, y: object) -> Self:
        """Validate configured client and row-wise feature metadata."""
        del y
        if self.client is None:
            raise OptionalDependencyError("Configured LLM annotation features require an explicit LLM client")
        frame = _llm_text_frame(x, self.text_column)
        _validate_feature_names(self.feature_names)
        self.feature_names_out_ = np.asarray(self.feature_names, dtype=object)
        self.n_features_in_ = len(frame.columns)
        self.sidecar_schema_ = "configured_llm_annotation_sidecar_v1"
        return self

    def transform(self, x: object) -> np.ndarray:
        """Execute configured row-wise LLM prompts and return numeric projections."""
        _require_fitted_attribute(self, "feature_names_out_")
        if self.client is None:
            raise OptionalDependencyError("Configured LLM annotation features require an explicit LLM client")
        frame = _llm_text_frame(x, self.text_column)
        rows: list[list[float]] = []
        sidecars: list[ConfiguredLLMAnnotationSidecar] = []
        for document_id, text in _iter_text_rows(frame, self.text_column):
            row_values: list[float] = []
            for feature_name in self.feature_names:
                value, sidecar = _execute_row_feature(self.client, feature_name, document_id, text, self.text_column)
                row_values.append(value)
                sidecars.append(sidecar)
            rows.append(row_values)
        self.last_sidecars_ = tuple(sidecars)
        self.last_diagnostics_ = tuple(diagnostic for sidecar in sidecars for diagnostic in sidecar.diagnostics)
        return np.asarray(rows, dtype=np.float64)

    def fit_transform(self, x: object, y: object) -> np.ndarray:
        """Fit and execute configured row-wise LLM prompts."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable configured LLM feature names."""
        del input_features
        _require_fitted_attribute(self, "feature_names_out_")
        return self.feature_names_out_


def configured_llm_annotation_transformer(
    client: LLMClientProtocol | None,
    text_column: str,
    feature_names: tuple[str, ...],
) -> ConfiguredLLMAnnotationTransformer:
    """Build a row-wise configured LLM annotation transformer."""
    return ConfiguredLLMAnnotationTransformer(client=client, text_column=text_column, feature_names=feature_names)


def configured_llm_row_feature_names() -> tuple[str, ...]:
    """Return row-wise configured LLM annotation feature names."""
    return llm_row_feature_names()


class ConfiguredLLMPairwiseEstimator(BaseEstimator):
    """Configured-provider pairwise LLM estimator."""

    def __init__(self, client: LLMClientProtocol | None, feature_names: tuple[str, ...]) -> None:
        self.client = client
        self.feature_names = feature_names

    def fit(self, x: object, y: object) -> Self:
        """Validate configured client and pairwise feature metadata."""
        del x, y
        if self.client is None:
            raise OptionalDependencyError("Configured pairwise LLM features require an explicit LLM client")
        _validate_pairwise_feature_names(self.feature_names)
        self.feature_names_out_ = np.asarray(self.feature_names, dtype=object)
        self.sidecar_schema_ = "configured_llm_pairwise_sidecar_v1"
        return self

    def transform(self, pairs: tuple[LLMPair, ...]) -> np.ndarray:
        """Execute configured pairwise LLM prompts and return numeric projections."""
        _require_fitted_attribute(self, "feature_names_out_")
        if self.client is None:
            raise OptionalDependencyError("Configured pairwise LLM features require an explicit LLM client")
        _validate_pairs(pairs)
        rows: list[list[float]] = []
        sidecars: list[ConfiguredLLMPairwiseSidecar] = []
        for pair in pairs:
            row_values: list[float] = []
            for feature_name in self.feature_names:
                value, sidecar = _execute_pairwise_feature(self.client, feature_name, pair)
                row_values.append(value)
                sidecars.append(sidecar)
            rows.append(row_values)
        self.last_sidecars_ = tuple(sidecars)
        self.last_diagnostics_ = tuple(diagnostic for sidecar in sidecars for diagnostic in sidecar.diagnostics)
        return np.asarray(rows, dtype=np.float64)

    def fit_transform(self, pairs: tuple[LLMPair, ...], y: object) -> np.ndarray:
        """Fit and execute configured pairwise LLM prompts."""
        return self.fit(pairs, y).transform(pairs)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable configured pairwise LLM feature names."""
        del input_features
        _require_fitted_attribute(self, "feature_names_out_")
        return self.feature_names_out_

    def reversed_pair_audit(self, pairs: tuple[LLMPair, ...]) -> ConfiguredLLMPairwiseAudit:
        """Run original and reversed pair-order prompts and return audit evidence."""
        original_values = self.transform(pairs)
        original_sidecars = self.last_sidecars_
        reversed_pairs = tuple(_reverse_pair(pair) for pair in pairs)
        reversed_values = self.transform(reversed_pairs)
        reversed_sidecars = self.last_sidecars_
        return ConfiguredLLMPairwiseAudit(
            feature_names=self.feature_names,
            original_pair_ids=tuple(pair.pair_id for pair in pairs),
            reversed_pair_ids=tuple(pair.pair_id for pair in reversed_pairs),
            original_values=tuple(tuple(float(value) for value in row) for row in original_values),
            reversed_values=tuple(tuple(float(value) for value in row) for row in reversed_values),
            sidecars=(*original_sidecars, *reversed_sidecars),
        )


def configured_llm_pairwise_estimator(
    client: LLMClientProtocol | None,
    feature_names: tuple[str, ...],
) -> ConfiguredLLMPairwiseEstimator:
    """Build a configured pairwise LLM estimator."""
    return ConfiguredLLMPairwiseEstimator(client=client, feature_names=feature_names)


def configured_llm_pairwise_feature_names() -> tuple[str, ...]:
    """Return pairwise configured LLM feature names."""
    return llm_pairwise_feature_names()


class LLMVectorTransformer(BaseEstimator):
    """Row-wise vector transformer for fake or user-provided LLM vector outputs."""

    def __init__(
        self,
        feature_name: str,
        provider: str,
        model: str,
        source: str,
        text_column: str,
        vectors: tuple[LLMVectorFixture, ...],
    ) -> None:
        self.feature_name = feature_name
        self.provider = provider
        self.model = model
        self.source = source
        self.text_column = text_column
        self.vectors = vectors

    def fit(self, x: object, y: object) -> Self:
        """Validate row identity and lock vector width."""
        del y
        frame = _llm_text_frame(x, self.text_column)
        _validate_vector_feature_name(self.feature_name)
        vector_map = _vector_fixture_map(self.vectors)
        row_ids = tuple(str(index) for index in frame.index)
        for document_id in vector_map:
            if document_id not in row_ids:
                raise ValueError(f"LLM vector fixture document id has no input row: {document_id}")
        for document_id in row_ids:
            if document_id not in vector_map:
                raise ValueError(f"Missing LLM vector fixture for row id: {document_id}")
        first_vector = next(iter(vector_map.values()))
        width = len(first_vector)
        for document_id, vector in vector_map.items():
            if len(vector) != width:
                raise ValueError(f"LLM vector fixture width mismatch for row id: {document_id}")
        self.vector_width_ = width
        self.vectors_ = vector_map
        self.feature_names_out_ = np.asarray(_vector_dimension_names(self.feature_name, width), dtype=object)
        return self

    def transform(self, x: object) -> np.ndarray:
        """Return vectors in input row order."""
        _require_fitted_attribute(self, "feature_names_out_")
        frame = _llm_text_frame(x, self.text_column)
        rows: list[tuple[float, ...]] = []
        sidecars: list[LLMVectorSidecar] = []
        for index in frame.index:
            document_id = str(index)
            if document_id not in self.vectors_:
                raise ValueError(f"Missing LLM vector fixture for row id: {document_id}")
            vector = self.vectors_[document_id]
            if len(vector) != self.vector_width_:
                raise ValueError(f"LLM vector fixture width mismatch for row id: {document_id}")
            rows.append(vector)
            sidecars.append(
                LLMVectorSidecar(
                    document_id=document_id,
                    feature_name=self.feature_name,
                    vector_width=self.vector_width_,
                    provider=self.provider,
                    model=self.model,
                    source=self.source,
                    sidecar_schema=llm_prompt_template(self.feature_name).sidecar_schema,
                    config_metadata={
                        "source": self.source,
                        "vector_width": self.vector_width_,
                    },
                    preprocessing_settings={
                        "input_kind": "row_vector_fixture",
                        "text_column": self.text_column,
                        "normalization": "none",
                    },
                    validation_status="valid",
                    diagnostics=(),
                )
            )
        self.last_sidecars_ = tuple(sidecars)
        return np.asarray(rows, dtype=np.float64)

    def fit_transform(self, x: object, y: object) -> np.ndarray:
        """Fit and return vectors in input row order."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable vector dimension names."""
        del input_features
        _require_fitted_attribute(self, "feature_names_out_")
        return self.feature_names_out_


def llm_vector_transformer(
    feature_name: str,
    provider: str,
    model: str,
    source: str,
    text_column: str,
    vectors: tuple[LLMVectorFixture, ...],
) -> LLMVectorTransformer:
    """Build a row-wise LLM vector transformer for fake or user-provided vectors."""
    return LLMVectorTransformer(
        feature_name=feature_name,
        provider=provider,
        model=model,
        source=source,
        text_column=text_column,
        vectors=vectors,
    )


def configured_llm_vector_feature_names() -> tuple[str, ...]:
    """Return vector-output configured LLM feature names."""
    return llm_vector_feature_names()


def _execute_row_feature(
    client: LLMClientProtocol,
    feature_name: str,
    document_id: str,
    text: str,
    text_column: str,
) -> tuple[float, ConfiguredLLMAnnotationSidecar]:
    template = llm_prompt_template(feature_name)
    rendered = render_llm_prompt(feature_name, document_id=document_id, text=text)
    try:
        response = client.complete(rendered.to_request())
    except LLMProviderError as error:
        return _invalid_value_and_sidecar(
            rendered=rendered,
            document_id=document_id,
            reason=error.diagnostic.reason,
            message=error.diagnostic.message,
            provider=error.diagnostic.provider,
            model=error.diagnostic.model,
            finish_reason="error",
            decoding_settings=_unavailable_decoding_settings(error.diagnostic.reason),
            preprocessing_settings=_row_preprocessing_settings(text_column),
            raw_response=None,
            parsed_response=None,
        )
    parsed_response: JsonObject | None = None
    try:
        parsed = _parse_response_content(feature_name, response.content)
        parsed_response = parsed
        validate_llm_schema_payload(feature_name, parsed)
        sidecar = _valid_sidecar(
            rendered,
            document_id,
            response.provider,
            response.model,
            response.finish_reason,
            response.decoding_settings,
            _row_preprocessing_settings(text_column),
            response.raw_response,
            parsed,
        )
        return _project_response(template, parsed), sidecar
    except json.JSONDecodeError as error:
        return _invalid_value_and_sidecar(
            rendered=rendered,
            document_id=document_id,
            reason=LLMDiagnosticReason.INVALID_JSON,
            message=f"LLM response content is not valid JSON: {error.msg}",
            provider=response.provider,
            model=response.model,
            finish_reason=response.finish_reason,
            decoding_settings=response.decoding_settings,
            preprocessing_settings=_row_preprocessing_settings(text_column),
            raw_response=response.raw_response,
            parsed_response=None,
        )
    except ValueError as error:
        reason = _schema_error_reason(str(error))
        return _invalid_value_and_sidecar(
            rendered=rendered,
            document_id=document_id,
            reason=reason,
            message=str(error),
            provider=response.provider,
            model=response.model,
            finish_reason=response.finish_reason,
            decoding_settings=response.decoding_settings,
            preprocessing_settings=_row_preprocessing_settings(text_column),
            raw_response=response.raw_response,
            parsed_response=parsed_response,
        )


def _execute_pairwise_feature(
    client: LLMClientProtocol,
    feature_name: str,
    pair: LLMPair,
) -> tuple[float, ConfiguredLLMPairwiseSidecar]:
    template = llm_prompt_template(feature_name)
    rendered = render_pairwise_llm_prompt(
        feature_name,
        pair_id=pair.pair_id,
        document_id_a=pair.document_id_a,
        text_a=pair.text_a,
        document_id_b=pair.document_id_b,
        text_b=pair.text_b,
    )
    try:
        response = client.complete(rendered.to_request())
    except LLMProviderError as error:
        return _invalid_pairwise_value_and_sidecar(
            rendered=rendered,
            pair=pair,
            reason=error.diagnostic.reason,
            message=error.diagnostic.message,
            provider=error.diagnostic.provider,
            model=error.diagnostic.model,
            finish_reason="error",
            decoding_settings=_unavailable_decoding_settings(error.diagnostic.reason),
            preprocessing_settings=_pairwise_preprocessing_settings(),
            raw_response=None,
            parsed_response=None,
        )
    parsed_response: JsonObject | None = None
    try:
        parsed = _parse_response_content(feature_name, response.content)
        parsed_response = parsed
        validate_llm_schema_payload(feature_name, parsed)
        sidecar = _valid_pairwise_sidecar(
            rendered,
            pair,
            response.provider,
            response.model,
            response.finish_reason,
            response.decoding_settings,
            _pairwise_preprocessing_settings(),
            response.raw_response,
            parsed,
        )
        return _project_response(template, parsed), sidecar
    except json.JSONDecodeError as error:
        return _invalid_pairwise_value_and_sidecar(
            rendered=rendered,
            pair=pair,
            reason=LLMDiagnosticReason.INVALID_JSON,
            message=f"LLM response content is not valid JSON: {error.msg}",
            provider=response.provider,
            model=response.model,
            finish_reason=response.finish_reason,
            decoding_settings=response.decoding_settings,
            preprocessing_settings=_pairwise_preprocessing_settings(),
            raw_response=response.raw_response,
            parsed_response=None,
        )
    except ValueError as error:
        reason = _schema_error_reason(str(error))
        return _invalid_pairwise_value_and_sidecar(
            rendered=rendered,
            pair=pair,
            reason=reason,
            message=str(error),
            provider=response.provider,
            model=response.model,
            finish_reason=response.finish_reason,
            decoding_settings=response.decoding_settings,
            preprocessing_settings=_pairwise_preprocessing_settings(),
            raw_response=response.raw_response,
            parsed_response=parsed_response,
        )


def _valid_sidecar(
    rendered: RenderedLLMPrompt,
    document_id: str,
    provider: str,
    model: str,
    finish_reason: str,
    decoding_settings: JsonObject,
    preprocessing_settings: JsonObject,
    raw_response: JsonObject,
    parsed_response: JsonObject,
) -> ConfiguredLLMAnnotationSidecar:
    return ConfiguredLLMAnnotationSidecar(
        document_id=document_id,
        feature_name=rendered.feature_name,
        prompt_version=rendered.prompt_version,
        schema_id=rendered.schema_id,
        prompt_hash=rendered.prompt_hash,
        sidecar_schema=llm_prompt_template(rendered.feature_name).sidecar_schema,
        validation_status="valid",
        diagnostics=(),
        llm_diagnostics=(),
        provider=provider,
        model=model,
        resolved_model_id=model,
        finish_reason=finish_reason,
        decoding_settings=decoding_settings,
        preprocessing_settings=preprocessing_settings,
        raw_response=raw_response,
        parsed_response=parsed_response,
    )


def _valid_pairwise_sidecar(
    rendered: RenderedLLMPrompt,
    pair: LLMPair,
    provider: str,
    model: str,
    finish_reason: str,
    decoding_settings: JsonObject,
    preprocessing_settings: JsonObject,
    raw_response: JsonObject,
    parsed_response: JsonObject,
) -> ConfiguredLLMPairwiseSidecar:
    return ConfiguredLLMPairwiseSidecar(
        pair_id=pair.pair_id,
        document_id_a=pair.document_id_a,
        document_id_b=pair.document_id_b,
        feature_name=rendered.feature_name,
        prompt_version=rendered.prompt_version,
        schema_id=rendered.schema_id,
        prompt_hash=rendered.prompt_hash,
        sidecar_schema=llm_prompt_template(rendered.feature_name).sidecar_schema,
        prompt_order=(pair.document_id_a, pair.document_id_b),
        validation_status="valid",
        diagnostics=(),
        llm_diagnostics=(),
        provider=provider,
        model=model,
        resolved_model_id=model,
        finish_reason=finish_reason,
        decoding_settings=decoding_settings,
        preprocessing_settings=preprocessing_settings,
        raw_response=raw_response,
        parsed_response=parsed_response,
    )


def _invalid_value_and_sidecar(
    rendered: RenderedLLMPrompt,
    document_id: str,
    reason: LLMDiagnosticReason,
    message: str,
    provider: str,
    model: str,
    finish_reason: str,
    decoding_settings: JsonObject,
    preprocessing_settings: JsonObject,
    raw_response: JsonObject | None,
    parsed_response: JsonObject | None,
) -> tuple[float, ConfiguredLLMAnnotationSidecar]:
    diagnostic = FeatureDiagnostic(
        feature_name=rendered.feature_name,
        status=FeatureStatus.UNDEFINED,
        reason=reason.value,
        warnings=(message,),
    )
    llm_diagnostic = _llm_feature_diagnostic(
        rendered=rendered,
        reason=reason,
        message=message,
        provider=provider,
        model=model,
        document_id=document_id,
        pair_id=None,
    )
    sidecar = ConfiguredLLMAnnotationSidecar(
        document_id=document_id,
        feature_name=rendered.feature_name,
        prompt_version=rendered.prompt_version,
        schema_id=rendered.schema_id,
        prompt_hash=rendered.prompt_hash,
        sidecar_schema=llm_prompt_template(rendered.feature_name).sidecar_schema,
        validation_status="invalid",
        diagnostics=(diagnostic,),
        llm_diagnostics=(llm_diagnostic,),
        provider=provider,
        model=model,
        resolved_model_id=model,
        finish_reason=finish_reason,
        decoding_settings=decoding_settings,
        preprocessing_settings=preprocessing_settings,
        raw_response=raw_response,
        parsed_response=parsed_response,
    )
    return float("nan"), sidecar


def _invalid_pairwise_value_and_sidecar(
    rendered: RenderedLLMPrompt,
    pair: LLMPair,
    reason: LLMDiagnosticReason,
    message: str,
    provider: str,
    model: str,
    finish_reason: str,
    decoding_settings: JsonObject,
    preprocessing_settings: JsonObject,
    raw_response: JsonObject | None,
    parsed_response: JsonObject | None,
) -> tuple[float, ConfiguredLLMPairwiseSidecar]:
    diagnostic = FeatureDiagnostic(
        feature_name=rendered.feature_name,
        status=FeatureStatus.UNDEFINED,
        reason=reason.value,
        warnings=(message,),
    )
    llm_diagnostic = _llm_feature_diagnostic(
        rendered=rendered,
        reason=reason,
        message=message,
        provider=provider,
        model=model,
        document_id=None,
        pair_id=pair.pair_id,
    )
    sidecar = ConfiguredLLMPairwiseSidecar(
        pair_id=pair.pair_id,
        document_id_a=pair.document_id_a,
        document_id_b=pair.document_id_b,
        feature_name=rendered.feature_name,
        prompt_version=rendered.prompt_version,
        schema_id=rendered.schema_id,
        prompt_hash=rendered.prompt_hash,
        sidecar_schema=llm_prompt_template(rendered.feature_name).sidecar_schema,
        prompt_order=(pair.document_id_a, pair.document_id_b),
        validation_status="invalid",
        diagnostics=(diagnostic,),
        llm_diagnostics=(llm_diagnostic,),
        provider=provider,
        model=model,
        resolved_model_id=model,
        finish_reason=finish_reason,
        decoding_settings=decoding_settings,
        preprocessing_settings=preprocessing_settings,
        raw_response=raw_response,
        parsed_response=parsed_response,
    )
    return float("nan"), sidecar


def _llm_feature_diagnostic(
    rendered: RenderedLLMPrompt,
    reason: LLMDiagnosticReason,
    message: str,
    provider: str,
    model: str,
    document_id: str | None,
    pair_id: str | None,
) -> LLMFeatureDiagnostic:
    return LLMFeatureDiagnostic(
        feature_name=rendered.feature_name,
        status=FeatureStatus.UNDEFINED,
        reason=reason.value,
        warnings=(message,),
        document_id=document_id,
        pair_id=pair_id,
        provider=provider,
        model=model,
        prompt_version=rendered.prompt_version,
        schema_id=rendered.schema_id,
    )


def _row_preprocessing_settings(text_column: str) -> JsonObject:
    return {
        "input_kind": "row_text",
        "text_column": text_column,
        "normalization": "none",
    }


def _pairwise_preprocessing_settings() -> JsonObject:
    return {
        "input_kind": "explicit_pair_text",
        "pair_order": "A_then_B",
        "normalization": "none",
    }


def _unavailable_decoding_settings(reason: LLMDiagnosticReason) -> JsonObject:
    return {
        "status": "unavailable",
        "reason": reason.value,
    }


def _parse_response_content(feature_name: str, content: str) -> JsonObject:
    payload: JsonValue = json.loads(content)
    if not isinstance(payload, dict):
        raise ValueError(f"{feature_name} response content must be a JSON object")
    return payload


def _project_response(template: LLMPromptTemplate, payload: JsonObject) -> float:
    if template.output_kind == "label_confidence":
        label = payload["label"]
        if not isinstance(label, str):
            raise ValueError(f"{template.feature_name} response label must be text")
        return project_llm_label(template.feature_name, label)
    if template.output_kind == "score_explanation":
        return _finite_float(template.feature_name, payload, "score")
    if template.output_kind in {"descriptor", "generated_features"}:
        return _finite_float(template.feature_name, payload, "confidence")
    raise ValueError(f"Unsupported row-wise LLM output kind: {template.output_kind}")


def _finite_float(feature_name: str, payload: JsonObject, key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
        raise ValueError(f"{feature_name} response field must be finite numeric: {key}")
    return float(value)


def _schema_error_reason(message: str) -> LLMDiagnosticReason:
    if "missing required field" in message:
        return LLMDiagnosticReason.MISSING_REQUIRED_FIELD
    return LLMDiagnosticReason.SCHEMA_MISMATCH


def _llm_text_frame(x: object, text_column: str) -> pd.DataFrame:
    if text_column == "":
        raise ValueError("text_column must not be empty")
    if not isinstance(x, pd.DataFrame):
        raise ValueError("Configured LLM annotations require pandas DataFrame input")
    if text_column not in x.columns:
        raise ValueError(f"Configured LLM annotations require text column: {text_column}")
    return x


def _iter_text_rows(frame: pd.DataFrame, text_column: str) -> tuple[tuple[str, str], ...]:
    rows: list[tuple[str, str]] = []
    for index, value in frame[text_column].items():
        if not isinstance(value, str) or value.strip() == "":
            raise ValueError(f"Configured LLM annotations require non-empty text for row id: {index}")
        rows.append((str(index), value))
    return tuple(rows)


def _validate_feature_names(feature_names: tuple[str, ...]) -> None:
    if len(feature_names) == 0:
        raise ValueError("Configured LLM feature_names must not be empty")
    allowed = set(llm_row_feature_names())
    seen: set[str] = set()
    for feature_name in feature_names:
        if feature_name in seen:
            raise ValueError(f"Duplicate configured LLM feature name: {feature_name}")
        if feature_name not in allowed:
            raise ValueError(f"Configured LLM feature is not row-wise scalar: {feature_name}")
        seen.add(feature_name)


def _validate_pairwise_feature_names(feature_names: tuple[str, ...]) -> None:
    if len(feature_names) == 0:
        raise ValueError("Configured pairwise LLM feature_names must not be empty")
    allowed = set(llm_pairwise_feature_names())
    seen: set[str] = set()
    for feature_name in feature_names:
        if feature_name in seen:
            raise ValueError(f"Duplicate configured pairwise LLM feature name: {feature_name}")
        if feature_name not in allowed:
            raise ValueError(f"Configured LLM feature is not pairwise: {feature_name}")
        seen.add(feature_name)


def _validate_pairs(pairs: tuple[LLMPair, ...]) -> None:
    if len(pairs) == 0:
        raise ValueError("Configured pairwise LLM pairs must not be empty")
    seen: set[str] = set()
    for pair in pairs:
        _validate_pair(pair)
        if pair.pair_id in seen:
            raise ValueError(f"Duplicate configured pairwise LLM pair id: {pair.pair_id}")
        seen.add(pair.pair_id)


def _validate_pair(pair: LLMPair) -> None:
    _require_non_empty("pair_id", pair.pair_id)
    _require_non_empty("document_id_a", pair.document_id_a)
    _require_non_empty("document_id_b", pair.document_id_b)
    _require_non_empty("text_a", pair.text_a)
    _require_non_empty("text_b", pair.text_b)


def _reverse_pair(pair: LLMPair) -> LLMPair:
    return LLMPair(
        pair_id=f"{pair.pair_id}::reversed",
        document_id_a=pair.document_id_b,
        text_a=pair.text_b,
        document_id_b=pair.document_id_a,
        text_b=pair.text_a,
    )


def _validate_vector_feature_name(feature_name: str) -> None:
    if feature_name not in set(llm_vector_feature_names()):
        raise ValueError(f"Configured LLM feature is not vector-output: {feature_name}")


def _validate_vector(document_id: str, vector: tuple[float, ...]) -> None:
    if len(vector) == 0:
        raise ValueError(f"LLM vector fixture must not be empty for row id: {document_id}")
    for value in vector:
        if not math.isfinite(value):
            raise ValueError(f"LLM vector fixture value must be finite for row id: {document_id}")


def _vector_fixture_map(vectors: tuple[LLMVectorFixture, ...]) -> dict[str, tuple[float, ...]]:
    if len(vectors) == 0:
        raise ValueError("LLM vector fixtures must not be empty")
    vector_map: dict[str, tuple[float, ...]] = {}
    for fixture in vectors:
        if fixture.document_id in vector_map:
            raise ValueError(f"Duplicate LLM vector fixture document id: {fixture.document_id}")
        vector_map[fixture.document_id] = fixture.vector
    return vector_map


def _vector_dimension_names(feature_name: str, width: int) -> tuple[str, ...]:
    return tuple(f"{feature_name}::dim_{index:04d}" for index in range(width))


def _require_non_empty(name: str, value: str) -> None:
    if value.strip() == "":
        raise ValueError(f"{name} must not be empty")


def _require_fitted_attribute(estimator: object, attribute_name: str) -> None:
    if not hasattr(estimator, attribute_name):
        raise NotFittedError(f"This configured LLM transformer is not fitted yet: missing {attribute_name}")
