"""Configured LLM client support for optional stylometry feature layers."""

from __future__ import annotations

import http.client
import json
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Literal, Protocol
from urllib.parse import ParseResult, urlparse

type JsonScalar = str | int | float | bool | None
type JsonValue = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
type JsonObject = dict[str, JsonValue]


class LLMDiagnosticReason(StrEnum):
    """Machine-readable LLM provider diagnostic reasons."""

    INVALID_JSON = "invalid_json"
    SCHEMA_MISMATCH = "schema_mismatch"
    MISSING_REQUIRED_FIELD = "missing_required_field"
    REFUSAL = "refusal"
    TIMEOUT = "timeout"
    PROVIDER_ERROR = "provider_error"
    TRUNCATION = "truncation"
    CONTEXT_WINDOW_OVERFLOW = "context_window_overflow"
    EMPTY_RESPONSE = "empty_response"
    UNSUPPORTED_ENDPOINT_CAPABILITY = "unsupported_endpoint_capability"


@dataclass(frozen=True)
class LLMDiagnostic:
    """Structured diagnostic for configured LLM execution failures."""

    reason: LLMDiagnosticReason
    message: str
    provider: str
    model: str
    endpoint: str


class LLMConfigError(ValueError):
    """Raised when configured LLM settings are missing or invalid."""


class LLMProviderError(RuntimeError):
    """Raised when an LLM provider returns an explicit failure."""

    def __init__(self, diagnostic: LLMDiagnostic) -> None:
        super().__init__(diagnostic.message)
        self.diagnostic = diagnostic


class LLMTimeoutError(LLMProviderError):
    """Raised when an LLM provider request times out."""


class LLMContextWindowError(LLMProviderError):
    """Raised when a rendered request exceeds the configured context budget."""


class LLMResponseFormatError(LLMProviderError):
    """Raised when the provider response cannot be parsed into the expected envelope."""


class LLMUnsupportedCapabilityError(LLMProviderError):
    """Raised when the endpoint does not support a requested LLM capability."""


@dataclass(frozen=True)
class LLMConfig:
    """Strict root-level `config.yaml` LLM configuration."""

    model: str
    api_base: str
    api_key: str
    context_window: int
    max_tokens: int
    temperature: float
    context_window_threshold: int
    max_retries: int
    retry_delay: float
    request_timeout_seconds: float

    def __post_init__(self) -> None:
        """Validate configured values without fallback defaults."""
        _require_non_empty_string("llm.model", self.model)
        _require_non_empty_string("llm.api_base", self.api_base)
        _require_non_empty_string("llm.api_key", self.api_key)
        parsed = urlparse(self.api_base)
        if parsed.scheme not in {"http", "https"} or parsed.netloc == "":
            raise LLMConfigError("llm.api_base must be an absolute http or https URL")
        _require_positive_int("llm.context_window", self.context_window)
        _require_positive_int("llm.max_tokens", self.max_tokens)
        if self.max_tokens > self.context_window:
            raise LLMConfigError("llm.max_tokens must be less than or equal to llm.context_window")
        if not math.isfinite(self.temperature) or self.temperature < 0.0 or self.temperature > 2.0:
            raise LLMConfigError("llm.temperature must be finite and between 0.0 and 2.0")
        if self.context_window_threshold < 1 or self.context_window_threshold > 100:
            raise LLMConfigError("llm.context_window_threshold must be between 1 and 100")
        if self.max_retries < 0:
            raise LLMConfigError("llm.max_retries must be greater than or equal to 0")
        _require_non_negative_float("llm.retry_delay", self.retry_delay)
        _require_positive_float("llm.request_timeout_seconds", self.request_timeout_seconds)

    @property
    def normalized_api_base(self) -> str:
        """Return the configured API base without a trailing slash."""
        return self.api_base.rstrip("/")


@dataclass(frozen=True)
class LLMMessage:
    """Provider-neutral chat message."""

    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self) -> None:
        """Validate message content."""
        _require_non_empty_string("llm.request.message.content", self.content)


@dataclass(frozen=True)
class LLMRequest:
    """Provider-neutral completion request for a rendered feature prompt."""

    feature_name: str
    prompt_version: str
    schema_id: str
    messages: tuple[LLMMessage, ...]
    response_schema: JsonObject | None

    def __post_init__(self) -> None:
        """Validate request metadata."""
        _require_non_empty_string("llm.request.feature_name", self.feature_name)
        _require_non_empty_string("llm.request.prompt_version", self.prompt_version)
        _require_non_empty_string("llm.request.schema_id", self.schema_id)
        if len(self.messages) == 0:
            raise ValueError("llm.request.messages must not be empty")


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral completion response."""

    content: str
    raw_response: JsonObject
    provider: str
    model: str
    finish_reason: str
    usage: JsonObject | None
    decoding_settings: JsonObject


@dataclass(frozen=True)
class LLMEmbeddingResponse:
    """Provider-neutral embedding response."""

    vectors: tuple[tuple[float, ...], ...]
    raw_response: JsonObject
    provider: str
    model: str
    usage: JsonObject | None
    config_metadata: JsonObject


@dataclass(frozen=True)
class HTTPTransportResponse:
    """Raw JSON HTTP transport response."""

    status_code: int
    body: str


class HTTPJSONTransport(Protocol):
    """Transport protocol used by OpenAI-compatible clients."""

    def request_json(
        self,
        method: Literal["GET", "POST"],
        url: str,
        headers: Mapping[str, str],
        payload: JsonObject | None,
        timeout_seconds: float,
    ) -> HTTPTransportResponse:
        """Execute an HTTP request and return the raw JSON body."""
        ...


class LLMClientProtocol(Protocol):
    """Provider-neutral LLM client protocol."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a completion request."""
        ...

    def list_models(self) -> tuple[str, ...]:
        """Return configured endpoint model ids."""
        ...


class LLMEmbeddingClientProtocol(Protocol):
    """Provider-neutral embedding client protocol."""

    def embed_texts(self, texts: tuple[str, ...]) -> LLMEmbeddingResponse:
        """Return provider embedding vectors for input texts."""
        ...


class UrllibJSONTransport:
    """Stdlib JSON transport for OpenAI-compatible HTTP endpoints."""

    def request_json(
        self,
        method: Literal["GET", "POST"],
        url: str,
        headers: Mapping[str, str],
        payload: JsonObject | None,
        timeout_seconds: float,
    ) -> HTTPTransportResponse:
        """Execute a JSON HTTP request."""
        body: bytes | None = None
        if payload is not None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("HTTP transport URL must be an absolute http or https URL")
        connection = _http_connection(parsed, timeout_seconds)
        path = parsed.path if parsed.path != "" else "/"
        if parsed.query != "":
            path = f"{path}?{parsed.query}"
        try:
            connection.request(method=method, url=path, body=body, headers=dict(headers))
            response = connection.getresponse()
            status_code = int(response.status)
            response_body = response.read().decode("utf-8")
        finally:
            connection.close()
        return HTTPTransportResponse(status_code=status_code, body=response_body)


class OpenAICompatibleLLMClient:
    """OpenAI-compatible client for local LM Studio configured endpoints."""

    provider = "openai-compatible"

    def __init__(self, config: LLMConfig, transport: HTTPJSONTransport) -> None:
        self.config = config
        self.transport = transport

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a chat completion request against the configured endpoint."""
        self._check_context_window(request)
        payload = self._completion_payload(request)
        response = self._request_with_retries("POST", f"{self.config.normalized_api_base}/chat/completions", payload)
        return self._parse_completion_response(response)

    def list_models(self) -> tuple[str, ...]:
        """Return model ids exposed by the configured endpoint."""
        response = self._request_with_retries("GET", f"{self.config.normalized_api_base}/models", None)
        invalid_json = self._diagnostic(LLMDiagnosticReason.INVALID_JSON, "Model list response is not valid JSON")
        payload = _loads_json_object(response.body, invalid_json)
        data = payload.get("data")
        if not isinstance(data, list):
            diagnostic = self._diagnostic(LLMDiagnosticReason.MISSING_REQUIRED_FIELD, "Model list response missing data array")
            raise LLMResponseFormatError(diagnostic)
        model_ids: list[str] = []
        for entry in data:
            if not isinstance(entry, dict):
                raise LLMResponseFormatError(self._diagnostic(LLMDiagnosticReason.SCHEMA_MISMATCH, "Model list entry is not an object"))
            model_id = entry.get("id")
            if not isinstance(model_id, str) or model_id == "":
                raise LLMResponseFormatError(self._diagnostic(LLMDiagnosticReason.MISSING_REQUIRED_FIELD, "Model list entry missing id"))
            model_ids.append(model_id)
        return tuple(model_ids)

    def embed_texts(self, texts: tuple[str, ...]) -> LLMEmbeddingResponse:
        """Return embedding vectors from the configured endpoint."""
        if len(texts) == 0:
            raise ValueError("llm.embedding.texts must not be empty")
        for text in texts:
            _require_non_empty_string("llm.embedding.text", text)
        payload: JsonObject = {"model": self.config.model, "input": list(texts)}
        try:
            response = self._request_with_retries("POST", f"{self.config.normalized_api_base}/embeddings", payload)
        except LLMProviderError as error:
            message = f"Configured endpoint does not support embeddings for model {self.config.model}: {error.diagnostic.message}"
            raise LLMUnsupportedCapabilityError(self._diagnostic(LLMDiagnosticReason.UNSUPPORTED_ENDPOINT_CAPABILITY, message)) from error
        return self._parse_embedding_response(response, expected_count=len(texts))

    def _completion_payload(self, request: LLMRequest) -> JsonObject:
        messages: list[JsonValue] = [{"role": message.role, "content": message.content} for message in request.messages]
        payload: JsonObject = {
            "model": self.config.model,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "stream": False,
        }
        if request.response_schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": request.schema_id, "schema": request.response_schema, "strict": True},
            }
        return payload

    def _request_with_retries(
        self,
        method: Literal["GET", "POST"],
        url: str,
        payload: JsonObject | None,
    ) -> HTTPTransportResponse:
        attempts = self.config.max_retries + 1
        last_error: LLMProviderError | None = None
        for attempt_index in range(attempts):
            try:
                response = self.transport.request_json(
                    method=method,
                    url=url,
                    headers=self._headers(),
                    payload=payload,
                    timeout_seconds=self.config.request_timeout_seconds,
                )
                if response.status_code >= 500:
                    diagnostic = self._diagnostic(LLMDiagnosticReason.PROVIDER_ERROR, f"Provider returned HTTP {response.status_code}")
                    raise LLMProviderError(diagnostic)
                if response.status_code >= 400:
                    raise LLMResponseFormatError(
                        self._diagnostic(LLMDiagnosticReason.PROVIDER_ERROR, f"Provider returned non-retriable HTTP {response.status_code}")
                    )
                return response
            except TimeoutError as error:
                last_error = LLMTimeoutError(self._diagnostic(LLMDiagnosticReason.TIMEOUT, f"Provider request timed out: {error}"))
            except (OSError, http.client.HTTPException) as error:
                diagnostic = self._diagnostic(LLMDiagnosticReason.PROVIDER_ERROR, f"Provider request failed: {error}")
                last_error = LLMProviderError(diagnostic)
            except LLMProviderError as error:
                last_error = error
                if isinstance(error, LLMResponseFormatError):
                    raise
            if attempt_index < attempts - 1 and self.config.retry_delay > 0.0:
                time.sleep(self.config.retry_delay)
        if last_error is None:
            raise LLMProviderError(self._diagnostic(LLMDiagnosticReason.PROVIDER_ERROR, "Provider request failed without diagnostic"))
        raise last_error

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _parse_completion_response(self, response: HTTPTransportResponse) -> LLMResponse:
        invalid_json = self._diagnostic(LLMDiagnosticReason.INVALID_JSON, "Completion response is not valid JSON")
        payload = _loads_json_object(response.body, invalid_json)
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) == 0:
            diagnostic = self._diagnostic(LLMDiagnosticReason.MISSING_REQUIRED_FIELD, "Completion response missing choices")
            raise LLMResponseFormatError(diagnostic)
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise LLMResponseFormatError(self._diagnostic(LLMDiagnosticReason.SCHEMA_MISMATCH, "Completion choice is not an object"))
        finish_reason_value = first_choice.get("finish_reason")
        if not isinstance(finish_reason_value, str) or finish_reason_value == "":
            diagnostic = self._diagnostic(LLMDiagnosticReason.MISSING_REQUIRED_FIELD, "Completion choice missing finish_reason")
            raise LLMResponseFormatError(diagnostic)
        if finish_reason_value == "length":
            raise LLMResponseFormatError(self._diagnostic(LLMDiagnosticReason.TRUNCATION, "Completion response was truncated"))
        if finish_reason_value == "content_filter":
            raise LLMResponseFormatError(self._diagnostic(LLMDiagnosticReason.REFUSAL, "Completion response was refused by content filter"))
        message = first_choice.get("message")
        if not isinstance(message, dict):
            raise LLMResponseFormatError(self._diagnostic(LLMDiagnosticReason.MISSING_REQUIRED_FIELD, "Completion choice missing message"))
        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip() != "":
            raise LLMResponseFormatError(self._diagnostic(LLMDiagnosticReason.REFUSAL, refusal))
        content = _completion_message_content(message)
        if content is None:
            raise LLMResponseFormatError(self._diagnostic(LLMDiagnosticReason.EMPTY_RESPONSE, "Completion response content is empty"))
        response_model = payload.get("model")
        model = response_model if isinstance(response_model, str) and response_model != "" else self.config.model
        usage = payload.get("usage")
        usage_payload = usage if isinstance(usage, dict) else None
        return LLMResponse(
            content=content,
            raw_response=payload,
            provider=self.provider,
            model=model,
            finish_reason=finish_reason_value,
            usage=usage_payload,
            decoding_settings=self._decoding_settings(),
        )

    def _parse_embedding_response(self, response: HTTPTransportResponse, expected_count: int) -> LLMEmbeddingResponse:
        invalid_json = self._diagnostic(LLMDiagnosticReason.INVALID_JSON, "Embedding response is not valid JSON")
        payload = _loads_json_object(response.body, invalid_json)
        data = payload.get("data")
        if not isinstance(data, list):
            diagnostic = self._diagnostic(LLMDiagnosticReason.MISSING_REQUIRED_FIELD, "Embedding response missing data array")
            raise LLMResponseFormatError(diagnostic)
        if len(data) != expected_count:
            diagnostic = self._diagnostic(LLMDiagnosticReason.SCHEMA_MISMATCH, "Embedding response count does not match inputs")
            raise LLMResponseFormatError(diagnostic)
        vectors = [_parse_embedding_entry(self.config.model, self.provider, self.config.normalized_api_base, entry) for entry in data]
        response_model = payload.get("model")
        model = response_model if isinstance(response_model, str) and response_model != "" else self.config.model
        usage = payload.get("usage")
        usage_payload = usage if isinstance(usage, dict) else None
        return LLMEmbeddingResponse(
            vectors=tuple(vectors),
            raw_response=payload,
            provider=self.provider,
            model=model,
            usage=usage_payload,
            config_metadata=self._embedding_config_metadata(),
        )

    def _check_context_window(self, request: LLMRequest) -> None:
        estimated_tokens = _estimate_request_tokens(request)
        allowed_tokens = math.floor(self.config.context_window * (self.config.context_window_threshold / 100.0))
        if estimated_tokens > allowed_tokens:
            message = f"Estimated prompt tokens {estimated_tokens} exceed configured context threshold {allowed_tokens}"
            raise LLMContextWindowError(self._diagnostic(LLMDiagnosticReason.CONTEXT_WINDOW_OVERFLOW, message))

    def _diagnostic(self, reason: LLMDiagnosticReason, message: str) -> LLMDiagnostic:
        return LLMDiagnostic(
            reason=reason,
            message=message,
            provider=self.provider,
            model=self.config.model,
            endpoint=self.config.normalized_api_base,
        )

    def _decoding_settings(self) -> JsonObject:
        return {
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "context_window": self.config.context_window,
            "context_window_threshold": self.config.context_window_threshold,
            "max_retries": self.config.max_retries,
            "retry_delay": self.config.retry_delay,
            "request_timeout_seconds": self.config.request_timeout_seconds,
        }

    def _embedding_config_metadata(self) -> JsonObject:
        return {
            "api_base": self.config.normalized_api_base,
            "max_retries": self.config.max_retries,
            "retry_delay": self.config.retry_delay,
            "request_timeout_seconds": self.config.request_timeout_seconds,
        }


def openai_compatible_llm_client(config: LLMConfig) -> OpenAICompatibleLLMClient:
    """Build the stdlib HTTP OpenAI-compatible client."""
    return OpenAICompatibleLLMClient(config=config, transport=UrllibJSONTransport())


def load_llm_config(path: str | Path) -> LLMConfig:
    """Load and validate the root-level LLM config block."""
    config_path = Path(path)
    if not config_path.exists():
        raise LLMConfigError(f"llm config missing: {config_path}")
    payload = _parse_project_config(config_path.read_text(encoding="utf-8"))
    llm_payload = payload.get("llm")
    if not isinstance(llm_payload, dict):
        raise LLMConfigError("config.yaml missing required llm block")
    missing_keys = [key for key in _REQUIRED_LLM_KEYS if key not in llm_payload]
    if len(missing_keys) > 0:
        raise LLMConfigError(f"config.yaml missing required llm key: {missing_keys[0]}")
    extra_keys = [key for key in llm_payload if key not in _REQUIRED_LLM_KEYS]
    if len(extra_keys) > 0:
        raise LLMConfigError(f"config.yaml contains unsupported llm key: {extra_keys[0]}")
    return LLMConfig(
        model=_required_str(llm_payload, "model"),
        api_base=_required_str(llm_payload, "api_base"),
        api_key=_required_str(llm_payload, "api_key"),
        context_window=_required_int(llm_payload, "context_window"),
        max_tokens=_required_int(llm_payload, "max_tokens"),
        temperature=_required_float(llm_payload, "temperature"),
        context_window_threshold=_required_int(llm_payload, "context_window_threshold"),
        max_retries=_required_int(llm_payload, "max_retries"),
        retry_delay=_required_float(llm_payload, "retry_delay"),
        request_timeout_seconds=_required_float(llm_payload, "request_timeout_seconds"),
    )


def _parse_project_config(text: str) -> dict[str, JsonObject]:
    parsed: dict[str, JsonObject] = {}
    active_block: str | None = None
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        active_block = _parse_project_config_line(parsed, active_block, line_number, raw_line)
    return parsed


def _parse_project_config_line(
    parsed: dict[str, JsonObject],
    active_block: str | None,
    line_number: int,
    raw_line: str,
) -> str | None:
    if raw_line.strip() == "" or raw_line.lstrip().startswith("#"):
        return active_block
    if "\t" in raw_line:
        raise LLMConfigError(f"config.yaml line {line_number} contains tabs")
    indent = len(raw_line) - len(raw_line.lstrip(" "))
    stripped = _strip_inline_comment(raw_line.strip())
    if stripped == "":
        return active_block
    if indent == 0:
        return _record_top_level_mapping(parsed, stripped, line_number)
    _record_nested_mapping(parsed, active_block, indent, stripped, line_number)
    return active_block


def _record_top_level_mapping(parsed: dict[str, JsonObject], stripped: str, line_number: int) -> str:
    if not stripped.endswith(":"):
        raise LLMConfigError(f"config.yaml line {line_number} must be a top-level mapping")
    active_block = stripped[:-1]
    if active_block == "":
        raise LLMConfigError(f"config.yaml line {line_number} has an empty mapping key")
    if active_block in parsed:
        raise LLMConfigError(f"config.yaml duplicates top-level key: {active_block}")
    parsed[active_block] = {}
    return active_block


def _record_nested_mapping(
    parsed: dict[str, JsonObject],
    active_block: str | None,
    indent: int,
    stripped: str,
    line_number: int,
) -> None:
    if active_block is None:
        raise LLMConfigError(f"config.yaml line {line_number} has an indented value before a top-level key")
    if indent != 2:
        raise LLMConfigError(f"config.yaml line {line_number} must use two-space indentation")
    key, value = _split_mapping_line(stripped, line_number)
    block = parsed[active_block]
    if key in block:
        raise LLMConfigError(f"config.yaml duplicates key in {active_block}: {key}")
    block[key] = _parse_scalar(value, line_number)


def _split_mapping_line(line: str, line_number: int) -> tuple[str, str]:
    key, separator, value = line.partition(":")
    if separator == "" or key == "" or value.strip() == "":
        raise LLMConfigError(f"config.yaml line {line_number} must be a scalar mapping")
    return key.strip(), value.strip()


def _parse_scalar(value: str, line_number: int) -> JsonScalar:
    if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
        return value[1:-1]
    if value in {"true", "false"}:
        return value == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError as error:
        if value == "":
            raise LLMConfigError(f"config.yaml line {line_number} has an empty scalar value") from error
        return value


def _strip_inline_comment(value: str) -> str:
    in_single_quote = False
    in_double_quote = False
    for index, character in enumerate(value):
        if character == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif character == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
        elif character == "#" and not in_single_quote and not in_double_quote:
            return value[:index].rstrip()
    return value


def _required_str(payload: Mapping[str, JsonValue], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise LLMConfigError(f"llm.{key} must be a string")
    return value


def _required_int(payload: Mapping[str, JsonValue], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise LLMConfigError(f"llm.{key} must be an integer")
    return value


def _required_float(payload: Mapping[str, JsonValue], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise LLMConfigError(f"llm.{key} must be numeric")
    return float(value)


def _require_non_empty_string(name: str, value: str) -> None:
    if value.strip() == "":
        raise LLMConfigError(f"{name} must not be empty")


def _require_positive_int(name: str, value: int) -> None:
    if value <= 0:
        raise LLMConfigError(f"{name} must be greater than 0")


def _require_positive_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise LLMConfigError(f"{name} must be finite and greater than 0")


def _require_non_negative_float(name: str, value: float) -> None:
    if not math.isfinite(value) or value < 0.0:
        raise LLMConfigError(f"{name} must be finite and greater than or equal to 0")


def _estimate_request_tokens(request: LLMRequest) -> int:
    characters = sum(len(message.role) + len(message.content) for message in request.messages)
    characters += len(request.feature_name) + len(request.prompt_version) + len(request.schema_id)
    if request.response_schema is not None:
        characters += len(json.dumps(request.response_schema, sort_keys=True))
    return max(1, math.ceil(characters / 4))


def _loads_json_object(text: str, diagnostic: LLMDiagnostic) -> JsonObject:
    try:
        payload: JsonValue = json.loads(text)
    except json.JSONDecodeError as error:
        raise LLMResponseFormatError(
            LLMDiagnostic(
                reason=diagnostic.reason,
                message=f"{diagnostic.message}: {error.msg}",
                provider=diagnostic.provider,
                model=diagnostic.model,
                endpoint=diagnostic.endpoint,
            )
        ) from error
    if not isinstance(payload, dict):
        raise LLMResponseFormatError(diagnostic)
    return payload


def _completion_message_content(message: JsonObject) -> str | None:
    content = message.get("content")
    if isinstance(content, str) and content.strip() != "":
        return content
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip() != "":
        return reasoning_content
    return None


def _parse_embedding_entry(model: str, provider: str, endpoint: str, entry: JsonValue) -> tuple[float, ...]:
    diagnostic = LLMDiagnostic(
        reason=LLMDiagnosticReason.SCHEMA_MISMATCH,
        message="Embedding response entry is not a valid numeric vector",
        provider=provider,
        model=model,
        endpoint=endpoint,
    )
    if not isinstance(entry, dict):
        raise LLMResponseFormatError(diagnostic)
    embedding = entry.get("embedding")
    if not isinstance(embedding, list) or len(embedding) == 0:
        raise LLMResponseFormatError(diagnostic)
    values: list[float] = []
    for value in embedding:
        if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise LLMResponseFormatError(diagnostic)
        values.append(float(value))
    return tuple(values)


def _http_connection(parsed_url: ParseResult, timeout_seconds: float) -> http.client.HTTPConnection:
    scheme = parsed_url.scheme
    host = parsed_url.hostname
    port = parsed_url.port
    if host is None:
        raise ValueError("HTTP transport URL must include scheme and host")
    if scheme == "https":
        return http.client.HTTPSConnection(host=host, port=port, timeout=timeout_seconds)
    if scheme == "http":
        return http.client.HTTPConnection(host=host, port=port, timeout=timeout_seconds)
    raise ValueError("HTTP transport URL must use http or https")


_REQUIRED_LLM_KEYS = (
    "model",
    "api_base",
    "api_key",
    "context_window",
    "max_tokens",
    "temperature",
    "context_window_threshold",
    "max_retries",
    "retry_delay",
    "request_timeout_seconds",
)
