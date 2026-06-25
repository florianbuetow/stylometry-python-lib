"""Tests for configured LLM config loading and OpenAI-compatible client behavior."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pytest

from stylometry_python_lib import (
    HTTPTransportResponse,
    LLMConfig,
    LLMConfigError,
    LLMContextWindowError,
    LLMDiagnosticReason,
    LLMMessage,
    LLMRequest,
    LLMResponseFormatError,
    LLMTimeoutError,
    LLMUnsupportedCapabilityError,
    OpenAICompatibleLLMClient,
    load_llm_config,
)
from stylometry_python_lib.llm import JsonObject


@dataclass(frozen=True)
class RecordedHTTPCall:
    method: Literal["GET", "POST"]
    url: str
    headers: dict[str, str]
    payload: JsonObject | None
    timeout_seconds: float


def _recorded_http_calls() -> list[RecordedHTTPCall]:
    return []


@dataclass
class FakeHTTPTransport:
    responses: list[HTTPTransportResponse | BaseException]
    calls: list[RecordedHTTPCall] = field(default_factory=_recorded_http_calls)

    def request_json(
        self,
        method: Literal["GET", "POST"],
        url: str,
        headers: Mapping[str, str],
        payload: JsonObject | None,
        timeout_seconds: float,
    ) -> HTTPTransportResponse:
        self.calls.append(
            RecordedHTTPCall(
                method=method,
                url=url,
                headers=dict(headers),
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
        )
        if len(self.responses) == 0:
            raise AssertionError("FakeHTTPTransport has no queued response")
        response = self.responses.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def test_load_llm_config_validates_exact_root_config(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "\n".join(
            (
                "llm:",
                "  model: qwen/qwen3.6-35b-a3b",
                "  api_base: http://127.0.0.1:1234/v1",
                "  api_key: lm-studio",
                "  context_window: 32786",
                "  max_tokens: 32768",
                "  temperature: 0.3",
                "  context_window_threshold: 90",
                "  max_retries: 3",
                "  retry_delay: 2.0",
                "  request_timeout_seconds: 1800.0",
            )
        ),
        encoding="utf-8",
    )

    config = load_llm_config(config_path)

    assert config == LLMConfig(
        model="qwen/qwen3.6-35b-a3b",
        api_base="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        context_window=32786,
        max_tokens=32768,
        temperature=0.3,
        context_window_threshold=90,
        max_retries=3,
        retry_delay=2.0,
        request_timeout_seconds=1800.0,
    )


def test_load_llm_config_fails_on_missing_file_missing_key_and_invalid_bounds(tmp_path: Path) -> None:
    with pytest.raises(LLMConfigError, match="llm config missing"):
        load_llm_config(tmp_path / "missing.yaml")

    missing_key_path = tmp_path / "missing-key.yaml"
    missing_key_path.write_text(
        "\n".join(
            (
                "llm:",
                "  model: qwen/qwen3.6-35b-a3b",
                "  api_base: http://127.0.0.1:1234/v1",
                "  api_key: lm-studio",
                "  context_window: 32786",
                "  max_tokens: 32768",
                "  temperature: 0.3",
                "  context_window_threshold: 90",
                "  max_retries: 3",
                "  retry_delay: 2.0",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(LLMConfigError, match="request_timeout_seconds"):
        load_llm_config(missing_key_path)

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text(
        "\n".join(
            (
                "llm:",
                "  model: qwen/qwen3.6-35b-a3b",
                "  api_base: http://127.0.0.1:1234/v1",
                "  api_key: lm-studio",
                "  context_window: 128",
                "  max_tokens: 129",
                "  temperature: 0.3",
                "  context_window_threshold: 90",
                "  max_retries: 3",
                "  retry_delay: 2.0",
                "  request_timeout_seconds: 1800.0",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(LLMConfigError, match="max_tokens"):
        load_llm_config(invalid_path)


def test_openai_client_builds_chat_completion_payload() -> None:
    transport = FakeHTTPTransport(responses=[HTTPTransportResponse(status_code=200, body=_completion_body('{"tone":"formal"}'))])
    client = OpenAICompatibleLLMClient(config=_config(max_retries=0), transport=transport)
    request = _request(
        response_schema={
            "type": "object",
            "properties": {"tone": {"type": "string"}},
            "required": ["tone"],
            "additionalProperties": False,
        }
    )

    response = client.complete(request)

    assert response.content == '{"tone":"formal"}'
    assert response.model == "fixture-model"
    assert response.finish_reason == "stop"
    assert response.decoding_settings == {
        "max_tokens": 64,
        "temperature": 0.3,
        "context_window": 4096,
        "context_window_threshold": 90,
        "max_retries": 0,
        "retry_delay": 0.0,
        "request_timeout_seconds": 5.0,
    }
    assert len(transport.calls) == 1
    call = transport.calls[0]
    assert call.method == "POST"
    assert call.url == "http://127.0.0.1:1234/v1/chat/completions"
    assert call.headers["Authorization"] == "Bearer lm-studio"
    assert call.timeout_seconds == 5.0
    assert call.payload is not None
    assert call.payload["model"] == "qwen/qwen3.6-35b-a3b"
    assert call.payload["temperature"] == 0.3
    assert call.payload["max_tokens"] == 64
    assert call.payload["stream"] is False
    assert call.payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "tone_schema_v1",
            "schema": request.response_schema,
            "strict": True,
        },
    }


def test_openai_client_retries_retriable_provider_errors() -> None:
    transport = FakeHTTPTransport(
        responses=[
            HTTPTransportResponse(status_code=500, body='{"error":"busy"}'),
            HTTPTransportResponse(status_code=200, body=_completion_body('{"tone":"formal"}')),
        ]
    )
    client = OpenAICompatibleLLMClient(config=_config(max_retries=1), transport=transport)

    response = client.complete(_request(response_schema=None))

    assert response.content == '{"tone":"formal"}'
    assert len(transport.calls) == 2


def test_openai_client_accepts_lm_studio_reasoning_content_when_content_is_empty() -> None:
    body = json.dumps(
        {
            "model": "fixture-model",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": '{"label":"formal","confidence":0.8,"evidence":"test"}',
                    },
                }
            ],
        }
    )
    transport = FakeHTTPTransport(responses=[HTTPTransportResponse(status_code=200, body=body)])
    client = OpenAICompatibleLLMClient(config=_config(max_retries=0), transport=transport)

    response = client.complete(_request(response_schema=None))

    assert response.content == '{"label":"formal","confidence":0.8,"evidence":"test"}'


def test_openai_client_maps_timeout_provider_and_malformed_response_diagnostics() -> None:
    timeout_client = OpenAICompatibleLLMClient(config=_config(max_retries=0), transport=FakeHTTPTransport(responses=[TimeoutError("slow")]))
    with pytest.raises(LLMTimeoutError) as timeout_error:
        timeout_client.complete(_request(response_schema=None))
    assert timeout_error.value.diagnostic.reason == LLMDiagnosticReason.TIMEOUT

    provider_client = OpenAICompatibleLLMClient(
        config=_config(max_retries=0),
        transport=FakeHTTPTransport(responses=[HTTPTransportResponse(status_code=400, body='{"error":"bad"}')]),
    )
    with pytest.raises(LLMResponseFormatError) as provider_error:
        provider_client.complete(_request(response_schema=None))
    assert provider_error.value.diagnostic.reason == LLMDiagnosticReason.PROVIDER_ERROR

    malformed_client = OpenAICompatibleLLMClient(
        config=_config(max_retries=0),
        transport=FakeHTTPTransport(responses=[HTTPTransportResponse(status_code=200, body="not-json")]),
    )
    with pytest.raises(LLMResponseFormatError) as malformed_error:
        malformed_client.complete(_request(response_schema=None))
    assert malformed_error.value.diagnostic.reason == LLMDiagnosticReason.INVALID_JSON


def test_openai_client_maps_structured_response_diagnostics() -> None:
    cases = (
        (
            HTTPTransportResponse(status_code=200, body=json.dumps({})),
            LLMDiagnosticReason.MISSING_REQUIRED_FIELD,
        ),
        (
            HTTPTransportResponse(status_code=200, body=json.dumps({"choices": ["not-an-object"]})),
            LLMDiagnosticReason.SCHEMA_MISMATCH,
        ),
        (
            HTTPTransportResponse(
                status_code=200,
                body=json.dumps({"choices": [{"finish_reason": "length", "message": {"role": "assistant", "content": "{}"}}]}),
            ),
            LLMDiagnosticReason.TRUNCATION,
        ),
        (
            HTTPTransportResponse(
                status_code=200,
                body=json.dumps({"choices": [{"finish_reason": "content_filter", "message": {"role": "assistant", "content": "{}"}}]}),
            ),
            LLMDiagnosticReason.REFUSAL,
        ),
        (
            HTTPTransportResponse(
                status_code=200,
                body=json.dumps({"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": ""}}]}),
            ),
            LLMDiagnosticReason.EMPTY_RESPONSE,
        ),
    )
    for response, reason in cases:
        client = OpenAICompatibleLLMClient(config=_config(max_retries=0), transport=FakeHTTPTransport(responses=[response]))
        with pytest.raises(LLMResponseFormatError) as error:
            client.complete(_request(response_schema=None))
        assert error.value.diagnostic.reason == reason


def test_openai_client_rejects_context_window_overflow() -> None:
    transport = FakeHTTPTransport(responses=[HTTPTransportResponse(status_code=200, body=_completion_body('{"tone":"formal"}'))])
    client = OpenAICompatibleLLMClient(
        config=_config(context_window=16, max_tokens=8, context_window_threshold=50, max_retries=0),
        transport=transport,
    )
    request = LLMRequest(
        feature_name="text::llm::tone",
        prompt_version="tone_prompt_v1",
        schema_id="tone_schema_v1",
        messages=(LLMMessage(role="user", content="x" * 200),),
        response_schema=None,
    )

    with pytest.raises(LLMContextWindowError) as error:
        client.complete(request)

    assert error.value.diagnostic.reason == LLMDiagnosticReason.CONTEXT_WINDOW_OVERFLOW
    assert transport.calls == []


def test_openai_client_lists_models() -> None:
    body = json.dumps({"data": [{"id": "qwen/qwen3.6-35b-a3b"}, {"id": "text-embedding-bge-large-en-v1.5@f16"}]})
    transport = FakeHTTPTransport(responses=[HTTPTransportResponse(status_code=200, body=body)])
    client = OpenAICompatibleLLMClient(config=_config(max_retries=0), transport=transport)

    models = client.list_models()

    assert models == ("qwen/qwen3.6-35b-a3b", "text-embedding-bge-large-en-v1.5@f16")
    assert transport.calls[0].method == "GET"
    assert transport.calls[0].url == "http://127.0.0.1:1234/v1/models"


def test_openai_client_builds_embedding_payload_and_parses_vectors() -> None:
    body = json.dumps(
        {
            "model": "embedding-model",
            "data": [{"embedding": [0.1, 0.2]}, {"embedding": [0.3, 0.4]}],
            "usage": {"total_tokens": 4},
        }
    )
    transport = FakeHTTPTransport(responses=[HTTPTransportResponse(status_code=200, body=body)])
    client = OpenAICompatibleLLMClient(config=_config(max_retries=0), transport=transport)

    response = client.embed_texts(("first text", "second text"))

    assert response.vectors == ((0.1, 0.2), (0.3, 0.4))
    assert response.model == "embedding-model"
    assert response.usage == {"total_tokens": 4}
    assert response.config_metadata == {
        "api_base": "http://127.0.0.1:1234/v1",
        "max_retries": 0,
        "retry_delay": 0.0,
        "request_timeout_seconds": 5.0,
    }
    assert transport.calls[0].method == "POST"
    assert transport.calls[0].url == "http://127.0.0.1:1234/v1/embeddings"
    assert transport.calls[0].payload == {"model": "qwen/qwen3.6-35b-a3b", "input": ["first text", "second text"]}


def test_openai_client_maps_embedding_provider_error_to_unsupported_capability() -> None:
    transport = FakeHTTPTransport(responses=[HTTPTransportResponse(status_code=400, body='{"error":"unsupported"}')])
    client = OpenAICompatibleLLMClient(config=_config(max_retries=0), transport=transport)

    with pytest.raises(LLMUnsupportedCapabilityError) as error:
        client.embed_texts(("first text",))

    assert error.value.diagnostic.reason == LLMDiagnosticReason.UNSUPPORTED_ENDPOINT_CAPABILITY


def _config(
    *,
    context_window: int = 4096,
    max_tokens: int = 64,
    context_window_threshold: int = 90,
    max_retries: int,
) -> LLMConfig:
    return LLMConfig(
        model="qwen/qwen3.6-35b-a3b",
        api_base="http://127.0.0.1:1234/v1",
        api_key="lm-studio",
        context_window=context_window,
        max_tokens=max_tokens,
        temperature=0.3,
        context_window_threshold=context_window_threshold,
        max_retries=max_retries,
        retry_delay=0.0,
        request_timeout_seconds=5.0,
    )


def _request(response_schema: JsonObject | None) -> LLMRequest:
    return LLMRequest(
        feature_name="text::llm::tone",
        prompt_version="tone_prompt_v1",
        schema_id="tone_schema_v1",
        messages=(LLMMessage(role="system", content="Return JSON only."), LLMMessage(role="user", content="Text: Hello.")),
        response_schema=response_schema,
    )


def _completion_body(content: str) -> str:
    return json.dumps(
        {
            "model": "fixture-model",
            "choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
            "usage": {"total_tokens": 10},
        }
    )
