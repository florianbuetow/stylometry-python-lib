"""Tests for LLM recorded-response cassette tooling."""

import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest

from stylometry_python_lib.llm import LLMMessage, LLMProviderError, LLMRequest, LLMResponse
from stylometry_python_lib.llm_replay import (
    LLMCassette,
    LLMCassetteEntry,
    RecordedResponseLLMClient,
    ReplayResponseLLMClient,
    read_cassette,
    record_key,
    write_cassette,
)
from stylometry_python_lib.llm_transformers import (
    configured_llm_annotation_transformer,
    configured_llm_row_feature_names,
)


def _stub_calls() -> list[LLMRequest]:
    return []


@dataclass
class _StubClient:
    response: LLMResponse
    calls: list[LLMRequest] = field(default_factory=_stub_calls)

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.calls.append(request)
        return self.response

    def list_models(self) -> tuple[str, ...]:
        return ("m",)


def _request() -> LLMRequest:
    return LLMRequest(
        feature_name="text::llm::tone",
        prompt_version="v1",
        schema_id="tone_v1",
        messages=(LLMMessage(role="user", content="Classify the tone."),),
        response_schema=None,
    )


def test_record_key_is_stable_and_order_sensitive() -> None:
    first = record_key(_request())
    second = record_key(_request())
    assert first == second
    assert len(first) == 64  # sha256 hexdigest


def test_cassette_round_trips_through_disk(tmp_path: Path) -> None:
    response = LLMResponse(
        content='{"tone": 0.5}',
        raw_response={"id": "x"},
        provider="openai-compatible",
        model="m",
        finish_reason="stop",
        usage=None,
        decoding_settings={"temperature": 0.0},
    )
    cassette = LLMCassette(entries=(LLMCassetteEntry(key="k", response=response),))
    path = tmp_path / "cassette.json"
    write_cassette(path, cassette)
    restored = read_cassette(path)
    assert restored == cassette


def test_recorded_client_delegates_and_captures() -> None:
    response = LLMResponse(content="{}", raw_response={}, provider="p", model="m", finish_reason="stop", usage=None, decoding_settings={})
    inner = _StubClient(response=response)
    client = RecordedResponseLLMClient(inner=inner)
    out = client.complete(_request())
    assert out == response
    assert len(inner.calls) == 1
    assert len(client.recorded_cassette().entries) == 1


def test_replay_returns_recorded_response() -> None:
    request = _request()
    response = LLMResponse(
        content='{"tone": 1.0}',
        raw_response={},
        provider="recorded",
        model="m",
        finish_reason="stop",
        usage=None,
        decoding_settings={},
    )
    cassette = LLMCassette(entries=(LLMCassetteEntry(key=record_key(request), response=response),))
    client = ReplayResponseLLMClient(cassette=cassette)
    assert client.complete(request) == response


def test_replay_raises_on_missing_key() -> None:
    client = ReplayResponseLLMClient(cassette=LLMCassette(entries=()))
    with pytest.raises(LLMProviderError, match="no recorded response"):
        client.complete(_request())


def test_replay_miss_surfaces_as_diagnostic_through_transformer() -> None:
    feature_names = configured_llm_row_feature_names()[:1]
    client = ReplayResponseLLMClient(cassette=LLMCassette(entries=()))
    transformer = configured_llm_annotation_transformer(client=client, text_column="text", feature_names=feature_names)
    x = pd.DataFrame({"text": ["hello"]}, index=["d1"])
    result = transformer.fit_transform(x, None)
    assert result.shape == (1, 1)
    assert math.isnan(result[0, 0])
