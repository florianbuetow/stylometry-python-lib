"""Tests for LLM recorded-response cassette tooling."""

from dataclasses import dataclass, field

from stylometry_python_lib.llm import LLMMessage, LLMRequest, LLMResponse
from stylometry_python_lib.llm_replay import (
    LLMCassette,
    LLMCassetteEntry,
    RecordedResponseLLMClient,
    read_cassette,
    record_key,
    write_cassette,
)


@dataclass
class _StubClient:
    response: LLMResponse
    calls: list[LLMRequest] = field(default_factory=list)

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


def test_cassette_round_trips_through_disk(tmp_path) -> None:
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
