"""Recorded-response and replay tooling for offline LLM tests."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from stylometry_python_lib.llm import JsonObject, JsonValue, LLMRequest, LLMResponse


def record_key(request: LLMRequest) -> str:
    """Return a stable sha256 key for a request's identity fields."""
    payload: JsonObject = {
        "feature_name": request.feature_name,
        "prompt_version": request.prompt_version,
        "schema_id": request.schema_id,
        "messages": [{"role": message.role, "content": message.content} for message in request.messages],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class LLMCassetteEntry:
    """One recorded request/response pair keyed by request identity."""

    key: str
    response: LLMResponse


@dataclass(frozen=True)
class LLMCassette:
    """An immutable collection of recorded LLM responses."""

    entries: tuple[LLMCassetteEntry, ...]


def _response_to_json(response: LLMResponse) -> JsonObject:
    payload: JsonObject = {
        "content": response.content,
        "raw_response": response.raw_response,
        "provider": response.provider,
        "model": response.model,
        "finish_reason": response.finish_reason,
        "usage": response.usage,
        "decoding_settings": response.decoding_settings,
    }
    return payload


def _require_str(payload: JsonObject, key: str) -> str:
    if key not in payload:
        raise ValueError(f"Cassette response missing required field: {key}")
    value = payload[key]
    if not isinstance(value, str):
        raise ValueError(f"Cassette response field '{key}' must be a string")
    return value


def _require_object(payload: JsonObject, key: str) -> JsonObject:
    if key not in payload:
        raise ValueError(f"Cassette response missing required field: {key}")
    value = payload[key]
    if not isinstance(value, dict):
        raise ValueError(f"Cassette response field '{key}' must be an object")
    return value


def _require_optional_object(payload: JsonObject, key: str) -> JsonObject | None:
    if key not in payload:
        raise ValueError(f"Cassette response missing required field: {key}")
    value = payload[key]
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(f"Cassette response field '{key}' must be an object or null")
    return value


def _response_from_json(payload: JsonObject) -> LLMResponse:
    return LLMResponse(
        content=_require_str(payload, "content"),
        raw_response=_require_object(payload, "raw_response"),
        provider=_require_str(payload, "provider"),
        model=_require_str(payload, "model"),
        finish_reason=_require_str(payload, "finish_reason"),
        usage=_require_optional_object(payload, "usage"),
        decoding_settings=_require_object(payload, "decoding_settings"),
    )


def write_cassette(path: str | Path, cassette: LLMCassette) -> None:
    """Write a cassette to disk as deterministic JSON."""
    serialized: JsonObject = {"entries": [{"key": entry.key, "response": _response_to_json(entry.response)} for entry in cassette.entries]}
    Path(path).write_text(json.dumps(serialized, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def read_cassette(path: str | Path) -> LLMCassette:
    """Read a cassette from disk, failing fast on malformed content."""
    loaded: JsonValue = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Cassette file must contain a JSON object: {path}")
    if "entries" not in loaded:
        raise ValueError(f"Cassette file missing 'entries': {path}")
    raw_entries = loaded["entries"]
    if not isinstance(raw_entries, list):
        raise ValueError(f"Cassette 'entries' must be a list: {path}")
    entries: list[LLMCassetteEntry] = []
    for item in raw_entries:
        if not isinstance(item, dict):
            raise ValueError(f"Cassette entry must be an object: {path}")
        entries.append(LLMCassetteEntry(key=_require_str(item, "key"), response=_response_from_json(_require_object(item, "response"))))
    return LLMCassette(entries=tuple(entries))
