"""Caching front-end that turns a parser adapter into reusable parsed structures.

The adapters produce the same canonical ParsedDocument structures consumed by
the offline parser-backed transformers, so a live spaCy or Stanza adapter can
feed every parser feature family through one cached parse step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from stylometry_python_lib.document import AnnotationLayerMetadata
from stylometry_python_lib.features.optional import ParsedDocument


class DocumentParserProtocol(Protocol):
    """Minimal adapter interface required for cached document parsing."""

    def parse_documents(self, texts: tuple[tuple[str, str], ...]) -> tuple[ParsedDocument, ...]:
        """Return canonical parsed documents for (document_id, text) pairs."""
        ...

    def provider_metadata(self) -> AnnotationLayerMetadata:
        """Return provider provenance metadata used as the cache namespace."""
        ...


def _empty_store() -> dict[tuple[str, str], ParsedDocument]:
    return {}


@dataclass
class ParserCache:
    """In-memory cache keyed by provider config hash and document id."""

    store: dict[tuple[str, str], ParsedDocument] = field(default_factory=_empty_store)

    def cache_key(self, metadata_hash: str, document_id: str) -> tuple[str, str]:
        """Return the cache key for one document under one provider config."""
        return (metadata_hash, document_id)


def parse_with_cache(adapter: DocumentParserProtocol, texts: tuple[tuple[str, str], ...], cache: ParserCache) -> tuple[ParsedDocument, ...]:
    """Parse documents, serving cached results and parsing only the misses."""
    metadata_hash = adapter.provider_metadata().config_hash
    missing = tuple((document_id, text) for document_id, text in texts if cache.cache_key(metadata_hash, document_id) not in cache.store)
    if len(missing) > 0:
        for parsed in adapter.parse_documents(missing):
            cache.store[cache.cache_key(metadata_hash, parsed.document_id)] = parsed
    return tuple(cache.store[cache.cache_key(metadata_hash, document_id)] for document_id, _ in texts)
