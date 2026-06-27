"""Tests for the caching parser pipeline feeding existing parser transformers."""

from __future__ import annotations

from dataclasses import dataclass

from stylometry_python_lib.document import AnnotationLayerMetadata
from stylometry_python_lib.features.optional import (
    ParsedDocument,
    ParsedToken,
    parser_backed_transformer,
)
from stylometry_python_lib.parser_pipeline import ParserCache, parse_with_cache


@dataclass
class _FixedAdapter:
    calls: int = 0

    def parse_documents(self, texts: tuple[tuple[str, str], ...]) -> tuple[ParsedDocument, ...]:
        self.calls += 1
        return tuple(
            ParsedDocument(document_id=document_id, tokens=(ParsedToken(text=text, upos="NOUN", morphology=()),), dependency_arcs=())
            for document_id, text in texts
        )

    def provider_metadata(self) -> AnnotationLayerMetadata:
        return AnnotationLayerMetadata(
            layer_name="parser",
            provider="fixed",
            model="m",
            version="1",
            language="en",
            tokenizer_settings="t",
            preprocessing_settings="p",
            config_hash="fixed:m:1",
            diagnostics=(),
        )


def test_cache_avoids_reparsing_and_feeds_existing_transformer() -> None:
    adapter = _FixedAdapter()
    cache = ParserCache()
    texts = (("d0", "alpha"),)
    first = parse_with_cache(adapter, texts, cache)
    second = parse_with_cache(adapter, texts, cache)
    assert first == second
    assert adapter.calls == 1  # second call served from cache

    transformer = parser_backed_transformer(
        provider="fake",
        model="m",
        version="1",
        text_column="text",
        output="pandas",
        parsed_documents=first,
    )
    assert transformer is not None
