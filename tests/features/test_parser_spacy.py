"""Offline and opt-in live tests for the spaCy parser adapter."""

from __future__ import annotations

import importlib.util
import os
from collections.abc import Iterator
from dataclasses import dataclass, field

import pytest

from stylometry_python_lib.errors import OptionalDependencyError
from stylometry_python_lib.features.optional import ParsedMorphologyFeature
from stylometry_python_lib.parser_spacy import (
    SpacyParserAdapter,
    spacy_document_to_parsed,
    spacy_parser_adapter,
)

_HAS_SPACY = importlib.util.find_spec("spacy") is not None


@dataclass
class _FakeToken:
    text: str
    pos_: str
    dep_: str
    i: int
    morph: str
    head_token: _FakeToken | None = None

    @property
    def head(self) -> _FakeToken:
        return self.head_token if self.head_token is not None else self


@dataclass
class _FakeSpan:
    text: str
    label_: str
    start: int
    end: int


def _no_spans() -> list[_FakeSpan]:
    return []


@dataclass
class _FakeDoc:
    tokens: list[_FakeToken]
    spans: list[_FakeSpan] = field(default_factory=_no_spans)

    def __iter__(self) -> Iterator[_FakeToken]:
        return iter(self.tokens)

    @property
    def ents(self) -> tuple[_FakeSpan, ...]:
        return tuple(self.spans)

    @property
    def sents(self) -> Iterator[_FakeSpan]:
        return iter([_FakeSpan(text="full", label_="", start=0, end=len(self.tokens))])


@dataclass
class _FakePipeline:
    doc: _FakeDoc
    lang: str = "en"

    def __call__(self, text: str) -> _FakeDoc:
        return self.doc


def _sentence_doc() -> _FakeDoc:
    the = _FakeToken(text="The", pos_="DET", dep_="det", i=0, morph="")
    cat = _FakeToken(text="cat", pos_="NOUN", dep_="nsubj", i=1, morph="Number=Sing")
    sat = _FakeToken(text="sat", pos_="VERB", dep_="ROOT", i=2, morph="Tense=Past")
    stop = _FakeToken(text=".", pos_="PUNCT", dep_="punct", i=3, morph="")
    the.head_token = cat
    cat.head_token = sat
    sat.head_token = sat
    stop.head_token = sat
    span = _FakeSpan(text="cat", label_="ANIMAL", start=1, end=2)
    return _FakeDoc(tokens=[the, cat, sat, stop], spans=[span])


def test_spacy_conversion_produces_canonical_parsed_document() -> None:
    parsed = spacy_document_to_parsed("d0", _sentence_doc())
    assert parsed.document_id == "d0"
    assert tuple(token.upos for token in parsed.tokens) == ("DET", "NOUN", "VERB", "PUNCT")
    assert parsed.tokens[1].morphology == (ParsedMorphologyFeature(attribute="Number", value="Sing"),)
    root_arcs = [arc for arc in parsed.dependency_arcs if arc.head_index is None]
    assert len(root_arcs) == 1
    assert root_arcs[0].dependent_index == 2


def test_spacy_adapter_parses_entities_and_counts() -> None:
    adapter = SpacyParserAdapter(pipeline=_FakePipeline(doc=_sentence_doc()), model="fake", version="0")
    entities = adapter.parse_named_entities((("d0", "The cat sat."),))
    assert entities[0].label == "ANIMAL"
    counts = adapter.parse_syntactic_counts((("d0", "The cat sat."),))
    assert counts[0].sentence_count == 1
    assert counts[0].verb_phrase_count == 1
    assert adapter.provider_metadata().provider == "spacy"


def test_spacy_adapter_missing_model_fails_fast() -> None:
    if not _HAS_SPACY:
        pytest.skip("spaCy not installed")
    with pytest.raises(OptionalDependencyError, match="model"):
        spacy_parser_adapter("stylometry_nonexistent_model_xyz")


@pytest.mark.skipif(not (_HAS_SPACY and os.environ.get("STYLOMETRY_LIVE_PARSER") == "1"), reason="opt-in live spaCy test")
def test_spacy_adapter_parses_live_document() -> None:
    adapter = spacy_parser_adapter("en_core_web_sm")
    parsed = adapter.parse_documents((("d0", "The cat sat on the mat."),))
    assert parsed[0].document_id == "d0"
    assert any(token.upos == "NOUN" for token in parsed[0].tokens)
    assert any(arc.head_index is None for arc in parsed[0].dependency_arcs)
