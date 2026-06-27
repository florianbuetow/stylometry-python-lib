"""Offline and opt-in live tests for the Stanza parser adapter."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field

import pytest

from stylometry_python_lib.features.optional import ParsedMorphologyFeature
from stylometry_python_lib.parser_stanza import (
    StanzaParserAdapter,
    stanza_document_to_parsed,
    stanza_parser_adapter,
)

_HAS_STANZA = importlib.util.find_spec("stanza") is not None


@dataclass
class _FakeWord:
    text: str
    upos: str
    feats: str | None
    head: int
    deprel: str
    id: int
    start_char: int
    end_char: int


@dataclass
class _FakeSpan:
    text: str
    type: str
    start_char: int
    end_char: int


def _no_words() -> list[_FakeWord]:
    return []


def _no_spans() -> list[_FakeSpan]:
    return []


@dataclass
class _FakeSentence:
    words: list[_FakeWord] = field(default_factory=_no_words)
    ents: list[_FakeSpan] = field(default_factory=_no_spans)


def _no_sentences() -> list[_FakeSentence]:
    return []


@dataclass
class _FakeDocument:
    sentences: list[_FakeSentence] = field(default_factory=_no_sentences)


@dataclass
class _FakePipeline:
    document: _FakeDocument

    def __call__(self, text: str) -> _FakeDocument:
        return self.document


def _document() -> _FakeDocument:
    the = _FakeWord(text="The", upos="DET", feats=None, head=2, deprel="det", id=1, start_char=0, end_char=3)
    cat = _FakeWord(text="cat", upos="NOUN", feats="Number=Sing", head=3, deprel="nsubj", id=2, start_char=4, end_char=7)
    sat = _FakeWord(text="sat", upos="VERB", feats="Tense=Past", head=0, deprel="root", id=3, start_char=8, end_char=11)
    ent = _FakeSpan(text="cat", type="ANIMAL", start_char=4, end_char=7)
    return _FakeDocument(sentences=[_FakeSentence(words=[the, cat, sat], ents=[ent])])


def test_stanza_conversion_produces_canonical_parsed_document() -> None:
    parsed = stanza_document_to_parsed("d0", _document())
    assert parsed.document_id == "d0"
    assert tuple(token.upos for token in parsed.tokens) == ("DET", "NOUN", "VERB")
    assert parsed.tokens[1].morphology == (ParsedMorphologyFeature(attribute="Number", value="Sing"),)
    root_arcs = [arc for arc in parsed.dependency_arcs if arc.head_index is None]
    assert len(root_arcs) == 1
    assert root_arcs[0].dependent_index == 2


def test_stanza_adapter_entities_and_counts() -> None:
    adapter = StanzaParserAdapter(pipeline=_FakePipeline(document=_document()), language="en", version="0")
    entities = adapter.parse_named_entities((("d0", "The cat sat"),))
    assert entities[0].label == "ANIMAL"
    assert entities[0].start_token_index == 1
    assert entities[0].end_token_index == 2
    counts = adapter.parse_syntactic_counts((("d0", "The cat sat"),))
    assert counts[0].sentence_count == 1
    assert counts[0].verb_phrase_count == 1
    assert adapter.provider_metadata().provider == "stanza"


@pytest.mark.skipif(not (_HAS_STANZA and os.environ.get("STYLOMETRY_LIVE_PARSER") == "1"), reason="opt-in live Stanza test")
def test_stanza_adapter_parses_live_document() -> None:
    adapter = stanza_parser_adapter("en")
    parsed = adapter.parse_documents((("d0", "The cat sat on the mat."),))
    assert parsed[0].document_id == "d0"
    assert any(arc.head_index is None for arc in parsed[0].dependency_arcs)
