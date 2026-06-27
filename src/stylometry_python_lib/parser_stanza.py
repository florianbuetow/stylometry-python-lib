"""Live Stanza parser adapter producing canonical parsed structures.

The adapter consumes a pipeline through a minimal structural protocol so the
conversion logic is testable offline with lightweight fakes; the factory builds
a real Stanza pipeline behind the parser-stanza extra.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, cast

from stylometry_python_lib.document import AnnotationLayerMetadata
from stylometry_python_lib.errors import OptionalDependencyError
from stylometry_python_lib.features.optional import (
    ParsedDependencyArc,
    ParsedDocument,
    ParsedMorphologyFeature,
    ParsedNamedEntity,
    ParsedSyntacticCounts,
    ParsedToken,
)

_DEPENDENT_CLAUSE_RELATIONS = frozenset({"ccomp", "advcl", "acl", "relcl", "csubj", "xcomp", "pcomp"})
_COMPLEX_NOMINAL_RELATIONS = frozenset({"nmod", "acl", "appos", "nummod"})


class StanzaWordLike(Protocol):
    """Minimal structural view of a Stanza word."""

    text: str
    upos: str
    feats: str | None
    head: int
    deprel: str
    id: int
    start_char: int
    end_char: int


class StanzaSpanLike(Protocol):
    """Minimal structural view of a Stanza entity span."""

    text: str
    type: str
    start_char: int
    end_char: int


class StanzaSentenceLike(Protocol):
    """Minimal structural view of a Stanza sentence."""

    @property
    def words(self) -> Sequence[StanzaWordLike]:
        """Return the sentence words."""
        ...

    @property
    def ents(self) -> Sequence[StanzaSpanLike]:
        """Return the sentence entity spans."""
        ...


class StanzaDocumentLike(Protocol):
    """Minimal structural view of a Stanza document."""

    @property
    def sentences(self) -> Sequence[StanzaSentenceLike]:
        """Return the document sentences."""
        ...


class StanzaPipelineLike(Protocol):
    """Minimal structural view of a callable Stanza pipeline."""

    def __call__(self, text: str) -> StanzaDocumentLike:
        """Annotate one text into a document."""
        ...


def _morphology(feats: str | None) -> tuple[ParsedMorphologyFeature, ...]:
    if feats is None or feats == "":
        return ()
    features: list[ParsedMorphologyFeature] = []
    for item in feats.split("|"):
        attribute, separator, value = item.partition("=")
        if separator == "=" and attribute != "" and value != "":
            features.append(ParsedMorphologyFeature(attribute=attribute, value=value))
    return tuple(features)


def stanza_document_to_parsed(document_id: str, document: StanzaDocumentLike) -> ParsedDocument:
    """Convert one Stanza document into a canonical ParsedDocument."""
    tokens: list[ParsedToken] = []
    arcs: list[ParsedDependencyArc] = []
    offset = 0
    for sentence in document.sentences:
        words = sentence.words
        for word in words:
            local_index = word.id - 1
            global_index = offset + local_index
            tokens.append(ParsedToken(text=word.text, upos=word.upos, morphology=_morphology(word.feats)))
            head_index = None if word.head == 0 else offset + (word.head - 1)
            arcs.append(ParsedDependencyArc(head_index=head_index, dependent_index=global_index, relation=word.deprel.lower()))
        offset += len(words)
    return ParsedDocument(document_id=document_id, tokens=tuple(tokens), dependency_arcs=tuple(arcs))


def stanza_document_to_entities(document_id: str, document: StanzaDocumentLike) -> tuple[ParsedNamedEntity, ...]:
    """Convert Stanza entity spans into canonical ParsedNamedEntity records by char overlap."""
    char_bounds: list[tuple[int, int]] = [
        (word.start_char, word.end_char) for sentence in document.sentences for word in sentence.words
    ]
    spans: list[StanzaSpanLike] = [span for sentence in document.sentences for span in sentence.ents]
    entities: list[ParsedNamedEntity] = []
    for span in spans:
        overlapping = [index for index, (start, end) in enumerate(char_bounds) if start < span.end_char and end > span.start_char]
        if len(overlapping) == 0:
            continue
        entities.append(
            ParsedNamedEntity(
                document_id=document_id,
                text=span.text,
                label=span.type,
                start_token_index=overlapping[0],
                end_token_index=overlapping[-1] + 1,
            )
        )
    return tuple(entities)


def stanza_document_to_syntactic_counts(document_id: str, document: StanzaDocumentLike) -> ParsedSyntacticCounts:
    """Derive clause and T-unit counts from Stanza dependency structure."""
    words: list[StanzaWordLike] = [word for sentence in document.sentences for word in sentence.words]
    sentence_count = len(document.sentences)
    dependent_clause_count = sum(1 for word in words if word.deprel.lower() in _DEPENDENT_CLAUSE_RELATIONS)
    return ParsedSyntacticCounts(
        document_id=document_id,
        word_count=len(words),
        sentence_count=sentence_count,
        clause_count=sentence_count + dependent_clause_count,
        t_unit_count=max(sentence_count, 1),
        dependent_clause_count=dependent_clause_count,
        coordinate_phrase_count=sum(1 for word in words if word.deprel.lower() == "conj"),
        complex_nominal_count=sum(1 for word in words if word.deprel.lower() in _COMPLEX_NOMINAL_RELATIONS),
        verb_phrase_count=sum(1 for word in words if word.upos == "VERB"),
    )


class StanzaParserAdapter:
    """Produce canonical parsed structures from an injected Stanza pipeline."""

    def __init__(self, pipeline: StanzaPipelineLike, language: str, version: str) -> None:
        self._pipeline = pipeline
        self._language = language
        self._version = version

    def parse_documents(self, texts: tuple[tuple[str, str], ...]) -> tuple[ParsedDocument, ...]:
        """Return token, morphology, and dependency annotations per document."""
        return tuple(stanza_document_to_parsed(document_id, self._pipeline(text)) for document_id, text in texts)

    def parse_named_entities(self, texts: tuple[tuple[str, str], ...]) -> tuple[ParsedNamedEntity, ...]:
        """Return named-entity spans across all documents."""
        entities: list[ParsedNamedEntity] = []
        for document_id, text in texts:
            entities.extend(stanza_document_to_entities(document_id, self._pipeline(text)))
        return tuple(entities)

    def parse_syntactic_counts(self, texts: tuple[tuple[str, str], ...]) -> tuple[ParsedSyntacticCounts, ...]:
        """Return clause/T-unit syntactic counts per document."""
        return tuple(stanza_document_to_syntactic_counts(document_id, self._pipeline(text)) for document_id, text in texts)

    def provider_metadata(self) -> AnnotationLayerMetadata:
        """Return Stanza provenance metadata for the DocumentView parser layer."""
        return AnnotationLayerMetadata(
            layer_name="parser",
            provider="stanza",
            model=self._language,
            version=self._version,
            language=self._language,
            tokenizer_settings="stanza_default",
            preprocessing_settings="stanza_default",
            config_hash=f"stanza:{self._language}:{self._version}",
            diagnostics=(),
        )


def stanza_parser_adapter(language: str) -> StanzaParserAdapter:
    """Build a live Stanza pipeline and adapter. Requires the parser-stanza extra and installed models."""
    from importlib.metadata import version

    try:
        import stanza
    except ImportError as exc:
        raise OptionalDependencyError("Live Stanza parsing requires the 'parser-stanza' extra (stanza)") from exc
    try:
        pipeline = stanza.Pipeline(lang=language, processors="tokenize,pos,lemma,depparse,ner", download_method=None)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise OptionalDependencyError(
            f"Stanza models for language '{language}' are not installed; download them before live parsing"
        ) from exc
    return StanzaParserAdapter(pipeline=cast(StanzaPipelineLike, pipeline), language=language, version=version("stanza"))
