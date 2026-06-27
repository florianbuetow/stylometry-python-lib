"""Live spaCy parser adapter producing canonical parsed structures.

The adapter consumes a pipeline through a minimal structural protocol so the
conversion logic is testable offline with lightweight fakes; the factory loads
a real spaCy model behind the parser-spacy extra.
"""

from __future__ import annotations

from collections.abc import Iterator
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


class SpacyTokenLike(Protocol):
    """Minimal structural view of a spaCy token."""

    text: str
    pos_: str
    dep_: str
    i: int

    @property
    def head(self) -> SpacyTokenLike:
        """Return the syntactic head token."""
        ...

    @property
    def morph(self) -> object:
        """Return the morphological analysis (stringified as Attr=Val pipes)."""
        ...


class SpacySpanLike(Protocol):
    """Minimal structural view of a spaCy entity span."""

    text: str
    label_: str
    start: int
    end: int


class SpacyDocLike(Protocol):
    """Minimal structural view of a spaCy document."""

    def __iter__(self) -> Iterator[SpacyTokenLike]:
        """Iterate document tokens."""
        ...

    @property
    def ents(self) -> tuple[SpacySpanLike, ...]:
        """Return named-entity spans."""
        ...

    @property
    def sents(self) -> Iterator[SpacySpanLike]:
        """Iterate sentence spans."""
        ...


class SpacyPipelineLike(Protocol):
    """Minimal structural view of a callable spaCy pipeline."""

    lang: str

    def __call__(self, text: str) -> SpacyDocLike:
        """Annotate one text into a document."""
        ...


def _morphology(token: SpacyTokenLike) -> tuple[ParsedMorphologyFeature, ...]:
    rendered = str(token.morph)
    if rendered == "":
        return ()
    features: list[ParsedMorphologyFeature] = []
    for item in rendered.split("|"):
        attribute, separator, value = item.partition("=")
        if separator == "=" and attribute != "" and value != "":
            features.append(ParsedMorphologyFeature(attribute=attribute, value=value))
    return tuple(features)


def spacy_document_to_parsed(document_id: str, doc: SpacyDocLike) -> ParsedDocument:
    """Convert one spaCy document into a canonical ParsedDocument."""
    tokens = tuple(ParsedToken(text=token.text, upos=token.pos_, morphology=_morphology(token)) for token in doc)
    arcs = tuple(
        ParsedDependencyArc(
            head_index=None if token.head.i == token.i else token.head.i,
            dependent_index=token.i,
            relation=token.dep_.lower(),
        )
        for token in doc
    )
    return ParsedDocument(document_id=document_id, tokens=tokens, dependency_arcs=arcs)


def spacy_document_to_entities(document_id: str, doc: SpacyDocLike) -> tuple[ParsedNamedEntity, ...]:
    """Convert spaCy entity spans into canonical ParsedNamedEntity records."""
    return tuple(
        ParsedNamedEntity(
            document_id=document_id, text=span.text, label=span.label_, start_token_index=span.start, end_token_index=span.end
        )
        for span in doc.ents
    )


def spacy_document_to_syntactic_counts(document_id: str, doc: SpacyDocLike) -> ParsedSyntacticCounts:
    """Derive clause and T-unit counts from spaCy dependency structure."""
    tokens = tuple(doc)
    sentence_count = sum(1 for _ in doc.sents)
    dependent_clause_count = sum(1 for token in tokens if token.dep_ in _DEPENDENT_CLAUSE_RELATIONS)
    return ParsedSyntacticCounts(
        document_id=document_id,
        word_count=len(tokens),
        sentence_count=sentence_count,
        clause_count=sentence_count + dependent_clause_count,
        t_unit_count=max(sentence_count, 1),
        dependent_clause_count=dependent_clause_count,
        coordinate_phrase_count=sum(1 for token in tokens if token.dep_ == "conj"),
        complex_nominal_count=sum(1 for token in tokens if token.dep_ in _COMPLEX_NOMINAL_RELATIONS),
        verb_phrase_count=sum(1 for token in tokens if token.pos_ == "VERB"),
    )


class SpacyParserAdapter:
    """Produce canonical parsed structures from an injected spaCy pipeline."""

    def __init__(self, pipeline: SpacyPipelineLike, model: str, version: str) -> None:
        self._pipeline = pipeline
        self._model = model
        self._version = version

    def parse_documents(self, texts: tuple[tuple[str, str], ...]) -> tuple[ParsedDocument, ...]:
        """Return token, morphology, and dependency annotations per document."""
        return tuple(spacy_document_to_parsed(document_id, self._pipeline(text)) for document_id, text in texts)

    def parse_named_entities(self, texts: tuple[tuple[str, str], ...]) -> tuple[ParsedNamedEntity, ...]:
        """Return named-entity spans across all documents."""
        entities: list[ParsedNamedEntity] = []
        for document_id, text in texts:
            entities.extend(spacy_document_to_entities(document_id, self._pipeline(text)))
        return tuple(entities)

    def parse_syntactic_counts(self, texts: tuple[tuple[str, str], ...]) -> tuple[ParsedSyntacticCounts, ...]:
        """Return clause/T-unit syntactic counts per document."""
        return tuple(spacy_document_to_syntactic_counts(document_id, self._pipeline(text)) for document_id, text in texts)

    def provider_metadata(self) -> AnnotationLayerMetadata:
        """Return spaCy provenance metadata for the DocumentView parser layer."""
        return AnnotationLayerMetadata(
            layer_name="parser",
            provider="spacy",
            model=self._model,
            version=self._version,
            language=self._pipeline.lang,
            tokenizer_settings="spacy_default",
            preprocessing_settings="spacy_default",
            config_hash=f"spacy:{self._model}:{self._version}",
            diagnostics=(),
        )


def spacy_parser_adapter(model: str) -> SpacyParserAdapter:
    """Load a live spaCy model and build an adapter. Requires the parser-spacy extra."""
    from importlib.metadata import version

    try:
        import spacy
    except ImportError as exc:
        raise OptionalDependencyError("Live spaCy parsing requires the 'parser-spacy' extra (spacy)") from exc
    try:
        pipeline = spacy.load(model)
    except OSError as exc:
        raise OptionalDependencyError(f"spaCy model '{model}' is not installed; download it before live parsing") from exc
    return SpacyParserAdapter(pipeline=cast(SpacyPipelineLike, pipeline), model=model, version=version("spacy"))
