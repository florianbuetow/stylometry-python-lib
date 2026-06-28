"""Optional parser-backed and LLM-backed feature gates."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib._tabular import text_series, validate_output_mode
from stylometry_python_lib.errors import OptionalDependencyError
from stylometry_python_lib.llm import LLMClientProtocol
from stylometry_python_lib.llm_transformers import (
    ConfiguredLLMAnnotationSidecar,
    ConfiguredLLMAnnotationTransformer,
    configured_llm_annotation_transformer,
)
from stylometry_python_lib.registry import FeatureRegistry
from stylometry_python_lib.specs import FeatureSpec, InputLayer, StabilityStatus, TopicDependence
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus

_UNIVERSAL_POS_TAGS = (
    "ADJ",
    "ADP",
    "ADV",
    "AUX",
    "CCONJ",
    "DET",
    "INTJ",
    "NOUN",
    "NUM",
    "PART",
    "PRON",
    "PROPN",
    "PUNCT",
    "SCONJ",
    "SYM",
    "VERB",
    "X",
)

# Full Universal Dependencies v2 universal feature inventory plus the
# treebank-specific features that the spaCy and Stanza English models emit
# (ConjType, NumForm, Punct*, AdpType, PartType, Style, Hyph). Real parser
# output must validate without rejecting standard attributes such as Definite,
# PronType, VerbForm, or PunctType.
_MORPHOLOGY_ATTRIBUTES = (
    "Abbr",
    "AdpType",
    "Animacy",
    "Aspect",
    "Case",
    "Clusivity",
    "ConjType",
    "Definite",
    "Degree",
    "Evident",
    "Foreign",
    "Gender",
    "Hyph",
    "Mood",
    "NounClass",
    "NumForm",
    "NumType",
    "Number",
    "PartType",
    "Person",
    "Polarity",
    "Polite",
    "Poss",
    "PronType",
    "PunctSide",
    "PunctType",
    "Reflex",
    "Style",
    "Tense",
    "Typo",
    "VerbForm",
    "Voice",
)

_UNIVERSAL_DEPENDENCY_RELATIONS = (
    "acl",
    "advcl",
    "advmod",
    "amod",
    "appos",
    "aux",
    "case",
    "cc",
    "ccomp",
    "clf",
    "compound",
    "conj",
    "cop",
    "csubj",
    "dep",
    "det",
    "discourse",
    "dislocated",
    "expl",
    "fixed",
    "flat",
    "goeswith",
    "iobj",
    "list",
    "mark",
    "nmod",
    "nsubj",
    "nummod",
    "obj",
    "obl",
    "orphan",
    "parataxis",
    "punct",
    "reparandum",
    "root",
    "vocative",
    "xcomp",
)

_PARSER_DISTRIBUTION_STATISTICS = (
    "count",
    "mean",
    "sample_std",
    "sample_variance",
    "min",
    "max",
    "p10",
    "p25",
    "p50",
    "p75",
    "p90",
    "skewness",
    "excess_kurtosis",
    "shannon_entropy",
)

_DEPENDENCY_STRUCTURE_KINDS = (
    "dependency_ngram",
    "dependency_path",
    "dependency_subtree",
    "dependency_dtgram",
)

_SYNTACTIC_COMPLEXITY_METRICS = (
    "mls",
    "mlt",
    "mlc",
    "c_per_s",
    "c_per_t",
    "cp_per_t",
    "dc_per_c",
    "cn_per_c",
    "cn_per_t",
    "vp_per_t",
    "t_per_s",
)

_LEXICAL_DENSITY_UPOS_TAGS = ("ADJ", "ADV", "NOUN", "VERB")
_NAMED_ENTITY_DENSITY_SCOPES = ("style_adjacent", "content_control")


@dataclass(frozen=True)
class ParsedSyntacticCounts:
    """Parser-derived syntactic complexity counts for fake-provider fixtures."""

    document_id: str
    word_count: int
    sentence_count: int
    clause_count: int
    t_unit_count: int
    dependent_clause_count: int
    coordinate_phrase_count: int
    complex_nominal_count: int
    verb_phrase_count: int

    def __post_init__(self) -> None:
        """Validate fake-parser syntactic count annotations."""
        if self.document_id == "":
            raise ValueError("ParsedSyntacticCounts document_id must not be empty")
        count_fields = {
            "word_count": self.word_count,
            "sentence_count": self.sentence_count,
            "clause_count": self.clause_count,
            "t_unit_count": self.t_unit_count,
            "dependent_clause_count": self.dependent_clause_count,
            "coordinate_phrase_count": self.coordinate_phrase_count,
            "complex_nominal_count": self.complex_nominal_count,
            "verb_phrase_count": self.verb_phrase_count,
        }
        for field_name, value in count_fields.items():
            if value < 0:
                raise ValueError(f"ParsedSyntacticCounts {field_name} must be non-negative")
        if self.dependent_clause_count > self.clause_count:
            raise ValueError("ParsedSyntacticCounts dependent_clause_count must not exceed clause_count")


@dataclass(frozen=True)
class ParsedNamedEntity:
    """Parser/NER fake-provider entity span keyed to one input row."""

    document_id: str
    text: str
    label: str
    start_token_index: int
    end_token_index: int

    def __post_init__(self) -> None:
        """Validate fake named-entity annotations."""
        if self.document_id == "":
            raise ValueError("ParsedNamedEntity document_id must not be empty")
        if self.text == "":
            raise ValueError("ParsedNamedEntity text must not be empty")
        if self.label == "":
            raise ValueError("ParsedNamedEntity label must not be empty")
        if self.start_token_index < 0:
            raise ValueError("ParsedNamedEntity start_token_index must be non-negative")
        if self.end_token_index <= self.start_token_index:
            raise ValueError("ParsedNamedEntity end_token_index must be greater than start_token_index")


@dataclass(frozen=True)
class ParserContentMaskingSidecar:
    """Parser-backed content masking sidecar for one fake-provider document."""

    document_id: str
    schema_version: str
    masked_text: str
    token_count: int
    masked_token_count: int
    replacement_token: str
    mask_upos_tags: tuple[str, ...]
    named_entity_count: int


@dataclass(frozen=True)
class ParsedMorphologyFeature:
    """Parser token morphology fixture with a UD attribute/value pair."""

    attribute: str
    value: str

    def __post_init__(self) -> None:
        """Validate fake-parser morphology metadata."""
        if self.attribute not in _MORPHOLOGY_ATTRIBUTES:
            raise ValueError(f"Unsupported Universal Dependencies morphology attribute: {self.attribute}")
        if self.value == "":
            raise ValueError("ParsedMorphologyFeature value must not be empty")


@dataclass(frozen=True)
class ParsedToken:
    """Parser token fixture with Universal POS and morphology metadata."""

    text: str
    upos: str
    morphology: tuple[ParsedMorphologyFeature, ...]

    def __post_init__(self) -> None:
        """Validate fake-parser token metadata."""
        if self.text == "":
            raise ValueError("ParsedToken text must not be empty")
        if self.upos not in _UNIVERSAL_POS_TAGS:
            raise ValueError(f"Unsupported Universal POS tag: {self.upos}")


@dataclass(frozen=True)
class ParsedDependencyArc:
    """Parser dependency arc fixture with a UD relation label."""

    head_index: int | None
    dependent_index: int
    relation: str

    def __post_init__(self) -> None:
        """Validate fake-parser dependency metadata."""
        if self.head_index is not None and self.head_index < 0:
            raise ValueError("ParsedDependencyArc head_index must be non-negative or None")
        if self.dependent_index < 0:
            raise ValueError("ParsedDependencyArc dependent_index must be non-negative")
        if _dependency_relation_base(self.relation) not in _UNIVERSAL_DEPENDENCY_RELATIONS:
            raise ValueError(f"Unsupported Universal Dependencies relation label: {self.relation}")
        if self.relation == "root" and self.head_index is not None:
            raise ValueError("ParsedDependencyArc root relation must use head_index=None")
        if self.relation != "root" and self.head_index is None:
            raise ValueError("ParsedDependencyArc non-root relation must provide head_index")


@dataclass(frozen=True)
class ParsedDocument:
    """Fake parser output fixture keyed to one input row."""

    document_id: str
    tokens: tuple[ParsedToken, ...]
    dependency_arcs: tuple[ParsedDependencyArc, ...]

    def __post_init__(self) -> None:
        """Validate fake-parser document metadata."""
        if self.document_id == "":
            raise ValueError("ParsedDocument document_id must not be empty")
        _validate_dependency_arc_indexes(self)


@dataclass(frozen=True)
class FakeLLMAnnotation:
    """Offline fake LLM response fixture keyed to one input row."""

    document_id: str
    feature_values: tuple[tuple[str, float], ...]
    structured_response: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        """Validate fake LLM annotation metadata."""
        if self.document_id == "":
            raise ValueError("FakeLLMAnnotation document_id must not be empty")
        _validate_fake_llm_feature_values(self.feature_values)
        for key, value in self.structured_response:
            if key == "":
                raise ValueError("FakeLLMAnnotation structured_response keys must not be empty")
            if value == "":
                raise ValueError("FakeLLMAnnotation structured_response values must not be empty")


@dataclass(frozen=True)
class LLMAnnotationSidecar:
    """Structured fake LLM response sidecar for one document."""

    document_id: str
    schema_version: str
    prompt_version: str
    response_schema: str
    structured_response: tuple[tuple[str, str], ...]


type LLMAnnotationSidecarLike = LLMAnnotationSidecar | ConfiguredLLMAnnotationSidecar


class ParserBackedTransformer(BaseEstimator):
    """Parser-backed features with explicit dependency metadata."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        feature_names: tuple[str, ...],
        text_column: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.feature_names = feature_names
        self.text_column = text_column
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit parser-backed extraction metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
        specs = parser_pos_frequency_feature_specs(provider=self.provider, model=self.model, version=self.version)
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_pos_frequency_feature_names(), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fake-provider parser-backed features or fail for unavailable providers."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[dict[str, float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            row, row_diagnostics = _pos_frequency_row(self.parser_documents_[document_id])
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_.tolist(), index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser-backed metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return generated parser-backed output feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def feature_specs(self) -> tuple[FeatureSpec, ...]:
        """Return parser-backed feature metadata for discovery."""
        return tuple(
            FeatureSpec(
                name=name,
                family="parser_backed",
                description="Parser-backed stylometry feature requiring optional NLP tooling",
                formula_or_rule="computed by configured parser/tagger model",
                input_layer=InputLayer.NLP,
                topic_dependence=TopicDependence.MIXED,
                text_length_policy="requires parser-compatible text; short fragments may be unstable",
                provenance=f"provider={self.provider}; model={self.model}; version={self.version}",
                output_dtype="float64",
                undefined_behavior="dependency error unless optional parser provider is installed",
                normalization="feature_specific",
                sparsity="dense_or_sparse",
                stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
            )
            for name in self.feature_names
        )

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser-backed features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserPOSNGramTransformer(BaseEstimator):
    """Fitted POS n-gram features over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        ngram_range: tuple[int, int],
        max_features: int | None,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit a deterministic POS n-gram vocabulary from fake parser fixtures."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        _validate_ngram_range(self.ngram_range)
        _validate_max_features(self.max_features)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        vocabulary_counter: Counter[str] = Counter()
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            vocabulary_counter.update(_pos_ngrams(document_map[document_id], self.ngram_range))
        sorted_items = sorted(vocabulary_counter.items(), key=lambda item: (-item[1], item[0]))
        selected_items = _select_items(sorted_items, self.max_features)
        if len(selected_items) == 0:
            raise ValueError("No parser POS n-grams found for fitted corpus")
        self.vocabulary_ = {gram: vocabulary_index for vocabulary_index, (gram, _count) in enumerate(selected_items)}
        self.feature_names_out_ = np.asarray([_pos_ngram_feature_name(gram) for gram, _count in selected_items], dtype=object)
        specs = parser_pos_ngram_feature_specs(
            feature_names=tuple(str(name) for name in self.feature_names_out_.tolist()),
            provider=self.provider,
            model=self.model,
            version=self.version,
            ngram_range=self.ngram_range,
        )
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Transform fake parser fixtures into fitted POS n-gram count columns."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "vocabulary_")
        series = text_series(x, self.text_column)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row_index, index in enumerate(series.index):
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            counts = Counter(_pos_ngrams(self.parser_documents_[document_id], self.ngram_range))
            for gram, count in counts.items():
                if gram in self.vocabulary_:
                    rows.append(row_index)
                    cols.append(self.vocabulary_[gram])
                    data.append(float(count))
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(series), len(self.vocabulary_)), dtype=np.float64)
        if self.output == "sparse":
            return matrix
        if self.output == "pandas":
            return pd.DataFrame(matrix.toarray(), columns=self.feature_names_out_, index=series.index)
        return matrix.toarray()

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser POS n-gram metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable fitted parser POS n-gram feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser POS n-gram features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserPOSSkipGramTransformer(BaseEstimator):
    """Fitted POS skip-gram features over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        skip_distance: int,
        max_features: int | None,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.skip_distance = skip_distance
        self.max_features = max_features
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit a deterministic POS skip-gram vocabulary from fake parser fixtures."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        _validate_skip_distance(self.skip_distance)
        _validate_max_features(self.max_features)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        vocabulary_counter: Counter[str] = Counter()
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            vocabulary_counter.update(_pos_skipgrams(document_map[document_id], self.skip_distance))
        sorted_items = sorted(vocabulary_counter.items(), key=lambda item: (-item[1], item[0]))
        selected_items = _select_items(sorted_items, self.max_features)
        if len(selected_items) == 0:
            raise ValueError("No parser POS skip-grams found for fitted corpus")
        self.vocabulary_ = {gram: vocabulary_index for vocabulary_index, (gram, _count) in enumerate(selected_items)}
        self.feature_names_out_ = np.asarray([_pos_skipgram_feature_name(gram) for gram, _count in selected_items], dtype=object)
        specs = parser_pos_skipgram_feature_specs(
            feature_names=tuple(str(name) for name in self.feature_names_out_.tolist()),
            provider=self.provider,
            model=self.model,
            version=self.version,
            skip_distance=self.skip_distance,
        )
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Transform fake parser fixtures into fitted POS skip-gram count columns."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "vocabulary_")
        series = text_series(x, self.text_column)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row_index, index in enumerate(series.index):
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            counts = Counter(_pos_skipgrams(self.parser_documents_[document_id], self.skip_distance))
            for gram, count in counts.items():
                if gram in self.vocabulary_:
                    rows.append(row_index)
                    cols.append(self.vocabulary_[gram])
                    data.append(float(count))
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(series), len(self.vocabulary_)), dtype=np.float64)
        if self.output == "sparse":
            return matrix
        if self.output == "pandas":
            return pd.DataFrame(matrix.toarray(), columns=self.feature_names_out_, index=series.index)
        return matrix.toarray()

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser POS skip-gram metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable fitted parser POS skip-gram feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser POS skip-gram features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserMorphologyTransformer(BaseEstimator):
    """Fitted morphology attribute/value features over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit a deterministic morphology vocabulary from fake parser fixtures."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        vocabulary_counter: Counter[tuple[str, str]] = Counter()
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            vocabulary_counter.update(_morphology_items(document_map[document_id]))
        sorted_items = sorted(vocabulary_counter.items(), key=lambda item: (-item[1], f"{item[0][0]}={item[0][1]}"))
        if len(sorted_items) == 0:
            raise ValueError("No parser morphology attributes found for fitted corpus")
        self.vocabulary_ = {item: vocabulary_index for vocabulary_index, (item, _count) in enumerate(sorted_items)}
        self.feature_names_out_ = np.asarray(
            [_morphology_feature_name(attribute, value) for (attribute, value), _count in sorted_items],
            dtype=object,
        )
        specs = parser_morphology_feature_specs(
            feature_names=tuple(str(name) for name in self.feature_names_out_.tolist()),
            provider=self.provider,
            model=self.model,
            version=self.version,
        )
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Transform fake parser fixtures into fitted morphology count columns."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "vocabulary_")
        series = text_series(x, self.text_column)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row_index, index in enumerate(series.index):
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            counts = Counter(_morphology_items(self.parser_documents_[document_id]))
            for item, count in counts.items():
                if item in self.vocabulary_:
                    rows.append(row_index)
                    cols.append(self.vocabulary_[item])
                    data.append(float(count))
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(series), len(self.vocabulary_)), dtype=np.float64)
        if self.output == "sparse":
            return matrix
        if self.output == "pandas":
            return pd.DataFrame(matrix.toarray(), columns=self.feature_names_out_, index=series.index)
        return matrix.toarray()

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser morphology metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable fitted parser morphology feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser morphology features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserDependencyRelationTransformer(BaseEstimator):
    """Universal Dependencies relation-frequency features over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit dependency-relation extraction metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
        specs = parser_dependency_relation_feature_specs(provider=self.provider, model=self.model, version=self.version)
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_dependency_relation_feature_names(), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fake-provider dependency-relation frequency columns."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[dict[str, float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            row, row_diagnostics = _dependency_relation_row(self.parser_documents_[document_id])
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_.tolist(), index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser dependency-relation metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return generated dependency-relation output feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser dependency-relation features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserHeadDependentPOSPairTransformer(BaseEstimator):
    """Fitted head-dependent Universal POS pair features over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit a deterministic head-dependent POS pair vocabulary from fake parser fixtures."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        vocabulary_counter: Counter[tuple[str, str]] = Counter()
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            vocabulary_counter.update(_head_dependent_pos_pairs(document_map[document_id]))
        sorted_items = sorted(vocabulary_counter.items(), key=lambda item: (-item[1], _head_dependent_pos_pair_key(item[0])))
        if len(sorted_items) == 0:
            raise ValueError("No parser head-dependent POS pairs found for fitted corpus")
        self.vocabulary_ = {item: vocabulary_index for vocabulary_index, (item, _count) in enumerate(sorted_items)}
        self.feature_names_out_ = np.asarray(
            [_head_dependent_pos_pair_feature_name(head_upos, dependent_upos) for (head_upos, dependent_upos), _count in sorted_items],
            dtype=object,
        )
        specs = parser_head_dependent_pos_pair_feature_specs(
            feature_names=tuple(str(name) for name in self.feature_names_out_.tolist()),
            provider=self.provider,
            model=self.model,
            version=self.version,
        )
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Transform fake parser fixtures into fitted head-dependent POS pair count columns."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "vocabulary_")
        series = text_series(x, self.text_column)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row_index, index in enumerate(series.index):
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            counts = Counter(_head_dependent_pos_pairs(self.parser_documents_[document_id]))
            for item, count in counts.items():
                if item in self.vocabulary_:
                    rows.append(row_index)
                    cols.append(self.vocabulary_[item])
                    data.append(float(count))
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(series), len(self.vocabulary_)), dtype=np.float64)
        if self.output == "sparse":
            return matrix
        if self.output == "pandas":
            return pd.DataFrame(matrix.toarray(), columns=self.feature_names_out_, index=series.index)
        return matrix.toarray()

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser head-dependent POS pair metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable fitted parser head-dependent POS pair feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser head-dependent POS pair features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserDependencyDistanceTransformer(BaseEstimator):
    """Dependency distance distribution statistics over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit dependency-distance extraction metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
        specs = parser_dependency_distance_feature_specs(provider=self.provider, model=self.model, version=self.version)
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_dependency_distance_feature_names(), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fake-provider dependency-distance distribution statistics."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            row, row_diagnostics = _dependency_distance_row(self.parser_documents_[document_id])
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_, index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser dependency-distance metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable parser dependency-distance feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser dependency-distance features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserParseDepthTransformer(BaseEstimator):
    """Dependency parse-depth distribution statistics over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit parse-depth extraction metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
        specs = parser_parse_depth_feature_specs(provider=self.provider, model=self.model, version=self.version)
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_parse_depth_feature_names(), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fake-provider dependency parse-depth distribution statistics."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            row, row_diagnostics = _parse_depth_row(self.parser_documents_[document_id])
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_, index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser parse-depth metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable parser parse-depth feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser parse-depth features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserSyntacticComplexityTransformer(BaseEstimator):
    """Syntactic complexity, clause, T-unit, subordination, and coordination scalars."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        output: str,
        syntactic_counts: tuple[ParsedSyntacticCounts, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.output = output
        self.syntactic_counts = syntactic_counts

    def fit(self, x: object, y: object) -> Self:
        """Fit syntactic-complexity metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        counts_map = _syntactic_counts_map(self.syntactic_counts)
        for index in series.index:
            document_id = str(index)
            if document_id not in counts_map:
                raise ValueError(f"Missing fake syntactic counts for row id: {document_id}")
        specs = parser_syntactic_complexity_feature_specs(provider=self.provider, model=self.model, version=self.version)
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_syntactic_complexity_feature_names(), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.syntactic_counts_ = counts_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fake-provider syntactic-complexity and clause/T-unit scalars."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[dict[str, float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.syntactic_counts_:
                raise ValueError(f"Missing fake syntactic counts for row id: {document_id}")
            row, row_diagnostics = _syntactic_complexity_row(self.syntactic_counts_[document_id])
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_.tolist(), index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser syntactic-complexity metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable syntactic-complexity feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.syntactic_counts is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser syntactic-complexity features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserPOSLexicalDensityTransformer(BaseEstimator):
    """Universal POS lexical-density scalar over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit POS lexical-density metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
        specs = parser_pos_lexical_density_feature_specs(provider=self.provider, model=self.model, version=self.version)
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_pos_lexical_density_feature_names(), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fake-provider POS lexical-density scalar."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[dict[str, float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            row, row_diagnostics = _pos_lexical_density_row(self.parser_documents_[document_id])
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_.tolist(), index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser POS lexical-density metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable POS lexical-density feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser POS lexical-density features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserPassiveVoiceTransformer(BaseEstimator):
    """Passive voice frequency over fake dependency and syntactic-count fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
        syntactic_counts: tuple[ParsedSyntacticCounts, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.output = output
        self.parsed_documents = parsed_documents
        self.syntactic_counts = syntactic_counts

    def fit(self, x: object, y: object) -> Self:
        """Fit passive-voice metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        counts_map = _syntactic_counts_map(self.syntactic_counts)
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            if document_id not in counts_map:
                raise ValueError(f"Missing fake syntactic counts for row id: {document_id}")
        specs = parser_passive_voice_feature_specs(provider=self.provider, model=self.model, version=self.version)
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_passive_voice_feature_names(), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.syntactic_counts_ = counts_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fake-provider passive voice frequency."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[dict[str, float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            if document_id not in self.syntactic_counts_:
                raise ValueError(f"Missing fake syntactic counts for row id: {document_id}")
            row, row_diagnostics = _passive_voice_row(
                self.parser_documents_[document_id],
                self.syntactic_counts_[document_id],
            )
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_.tolist(), index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser passive-voice metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable passive-voice feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None and self.syntactic_counts is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser passive-voice features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserNamedEntityDensityTransformer(BaseEstimator):
    """Named-entity density vectors over fake parser/NER fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        entity_types: tuple[str, ...],
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
        named_entities: tuple[ParsedNamedEntity, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.entity_types = entity_types
        self.output = output
        self.parsed_documents = parsed_documents
        self.named_entities = named_entities

    def fit(self, x: object, y: object) -> Self:
        """Fit named-entity density metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        _validate_entity_types(self.entity_types)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        entity_map = _named_entity_map(self.named_entities)
        for document_id in entity_map:
            if document_id not in document_map:
                raise ValueError(f"Fake named entity document id has no parsed document: {document_id}")
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            entities = _entities_for_document(entity_map, document_id)
            for entity in entities:
                if entity.label not in self.entity_types:
                    raise ValueError(f"Fake named entity label not configured: {entity.label}")
                _validate_named_entity_span(document_map[document_id], entity)
        specs = parser_named_entity_density_feature_specs(
            entity_types=self.entity_types,
            provider=self.provider,
            model=self.model,
            version=self.version,
        )
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_named_entity_density_feature_names(self.entity_types), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.named_entities_ = entity_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fake-provider named-entity density columns."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[dict[str, float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            entities = _entities_for_document(self.named_entities_, document_id)
            row, row_diagnostics = _named_entity_density_row(
                self.parser_documents_[document_id],
                entities,
                self.entity_types,
            )
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_.tolist(), index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser named-entity density metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable named-entity density feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None and self.named_entities is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser named-entity density features require an installed parser or NER extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserContentMaskingTransformer(BaseEstimator):
    """Parser-backed content masking and topic-neutral distortion over fake fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        mask_upos_tags: tuple[str, ...],
        replacement_token: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
        named_entities: tuple[ParsedNamedEntity, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.mask_upos_tags = mask_upos_tags
        self.replacement_token = replacement_token
        self.output = output
        self.parsed_documents = parsed_documents
        self.named_entities = named_entities

    def fit(self, x: object, y: object) -> Self:
        """Fit parser-backed content masking metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        _validate_mask_upos_tags(self.mask_upos_tags)
        _validate_replacement_token(self.replacement_token)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        entity_map = _named_entity_map(self.named_entities)
        for document_id in entity_map:
            if document_id not in document_map:
                raise ValueError(f"Fake named entity document id has no parsed document: {document_id}")
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            entities = _entities_for_document(entity_map, document_id)
            for entity in entities:
                _validate_named_entity_span(document_map[document_id], entity)
        specs = parser_content_masking_feature_specs(provider=self.provider, model=self.model, version=self.version)
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_content_masking_feature_names(), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.named_entities_ = entity_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute parser-backed topic-neutral distortion ratio and masked-text sidecars."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[dict[str, float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[ParserContentMaskingSidecar] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            entities = _entities_for_document(self.named_entities_, document_id)
            row, row_diagnostics, sidecar = _content_masking_row(
                self.parser_documents_[document_id],
                entities,
                self.mask_upos_tags,
                self.replacement_token,
            )
            rows.append(row)
            diagnostics.append(row_diagnostics)
            sidecars.append(sidecar)
        self.last_diagnostics_ = tuple(diagnostics)
        self.last_sidecars_ = tuple(sidecars)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_.tolist(), index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser-backed content masking metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable content-masking feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None and self.named_entities is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser content-masking features require an installed parser or NER extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserRootStatisticsTransformer(BaseEstimator):
    """Root node statistics over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit root-statistics extraction metadata or fail for unavailable providers."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
        specs = parser_root_statistics_feature_specs(provider=self.provider, model=self.model, version=self.version)
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(parser_root_statistics_feature_names(), dtype=object)
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fake-provider root statistics."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[dict[str, float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for index in series.index:
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            row, row_diagnostics = _root_statistics_row(self.parser_documents_[document_id])
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_.tolist(), index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser root-statistics metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable parser root-statistics feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser root-statistics features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


class ParserDependencyStructureTransformer(BaseEstimator):
    """Fitted dependency n-gram, path, subtree, and DT-gram features over fake parser fixtures."""

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        text_column: str,
        ngram_range: tuple[int, int],
        max_features_per_kind: int | None,
        output: str,
        parsed_documents: tuple[ParsedDocument, ...] | None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.text_column = text_column
        self.ngram_range = ngram_range
        self.max_features_per_kind = max_features_per_kind
        self.output = output
        self.parsed_documents = parsed_documents

    def fit(self, x: object, y: object) -> Self:
        """Fit deterministic dependency-structure vocabularies from fake parser fixtures."""
        del y
        if not self._uses_fake_provider():
            raise self._dependency_error()
        validate_output_mode(self.output)
        _validate_ngram_range(self.ngram_range)
        _validate_max_features(self.max_features_per_kind)
        series = text_series(x, self.text_column)
        document_map = _parsed_document_map(self.parsed_documents)
        counters = {kind: Counter[str]() for kind in _DEPENDENCY_STRUCTURE_KINDS}
        for index in series.index:
            document_id = str(index)
            if document_id not in document_map:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            structure_items = _dependency_structure_items(document_map[document_id], self.ngram_range)
            for kind in _DEPENDENCY_STRUCTURE_KINDS:
                counters[kind].update(structure_items[kind])
        selected_items: list[tuple[str, str]] = []
        for kind in _DEPENDENCY_STRUCTURE_KINDS:
            sorted_items = sorted(counters[kind].items(), key=lambda item: (-item[1], item[0]))
            selected_items.extend((kind, item) for item, _count in _select_items(sorted_items, self.max_features_per_kind))
        if len(selected_items) == 0:
            raise ValueError("No parser dependency structures found for fitted corpus")
        self.vocabulary_ = {item: vocabulary_index for vocabulary_index, item in enumerate(selected_items)}
        self.feature_names_out_ = np.asarray(
            [_dependency_structure_feature_name(kind, item) for kind, item in selected_items], dtype=object
        )
        specs = parser_dependency_structure_feature_specs(
            feature_names=tuple(str(name) for name in self.feature_names_out_.tolist()),
            provider=self.provider,
            model=self.model,
            version=self.version,
            ngram_range=self.ngram_range,
        )
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        self.n_features_in_ = 1
        self.parser_documents_ = document_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Transform fake parser fixtures into fitted dependency-structure count columns."""
        if not self._uses_fake_provider():
            raise self._dependency_error()
        require_fitted(self, "vocabulary_")
        series = text_series(x, self.text_column)
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        for row_index, index in enumerate(series.index):
            document_id = str(index)
            if document_id not in self.parser_documents_:
                raise ValueError(f"Missing fake parser document for row id: {document_id}")
            structure_items = _dependency_structure_items(self.parser_documents_[document_id], self.ngram_range)
            counts: Counter[tuple[str, str]] = Counter()
            for kind in _DEPENDENCY_STRUCTURE_KINDS:
                counts.update((kind, item) for item in structure_items[kind])
            for item, count in counts.items():
                if item in self.vocabulary_:
                    rows.append(row_index)
                    cols.append(self.vocabulary_[item])
                    data.append(float(count))
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(series), len(self.vocabulary_)), dtype=np.float64)
        if self.output == "sparse":
            return matrix
        if self.output == "pandas":
            return pd.DataFrame(matrix.toarray(), columns=self.feature_names_out_, index=series.index)
        return matrix.toarray()

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with parser dependency-structure metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable fitted parser dependency-structure feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.parsed_documents is not None

    def _dependency_error(self) -> OptionalDependencyError:
        message = (
            "Parser dependency-structure features require an installed parser extra; "
            f"provider={self.provider}, model={self.model}, version={self.version}"
        )
        return OptionalDependencyError(message)


def parser_pos_frequency_feature_names() -> tuple[str, ...]:
    """Return generated Universal POS frequency feature names."""
    return tuple(f"text::syntax::pos_frequency::upos={upos}" for upos in _UNIVERSAL_POS_TAGS)


def parser_dependency_relation_feature_names() -> tuple[str, ...]:
    """Return generated Universal Dependencies relation-frequency feature names."""
    return tuple(f"text::syntax::dependency_relation_frequency::deprel={relation}" for relation in _UNIVERSAL_DEPENDENCY_RELATIONS)


def parser_dependency_distance_feature_names() -> tuple[str, ...]:
    """Return dependency-distance distribution-statistic feature names."""
    return tuple(f"text::syntax::dependency_distance_{statistic}" for statistic in _PARSER_DISTRIBUTION_STATISTICS)


def parser_parse_depth_feature_names() -> tuple[str, ...]:
    """Return dependency parse-depth distribution-statistic feature names."""
    return tuple(f"text::syntax::parse_depth_{statistic}" for statistic in _PARSER_DISTRIBUTION_STATISTICS)


def parser_syntactic_complexity_metric_feature_names() -> tuple[str, ...]:
    """Return classic syntactic-complexity metric feature names."""
    return tuple(f"text::syntax::syntactic_complexity::{metric}" for metric in _SYNTACTIC_COMPLEXITY_METRICS)


def parser_syntactic_count_feature_names() -> tuple[str, ...]:
    """Return parser-backed clause and T-unit count feature names."""
    return ("text::syntax::clause_count", "text::syntax::t_unit_count")


def parser_subordination_feature_names() -> tuple[str, ...]:
    """Return parser-backed subordination feature names."""
    return ("text::syntax::subordination_ratio",)


def parser_coordination_feature_names() -> tuple[str, ...]:
    """Return parser-backed coordination feature names."""
    return ("text::syntax::coordination_ratio",)


def parser_syntactic_complexity_feature_names() -> tuple[str, ...]:
    """Return parser-backed syntactic-complexity, count, subordination, and coordination names."""
    return (
        *parser_syntactic_complexity_metric_feature_names(),
        *parser_syntactic_count_feature_names(),
        *parser_subordination_feature_names(),
        *parser_coordination_feature_names(),
    )


def parser_pos_lexical_density_feature_names() -> tuple[str, ...]:
    """Return parser-backed POS lexical-density feature names."""
    return ("text::syntax::pos_lexical_density",)


def parser_passive_voice_feature_names() -> tuple[str, ...]:
    """Return parser-backed passive voice feature names."""
    return ("text::syntax::passive_voice_frequency",)


def parser_named_entity_density_feature_names(entity_types: tuple[str, ...]) -> tuple[str, ...]:
    """Return parser-backed named-entity density feature names for configured entity types."""
    _validate_entity_types(entity_types)
    return tuple(
        f"text::{scope}::named_entity_density::entity_type={entity_type}"
        for scope in _NAMED_ENTITY_DENSITY_SCOPES
        for entity_type in entity_types
    )


def parser_content_masking_feature_names() -> tuple[str, ...]:
    """Return parser-backed topic-neutral distortion feature names."""
    return ("text::content_control::topic_neutral_distortion::masked_token_ratio",)


def parser_root_statistics_feature_names() -> tuple[str, ...]:
    """Return root-statistics feature names."""
    return (
        "text::syntax::root_statistics::root_count",
        "text::syntax::root_statistics::root_per_token",
        *(f"text::syntax::root_statistics::root_upos_ratio::upos={upos}" for upos in _UNIVERSAL_POS_TAGS),
    )


def parser_pos_frequency_feature_specs(provider: str, model: str, version: str) -> tuple[FeatureSpec, ...]:
    """Return generated Universal POS frequency metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_pos_frequency",
            description=f"Universal POS frequency for {name}",
            formula_or_rule="count Universal POS tag divided by parsed token count",
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MIXED,
            text_length_policy="NaN when parser token denominator is zero; short fragments may be unstable",
            provenance=(
                f"provider={provider}; model={model}; version={version}; tagset=Universal POS; "
                "tokenizer=fake_parser_fixture; preprocessing_config"
            ),
            output_dtype="float64",
            undefined_behavior="NaN with FeatureDiagnostic reason zero_parser_tokens",
            normalization="ratio",
            sparsity="dense_or_sparse",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for name in parser_pos_frequency_feature_names()
    )


def parser_dependency_distance_feature_specs(provider: str, model: str, version: str) -> tuple[FeatureSpec, ...]:
    """Return dependency-distance distribution-statistic metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_dependency_distance",
            description=f"Dependency distance distribution statistic: {statistic}",
            formula_or_rule=_parser_distribution_formula(statistic),
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
            text_length_policy=(
                "count is always defined; continuous statistics require at least one non-root dependency arc; "
                "sample and moment statistics require two non-root dependency arcs"
            ),
            provenance=(
                f"provider={provider}; model={model}; version={version}; dependency_schema=Universal Dependencies; "
                "distance=abs(head_index-dependent_index); std_type=sample; percentile_method=numpy_default_linear; "
                "moment_statistics=population_central_moments; entropy_log_base=e; tokenizer=fake_parser_fixture"
            ),
            output_dtype="float64",
            undefined_behavior=_parser_distribution_undefined_behavior(statistic, "zero_dependency_distances"),
            normalization="distribution_statistic",
            sparsity="dense_scalar",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for statistic, name in zip(_PARSER_DISTRIBUTION_STATISTICS, parser_dependency_distance_feature_names(), strict=True)
    )


def parser_parse_depth_feature_specs(provider: str, model: str, version: str) -> tuple[FeatureSpec, ...]:
    """Return dependency parse-depth distribution-statistic metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_parse_depth",
            description=f"Dependency parse-depth distribution statistic: {statistic}",
            formula_or_rule=_parse_depth_distribution_formula(statistic),
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
            text_length_policy=(
                "count is always defined; continuous statistics require at least one dependency parse depth; "
                "sample and moment statistics require two dependency parse depths"
            ),
            provenance=(
                f"provider={provider}; model={model}; version={version}; dependency_schema=Universal Dependencies; "
                "root_depth=0; depth=edge_count_to_root; std_type=sample; percentile_method=numpy_default_linear; "
                "moment_statistics=population_central_moments; entropy_log_base=e; tokenizer=fake_parser_fixture"
            ),
            output_dtype="float64",
            undefined_behavior=_parser_distribution_undefined_behavior(statistic, "zero_parse_depths"),
            normalization="distribution_statistic",
            sparsity="dense_scalar",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for statistic, name in zip(_PARSER_DISTRIBUTION_STATISTICS, parser_parse_depth_feature_names(), strict=True)
    )


def parser_syntactic_complexity_feature_specs(provider: str, model: str, version: str) -> tuple[FeatureSpec, ...]:
    """Return syntactic-complexity and clause/T-unit metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family=_syntactic_complexity_family(name),
            description=_syntactic_complexity_description(name),
            formula_or_rule=_syntactic_complexity_formula(name),
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MIXED,
            text_length_policy="counts are defined; ratios are NaN when their sentence, clause, or T-unit denominator is zero",
            provenance=(
                f"provider={provider}; model={model}; version={version}; annotation_source=fake_syntactic_counts; "
                "metrics=MLS,MLT,MLC,C/S,C/T,CP/T,DC/C,CN/C,CN/T,VP/T,T/S"
            ),
            output_dtype="float64",
            undefined_behavior=_syntactic_complexity_undefined_behavior(name),
            normalization=_syntactic_complexity_normalization(name),
            sparsity="dense_scalar",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for name in parser_syntactic_complexity_feature_names()
    )


def parser_pos_lexical_density_feature_specs(provider: str, model: str, version: str) -> tuple[FeatureSpec, ...]:
    """Return POS lexical-density metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_pos_lexical_density",
            description="Universal POS content-token ratio",
            formula_or_rule="count parser tokens whose Universal POS tag is ADJ, ADV, NOUN, or VERB divided by parser token count",
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MIXED,
            text_length_policy="NaN when parser token denominator is zero; short fragments may be unstable",
            provenance=(
                f"provider={provider}; model={model}; version={version}; tagset=Universal POS; "
                "content_upos=ADJ,ADV,NOUN,VERB; tokenizer=fake_parser_fixture"
            ),
            output_dtype="float64",
            undefined_behavior="NaN with FeatureDiagnostic reason zero_parser_tokens",
            normalization="ratio",
            sparsity="dense_scalar",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for name in parser_pos_lexical_density_feature_names()
    )


def parser_passive_voice_feature_specs(provider: str, model: str, version: str) -> tuple[FeatureSpec, ...]:
    """Return passive voice metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_passive_voice",
            description="Passive voice constructions divided by parser-derived clause count",
            formula_or_rule=(
                "count predicate heads with Voice=Pass morphology plus aux:pass and nsubj:pass dependents divided by clause_count"
            ),
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MIXED,
            text_length_policy="NaN when clause_count denominator is zero; short fragments may be unstable",
            provenance=(
                f"provider={provider}; model={model}; version={version}; dependency_schema=Universal Dependencies; "
                "morphology_schema=Universal Dependencies; rule=Voice=Pass+aux:pass+nsubj:pass; "
                "denominator=clause_count; tokenizer=fake_parser_fixture"
            ),
            output_dtype="float64",
            undefined_behavior="NaN with FeatureDiagnostic reason zero_clause_count",
            normalization="ratio",
            sparsity="dense_scalar",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for name in parser_passive_voice_feature_names()
    )


def parser_named_entity_density_feature_specs(
    entity_types: tuple[str, ...],
    provider: str,
    model: str,
    version: str,
) -> tuple[FeatureSpec, ...]:
    """Return named-entity density metadata for configured entity types."""
    return tuple(
        FeatureSpec(
            name=name,
            family=_named_entity_density_family(name),
            description="Named-entity label density exposed under an intent-specific namespace",
            formula_or_rule="count configured named-entity spans for the requested label divided by parser token count",
            input_layer=InputLayer.NLP,
            topic_dependence=_named_entity_density_topic_dependence(name),
            text_length_policy="NaN when parser token denominator is zero; short fragments may be unstable",
            provenance=(
                f"provider={provider}; model={model}; version={version}; entity_types={','.join(entity_types)}; "
                "annotation_source=fake_named_entities; denominator=parser_token_count"
            ),
            output_dtype="float64",
            undefined_behavior="NaN with FeatureDiagnostic reason zero_parser_tokens",
            normalization="ratio",
            sparsity="dense_vector",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for name in parser_named_entity_density_feature_names(entity_types)
    )


def parser_content_masking_feature_specs(provider: str, model: str, version: str) -> tuple[FeatureSpec, ...]:
    """Return parser-backed content masking metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_content_masking_topic_neutral_distortion",
            description="Parser-backed masked-token ratio with masked text sidecar",
            formula_or_rule=(
                "unique parser tokens masked by configured Universal POS tags or named-entity spans divided by parser token count"
            ),
            input_layer=InputLayer.MULTI,
            topic_dependence=TopicDependence.TOPIC_CONTROL,
            text_length_policy="NaN when parser token denominator is zero; masked text sidecar is always emitted",
            provenance=(
                f"provider={provider}; model={model}; version={version}; mask_source=Universal POS plus fake named entities; "
                "sidecar_schema=parser_content_masking_v1"
            ),
            output_dtype="float64",
            undefined_behavior="NaN with FeatureDiagnostic reason zero_parser_tokens",
            normalization="ratio",
            sparsity="dense_scalar",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for name in parser_content_masking_feature_names()
    )


def parser_dependency_relation_feature_specs(provider: str, model: str, version: str) -> tuple[FeatureSpec, ...]:
    """Return generated Universal Dependencies relation-frequency metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_dependency_relation_frequency",
            description=f"Universal Dependencies relation frequency for {name}",
            formula_or_rule="count Universal Dependencies basic relation label divided by dependency arc count",
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MIXED,
            text_length_policy="NaN when dependency arc denominator is zero; short fragments may be unstable",
            provenance=(
                f"provider={provider}; model={model}; version={version}; deprel_schema=Universal Dependencies basic relations; "
                "tokenizer=fake_parser_fixture; preprocessing_config"
            ),
            output_dtype="float64",
            undefined_behavior="NaN with FeatureDiagnostic reason zero_dependency_arcs",
            normalization="ratio",
            sparsity="dense_or_sparse",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for name in parser_dependency_relation_feature_names()
    )


def parser_root_statistics_feature_specs(provider: str, model: str, version: str) -> tuple[FeatureSpec, ...]:
    """Return root-statistics metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_root_statistics",
            description=_root_statistics_description(name),
            formula_or_rule=_root_statistics_formula(name),
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
            text_length_policy="root count is always defined; root-per-token requires parsed tokens; root-UPOS ratios require root arcs",
            provenance=(
                f"provider={provider}; model={model}; version={version}; tagset=Universal POS; "
                "dependency_schema=Universal Dependencies; tokenizer=fake_parser_fixture"
            ),
            output_dtype="float64",
            undefined_behavior=_root_statistics_undefined_behavior(name),
            normalization="ratio_or_count",
            sparsity="dense_scalar",
            stability_status=StabilityStatus.PARSER_MODEL_DEPENDENT,
        )
        for name in parser_root_statistics_feature_names()
    )


def parser_dependency_structure_feature_specs(
    feature_names: tuple[str, ...],
    provider: str,
    model: str,
    version: str,
    ngram_range: tuple[int, int],
) -> tuple[FeatureSpec, ...]:
    """Return fitted dependency-structure metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_dependency_structures",
            description="Fitted parser dependency n-gram, path, subtree, or DT-gram count feature",
            formula_or_rule="count fitted dependency structure over Universal Dependencies fake-parser tree",
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MIXED,
            text_length_policy="defined as zero when a fitted dependency structure is absent from a parsed document",
            provenance=(
                f"provider={provider}; model={model}; version={version}; dependency_schema=Universal Dependencies; "
                f"ngram_range={ngram_range[0]}-{ngram_range[1]}; tokenizer=fake_parser_fixture; fitted_vocabulary"
            ),
            output_dtype="float64",
            undefined_behavior="not undefined after fit; absent fitted dependency structures produce valid zero counts",
            normalization="raw_count",
            sparsity="sparse_vector",
            stability_status=StabilityStatus.STATISTICAL_FIT_DEPENDENT,
        )
        for name in feature_names
    )


def parser_head_dependent_pos_pair_feature_specs(
    feature_names: tuple[str, ...],
    provider: str,
    model: str,
    version: str,
) -> tuple[FeatureSpec, ...]:
    """Return fitted head-dependent Universal POS pair metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_head_dependent_pos_pair_frequency",
            description="Fitted parser head-dependent Universal POS pair count feature",
            formula_or_rule="count fitted Universal POS head/dependent pair over non-root dependency arcs",
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MIXED,
            text_length_policy="defined as zero when a fitted head-dependent POS pair is absent from a parsed document",
            provenance=(
                f"provider={provider}; model={model}; version={version}; tagset=Universal POS; "
                "dependency_schema=Universal Dependencies; tokenizer=fake_parser_fixture; fitted_vocabulary"
            ),
            output_dtype="float64",
            undefined_behavior="not undefined after fit; absent fitted head-dependent POS pairs produce valid zero counts",
            normalization="raw_count",
            sparsity="sparse_vector",
            stability_status=StabilityStatus.STATISTICAL_FIT_DEPENDENT,
        )
        for name in feature_names
    )


def parser_pos_ngram_feature_specs(
    feature_names: tuple[str, ...],
    provider: str,
    model: str,
    version: str,
    ngram_range: tuple[int, int],
) -> tuple[FeatureSpec, ...]:
    """Return fitted Universal POS n-gram metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_pos_ngram",
            description="Fitted parser POS n-gram count feature",
            formula_or_rule="count fitted contiguous Universal POS n-gram in the parser token sequence",
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
            text_length_policy="defined as zero when a fitted POS n-gram is absent from a parsed document",
            provenance=(
                f"provider={provider}; model={model}; version={version}; tagset=Universal POS; "
                f"ngram_range={ngram_range[0]}-{ngram_range[1]}; tokenizer=fake_parser_fixture; fitted_vocabulary"
            ),
            output_dtype="float64",
            undefined_behavior="not undefined after fit; absent fitted grams produce valid zero counts",
            normalization="raw_count",
            sparsity="sparse_vector",
            stability_status=StabilityStatus.STATISTICAL_FIT_DEPENDENT,
        )
        for name in feature_names
    )


def parser_pos_skipgram_feature_specs(
    feature_names: tuple[str, ...],
    provider: str,
    model: str,
    version: str,
    skip_distance: int,
) -> tuple[FeatureSpec, ...]:
    """Return fitted Universal POS skip-gram metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_pos_skipgram",
            description="Fitted parser POS skip-gram count feature",
            formula_or_rule="count fitted non-contiguous Universal POS skip-bigram in the parser token sequence",
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
            text_length_policy="defined as zero when a fitted POS skip-gram is absent from a parsed document",
            provenance=(
                f"provider={provider}; model={model}; version={version}; tagset=Universal POS; "
                f"skip_distance={skip_distance}; tokenizer=fake_parser_fixture; fitted_vocabulary"
            ),
            output_dtype="float64",
            undefined_behavior="not undefined after fit; absent fitted skip-grams produce valid zero counts",
            normalization="raw_count",
            sparsity="sparse_vector",
            stability_status=StabilityStatus.STATISTICAL_FIT_DEPENDENT,
        )
        for name in feature_names
    )


def parser_morphology_feature_specs(
    feature_names: tuple[str, ...],
    provider: str,
    model: str,
    version: str,
) -> tuple[FeatureSpec, ...]:
    """Return fitted Universal Dependencies morphology metadata."""
    return tuple(
        FeatureSpec(
            name=name,
            family="parser_morphology_frequency",
            description="Fitted parser morphology attribute/value count feature",
            formula_or_rule="count fitted Universal Dependencies morphology attribute/value observation in the parser token sequence",
            input_layer=InputLayer.NLP,
            topic_dependence=TopicDependence.MIXED,
            text_length_policy="defined as zero when a fitted morphology attribute/value is absent from a parsed document",
            provenance=(
                f"provider={provider}; model={model}; version={version}; morphology_schema=Universal Dependencies; "
                "tokenizer=fake_parser_fixture; fitted_vocabulary"
            ),
            output_dtype="float64",
            undefined_behavior="not undefined after fit; absent fitted morphology attributes produce valid zero counts",
            normalization="raw_count",
            sparsity="sparse_vector",
            stability_status=StabilityStatus.STATISTICAL_FIT_DEPENDENT,
        )
        for name in feature_names
    )


def _parsed_document_map(parsed_documents: tuple[ParsedDocument, ...] | None) -> dict[str, ParsedDocument]:
    if parsed_documents is None:
        raise ValueError("Fake parser provider requires parsed_documents")
    document_map: dict[str, ParsedDocument] = {}
    for document in parsed_documents:
        if document.document_id in document_map:
            raise ValueError(f"Duplicate fake parser document id: {document.document_id}")
        document_map[document.document_id] = document
    return document_map


def _syntactic_counts_map(syntactic_counts: tuple[ParsedSyntacticCounts, ...] | None) -> dict[str, ParsedSyntacticCounts]:
    if syntactic_counts is None:
        raise ValueError("Fake parser provider requires syntactic_counts")
    counts_map: dict[str, ParsedSyntacticCounts] = {}
    for counts in syntactic_counts:
        if counts.document_id in counts_map:
            raise ValueError(f"Duplicate fake syntactic counts document id: {counts.document_id}")
        counts_map[counts.document_id] = counts
    return counts_map


def _named_entity_map(named_entities: tuple[ParsedNamedEntity, ...] | None) -> dict[str, tuple[ParsedNamedEntity, ...]]:
    if named_entities is None:
        raise ValueError("Fake parser provider requires named_entities")
    entity_map: dict[str, list[ParsedNamedEntity]] = {}
    for entity in named_entities:
        entity_map.setdefault(entity.document_id, []).append(entity)
    return {document_id: tuple(entities) for document_id, entities in entity_map.items()}


def _entities_for_document(entity_map: dict[str, tuple[ParsedNamedEntity, ...]], document_id: str) -> tuple[ParsedNamedEntity, ...]:
    try:
        return entity_map[document_id]
    except KeyError:
        return ()


def _pos_frequency_row(document: ParsedDocument) -> tuple[dict[str, float], tuple[FeatureDiagnostic, ...]]:
    token_count = len(document.tokens)
    if token_count == 0:
        row = {name: float("nan") for name in parser_pos_frequency_feature_names()}
        diagnostics = tuple(
            FeatureDiagnostic(feature_name=name, status=FeatureStatus.UNDEFINED, reason="zero_parser_tokens", warnings=())
            for name in parser_pos_frequency_feature_names()
        )
        return row, diagnostics
    counts = Counter(token.upos for token in document.tokens)
    row = {f"text::syntax::pos_frequency::upos={upos}": float(counts[upos]) / float(token_count) for upos in _UNIVERSAL_POS_TAGS}
    return row, ()


def _dependency_relation_row(document: ParsedDocument) -> tuple[dict[str, float], tuple[FeatureDiagnostic, ...]]:
    arc_count = len(document.dependency_arcs)
    if arc_count == 0:
        row = {name: float("nan") for name in parser_dependency_relation_feature_names()}
        diagnostics = tuple(
            FeatureDiagnostic(feature_name=name, status=FeatureStatus.UNDEFINED, reason="zero_dependency_arcs", warnings=())
            for name in parser_dependency_relation_feature_names()
        )
        return row, diagnostics
    counts = Counter(_dependency_relation_base(arc.relation) for arc in document.dependency_arcs)
    row = {
        f"text::syntax::dependency_relation_frequency::deprel={relation}": float(counts[relation]) / float(arc_count)
        for relation in _UNIVERSAL_DEPENDENCY_RELATIONS
    }
    return row, ()


def _dependency_distance_row(document: ParsedDocument) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    distances = _dependency_distances(document)
    values: list[float] = []
    diagnostics: list[FeatureDiagnostic] = []
    for statistic, feature_name in zip(_PARSER_DISTRIBUTION_STATISTICS, parser_dependency_distance_feature_names(), strict=True):
        value, diagnostic = _parser_distribution_statistic_value(
            feature_name=feature_name,
            statistic=statistic,
            values=distances,
            zero_reason="zero_dependency_distances",
        )
        values.append(value)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return values, tuple(diagnostics)


def _parse_depth_row(document: ParsedDocument) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    depths = _parse_depths(document)
    values: list[float] = []
    diagnostics: list[FeatureDiagnostic] = []
    for statistic, feature_name in zip(_PARSER_DISTRIBUTION_STATISTICS, parser_parse_depth_feature_names(), strict=True):
        value, diagnostic = _parser_distribution_statistic_value(
            feature_name=feature_name,
            statistic=statistic,
            values=depths,
            zero_reason="zero_parse_depths",
        )
        values.append(value)
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    return values, tuple(diagnostics)


def _syntactic_complexity_row(counts: ParsedSyntacticCounts) -> tuple[dict[str, float], tuple[FeatureDiagnostic, ...]]:
    diagnostics: list[FeatureDiagnostic] = []
    metric_inputs = {
        "mls": (counts.word_count, counts.sentence_count, "zero_sentence_count"),
        "mlt": (counts.word_count, counts.t_unit_count, "zero_t_unit_count"),
        "mlc": (counts.word_count, counts.clause_count, "zero_clause_count"),
        "c_per_s": (counts.clause_count, counts.sentence_count, "zero_sentence_count"),
        "c_per_t": (counts.clause_count, counts.t_unit_count, "zero_t_unit_count"),
        "cp_per_t": (counts.coordinate_phrase_count, counts.t_unit_count, "zero_t_unit_count"),
        "dc_per_c": (counts.dependent_clause_count, counts.clause_count, "zero_clause_count"),
        "cn_per_c": (counts.complex_nominal_count, counts.clause_count, "zero_clause_count"),
        "cn_per_t": (counts.complex_nominal_count, counts.t_unit_count, "zero_t_unit_count"),
        "vp_per_t": (counts.verb_phrase_count, counts.t_unit_count, "zero_t_unit_count"),
        "t_per_s": (counts.t_unit_count, counts.sentence_count, "zero_sentence_count"),
    }
    row: dict[str, float] = {}
    for metric in _SYNTACTIC_COMPLEXITY_METRICS:
        numerator, denominator, zero_reason = metric_inputs[metric]
        feature_name = f"text::syntax::syntactic_complexity::{metric}"
        value, diagnostic = _parser_ratio(feature_name, numerator, denominator, zero_reason)
        row[feature_name] = value
        if diagnostic is not None:
            diagnostics.append(diagnostic)
    row["text::syntax::clause_count"] = float(counts.clause_count)
    row["text::syntax::t_unit_count"] = float(counts.t_unit_count)
    subordination_value, subordination_diagnostic = _parser_ratio(
        "text::syntax::subordination_ratio",
        counts.dependent_clause_count,
        counts.clause_count,
        "zero_clause_count",
    )
    row["text::syntax::subordination_ratio"] = subordination_value
    if subordination_diagnostic is not None:
        diagnostics.append(subordination_diagnostic)
    coordination_value, coordination_diagnostic = _parser_ratio(
        "text::syntax::coordination_ratio",
        counts.coordinate_phrase_count,
        counts.t_unit_count,
        "zero_t_unit_count",
    )
    row["text::syntax::coordination_ratio"] = coordination_value
    if coordination_diagnostic is not None:
        diagnostics.append(coordination_diagnostic)
    return row, tuple(diagnostics)


def _pos_lexical_density_row(document: ParsedDocument) -> tuple[dict[str, float], tuple[FeatureDiagnostic, ...]]:
    feature_name = "text::syntax::pos_lexical_density"
    token_count = len(document.tokens)
    if token_count == 0:
        return {feature_name: float("nan")}, (_parser_undefined(feature_name, "zero_parser_tokens"),)
    content_count = sum(1 for token in document.tokens if token.upos in _LEXICAL_DENSITY_UPOS_TAGS)
    return {feature_name: float(content_count) / float(token_count)}, ()


def _passive_voice_row(
    document: ParsedDocument,
    counts: ParsedSyntacticCounts,
) -> tuple[dict[str, float], tuple[FeatureDiagnostic, ...]]:
    feature_name = "text::syntax::passive_voice_frequency"
    passive_count = _passive_voice_count(document)
    value, diagnostic = _parser_ratio(feature_name, passive_count, counts.clause_count, "zero_clause_count")
    if diagnostic is None:
        return {feature_name: value}, ()
    return {feature_name: value}, (diagnostic,)


def _passive_voice_count(document: ParsedDocument) -> int:
    children_by_head: dict[int, list[ParsedDependencyArc]] = {}
    for arc in document.dependency_arcs:
        if arc.head_index is not None:
            children_by_head.setdefault(arc.head_index, []).append(arc)
    passive_count = 0
    for head_index, child_arcs in children_by_head.items():
        if _is_passive_predicate(document.tokens[head_index], child_arcs):
            passive_count += 1
    return passive_count


def _is_passive_predicate(token: ParsedToken, child_arcs: list[ParsedDependencyArc]) -> bool:
    if not any(feature.attribute == "Voice" and feature.value == "Pass" for feature in token.morphology):
        return False
    relation_labels = {arc.relation for arc in child_arcs}
    return "aux:pass" in relation_labels and "nsubj:pass" in relation_labels


def _named_entity_density_row(
    document: ParsedDocument,
    entities: tuple[ParsedNamedEntity, ...],
    entity_types: tuple[str, ...],
) -> tuple[dict[str, float], tuple[FeatureDiagnostic, ...]]:
    token_count = len(document.tokens)
    row: dict[str, float] = {}
    feature_names = parser_named_entity_density_feature_names(entity_types)
    if token_count == 0:
        for feature_name in feature_names:
            row[feature_name] = float("nan")
        diagnostics = tuple(_parser_undefined(feature_name, "zero_parser_tokens") for feature_name in feature_names)
        return row, diagnostics
    counts = Counter(entity.label for entity in entities)
    for scope in _NAMED_ENTITY_DENSITY_SCOPES:
        for entity_type in entity_types:
            feature_name = f"text::{scope}::named_entity_density::entity_type={entity_type}"
            row[feature_name] = float(counts[entity_type]) / float(token_count)
    return row, ()


def _validate_named_entity_span(document: ParsedDocument, entity: ParsedNamedEntity) -> None:
    token_count = len(document.tokens)
    if entity.end_token_index > token_count:
        raise ValueError(f"Named entity span out of range for document: {document.document_id}")


def _content_masking_row(
    document: ParsedDocument,
    entities: tuple[ParsedNamedEntity, ...],
    mask_upos_tags: tuple[str, ...],
    replacement_token: str,
) -> tuple[dict[str, float], tuple[FeatureDiagnostic, ...], ParserContentMaskingSidecar]:
    feature_name = "text::content_control::topic_neutral_distortion::masked_token_ratio"
    token_count = len(document.tokens)
    masked_token_indexes = _masked_token_indexes(document, entities, mask_upos_tags)
    masked_tokens = [replacement_token if index in masked_token_indexes else token.text for index, token in enumerate(document.tokens)]
    sidecar = ParserContentMaskingSidecar(
        document_id=document.document_id,
        schema_version="parser_content_masking_v1",
        masked_text=" ".join(masked_tokens),
        token_count=token_count,
        masked_token_count=len(masked_token_indexes),
        replacement_token=replacement_token,
        mask_upos_tags=mask_upos_tags,
        named_entity_count=len(entities),
    )
    if token_count == 0:
        return {feature_name: float("nan")}, (_parser_undefined(feature_name, "zero_parser_tokens"),), sidecar
    return {feature_name: float(len(masked_token_indexes)) / float(token_count)}, (), sidecar


def _masked_token_indexes(
    document: ParsedDocument,
    entities: tuple[ParsedNamedEntity, ...],
    mask_upos_tags: tuple[str, ...],
) -> frozenset[int]:
    masked_indexes = {index for index, token in enumerate(document.tokens) if token.upos in mask_upos_tags}
    for entity in entities:
        masked_indexes.update(range(entity.start_token_index, entity.end_token_index))
    return frozenset(masked_indexes)


def _root_statistics_row(document: ParsedDocument) -> tuple[dict[str, float], tuple[FeatureDiagnostic, ...]]:
    root_arcs = [arc for arc in document.dependency_arcs if arc.relation == "root"]
    root_count = len(root_arcs)
    token_count = len(document.tokens)
    row: dict[str, float] = {"text::syntax::root_statistics::root_count": float(root_count)}
    diagnostics: list[FeatureDiagnostic] = []
    root_rate_feature_name = "text::syntax::root_statistics::root_per_token"
    if token_count == 0:
        row[root_rate_feature_name] = float("nan")
        diagnostics.append(_parser_undefined(root_rate_feature_name, "zero_parser_tokens"))
    else:
        row[root_rate_feature_name] = float(root_count) / float(token_count)
    root_upos_counts = Counter(document.tokens[arc.dependent_index].upos for arc in root_arcs)
    for upos in _UNIVERSAL_POS_TAGS:
        feature_name = f"text::syntax::root_statistics::root_upos_ratio::upos={upos}"
        if root_count == 0:
            row[feature_name] = float("nan")
            diagnostics.append(_parser_undefined(feature_name, "zero_dependency_roots"))
        else:
            row[feature_name] = float(root_upos_counts[upos]) / float(root_count)
    return row, tuple(diagnostics)


def _dependency_distances(document: ParsedDocument) -> list[int]:
    return [abs(arc.head_index - arc.dependent_index) for arc in document.dependency_arcs if arc.head_index is not None]


def _parse_depths(document: ParsedDocument) -> list[int]:
    arc_by_dependent = {arc.dependent_index: arc for arc in document.dependency_arcs}
    return [_dependency_depth(arc, arc_by_dependent) for arc in sorted(document.dependency_arcs, key=lambda item: item.dependent_index)]


def _dependency_depth(arc: ParsedDependencyArc, arc_by_dependent: dict[int, ParsedDependencyArc]) -> int:
    depth = 0
    current_arc = arc
    visited_dependents: set[int] = set()
    while current_arc.head_index is not None:
        if current_arc.dependent_index in visited_dependents:
            raise ValueError("Dependency depth contains a cycle")
        visited_dependents.add(current_arc.dependent_index)
        depth += 1
        head_arc = arc_by_dependent.get(current_arc.head_index)
        if head_arc is None:
            break
        current_arc = head_arc
    return depth


def _head_dependent_pos_pairs(document: ParsedDocument) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for arc in document.dependency_arcs:
        if arc.head_index is None:
            continue
        head_upos = document.tokens[arc.head_index].upos
        dependent_upos = document.tokens[arc.dependent_index].upos
        pairs.append((head_upos, dependent_upos))
    return pairs


def _dependency_structure_items(document: ParsedDocument, ngram_range: tuple[int, int]) -> dict[str, list[str]]:
    return {
        "dependency_ngram": _dependency_ngrams(document, ngram_range),
        "dependency_path": _dependency_paths(document),
        "dependency_subtree": _dependency_subtrees(document),
        "dependency_dtgram": _dependency_dtgrams(document),
    }


def _dependency_ngrams(document: ParsedDocument, ngram_range: tuple[int, int]) -> list[str]:
    relations = [_dependency_relation_base(arc.relation) for arc in _non_root_arcs_by_dependent(document)]
    grams: list[str] = []
    lower, upper = ngram_range
    for ngram_size in range(lower, upper + 1):
        if len(relations) < ngram_size:
            continue
        grams.extend(" ".join(relations[index : index + ngram_size]) for index in range(0, len(relations) - ngram_size + 1))
    return grams


def _dependency_paths(document: ParsedDocument) -> list[str]:
    arc_by_dependent = {arc.dependent_index: arc for arc in document.dependency_arcs}
    paths: list[str] = []
    for arc in _non_root_arcs_by_dependent(document):
        relation_path = _dependency_relation_path(arc, arc_by_dependent)
        dependent_upos = document.tokens[arc.dependent_index].upos
        paths.append(f"{relation_path}:{dependent_upos}")
    return paths


def _dependency_relation_path(arc: ParsedDependencyArc, arc_by_dependent: dict[int, ParsedDependencyArc]) -> str:
    relations: list[str] = []
    current_arc = arc
    visited_dependents: set[int] = set()
    while current_arc.head_index is not None:
        if current_arc.dependent_index in visited_dependents:
            raise ValueError("Dependency path contains a cycle")
        visited_dependents.add(current_arc.dependent_index)
        relations.append(_dependency_relation_base(current_arc.relation))
        head_arc = arc_by_dependent.get(current_arc.head_index)
        if head_arc is None:
            break
        current_arc = head_arc
    relations.reverse()
    return ">".join(relations)


def _dependency_subtrees(document: ParsedDocument) -> list[str]:
    child_arcs_by_head: dict[int, list[ParsedDependencyArc]] = {}
    for arc in document.dependency_arcs:
        if arc.head_index is None:
            continue
        child_arcs_by_head.setdefault(arc.head_index, []).append(arc)
    subtrees: list[str] = []
    for head_index in sorted(child_arcs_by_head):
        child_parts = [
            f"{_dependency_relation_base(arc.relation)}:{document.tokens[arc.dependent_index].upos}"
            for arc in sorted(
                child_arcs_by_head[head_index],
                key=lambda child_arc: (
                    _dependency_relation_base(child_arc.relation),
                    document.tokens[child_arc.dependent_index].upos,
                    child_arc.dependent_index,
                ),
            )
        ]
        subtrees.append(f"{document.tokens[head_index].upos}({','.join(child_parts)})")
    return subtrees


def _dependency_dtgrams(document: ParsedDocument) -> list[str]:
    return [
        (f"{document.tokens[arc.head_index].upos}>{_dependency_relation_base(arc.relation)}>{document.tokens[arc.dependent_index].upos}")
        for arc in _non_root_arcs_by_dependent(document)
        if arc.head_index is not None
    ]


def _non_root_arcs_by_dependent(document: ParsedDocument) -> list[ParsedDependencyArc]:
    return sorted((arc for arc in document.dependency_arcs if arc.head_index is not None), key=lambda arc: arc.dependent_index)


def _dependency_relation_base(relation: str) -> str:
    if relation == "":
        raise ValueError("ParsedDependencyArc relation must not be empty")
    if relation.endswith(":"):
        raise ValueError("ParsedDependencyArc relation subtype must not be empty")
    if relation.startswith("root:"):
        raise ValueError(f"Unsupported Universal Dependencies relation label: {relation}")
    return relation.split(":", 1)[0]


def _validate_dependency_arc_indexes(document: ParsedDocument) -> None:
    token_count = len(document.tokens)
    for arc in document.dependency_arcs:
        if arc.dependent_index >= token_count:
            raise ValueError(f"Dependency arc dependent_index out of range for document: {document.document_id}")
        if arc.head_index is not None and arc.head_index >= token_count:
            raise ValueError(f"Dependency arc head_index out of range for document: {document.document_id}")
        if arc.head_index == arc.dependent_index:
            raise ValueError(f"Dependency arc head_index must not equal dependent_index for document: {document.document_id}")


def _parser_distribution_statistic_value(
    feature_name: str,
    statistic: str,
    values: list[int],
    zero_reason: str,
) -> tuple[float, FeatureDiagnostic | None]:
    if statistic == "count":
        return float(len(values)), None
    if len(values) == 0:
        return float("nan"), _parser_undefined(feature_name, zero_reason)
    array = np.asarray(values, dtype=float)
    if statistic in {"sample_std", "sample_variance"}:
        return _parser_sample_statistic(feature_name, statistic, array)
    if statistic in {"mean", "min", "max"}:
        return _parser_array_statistic(statistic, array), None
    if statistic.startswith("p"):
        return float(np.percentile(array, float(statistic.removeprefix("p")))), None
    if statistic in {"skewness", "excess_kurtosis"}:
        return _parser_moment_statistic(feature_name, statistic, array)
    if statistic == "shannon_entropy":
        return _parser_shannon_entropy(array), None
    raise ValueError(f"Unsupported parser distribution statistic: {statistic}")


def _parser_sample_statistic(feature_name: str, statistic: str, array: np.ndarray) -> tuple[float, FeatureDiagnostic | None]:
    if array.size < 2:
        return float("nan"), _parser_undefined(feature_name, "insufficient_values_for_sample_statistic")
    if statistic == "sample_std":
        return float(np.std(array, ddof=1)), None
    return float(np.var(array, ddof=1)), None


def _parser_array_statistic(statistic: str, array: np.ndarray) -> float:
    if statistic == "mean":
        return float(np.mean(array))
    if statistic == "min":
        return float(np.min(array))
    if statistic == "max":
        return float(np.max(array))
    raise ValueError(f"Unsupported parser array statistic: {statistic}")


def _parser_moment_statistic(feature_name: str, statistic: str, array: np.ndarray) -> tuple[float, FeatureDiagnostic | None]:
    if array.size < 2:
        return float("nan"), _parser_undefined(feature_name, "insufficient_values_for_moment_statistic")
    centered = array - float(np.mean(array))
    second_moment = float(np.mean(np.power(centered, 2)))
    if second_moment == 0.0:
        return float("nan"), _parser_undefined(feature_name, "zero_variance_distribution")
    if statistic == "skewness":
        third_moment = float(np.mean(np.power(centered, 3)))
        return third_moment / float(second_moment**1.5), None
    fourth_moment = float(np.mean(np.power(centered, 4)))
    return fourth_moment / float(second_moment**2) - 3.0, None


def _parser_shannon_entropy(array: np.ndarray) -> float:
    unique_values, counts = np.unique(array, return_counts=True)
    del unique_values
    probabilities = counts.astype(float) / float(array.size)
    return float(-np.sum(probabilities * np.log(probabilities)))


def _parser_distribution_formula(statistic: str) -> str:
    formulas = {
        "count": "number of non-root dependency arcs in the distribution",
        "mean": "arithmetic mean over absolute head-dependent token distances",
        "sample_std": "sample standard deviation over absolute head-dependent token distances with ddof=1",
        "sample_variance": "sample variance over absolute head-dependent token distances with ddof=1",
        "min": "minimum absolute head-dependent token distance",
        "max": "maximum absolute head-dependent token distance",
        "skewness": "population third central moment divided by population variance to the 3/2 power",
        "excess_kurtosis": "population fourth central moment divided by squared population variance, minus 3",
        "shannon_entropy": "empirical Shannon entropy over distance value frequencies using natural logarithms",
    }
    if statistic in formulas:
        return formulas[statistic]
    if statistic.startswith("p"):
        return f"{statistic.removeprefix('p')}th percentile using numpy default linear interpolation"
    raise ValueError(f"Unsupported parser distribution statistic: {statistic}")


def _parse_depth_distribution_formula(statistic: str) -> str:
    formulas = {
        "count": "number of dependency parse depths in the distribution",
        "mean": "arithmetic mean over dependency edge depths from root to token",
        "sample_std": "sample standard deviation over dependency edge depths with ddof=1",
        "sample_variance": "sample variance over dependency edge depths with ddof=1",
        "min": "minimum dependency edge depth",
        "max": "maximum dependency edge depth",
        "skewness": "population third central moment divided by population variance to the 3/2 power",
        "excess_kurtosis": "population fourth central moment divided by squared population variance, minus 3",
        "shannon_entropy": "empirical Shannon entropy over depth value frequencies using natural logarithms",
    }
    if statistic in formulas:
        return formulas[statistic]
    if statistic.startswith("p"):
        return f"{statistic.removeprefix('p')}th percentile over dependency edge depths using numpy default linear interpolation"
    raise ValueError(f"Unsupported parser distribution statistic: {statistic}")


def _parser_distribution_undefined_behavior(statistic: str, zero_reason: str) -> str:
    if statistic == "count":
        return "defined as zero when the distribution is empty"
    if statistic in {"sample_std", "sample_variance"}:
        return "NaN with FeatureDiagnostic reason insufficient_values_for_sample_statistic when fewer than two values exist"
    if statistic in {"skewness", "excess_kurtosis"}:
        return (
            "NaN with FeatureDiagnostic reason insufficient_values_for_moment_statistic when fewer than two values exist; "
            "NaN with reason zero_variance_distribution when all values are identical"
        )
    return f"NaN with FeatureDiagnostic reason {zero_reason} when the distribution is empty"


def _root_statistics_description(name: str) -> str:
    if name.endswith("root_count"):
        return "Count of root dependency arcs"
    if name.endswith("root_per_token"):
        return "Root dependency arcs divided by parsed token count"
    return "Universal POS ratio among dependency roots"


def _root_statistics_formula(name: str) -> str:
    if name.endswith("root_count"):
        return "count dependency arcs with relation root"
    if name.endswith("root_per_token"):
        return "count root dependency arcs divided by parsed token count"
    return "count root arcs whose dependent token has the requested Universal POS tag divided by root arc count"


def _root_statistics_undefined_behavior(name: str) -> str:
    if name.endswith("root_count"):
        return "defined as zero when no root arcs exist"
    if name.endswith("root_per_token"):
        return "NaN with FeatureDiagnostic reason zero_parser_tokens when parsed token count is zero"
    return "NaN with FeatureDiagnostic reason zero_dependency_roots when no root arcs exist"


def _syntactic_complexity_family(name: str) -> str:
    if name.startswith("text::syntax::syntactic_complexity::"):
        return "parser_syntactic_complexity"
    if name in parser_syntactic_count_feature_names():
        return "parser_clause_count"
    if name in parser_subordination_feature_names():
        return "parser_subordination"
    if name in parser_coordination_feature_names():
        return "parser_coordination"
    raise ValueError(f"Unsupported parser syntactic-complexity feature name: {name}")


def _syntactic_complexity_description(name: str) -> str:
    descriptions = {
        "text::syntax::syntactic_complexity::mls": "Mean length of sentence",
        "text::syntax::syntactic_complexity::mlt": "Mean length of T-unit",
        "text::syntax::syntactic_complexity::mlc": "Mean length of clause",
        "text::syntax::syntactic_complexity::c_per_s": "Clauses per sentence",
        "text::syntax::syntactic_complexity::c_per_t": "Clauses per T-unit",
        "text::syntax::syntactic_complexity::cp_per_t": "Coordinate phrases per T-unit",
        "text::syntax::syntactic_complexity::dc_per_c": "Dependent clauses per clause",
        "text::syntax::syntactic_complexity::cn_per_c": "Complex nominals per clause",
        "text::syntax::syntactic_complexity::cn_per_t": "Complex nominals per T-unit",
        "text::syntax::syntactic_complexity::vp_per_t": "Verb phrases per T-unit",
        "text::syntax::syntactic_complexity::t_per_s": "T-units per sentence",
        "text::syntax::clause_count": "Parser-derived clause count",
        "text::syntax::t_unit_count": "Parser-derived T-unit count",
        "text::syntax::subordination_ratio": "Dependent clauses divided by clauses",
        "text::syntax::coordination_ratio": "Coordinate phrases divided by T-units",
    }
    if name not in descriptions:
        raise ValueError(f"Unsupported parser syntactic-complexity feature name: {name}")
    return descriptions[name]


def _syntactic_complexity_formula(name: str) -> str:
    formulas = {
        "text::syntax::syntactic_complexity::mls": "word_count / sentence_count",
        "text::syntax::syntactic_complexity::mlt": "word_count / t_unit_count",
        "text::syntax::syntactic_complexity::mlc": "word_count / clause_count",
        "text::syntax::syntactic_complexity::c_per_s": "clause_count / sentence_count",
        "text::syntax::syntactic_complexity::c_per_t": "clause_count / t_unit_count",
        "text::syntax::syntactic_complexity::cp_per_t": "coordinate_phrase_count / t_unit_count",
        "text::syntax::syntactic_complexity::dc_per_c": "dependent_clause_count / clause_count",
        "text::syntax::syntactic_complexity::cn_per_c": "complex_nominal_count / clause_count",
        "text::syntax::syntactic_complexity::cn_per_t": "complex_nominal_count / t_unit_count",
        "text::syntax::syntactic_complexity::vp_per_t": "verb_phrase_count / t_unit_count",
        "text::syntax::syntactic_complexity::t_per_s": "t_unit_count / sentence_count",
        "text::syntax::clause_count": "clause_count",
        "text::syntax::t_unit_count": "t_unit_count",
        "text::syntax::subordination_ratio": "dependent_clause_count / clause_count",
        "text::syntax::coordination_ratio": "coordinate_phrase_count / t_unit_count",
    }
    if name not in formulas:
        raise ValueError(f"Unsupported parser syntactic-complexity feature name: {name}")
    return formulas[name]


def _syntactic_complexity_undefined_behavior(name: str) -> str:
    if name in parser_syntactic_count_feature_names():
        return "defined as zero when the parser-derived count is zero"
    formula = _syntactic_complexity_formula(name)
    if formula.endswith("/ sentence_count"):
        return "NaN with FeatureDiagnostic reason zero_sentence_count when sentence_count is zero"
    if formula.endswith("/ t_unit_count"):
        return "NaN with FeatureDiagnostic reason zero_t_unit_count when t_unit_count is zero"
    if formula.endswith("/ clause_count"):
        return "NaN with FeatureDiagnostic reason zero_clause_count when clause_count is zero"
    raise ValueError(f"Unsupported parser syntactic-complexity formula: {formula}")


def _syntactic_complexity_normalization(name: str) -> str:
    if name in parser_syntactic_count_feature_names():
        return "raw_count"
    return "ratio"


def _named_entity_density_family(name: str) -> str:
    if name.startswith("text::style_adjacent::named_entity_density::"):
        return "parser_named_entity_density_style_adjacent"
    if name.startswith("text::content_control::named_entity_density::"):
        return "parser_named_entity_density_topic_control"
    raise ValueError(f"Unsupported named-entity density feature name: {name}")


def _named_entity_density_topic_dependence(name: str) -> TopicDependence:
    if name.startswith("text::style_adjacent::named_entity_density::"):
        return TopicDependence.MIXED
    if name.startswith("text::content_control::named_entity_density::"):
        return TopicDependence.TOPIC_SENSITIVE
    raise ValueError(f"Unsupported named-entity density feature name: {name}")


def _parser_undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _parser_ratio(feature_name: str, numerator: int, denominator: int, zero_reason: str) -> tuple[float, FeatureDiagnostic | None]:
    if denominator == 0:
        return float("nan"), _parser_undefined(feature_name, zero_reason)
    return float(numerator) / float(denominator), None


def _validate_entity_types(entity_types: tuple[str, ...]) -> None:
    if len(entity_types) == 0:
        raise ValueError("entity_types must not be empty")
    seen: set[str] = set()
    for entity_type in entity_types:
        if entity_type == "":
            raise ValueError("entity_types must not contain empty values")
        if entity_type in seen:
            raise ValueError(f"Duplicate entity type: {entity_type}")
        seen.add(entity_type)


def _validate_mask_upos_tags(mask_upos_tags: tuple[str, ...]) -> None:
    if len(mask_upos_tags) == 0:
        raise ValueError("mask_upos_tags must not be empty")
    seen: set[str] = set()
    for upos in mask_upos_tags:
        if upos not in _UNIVERSAL_POS_TAGS:
            raise ValueError(f"Unsupported Universal POS tag for content masking: {upos}")
        if upos in seen:
            raise ValueError(f"Duplicate mask UPOS tag: {upos}")
        seen.add(upos)


def _validate_replacement_token(replacement_token: str) -> None:
    if len(replacement_token) == 0:
        raise ValueError("replacement_token must not be empty")


def _validate_ngram_range(ngram_range: tuple[int, int]) -> None:
    lower, upper = ngram_range
    if lower <= 0:
        raise ValueError("ngram_range lower bound must be positive")
    if upper < lower:
        raise ValueError("ngram_range upper bound must be greater than or equal to lower bound")


def _validate_max_features(max_features: int | None) -> None:
    if max_features is not None and max_features <= 0:
        raise ValueError("max_features must be positive when provided")


def _validate_skip_distance(skip_distance: int) -> None:
    if skip_distance <= 0:
        raise ValueError("skip_distance must be positive")


def _pos_ngrams(document: ParsedDocument, ngram_range: tuple[int, int]) -> list[str]:
    tags = [token.upos for token in document.tokens]
    grams: list[str] = []
    lower, upper = ngram_range
    for ngram_size in range(lower, upper + 1):
        if len(tags) < ngram_size:
            continue
        grams.extend(" ".join(tags[index : index + ngram_size]) for index in range(0, len(tags) - ngram_size + 1))
    return grams


def _pos_skipgrams(document: ParsedDocument, skip_distance: int) -> list[str]:
    tags = [token.upos for token in document.tokens]
    grams: list[str] = []
    offset = skip_distance + 1
    if len(tags) <= offset:
        return grams
    return [f"{tags[index]} *{skip_distance} {tags[index + offset]}" for index in range(0, len(tags) - offset)]


def _morphology_items(document: ParsedDocument) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for token in document.tokens:
        items.extend((feature.attribute, feature.value) for feature in token.morphology)
    return items


def _select_items(sorted_items: list[tuple[str, int]], max_features: int | None) -> list[tuple[str, int]]:
    if max_features is None:
        return sorted_items
    return sorted_items[:max_features]


def _pos_ngram_feature_name(gram: str) -> str:
    return f"text::syntax::pos_ngram::gram={gram}"


def _pos_skipgram_feature_name(gram: str) -> str:
    return f"text::syntax::pos_skipgram::gram={gram}"


def _morphology_feature_name(attribute: str, value: str) -> str:
    return f"text::syntax::morphology_frequency::attribute={attribute}::value={value}"


def _head_dependent_pos_pair_key(pair: tuple[str, str]) -> str:
    head_upos, dependent_upos = pair
    return f"{head_upos}->{dependent_upos}"


def _head_dependent_pos_pair_feature_name(head_upos: str, dependent_upos: str) -> str:
    return f"text::syntax::head_dependent_pos_pair_frequency::head_upos={head_upos}::dependent_upos={dependent_upos}"


def _dependency_structure_feature_name(kind: str, item: str) -> str:
    if kind == "dependency_ngram":
        return f"text::syntax::dependency_ngram::gram={item}"
    if kind == "dependency_path":
        return f"text::syntax::dependency_path::path={item}"
    if kind == "dependency_subtree":
        return f"text::syntax::dependency_subtree::signature={item}"
    if kind == "dependency_dtgram":
        return f"text::syntax::dependency_dtgram::gram={item}"
    raise ValueError(f"Unsupported dependency structure kind: {kind}")


def parser_backed_feature_names() -> tuple[str, ...]:
    """Return the built-in parser-backed feature catalog names."""
    return (
        "text::syntax::pos_frequency",
        "text::syntax::pos_ngram",
        "text::syntax::pos_skipgram",
        "text::syntax::morphology_frequency",
        "text::syntax::dependency_relation_frequency",
        "text::syntax::head_dependent_pos_pair_frequency",
        "text::syntax::dependency_ngram",
        "text::syntax::dependency_path",
        "text::syntax::dependency_subtree",
        "text::syntax::dependency_dtgram",
        "text::syntax::dependency_distance_mean",
        "text::syntax::root_statistics",
        "text::syntax::parse_depth",
        "text::syntax::syntactic_complexity",
        "text::syntax::clause_count",
        "text::syntax::t_unit_count",
        "text::syntax::subordination_ratio",
        "text::syntax::coordination_ratio",
        "text::syntax::passive_voice_frequency",
        "text::syntax::pos_lexical_density",
        "text::content_control::named_entity_density",
        "text::content_control::content_masking",
        "text::content_control::topic_neutral_distortion",
    )


def parser_backed_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserBackedTransformer:
    """Build a parser-backed transformer gate for the built-in parser feature catalog."""
    return ParserBackedTransformer(
        provider=provider,
        model=model,
        version=version,
        feature_names=parser_backed_feature_names(),
        text_column=text_column,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_pos_ngram_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    ngram_range: tuple[int, int],
    max_features: int | None,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserPOSNGramTransformer:
    """Build a fitted POS n-gram transformer for parser-backed features."""
    return ParserPOSNGramTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        ngram_range=ngram_range,
        max_features=max_features,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_pos_skipgram_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    skip_distance: int,
    max_features: int | None,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserPOSSkipGramTransformer:
    """Build a fitted POS skip-gram transformer for parser-backed features."""
    return ParserPOSSkipGramTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        skip_distance=skip_distance,
        max_features=max_features,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_morphology_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserMorphologyTransformer:
    """Build a fitted morphology transformer for parser-backed features."""
    return ParserMorphologyTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_dependency_relation_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserDependencyRelationTransformer:
    """Build a dependency-relation frequency transformer for parser-backed features."""
    return ParserDependencyRelationTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_head_dependent_pos_pair_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserHeadDependentPOSPairTransformer:
    """Build a fitted head-dependent POS pair transformer for parser-backed features."""
    return ParserHeadDependentPOSPairTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_dependency_distance_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserDependencyDistanceTransformer:
    """Build a dependency-distance distribution transformer for parser-backed features."""
    return ParserDependencyDistanceTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_parse_depth_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserParseDepthTransformer:
    """Build a parse-depth distribution transformer for parser-backed features."""
    return ParserParseDepthTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_syntactic_complexity_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    syntactic_counts: tuple[ParsedSyntacticCounts, ...] | None,
) -> ParserSyntacticComplexityTransformer:
    """Build a syntactic-complexity transformer for parser-backed features."""
    return ParserSyntacticComplexityTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        output=output,
        syntactic_counts=syntactic_counts,
    )


def parser_pos_lexical_density_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserPOSLexicalDensityTransformer:
    """Build a POS lexical-density transformer for parser-backed features."""
    return ParserPOSLexicalDensityTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_passive_voice_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
    syntactic_counts: tuple[ParsedSyntacticCounts, ...] | None,
) -> ParserPassiveVoiceTransformer:
    """Build a passive voice transformer for parser-backed features."""
    return ParserPassiveVoiceTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        output=output,
        parsed_documents=parsed_documents,
        syntactic_counts=syntactic_counts,
    )


def parser_named_entity_density_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    entity_types: tuple[str, ...],
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
    named_entities: tuple[ParsedNamedEntity, ...] | None,
) -> ParserNamedEntityDensityTransformer:
    """Build a named-entity density transformer for parser/NER-backed features."""
    return ParserNamedEntityDensityTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        entity_types=entity_types,
        output=output,
        parsed_documents=parsed_documents,
        named_entities=named_entities,
    )


def parser_content_masking_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    mask_upos_tags: tuple[str, ...],
    replacement_token: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
    named_entities: tuple[ParsedNamedEntity, ...] | None,
) -> ParserContentMaskingTransformer:
    """Build a parser-backed content masking transformer."""
    return ParserContentMaskingTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        mask_upos_tags=mask_upos_tags,
        replacement_token=replacement_token,
        output=output,
        parsed_documents=parsed_documents,
        named_entities=named_entities,
    )


def parser_root_statistics_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserRootStatisticsTransformer:
    """Build a root-statistics transformer for parser-backed features."""
    return ParserRootStatisticsTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        output=output,
        parsed_documents=parsed_documents,
    )


def parser_dependency_structure_transformer(
    provider: str,
    model: str,
    version: str,
    text_column: str,
    ngram_range: tuple[int, int],
    max_features_per_kind: int | None,
    output: str,
    parsed_documents: tuple[ParsedDocument, ...] | None,
) -> ParserDependencyStructureTransformer:
    """Build a fitted dependency-structure transformer for parser-backed features."""
    return ParserDependencyStructureTransformer(
        provider=provider,
        model=model,
        version=version,
        text_column=text_column,
        ngram_range=ngram_range,
        max_features_per_kind=max_features_per_kind,
        output=output,
        parsed_documents=parsed_documents,
    )


class LLMAnnotationTransformer(BaseEstimator):
    """Placeholder gate for LLM annotation features with required provenance fields."""

    configured_transformer_: ConfiguredLLMAnnotationTransformer
    last_sidecars_: tuple[LLMAnnotationSidecarLike, ...]

    def __init__(
        self,
        provider: str,
        model: str,
        version: str,
        prompt_version: str,
        response_schema: str,
        feature_names: tuple[str, ...],
        fake_annotations: tuple[FakeLLMAnnotation, ...] | None,
        client: LLMClientProtocol | None,
        text_column: str,
    ) -> None:
        self.provider = provider
        self.model = model
        self.version = version
        self.prompt_version = prompt_version
        self.response_schema = response_schema
        self.feature_names = feature_names
        self.fake_annotations = fake_annotations
        self.client = client
        self.text_column = text_column

    def fit(self, x: object, y: object) -> Self:
        """Fit fake or configured LLM annotation metadata."""
        del y
        if not self._uses_fake_provider():
            if self.client is None:
                raise self._dependency_error()
            configured = configured_llm_annotation_transformer(
                client=self.client,
                text_column=self.text_column,
                feature_names=self.feature_names,
            ).fit(x, None)
            registry = FeatureRegistry(specs=self.feature_specs())
            registry.require_complete()
            self.feature_names_out_ = configured.get_feature_names_out(None)
            self.n_features_in_ = configured.n_features_in_
            self.registry_ = registry
            self.configured_transformer_ = configured
            return self
        row_ids = _llm_row_ids(x)
        annotation_map = _fake_llm_annotation_map(self.fake_annotations)
        expected_names = set(self.feature_names)
        for document_id, annotation in annotation_map.items():
            _validate_fake_llm_annotation_features(annotation, expected_names)
            if document_id not in row_ids:
                raise ValueError(f"Fake LLM annotation document id has no input row: {document_id}")
        for document_id in row_ids:
            if document_id not in annotation_map:
                raise ValueError(f"Missing fake LLM annotation for row id: {document_id}")
        registry = FeatureRegistry(specs=self.feature_specs())
        registry.require_complete()
        self.feature_names_out_ = np.asarray(self.feature_names, dtype=object)
        self.n_features_in_ = 1
        self.fake_annotations_ = annotation_map
        self.registry_ = registry
        return self

    def transform(self, x: object) -> np.ndarray:
        """Return fake or configured-provider LLM annotation values."""
        if not self._uses_fake_provider():
            if self.client is None:
                raise self._dependency_error()
            require_fitted(self, "configured_transformer_")
            result = self.configured_transformer_.transform(x)
            self.last_sidecars_ = self.configured_transformer_.last_sidecars_
            self.last_diagnostics_ = self.configured_transformer_.last_diagnostics_
            return result
        require_fitted(self, "feature_names_out_")
        row_ids = _llm_row_ids(x)
        rows: list[list[float]] = []
        sidecars: list[LLMAnnotationSidecar] = []
        for document_id in row_ids:
            if document_id not in self.fake_annotations_:
                raise ValueError(f"Missing fake LLM annotation for row id: {document_id}")
            annotation = self.fake_annotations_[document_id]
            value_map = dict(annotation.feature_values)
            rows.append([float(value_map[feature_name]) for feature_name in self.feature_names])
            sidecars.append(
                LLMAnnotationSidecar(
                    document_id=document_id,
                    schema_version="fake_llm_annotation_v1",
                    prompt_version=self.prompt_version,
                    response_schema=self.response_schema,
                    structured_response=annotation.structured_response,
                )
            )
        self.last_sidecars_ = tuple(sidecars)
        return np.asarray(rows, dtype=np.float64)

    def fit_transform(self, x: object, y: object) -> np.ndarray:
        """Fit, then transform with fake or configured LLM annotations."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable LLM annotation feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_

    def feature_specs(self) -> tuple[FeatureSpec, ...]:
        """Return LLM feature metadata for discovery."""
        return tuple(
            FeatureSpec(
                name=name,
                family="llm_annotation",
                description="Optional model-mediated stylometry annotation",
                formula_or_rule=f"prompt_version={self.prompt_version}; response_schema={self.response_schema}",
                input_layer=InputLayer.LLM,
                topic_dependence=TopicDependence.MIXED,
                text_length_policy="requires prompt-specific context length and repeated-run stability audit",
                provenance=f"provider={self.provider}; model={self.model}; version={self.version}; prompt_version={self.prompt_version}",
                output_dtype="float64_or_structured",
                undefined_behavior="dependency/provider error unless optional LLM provider is installed and configured",
                normalization="schema_specific",
                sparsity="dense_or_structured",
                stability_status=StabilityStatus.LLM_DEPENDENT,
            )
            for name in self.feature_names
        )

    def _uses_fake_provider(self) -> bool:
        return self.provider == "fake" and self.fake_annotations is not None

    def _dependency_error(self) -> OptionalDependencyError:
        return OptionalDependencyError(
            "LLM annotation features require an explicit configured LLM client; "
            f"provider={self.provider}, model={self.model}, version={self.version}, prompt_version={self.prompt_version}"
        )


def llm_annotation_feature_names() -> tuple[str, ...]:
    """Return the built-in optional LLM annotation feature catalog names."""
    return (
        "text::llm::tone",
        "text::llm::register",
        "text::llm::persona",
        "text::llm::narrative_perspective",
        "text::llm::sentence_intent",
        "text::llm::discourse_function",
        "text::llm::rhetorical_structure",
        "text::llm::argumentation_style",
        "text::llm::cohesion_judgment",
        "text::llm::style_topic_separation",
        "text::llm::stylistic_similarity",
        "text::llm::pairwise_style_comparison",
        "text::llm::style_difference_explanation",
        "text::llm::style_transfer_descriptor",
        "text::llm::authorial_habit_summary",
        "text::llm::prompt_derived_vector",
        "text::llm::embedding",
        "text::llm::style_tuned_embedding",
        "text::llm::same_author_prediction",
        "text::llm::generated_feature_extraction",
    )


def _validate_fake_llm_feature_values(feature_values: tuple[tuple[str, float], ...]) -> None:
    if len(feature_values) == 0:
        raise ValueError("FakeLLMAnnotation feature_values must not be empty")
    seen: set[str] = set()
    for feature_name, value in feature_values:
        if feature_name == "":
            raise ValueError("FakeLLMAnnotation feature names must not be empty")
        if feature_name in seen:
            raise ValueError(f"Duplicate fake LLM feature value: {feature_name}")
        if not np.isfinite(value):
            raise ValueError(f"Fake LLM feature value must be finite: {feature_name}")
        seen.add(feature_name)


def _fake_llm_annotation_map(fake_annotations: tuple[FakeLLMAnnotation, ...] | None) -> dict[str, FakeLLMAnnotation]:
    if fake_annotations is None:
        raise ValueError("fake_annotations must be provided for fake LLM provider")
    annotation_map: dict[str, FakeLLMAnnotation] = {}
    for annotation in fake_annotations:
        if annotation.document_id in annotation_map:
            raise ValueError(f"Duplicate fake LLM annotation document id: {annotation.document_id}")
        annotation_map[annotation.document_id] = annotation
    return annotation_map


def _validate_fake_llm_annotation_features(annotation: FakeLLMAnnotation, expected_names: set[str]) -> None:
    actual_names = {feature_name for feature_name, _value in annotation.feature_values}
    missing_names = sorted(expected_names - actual_names)
    extra_names = sorted(actual_names - expected_names)
    if len(missing_names) > 0:
        raise ValueError(f"Fake LLM annotation missing feature value: {missing_names[0]}")
    if len(extra_names) > 0:
        raise ValueError(f"Fake LLM annotation has unexpected feature value: {extra_names[0]}")


def _llm_row_ids(x: object) -> tuple[str, ...]:
    if isinstance(x, pd.DataFrame | pd.Series):
        return tuple(str(index) for index in x.index)
    raise ValueError("Fake LLM annotations require pandas input with row identifiers")


def llm_annotation_transformer(
    provider: str,
    model: str,
    version: str,
    prompt_version: str,
    response_schema: str,
    feature_names: tuple[str, ...],
    fake_annotations: tuple[FakeLLMAnnotation, ...] | None,
    client: LLMClientProtocol | None,
    text_column: str,
) -> LLMAnnotationTransformer:
    """Build an LLM annotation transformer gate for the built-in LLM feature catalog."""
    return LLMAnnotationTransformer(
        provider=provider,
        model=model,
        version=version,
        prompt_version=prompt_version,
        response_schema=response_schema,
        feature_names=feature_names,
        fake_annotations=fake_annotations,
        client=client,
        text_column=text_column,
    )
