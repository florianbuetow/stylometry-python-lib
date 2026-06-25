"""Frequency feature blocks for fixed and fitted vocabularies."""

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
from stylometry_python_lib.document import DocumentView, PreprocessingConfig
from stylometry_python_lib.features.deterministic import function_words_lexicon
from stylometry_python_lib.lexicons import LexiconEntry, VersionedLexicon, load_lexicon
from stylometry_python_lib.registry import FeatureRegistry
from stylometry_python_lib.specs import FeatureSpec, InputLayer, StabilityStatus, TopicDependence
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus


@dataclass(frozen=True)
class ContractionMatch:
    """One contraction match recorded in the structured expansion sidecar."""

    token: str
    orthographic_token_index: int
    expansion: tuple[str, ...]
    expansion_alternatives: tuple[tuple[str, ...], ...]
    groups: tuple[str, ...]
    ambiguous: bool
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ContractionExpansionSidecar:
    """Contraction expansion and ambiguity metadata for one document."""

    document_id: str
    lexicon_id: str
    language: str
    version: str
    normalization: str
    schema_version: str
    token_count: int
    match_count: int
    ambiguous_match_count: int
    matches: tuple[ContractionMatch, ...]
    warnings: tuple[str, ...]


class FixedVocabularyFrequencyTransformer(BaseEstimator):
    """Compute frequencies for an explicit token vocabulary."""

    def __init__(
        self,
        text_column: str,
        config: PreprocessingConfig,
        vocabulary: tuple[str, ...],
        family: str,
        topic_dependence: TopicDependence,
        output: str,
    ) -> None:
        self.text_column = text_column
        self.config = config
        self.vocabulary = vocabulary
        self.family = family
        self.topic_dependence = topic_dependence
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Freeze feature names for the configured vocabulary."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.vocabulary_ = tuple(self.vocabulary)
        self.feature_names_out_ = np.asarray([_token_feature_name(self.family, token) for token in self.vocabulary_], dtype=object)
        self.registry_ = FeatureRegistry(
            specs=tuple(_token_frequency_spec(str(name), self.family, self.topic_dependence) for name in self.feature_names_out_.tolist())
        )
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute per-token vocabulary frequencies."""
        require_fitted(self, "vocabulary_")
        rows: list[list[float]] = []
        series = text_series(x, self.text_column)
        for row_index, text in enumerate(series.tolist()):
            view = DocumentView.from_text(str(text), self.config, document_id=str(series.index[row_index]))
            rows.append(_token_frequency_row(view.tokens, self.vocabulary_))
        frame = pd.DataFrame(rows, columns=self.feature_names_out_, index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable frequency feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


class CharacterFrequencyTransformer(BaseEstimator):
    """Compute frequencies for explicit raw-text characters."""

    def __init__(self, text_column: str, config: PreprocessingConfig, characters: tuple[str, ...], family: str, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.characters = characters
        self.family = family
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Freeze character feature names."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.characters_ = tuple(self.characters)
        self.feature_names_out_ = np.asarray(
            [_character_feature_name(self.family, character) for character in self.characters_], dtype=object
        )
        self.registry_ = FeatureRegistry(
            specs=tuple(_character_frequency_spec(str(name), self.family) for name in self.feature_names_out_.tolist())
        )
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute per-character raw-text frequencies."""
        require_fitted(self, "characters_")
        series = text_series(x, self.text_column)
        rows = [_character_frequency_row(str(text), self.characters_) for text in series.tolist()]
        frame = pd.DataFrame(rows, columns=self.feature_names_out_, index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable character frequency feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


class MostFrequentWordsTransformer(BaseEstimator):
    """Fit corpus most-frequent-word features and emit normalized counts."""

    def __init__(self, text_column: str, config: PreprocessingConfig, max_features: int, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.max_features = max_features
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Fit the most frequent word vocabulary from the corpus."""
        del y
        validate_output_mode(self.output)
        if self.max_features <= 0:
            raise ValueError("max_features must be positive")
        counter: Counter[str] = Counter()
        for row_index, text in enumerate(text_series(x, self.text_column).tolist()):
            view = DocumentView.from_text(str(text), self.config, document_id=str(row_index))
            counter.update(view.tokens)
        selected = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[: self.max_features]
        self.vocabulary_ = tuple(token for token, _count in selected)
        self.feature_names_out_ = np.asarray(
            [_token_feature_name("most_frequent_words", token) for token in self.vocabulary_], dtype=object
        )
        self.registry_ = FeatureRegistry(
            specs=tuple(
                _token_frequency_spec(str(name), "most_frequent_words", TopicDependence.TOPIC_SENSITIVE)
                for name in self.feature_names_out_.tolist()
            )
        )
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute fitted most-frequent-word frequencies."""
        require_fitted(self, "vocabulary_")
        rows: list[list[float]] = []
        series = text_series(x, self.text_column)
        for row_index, text in enumerate(series.tolist()):
            view = DocumentView.from_text(str(text), self.config, document_id=str(series.index[row_index]))
            rows.append(_token_frequency_row(view.tokens, self.vocabulary_))
        frame = pd.DataFrame(rows, columns=self.feature_names_out_, index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable fitted most-frequent-word feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


class ClosedClassLexiconTransformer(BaseEstimator):
    """Emit per-item and grouped closed-class lexicon counts plus rates."""

    def __init__(
        self,
        text_column: str,
        config: PreprocessingConfig,
        lexicon_name: str,
        family: str,
        topic_dependence: TopicDependence,
        token_layer: InputLayer,
        output: str,
    ) -> None:
        self.text_column = text_column
        self.config = config
        self.lexicon_name = lexicon_name
        self.family = family
        self.topic_dependence = topic_dependence
        self.token_layer = token_layer
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Load lexicon metadata and freeze output names."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        lexicon = load_lexicon(self.lexicon_name)
        self.lexicon_ = lexicon
        self.feature_names_out_ = np.asarray(_closed_class_feature_names(self.family, lexicon), dtype=object)
        if self.token_layer not in {InputLayer.TOKENS, InputLayer.ORTHOGRAPHIC_TOKENS}:
            raise ValueError("ClosedClassLexiconTransformer supports only token or orthographic token layers")
        self.registry_ = FeatureRegistry(
            specs=tuple(
                _closed_class_feature_spec(str(name), self.family, lexicon, self.topic_dependence, self.token_layer)
                for name in self.feature_names_out_.tolist()
            )
        )
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute per-item and grouped closed-class lexicon outputs."""
        require_fitted(self, "lexicon_")
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[ContractionExpansionSidecar] = []
        series = text_series(x, self.text_column)
        for row_index, text in enumerate(series.tolist()):
            view = DocumentView.from_text(str(text), self.config, document_id=str(series.index[row_index]))
            row, row_diagnostics = _closed_class_row(view, self.lexicon_, self.family, self.token_layer)
            if self.family == "contraction":
                sidecar, sidecar_diagnostics = _contraction_sidecar_and_diagnostics(view, self.lexicon_)
                row_diagnostics = row_diagnostics + sidecar_diagnostics
                sidecars.append(sidecar)
            rows.append(row)
            diagnostics.append(row_diagnostics)
        self.last_diagnostics_ = tuple(diagnostics)
        if self.family == "contraction":
            self.last_sidecars_ = tuple(sidecars)
        elif hasattr(self, "last_sidecars_"):
            del self.last_sidecars_
        frame = pd.DataFrame(rows, columns=self.feature_names_out_, index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable closed-class lexicon feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def function_word_frequency_transformer(text_column: str, config: PreprocessingConfig, output: str) -> FixedVocabularyFrequencyTransformer:
    """Build a fixed-vocabulary function-word frequency transformer."""
    vocabulary = tuple(sorted(function_words_lexicon()))
    return FixedVocabularyFrequencyTransformer(
        text_column=text_column,
        config=config,
        vocabulary=vocabulary,
        family="function_word_frequency",
        topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
        output=output,
    )


def function_word_lexicon_transformer(text_column: str, config: PreprocessingConfig, output: str) -> ClosedClassLexiconTransformer:
    """Build a versioned function-word per-item and grouped transformer."""
    return ClosedClassLexiconTransformer(
        text_column=text_column,
        config=config,
        lexicon_name="function_words",
        family="function_word",
        topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
        token_layer=InputLayer.TOKENS,
        output=output,
    )


def stopword_lexicon_transformer(text_column: str, config: PreprocessingConfig, output: str) -> ClosedClassLexiconTransformer:
    """Build a versioned stopword per-item and grouped transformer."""
    return ClosedClassLexiconTransformer(
        text_column=text_column,
        config=config,
        lexicon_name="stopwords",
        family="stopword",
        topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
        token_layer=InputLayer.TOKENS,
        output=output,
    )


def pronoun_lexicon_transformer(text_column: str, config: PreprocessingConfig, output: str) -> ClosedClassLexiconTransformer:
    """Build a versioned pronoun per-item and grouped transformer."""
    return ClosedClassLexiconTransformer(
        text_column=text_column,
        config=config,
        lexicon_name="pronouns",
        family="pronoun",
        topic_dependence=TopicDependence.MIXED,
        token_layer=InputLayer.TOKENS,
        output=output,
    )


def modal_lexicon_transformer(text_column: str, config: PreprocessingConfig, output: str) -> ClosedClassLexiconTransformer:
    """Build a versioned modal per-item and grouped transformer."""
    return ClosedClassLexiconTransformer(
        text_column=text_column,
        config=config,
        lexicon_name="modals",
        family="modal",
        topic_dependence=TopicDependence.MIXED,
        token_layer=InputLayer.TOKENS,
        output=output,
    )


def auxiliary_lexicon_transformer(text_column: str, config: PreprocessingConfig, output: str) -> ClosedClassLexiconTransformer:
    """Build a versioned auxiliary per-item and grouped transformer."""
    return ClosedClassLexiconTransformer(
        text_column=text_column,
        config=config,
        lexicon_name="auxiliaries",
        family="auxiliary",
        topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
        token_layer=InputLayer.TOKENS,
        output=output,
    )


def contraction_lexicon_transformer(text_column: str, config: PreprocessingConfig, output: str) -> ClosedClassLexiconTransformer:
    """Build a versioned contraction per-item and grouped transformer."""
    return ClosedClassLexiconTransformer(
        text_column=text_column,
        config=config,
        lexicon_name="contractions",
        family="contraction",
        topic_dependence=TopicDependence.MIXED,
        token_layer=InputLayer.ORTHOGRAPHIC_TOKENS,
        output=output,
    )


def letter_frequency_transformer(text_column: str, config: PreprocessingConfig, output: str) -> CharacterFrequencyTransformer:
    """Build a lowercase English letter frequency transformer."""
    characters = tuple("abcdefghijklmnopqrstuvwxyz")
    return CharacterFrequencyTransformer(
        text_column=text_column, config=config, characters=characters, family="letter_frequency", output=output
    )


def punctuation_frequency_transformer(text_column: str, config: PreprocessingConfig, output: str) -> CharacterFrequencyTransformer:
    """Build a punctuation frequency transformer."""
    characters = (".", ",", ";", ":", "!", "?", "-", "(", ")", "'", '"')
    return CharacterFrequencyTransformer(
        text_column=text_column, config=config, characters=characters, family="punctuation_frequency", output=output
    )


def _token_frequency_row(tokens: tuple[str, ...], vocabulary: tuple[str, ...]) -> list[float]:
    denominator = len(tokens)
    counts = Counter(tokens)
    if denominator == 0:
        return [float("nan") for _token in vocabulary]
    return [float(counts[token]) / float(denominator) for token in vocabulary]


def _character_frequency_row(text: str, characters: tuple[str, ...]) -> list[float]:
    denominator = sum(1 for character in text if character.isalpha())
    lower_text = text.lower()
    if denominator == 0:
        return [float("nan") for _character in characters]
    return [float(lower_text.count(character)) / float(denominator) for character in characters]


def _closed_class_feature_names(family: str, lexicon: VersionedLexicon) -> tuple[str, ...]:
    names: list[str] = []
    for token in lexicon.tokens():
        names.extend((_closed_class_item_name(family, token, "count"), _closed_class_item_name(family, token, "per_1000_tokens")))
    for group in lexicon.groups():
        names.extend((_closed_class_group_name(family, group, "count"), _closed_class_group_name(family, group, "per_1000_tokens")))
    return tuple(names)


def _closed_class_row(
    view: DocumentView, lexicon: VersionedLexicon, family: str, token_layer: InputLayer
) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    observed_tokens = view.tokens
    if token_layer == InputLayer.ORTHOGRAPHIC_TOKENS:
        observed_tokens = view.orthographic_tokens
    token_count = len(view.tokens)
    counts = Counter(observed_tokens)
    values: list[float] = []
    diagnostics: list[FeatureDiagnostic] = []
    for entry in lexicon.entries:
        item_count = float(counts[entry.token])
        values.append(item_count)
        values.append(_per_1000_value(item_count, token_count))
        _add_rate_diagnostic(diagnostics, _closed_class_item_name(family, entry.token, "per_1000_tokens"), token_count)
    group_counts = _closed_class_group_counts(counts, lexicon)
    for group in lexicon.groups():
        group_count = float(group_counts[group])
        values.append(group_count)
        values.append(_per_1000_value(group_count, token_count))
        _add_rate_diagnostic(diagnostics, _closed_class_group_name(family, group, "per_1000_tokens"), token_count)
    return values, tuple(diagnostics)


def _closed_class_group_counts(counts: Counter[str], lexicon: VersionedLexicon) -> dict[str, int]:
    group_counts = {group: 0 for group in lexicon.groups()}
    for entry in lexicon.entries:
        count = counts[entry.token]
        for group in entry.groups:
            group_counts[group] += count
    return group_counts


def _contraction_sidecar_and_diagnostics(
    view: DocumentView, lexicon: VersionedLexicon
) -> tuple[ContractionExpansionSidecar, tuple[FeatureDiagnostic, ...]]:
    entry_by_token = {entry.token: entry for entry in lexicon.entries}
    matches: list[ContractionMatch] = []
    diagnostics: list[FeatureDiagnostic] = []
    for token_index, token in enumerate(view.orthographic_tokens):
        if token not in entry_by_token:
            continue
        entry = entry_by_token[token]
        match = _contraction_match(token_index, entry)
        matches.append(match)
        if match.ambiguous:
            diagnostics.append(_ambiguous_contraction_diagnostic(entry))
    ambiguous_match_count = sum(1 for match in matches if match.ambiguous)
    warnings: tuple[str, ...] = ()
    if ambiguous_match_count > 0:
        warnings = ("ambiguous_contraction_expansion",)
    return (
        ContractionExpansionSidecar(
            document_id=view.document_id,
            lexicon_id=lexicon.lexicon_id,
            language=lexicon.language,
            version=lexicon.version,
            normalization=lexicon.normalization,
            schema_version="contraction_expansion_v1",
            token_count=len(view.tokens),
            match_count=len(matches),
            ambiguous_match_count=ambiguous_match_count,
            matches=tuple(matches),
            warnings=warnings,
        ),
        tuple(diagnostics),
    )


def _contraction_match(token_index: int, entry: LexiconEntry) -> ContractionMatch:
    ambiguous = False
    if "ambiguous" in entry.groups:
        ambiguous = True
    if len(entry.expansion_alternatives) > 1:
        ambiguous = True
    warnings: tuple[str, ...] = ()
    if ambiguous:
        warnings = ("ambiguous_contraction_expansion",)
    return ContractionMatch(
        token=entry.token,
        orthographic_token_index=token_index,
        expansion=entry.expansion,
        expansion_alternatives=entry.expansion_alternatives,
        groups=entry.groups,
        ambiguous=ambiguous,
        warnings=warnings,
    )


def _ambiguous_contraction_diagnostic(entry: LexiconEntry) -> FeatureDiagnostic:
    alternatives = tuple(" ".join(alternative) for alternative in entry.expansion_alternatives)
    return FeatureDiagnostic(
        feature_name=_closed_class_item_name("contraction", entry.token, "count"),
        status=FeatureStatus.WARNING,
        reason="ambiguous_contraction_expansion",
        warnings=(f"expansion_alternatives={';'.join(alternatives)}",),
    )


def _per_1000_value(count: float, token_count: int) -> float:
    if token_count == 0:
        return float("nan")
    return count * 1000.0 / float(token_count)


def _add_rate_diagnostic(diagnostics: list[FeatureDiagnostic], feature_name: str, token_count: int) -> None:
    if token_count == 0:
        diagnostics.append(FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason="zero_tokens", warnings=()))


def _token_feature_name(family: str, token: str) -> str:
    return f"text::{family}::token={token}"


def _closed_class_item_name(family: str, token: str, measure: str) -> str:
    return f"text::{family}::item={token}::{measure}"


def _closed_class_group_name(family: str, group: str, measure: str) -> str:
    return f"text::{family}::group={group}::{measure}"


def _character_feature_name(family: str, character: str) -> str:
    escaped = character.replace("\n", "\\n")
    return f"text::{family}::char={escaped}"


def _token_frequency_spec(name: str, family: str, topic_dependence: TopicDependence) -> FeatureSpec:
    stability_status = StabilityStatus.DETERMINISTIC
    if family == "most_frequent_words":
        stability_status = StabilityStatus.STATISTICAL_FIT_DEPENDENT
    return FeatureSpec(
        name=name,
        family=family,
        description="Normalized token frequency for a fixed or fitted vocabulary item",
        formula_or_rule="count token occurrences divided by total token count",
        input_layer=InputLayer.TOKENS,
        topic_dependence=topic_dependence,
        text_length_policy="undefined as NaN when token count is zero",
        provenance="built_in_frequency_rules:v1; preprocessing_config; fitted_vocabulary_when_applicable",
        output_dtype="float64",
        undefined_behavior="NaN when zero tokens are present",
        normalization="per_token_ratio",
        sparsity="dense_scalar",
        stability_status=stability_status,
    )


def _character_frequency_spec(name: str, family: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family=family,
        description="Normalized raw character frequency",
        formula_or_rule="count character occurrences divided by total alphabetic character count",
        input_layer=InputLayer.RAW,
        topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
        text_length_policy="undefined as NaN when alphabetic character count is zero",
        provenance="built_in_character_frequency_rules:v1; preprocessing_config",
        output_dtype="float64",
        undefined_behavior="NaN when zero alphabetic characters are present",
        normalization="per_letter_ratio",
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )


def _closed_class_feature_spec(
    name: str,
    family: str,
    lexicon: VersionedLexicon,
    topic_dependence: TopicDependence,
    input_layer: InputLayer,
) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "count is zero when the item or group is absent and the token layer exists"
    if name.endswith("per_1000_tokens"):
        normalization = "per_1000_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_tokens when token count is zero"
    return FeatureSpec(
        name=name,
        family=family,
        description=f"Versioned closed-class lexicon feature from {lexicon.lexicon_id}",
        formula_or_rule="count exact lowercase lexicon item or grouped lexicon category; rate = count * 1000 / token_count",
        input_layer=input_layer,
        topic_dependence=topic_dependence,
        text_length_policy="counts are defined for an existing token layer; per-1,000-token rates are undefined when token count is zero",
        provenance=(
            f"lexicon_id={lexicon.lexicon_id}; language={lexicon.language}; version={lexicon.version}; "
            f"source={lexicon.source}; license_note={lexicon.license_note}; normalization={lexicon.normalization}"
        ),
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
