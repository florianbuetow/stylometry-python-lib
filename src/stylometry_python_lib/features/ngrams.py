"""Sparse n-gram stylometry feature blocks."""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Literal, Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib._tabular import text_series, validate_output_mode
from stylometry_python_lib.document import DocumentView, PreprocessingConfig
from stylometry_python_lib.features.deterministic import function_words_lexicon
from stylometry_python_lib.registry import FeatureRegistry
from stylometry_python_lib.specs import FeatureSpec, InputLayer, StabilityStatus, TopicDependence

Analyzer = Literal["word", "function_word", "char", "punctuation"]


class NGramStylometryTransformer(BaseEstimator):
    """Sklearn-compatible sparse word/function-word/character/punctuation n-gram transformer."""

    def __init__(
        self,
        text_column: str,
        config: PreprocessingConfig,
        analyzer: Analyzer,
        ngram_range: tuple[int, int],
        max_features: int | None,
        output: str,
    ) -> None:
        self.text_column = text_column
        self.config = config
        self.analyzer: Analyzer = analyzer
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Learn a stable n-gram vocabulary from the training corpus."""
        del y
        validate_output_mode(self.output)
        _validate_ngram_range(self.ngram_range)
        vocabulary_counter: Counter[str] = Counter()
        series = text_series(x, self.text_column)
        for row_index, text in enumerate(series.tolist()):
            view = DocumentView.from_text(str(text), self.config, document_id=str(series.index[row_index]))
            vocabulary_counter.update(_extract_ngrams(view, self.analyzer, self.ngram_range))
        sorted_items = sorted(vocabulary_counter.items(), key=lambda item: (-item[1], item[0]))
        selected_items = _select_items(sorted_items, self.max_features)
        self.vocabulary_ = {gram: index for index, (gram, _count) in enumerate(selected_items)}
        self.feature_names_out_ = np.asarray([_feature_name(self.analyzer, gram) for gram, _count in selected_items], dtype=object)
        specs = tuple(_spec_for_name(str(name), self.analyzer) for name in self.feature_names_out_.tolist())
        self.registry_ = FeatureRegistry(specs=specs)
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> sparse.csr_matrix | pd.DataFrame | np.ndarray:
        """Transform texts into an n-gram document-term matrix."""
        require_fitted(self, "vocabulary_")
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        series = text_series(x, self.text_column)
        texts = series.tolist()
        for row_index, text in enumerate(texts):
            view = DocumentView.from_text(str(text), self.config, document_id=str(series.index[row_index]))
            counts = Counter(_extract_ngrams(view, self.analyzer, self.ngram_range))
            for gram, count in counts.items():
                if gram in self.vocabulary_:
                    rows.append(row_index)
                    cols.append(self.vocabulary_[gram])
                    data.append(float(count))
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(len(texts), len(self.vocabulary_)), dtype=np.float64)
        if self.output == "sparse":
            return matrix
        if self.output == "pandas":
            return pd.DataFrame(matrix.toarray(), columns=self.feature_names_out_, index=series.index)
        return matrix.toarray()

    def fit_transform(self, x: object, y: object) -> sparse.csr_matrix | pd.DataFrame | np.ndarray:
        """Fit, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable fitted n-gram feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def _validate_ngram_range(ngram_range: tuple[int, int]) -> None:
    lower, upper = ngram_range
    if lower <= 0:
        raise ValueError("ngram_range lower bound must be positive")
    if upper < lower:
        raise ValueError("ngram_range upper bound must be greater than or equal to lower bound")


def _extract_ngrams(view: DocumentView, analyzer: Analyzer, ngram_range: tuple[int, int]) -> list[str]:
    if analyzer == "char":
        return _char_ngrams(view.raw, ngram_range)
    if analyzer == "punctuation":
        return _punctuation_ngrams(view.raw, ngram_range)
    tokens = list(view.tokens)
    if analyzer == "function_word":
        function_words = function_words_lexicon()
        tokens = [token for token in tokens if token in function_words]
    return _token_ngrams(tokens, ngram_range)


def _token_ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    grams: list[str] = []
    lower, upper = ngram_range
    for ngram_size in range(lower, upper + 1):
        if len(tokens) < ngram_size:
            continue
        grams.extend(" ".join(tokens[index : index + ngram_size]) for index in range(0, len(tokens) - ngram_size + 1))
    return grams


def _char_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    grams: list[str] = []
    lower, upper = ngram_range
    for ngram_size in range(lower, upper + 1):
        if len(text) < ngram_size:
            continue
        grams.extend(text[index : index + ngram_size] for index in range(0, len(text) - ngram_size + 1))
    return grams


def _punctuation_ngrams(text: str, ngram_range: tuple[int, int]) -> list[str]:
    punctuation_marks = [character for character in text if unicodedata.category(character).startswith("P")]
    grams: list[str] = []
    lower, upper = ngram_range
    for ngram_size in range(lower, upper + 1):
        if len(punctuation_marks) < ngram_size:
            continue
        grams.extend("".join(punctuation_marks[index : index + ngram_size]) for index in range(0, len(punctuation_marks) - ngram_size + 1))
    return grams


def _select_items(sorted_items: list[tuple[str, int]], max_features: int | None) -> list[tuple[str, int]]:
    if max_features is None:
        return sorted_items
    return sorted_items[:max_features]


def _feature_name(analyzer: Analyzer, gram: str) -> str:
    escaped = gram.replace("\n", "\\n")
    if analyzer == "word":
        return f"text::word_ngram::gram={escaped}"
    if analyzer == "function_word":
        return f"text::function_word_ngram::gram={escaped}"
    if analyzer == "punctuation":
        return f"text::punctuation_ngram::gram={escaped}"
    return f"text::char_ngram::gram={escaped}"


def _spec_for_name(name: str, analyzer: Analyzer) -> FeatureSpec:
    topic_dependence = TopicDependence.MIXED
    if analyzer == "function_word":
        topic_dependence = TopicDependence.MOSTLY_TOPIC_INDEPENDENT
    if analyzer == "punctuation":
        topic_dependence = TopicDependence.MOSTLY_TOPIC_INDEPENDENT
    if analyzer == "word":
        topic_dependence = TopicDependence.TOPIC_SENSITIVE
    input_layer = InputLayer.TOKENS
    if analyzer in {"char", "punctuation"}:
        input_layer = InputLayer.RAW
    return FeatureSpec(
        name=name,
        family=f"{analyzer}_ngram",
        description="Fitted sparse n-gram count feature",
        formula_or_rule="count fitted n-gram occurrences in the configured DocumentView layer",
        input_layer=input_layer,
        topic_dependence=topic_dependence,
        text_length_policy="sparse count is defined as zero when the fitted gram is absent",
        provenance="built_in_ngram_rules:v1; fitted_vocabulary; preprocessing_config",
        output_dtype="float64",
        undefined_behavior="not undefined after fit; absent fitted grams produce valid zero counts",
        normalization="raw_count",
        sparsity="sparse_vector",
        stability_status=StabilityStatus.STATISTICAL_FIT_DEPENDENT,
    )
