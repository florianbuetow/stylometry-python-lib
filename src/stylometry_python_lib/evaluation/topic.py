"""Utilities for topic-leakage and style-vs-topic evaluation."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib.evaluation.distances import as_float_matrix


@dataclass(frozen=True)
class TopicPredictionReport:
    """Cross-validated topic-prediction control report."""

    leakage_score: float
    fold_scores: tuple[float, ...]
    true_topics: tuple[str, ...]
    predicted_topics: tuple[str, ...]
    classes: tuple[str, ...]
    train_indices_by_fold: tuple[tuple[int, ...], ...]
    test_indices_by_fold: tuple[tuple[int, ...], ...]
    classifier_name: str
    cv_folds: int
    random_state: int


class TopicPredictionControl(BaseEstimator):
    """Sklearn-compatible topic classifier control for leakage checks."""

    def __init__(self, cv_folds: int, random_state: int, max_iter: int) -> None:
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.max_iter = max_iter

    def fit(self, x: object, y: object) -> Self:
        """Evaluate cross-validated topic leakage and fit a final topic classifier."""
        matrix = as_float_matrix(x)
        labels = _validate_topic_labels(y, matrix.shape[0])
        report = _topic_prediction_cv_report(
            matrix=matrix,
            labels=labels,
            cv_folds=self.cv_folds,
            random_state=self.random_state,
            max_iter=self.max_iter,
        )
        classifier = LogisticRegression(max_iter=self.max_iter, random_state=self.random_state)
        classifier.fit(matrix, labels)
        self.classifier_ = classifier
        self.report_ = report
        self.classes_ = np.asarray(report.classes, dtype=object)
        self.n_features_in_ = matrix.shape[1]
        return self

    def predict(self, x: object) -> NDArray[np.object_]:
        """Predict topic labels with the fitted final classifier."""
        require_fitted(self, "classifier_")
        matrix = as_float_matrix(x)
        if matrix.shape[1] != self.n_features_in_:
            raise ValueError("Input feature width changed after fit")
        return np.asarray(self.classifier_.predict(matrix), dtype=object)

    def score(self, x: object, y: object) -> float:
        """Return topic-classification accuracy for the fitted final classifier."""
        predictions = self.predict(x)
        labels = _validate_topic_labels(y, predictions.shape[0])
        return float(accuracy_score(labels, predictions))


def content_mask_text(text: str, content_words: frozenset[str], replacement_token: str) -> str:
    """Mask configured content words with a replacement token."""
    if len(replacement_token) == 0:
        raise ValueError("replacement_token must not be empty")
    pattern = re.compile(r"\b\w+\b")

    def replace(match: re.Match[str]) -> str:
        word = match.group(0)
        if word.lower() in content_words:
            return replacement_token
        return word

    return pattern.sub(replace, text)


def cross_topic_holdout_indices(topics: Sequence[str], holdout_topic: str) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Return train/test indices for one held-out topic."""
    train: list[int] = []
    test: list[int] = []
    for index, topic in enumerate(topics):
        if topic == holdout_topic:
            test.append(index)
        else:
            train.append(index)
    if len(train) == 0 or len(test) == 0:
        raise ValueError("holdout_topic must leave at least one train and one test row")
    return np.asarray(train, dtype=np.int64), np.asarray(test, dtype=np.int64)


def same_topic_hard_negative_pairs(authors: Sequence[str], topics: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Return same-topic/different-author index pairs."""
    _validate_parallel_labels(authors, topics)
    pairs = [
        (left, right)
        for left in range(len(authors))
        for right in range(left + 1, len(authors))
        if topics[left] == topics[right] and authors[left] != authors[right]
    ]
    return tuple(pairs)


def same_author_topic_shift_pairs(authors: Sequence[str], topics: Sequence[str]) -> tuple[tuple[int, int], ...]:
    """Return same-author/different-topic index pairs."""
    _validate_parallel_labels(authors, topics)
    pairs = [
        (left, right)
        for left in range(len(authors))
        for right in range(left + 1, len(authors))
        if authors[left] == authors[right] and topics[left] != topics[right]
    ]
    return tuple(pairs)


def topic_prediction_leakage_score(features: object, topics: Sequence[str], cv_folds: int, random_state: int) -> float:
    """Estimate how well features predict topic; high scores warn of topic leakage."""
    report = topic_prediction_control_report(
        features=features,
        topics=topics,
        cv_folds=cv_folds,
        random_state=random_state,
        max_iter=1000,
    )
    return report.leakage_score


def topic_prediction_control_report(
    features: object,
    topics: Sequence[str],
    cv_folds: int,
    random_state: int,
    max_iter: int,
) -> TopicPredictionReport:
    """Return a deterministic cross-validated topic-prediction report."""
    matrix = as_float_matrix(features)
    labels = _validate_topic_labels(topics, matrix.shape[0])
    return _topic_prediction_cv_report(
        matrix=matrix,
        labels=labels,
        cv_folds=cv_folds,
        random_state=random_state,
        max_iter=max_iter,
    )


def _topic_prediction_cv_report(
    matrix: NDArray[np.float64],
    labels: NDArray[np.object_],
    cv_folds: int,
    random_state: int,
    max_iter: int,
) -> TopicPredictionReport:
    _validate_topic_prediction_config(cv_folds, max_iter, matrix.shape[0])
    classes = tuple(str(label) for label in np.unique(labels).tolist())
    if len(classes) < 2:
        raise ValueError("topics must contain at least two distinct labels")
    predicted_topics = np.empty(labels.shape, dtype=object)
    fold_scores: list[float] = []
    train_indices_by_fold: list[tuple[int, ...]] = []
    test_indices_by_fold: list[tuple[int, ...]] = []
    for fold_index in range(cv_folds):
        train_indices = [index for index in range(matrix.shape[0]) if index % cv_folds != fold_index]
        test_indices = [index for index in range(matrix.shape[0]) if index % cv_folds == fold_index]
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise ValueError("Each cross-validation fold must contain train and test rows")
        train_labels = labels[train_indices]
        if len(np.unique(train_labels)) < 2:
            raise ValueError("Each topic-prediction fold must contain at least two train topic labels")
        model = LogisticRegression(max_iter=max_iter, random_state=random_state)
        model.fit(matrix[train_indices], labels[train_indices])
        predictions = model.predict(matrix[test_indices])
        predicted_topics[test_indices] = predictions
        fold_scores.append(float(accuracy_score(labels[test_indices], predictions)))
        train_indices_by_fold.append(tuple(train_indices))
        test_indices_by_fold.append(tuple(test_indices))
    leakage_score = float(np.mean(np.asarray(fold_scores, dtype=np.float64)))
    return TopicPredictionReport(
        leakage_score=leakage_score,
        fold_scores=tuple(fold_scores),
        true_topics=tuple(str(label) for label in labels.tolist()),
        predicted_topics=tuple(str(label) for label in predicted_topics.tolist()),
        classes=classes,
        train_indices_by_fold=tuple(train_indices_by_fold),
        test_indices_by_fold=tuple(test_indices_by_fold),
        classifier_name="LogisticRegression",
        cv_folds=cv_folds,
        random_state=random_state,
    )


def ablation_scores(
    features_by_family: dict[str, object],
    labels: Sequence[str],
    scorer: Callable[[NDArray[np.float64], Sequence[str]], float],
) -> dict[str, float]:
    """Compute one score per feature family for ablation studies."""
    scores: dict[str, float] = {}
    for family, features in features_by_family.items():
        matrix = as_float_matrix(features)
        if len(labels) != matrix.shape[0]:
            raise ValueError(f"labels length does not match row count for family {family}")
        scores[family] = scorer(matrix, labels)
    return scores


def length_sensitivity(features: object, token_counts: Sequence[int]) -> NDArray[np.float64]:
    """Compute per-feature absolute correlation with text length."""
    matrix = as_float_matrix(features)
    if len(token_counts) != matrix.shape[0]:
        raise ValueError("token_counts length must match feature row count")
    lengths = np.asarray(token_counts, dtype=np.float64)
    if np.std(lengths) == 0.0:
        raise ValueError("length sensitivity is undefined when token counts have zero variance")
    correlations: list[float] = []
    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        if np.nanstd(column) == 0.0:
            correlations.append(0.0)
        else:
            corr = np.corrcoef(lengths, column)[0, 1]
            correlations.append(abs(float(corr)))
    return np.asarray(correlations, dtype=np.float64)


def two_way_effect_sizes(values: Sequence[float], authors: Sequence[str], topics: Sequence[str]) -> dict[str, float]:
    """Estimate author, topic, and residual variance shares for one feature."""
    if len(values) != len(authors) or len(values) != len(topics):
        raise ValueError("values, authors, and topics must have the same length")
    y = np.asarray(values, dtype=np.float64)
    grand_mean = float(np.mean(y))
    total_ss = float(np.sum((y - grand_mean) ** 2.0))
    if total_ss == 0.0:
        raise ValueError("two-way effect sizes are undefined for zero total variance")
    author_ss = _group_sum_squares(y, authors, grand_mean)
    topic_ss = _group_sum_squares(y, topics, grand_mean)
    explained = author_ss + topic_ss
    residual = total_ss - explained
    if residual < 0.0:
        residual = 0.0
    return {
        "author_effect": author_ss / total_ss,
        "topic_effect": topic_ss / total_ss,
        "residual_effect": residual / total_ss,
    }


def simple_logistic_accuracy(features: NDArray[np.float64], labels: Sequence[str]) -> float:
    """Fit-and-score logistic regression on the same matrix for small ablation smoke tests."""
    if len(labels) != features.shape[0]:
        raise ValueError("labels length must match feature row count")
    model = LogisticRegression(max_iter=1000)
    model.fit(features, np.asarray(labels, dtype=object))
    predictions = model.predict(features)
    return float(accuracy_score(np.asarray(labels, dtype=object), predictions))


def _validate_topic_prediction_config(cv_folds: int, max_iter: int, sample_count: int) -> None:
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least 2")
    if cv_folds > sample_count:
        raise ValueError("cv_folds must not exceed feature row count")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")


def _validate_topic_labels(topics: object, sample_count: int) -> NDArray[np.object_]:
    if topics is None:
        raise ValueError("topic labels are required")
    labels = np.asarray(topics, dtype=object)
    if labels.ndim != 1:
        raise ValueError("topic labels must be one-dimensional")
    if labels.shape[0] != sample_count:
        raise ValueError("topics length must match feature row count")
    for label in labels.tolist():
        if not isinstance(label, str):
            raise ValueError("topic labels must be strings")
        if len(label) == 0:
            raise ValueError("topic labels must not be empty")
    return labels


def _validate_parallel_labels(authors: Sequence[str], topics: Sequence[str]) -> None:
    if len(authors) != len(topics):
        raise ValueError("authors and topics must have the same length")


def _group_sum_squares(values: NDArray[np.float64], labels: Sequence[str], grand_mean: float) -> float:
    groups: defaultdict[str, list[float]] = defaultdict(list)
    for value, label in zip(values.tolist(), labels, strict=True):
        groups[label].append(float(value))
    total = 0.0
    for grouped_values in groups.values():
        group_mean = float(np.mean(np.asarray(grouped_values, dtype=np.float64)))
        total += float(len(grouped_values)) * ((group_mean - grand_mean) ** 2.0)
    return total


@dataclass(frozen=True)
class TopicModelReport:
    """Topic-model decomposition with per-document topic loadings and top terms."""

    method: str
    n_topics: int
    document_topic_matrix: tuple[tuple[float, ...], ...]
    top_terms_by_topic: tuple[tuple[str, ...], ...]
    perplexity: float


def topic_model_report(
    counts: object,
    feature_names: Sequence[str],
    n_topics: int,
    method: str,
    random_state: int,
    top_terms: int,
) -> TopicModelReport:
    """Fit an LDA or NMF topic model and return loadings and top terms per topic."""
    matrix = np.asarray(counts, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError("counts must be a 2D document-term matrix")
    names = tuple(str(name) for name in feature_names)
    if len(names) != matrix.shape[1]:
        raise ValueError("feature_names length must match counts columns")
    if n_topics < 1:
        raise ValueError("n_topics must be >= 1")
    if top_terms < 1:
        raise ValueError("top_terms must be >= 1")
    if method == "lda":
        lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state)
        lda.fit(matrix)
        loadings = np.asarray(lda.transform(matrix), dtype=np.float64)
        components = np.asarray(lda.components_, dtype=np.float64)
        perplexity = float(lda.perplexity(matrix))
    elif method == "nmf":
        nmf = NMF(n_components=n_topics, random_state=random_state, init="nndsvda", max_iter=500)
        loadings = np.asarray(nmf.fit_transform(matrix), dtype=np.float64)
        components = np.asarray(nmf.components_, dtype=np.float64)
        perplexity = float("nan")
    else:
        raise ValueError(f"Unsupported topic-model method: {method}")
    top = tuple(tuple(names[int(i)] for i in np.argsort(component)[::-1][:top_terms]) for component in components)
    document_topic_matrix = tuple(tuple(float(value) for value in row) for row in loadings)
    return TopicModelReport(
        method=method,
        n_topics=n_topics,
        document_topic_matrix=document_topic_matrix,
        top_terms_by_topic=top,
        perplexity=perplexity,
    )
