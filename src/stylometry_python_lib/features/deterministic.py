"""Deterministic stylometry feature extraction."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib._tabular import text_series, validate_output_mode
from stylometry_python_lib.document import DocumentView, PreprocessingConfig
from stylometry_python_lib.lexicons import load_lexicon, load_spelling_variants
from stylometry_python_lib.registry import FeatureRegistry
from stylometry_python_lib.specs import FeatureSpec, InputLayer, StabilityStatus, TopicDependence
from stylometry_python_lib.text_metrics import syllable_count
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus, FeatureValue, defined_value, undefined_value

_VOCD_MIN_SAMPLE_SIZE = 35
_VOCD_MAX_SAMPLE_SIZE = 50
_VOCD_SAMPLES_PER_SIZE = 100
_VOCD_D_LOWER_BOUND = 0.01
_VOCD_D_UPPER_BOUND = 10000.0
_VOCD_FIT_ITERATIONS = 80
_VOCD_RANDOM_SEED = "stylometry_python_lib_vocd_d_v1"
_LEXICAL_RICHNESS_FORMULAS = {
    "text::lexical_richness::ttr": "types / tokens",
    "text::lexical_richness::cttr": "V / sqrt(2N)",
    "text::lexical_richness::herdan_c": "log(V) / log(N)",
    "text::lexical_richness::msttr": "mean type-token ratio over contiguous 50-token segments",
    "text::lexical_richness::mattr": "mean type-token ratio over a moving 50-token window",
    "text::lexical_richness::mtld": "bidirectional mean segment length before TTR falls below 0.72",
    "text::lexical_richness::hdd": "hypergeometric diversity over a sample size of min(42, N)",
    "text::lexical_richness::vocd_d": "vocd-D hash-sampled TTR curve fit over 100 deterministic samples for each size 35 through 50",
    "text::lexical_richness::yules_k": "10000 * (sum(i^2 * V_i) - N) / N^2",
    "text::lexical_richness::honore_r": "100 * log(N) / (1 - V1 / V)",
    "text::lexical_richness::guiraud_r": "V / sqrt(N)",
    "text::lexical_richness::sichel_s": "V2 / V",
    "text::lexical_richness::simpson_d": "sum(f_i * (f_i - 1)) / (N * (N - 1))",
    "text::lexical_richness::renyi_entropy_alpha_2": "-log(sum((f_i / N)^2))",
}


class DeterministicStylometryTransformer(BaseEstimator):
    """Sklearn-compatible deterministic scalar stylometry transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze scalar feature metadata."""
        del y
        validate_output_mode(self.output)
        names = deterministic_feature_names()
        specs = deterministic_feature_specs()
        registry = FeatureRegistry(specs=specs)
        registry.require_complete()
        self.feature_names_out_ = np.asarray(names, dtype=object)
        self.registry_ = registry
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        _ = text_series(x, self.text_column)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute deterministic scalar stylometry features."""
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[dict[str, float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        for row_index, text in enumerate(series.tolist()):
            view = DocumentView.from_text(str(text), self.config, document_id=str(series.index[row_index]))
            values = compute_deterministic_values(view)
            rows.append({name: values[name].value for name in self.feature_names_out_.tolist()})
            diagnostics.append(_diagnostics(values))
        self.last_diagnostics_ = tuple(diagnostics)
        frame = pd.DataFrame(rows, columns=self.feature_names_out_.tolist(), index=series.index)
        if self.output == "pandas":
            return frame
        if self.output == "sparse":
            return sparse.csr_matrix(frame.to_numpy(dtype=float))
        return frame.to_numpy(dtype=float)

    def fit_transform(self, x: object, y: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Fit, then transform with explicit target metadata."""
        return self.fit(x, y).transform(x)

    def get_feature_names_out(self, input_features: object) -> np.ndarray:
        """Return stable scalar feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def deterministic_feature_names() -> tuple[str, ...]:
    """Return deterministic scalar feature names in output order."""
    return (
        "text::counts::token_count",
        "text::counts::type_count",
        "text::counts::sentence_count",
        "text::counts::paragraph_count",
        "text::counts::character_count",
        "text::counts::letter_count",
        "text::lexical_richness::ttr",
        "text::lexical_richness::cttr",
        "text::lexical_richness::herdan_c",
        "text::lexical_richness::msttr",
        "text::lexical_richness::mattr",
        "text::lexical_richness::mtld",
        "text::lexical_richness::hdd",
        "text::lexical_richness::vocd_d",
        "text::lexical_richness::hapax_count",
        "text::lexical_richness::dis_legomena_count",
        "text::lexical_richness::yules_k",
        "text::lexical_richness::honore_r",
        "text::lexical_richness::guiraud_r",
        "text::lexical_richness::sichel_s",
        "text::lexical_richness::simpson_d",
        "text::lexical_richness::renyi_entropy_alpha_2",
        "text::length::word_mean",
        "text::length::word_median",
        "text::length::word_std",
        "text::length::syllables_per_word_mean",
        "text::length::sentence_tokens_mean",
        "text::length::sentence_tokens_std",
        "text::length::paragraph_tokens_mean",
        "text::length::line_characters_mean",
        "text::closed_class::function_word_ratio",
        "text::closed_class::stopword_ratio",
        "text::stance::pronoun_ratio",
        "text::stance::first_person_ratio",
        "text::stance::second_person_ratio",
        "text::stance::third_person_ratio",
        "text::stance::modal_verb_ratio",
        "text::grammar::auxiliary_verb_ratio",
        "text::register::contraction_count",
        "text::lexical_density::content_lexicon_ratio",
        "text::orthography::punctuation_count",
        "text::orthography::punctuation_per_token",
        "text::orthography::punctuation_sequence_count",
        "text::orthography::uppercase_character_ratio",
        "text::orthography::all_caps_word_count",
        "text::orthography::titlecase_line_count",
        "text::orthography::spelling_variant_count",
        "text::orthography::hyphenated_token_count",
        "text::orthography::abbreviation_count",
        "text::orthography::acronym_count",
        "text::discourse::quote_marker_count",
        "text::discourse::dialogue_dash_count",
        "text::discourse::discourse_marker_count",
        "text::discourse::transition_phrase_count",
        "text::layout::heading_line_count",
        "text::layout::bullet_line_count",
        "text::layout::numbered_line_count",
        "text::layout::code_fence_count",
        "text::layout::table_line_count",
        "text::layout::greeting_count",
        "text::layout::signoff_count",
        "text::whitespace::blank_line_count",
        "text::whitespace::line_break_count",
        "text::readability::flesch_reading_ease",
        "text::readability::flesch_kincaid_grade",
        "text::readability::gunning_fog",
        "text::readability::coleman_liau",
        "text::readability::smog",
        "text::readability::ari",
        "text::readability::dale_chall",
        "text::readability::forcast",
        "text::readability::linsear_write",
        "text::readability::lix",
    )


def deterministic_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return metadata for deterministic scalar features."""
    specs: list[FeatureSpec] = []
    for name in deterministic_feature_names():
        family = _family_from_name(name)
        specs.append(
            FeatureSpec(
                name=name,
                family=family,
                description=f"Deterministic stylometry scalar for {name}",
                formula_or_rule=_formula_for_name(name),
                input_layer=InputLayer.MULTI,
                topic_dependence=_topic_dependence_for_name(name),
                text_length_policy=_length_policy_for_name(name),
                provenance=_provenance_for_name(name),
                output_dtype="float64",
                undefined_behavior="NaN with FeatureDiagnostic reason for empty text, zero denominator, or unstable formula",
                normalization=_normalization_for_name(name),
                sparsity="dense_scalar",
                stability_status=StabilityStatus.DETERMINISTIC,
            )
        )
    return tuple(specs)


def compute_deterministic_values(view: DocumentView) -> dict[str, FeatureValue]:
    """Compute deterministic scalar features for one DocumentView."""
    values: dict[str, FeatureValue] = {}
    tokens = view.tokens
    token_count = len(tokens)
    token_counts = Counter(tokens)
    type_count = len(token_counts)
    sentence_token_lengths = _sentence_token_lengths(view)
    paragraph_token_lengths = _paragraph_token_lengths(view)
    syllable_counts = [syllable_count(token) for token in tokens]
    word_lengths = [len(token) for token in tokens]
    raw = view.raw
    lower_raw = raw.lower()

    _add_count_features(values, view, token_count, type_count)
    _add_richness_features(values, tokens, token_counts)
    _add_length_features(values, word_lengths, syllable_counts, sentence_token_lengths, paragraph_token_lengths, raw)
    _add_closed_class_features(values, tokens)
    _add_orthography_features(values, view, lower_raw, token_count)
    _add_discourse_layout_features(values, raw, lower_raw)
    _add_readability_features(values, tokens, sentence_token_lengths, syllable_counts, word_lengths)

    for name in deterministic_feature_names():
        if name not in values:
            raise KeyError(f"Missing deterministic feature computation: {name}")
    return values


def _add_count_features(values: dict[str, FeatureValue], view: DocumentView, token_count: int, type_count: int) -> None:
    values["text::counts::token_count"] = defined_value("text::counts::token_count", float(token_count), ())
    values["text::counts::type_count"] = defined_value("text::counts::type_count", float(type_count), ())
    values["text::counts::sentence_count"] = defined_value("text::counts::sentence_count", float(len(view.sentences)), ())
    values["text::counts::paragraph_count"] = defined_value("text::counts::paragraph_count", float(len(view.paragraphs)), ())
    values["text::counts::character_count"] = defined_value("text::counts::character_count", float(len(view.raw)), ())
    letter_count = sum(1 for char in view.raw if char.isalpha())
    values["text::counts::letter_count"] = defined_value("text::counts::letter_count", float(letter_count), ())


def _add_richness_features(values: dict[str, FeatureValue], tokens: tuple[str, ...], token_counts: Counter[str]) -> None:
    token_count = len(tokens)
    type_count = len(token_counts)
    hapax_count = sum(1 for count in token_counts.values() if count == 1)
    dis_count = sum(1 for count in token_counts.values() if count == 2)
    values["text::lexical_richness::ttr"] = _ratio_value("text::lexical_richness::ttr", type_count, token_count, "zero_tokens")
    values["text::lexical_richness::cttr"] = _cttr(token_count, type_count)
    values["text::lexical_richness::herdan_c"] = _herdan_c(token_count, type_count)
    values["text::lexical_richness::msttr"] = _msttr(tokens)
    values["text::lexical_richness::mattr"] = _mattr(tokens)
    values["text::lexical_richness::mtld"] = _mtld(tokens)
    values["text::lexical_richness::hdd"] = _hdd(tokens, token_counts)
    values["text::lexical_richness::vocd_d"] = _vocd_d(tokens)
    values["text::lexical_richness::hapax_count"] = defined_value(
        "text::lexical_richness::hapax_count", float(hapax_count), _short_warning(token_count)
    )
    values["text::lexical_richness::dis_legomena_count"] = defined_value(
        "text::lexical_richness::dis_legomena_count", float(dis_count), _short_warning(token_count)
    )
    values["text::lexical_richness::yules_k"] = _yules_k(token_count, token_counts)
    values["text::lexical_richness::honore_r"] = _honore_r(token_count, type_count, hapax_count)
    values["text::lexical_richness::guiraud_r"] = _guiraud_r(token_count, type_count)
    values["text::lexical_richness::sichel_s"] = _ratio_value("text::lexical_richness::sichel_s", dis_count, type_count, "zero_types")
    values["text::lexical_richness::simpson_d"] = _simpson_d(token_count, token_counts)
    values["text::lexical_richness::renyi_entropy_alpha_2"] = _renyi_entropy(token_count, token_counts)


def _add_length_features(
    values: dict[str, FeatureValue],
    word_lengths: list[int],
    syllable_counts: list[int],
    sentence_lengths: list[int],
    paragraph_lengths: list[int],
    raw: str,
) -> None:
    values["text::length::word_mean"] = _mean_value("text::length::word_mean", word_lengths, "zero_tokens")
    values["text::length::word_median"] = _median_value("text::length::word_median", word_lengths, "zero_tokens")
    values["text::length::word_std"] = _std_value("text::length::word_std", word_lengths, "zero_tokens")
    values["text::length::syllables_per_word_mean"] = _mean_value("text::length::syllables_per_word_mean", syllable_counts, "zero_tokens")
    values["text::length::sentence_tokens_mean"] = _mean_value("text::length::sentence_tokens_mean", sentence_lengths, "zero_sentences")
    values["text::length::sentence_tokens_std"] = _std_value("text::length::sentence_tokens_std", sentence_lengths, "zero_sentences")
    values["text::length::paragraph_tokens_mean"] = _mean_value("text::length::paragraph_tokens_mean", paragraph_lengths, "zero_paragraphs")
    line_lengths = [len(line) for line in raw.splitlines()]
    values["text::length::line_characters_mean"] = _mean_value("text::length::line_characters_mean", line_lengths, "zero_lines")


def _add_closed_class_features(values: dict[str, FeatureValue], tokens: tuple[str, ...]) -> None:
    token_count = len(tokens)
    function_words = function_words_lexicon()
    stopwords = _stopwords()
    pronouns = _pronouns()
    values["text::closed_class::function_word_ratio"] = _ratio_value(
        "text::closed_class::function_word_ratio", sum(1 for token in tokens if token in function_words), token_count, "zero_tokens"
    )
    values["text::closed_class::stopword_ratio"] = _ratio_value(
        "text::closed_class::stopword_ratio", sum(1 for token in tokens if token in stopwords), token_count, "zero_tokens"
    )
    values["text::stance::pronoun_ratio"] = _ratio_value(
        "text::stance::pronoun_ratio", sum(1 for token in tokens if token in pronouns), token_count, "zero_tokens"
    )
    values["text::stance::first_person_ratio"] = _ratio_value(
        "text::stance::first_person_ratio", sum(1 for token in tokens if token in _first_person_pronouns()), token_count, "zero_tokens"
    )
    values["text::stance::second_person_ratio"] = _ratio_value(
        "text::stance::second_person_ratio", sum(1 for token in tokens if token in _second_person_pronouns()), token_count, "zero_tokens"
    )
    values["text::stance::third_person_ratio"] = _ratio_value(
        "text::stance::third_person_ratio", sum(1 for token in tokens if token in _third_person_pronouns()), token_count, "zero_tokens"
    )
    values["text::stance::modal_verb_ratio"] = _ratio_value(
        "text::stance::modal_verb_ratio", sum(1 for token in tokens if token in _modal_verbs()), token_count, "zero_tokens"
    )
    values["text::grammar::auxiliary_verb_ratio"] = _ratio_value(
        "text::grammar::auxiliary_verb_ratio", sum(1 for token in tokens if token in _auxiliary_verbs()), token_count, "zero_tokens"
    )
    content_count = sum(1 for token in tokens if token.isalpha() and token not in stopwords)
    values["text::lexical_density::content_lexicon_ratio"] = _ratio_value(
        "text::lexical_density::content_lexicon_ratio", content_count, token_count, "zero_tokens"
    )


def _add_orthography_features(values: dict[str, FeatureValue], view: DocumentView, lower_raw: str, token_count: int) -> None:
    punctuation_count = sum(1 for char in view.raw if char in _punctuation_characters())
    values["text::register::contraction_count"] = defined_value(
        "text::register::contraction_count", float(len(re.findall(r"\b\w+'\w+\b", view.raw))), ()
    )
    values["text::orthography::punctuation_count"] = defined_value("text::orthography::punctuation_count", float(punctuation_count), ())
    values["text::orthography::punctuation_per_token"] = _ratio_value(
        "text::orthography::punctuation_per_token", punctuation_count, token_count, "zero_tokens"
    )
    values["text::orthography::punctuation_sequence_count"] = defined_value(
        "text::orthography::punctuation_sequence_count", float(len(re.findall(r"[.!?,;:\-]{2,}", view.raw))), ()
    )
    letter_count = sum(1 for char in view.raw if char.isalpha())
    uppercase_count = sum(1 for char in view.raw if char.isupper())
    values["text::orthography::uppercase_character_ratio"] = _ratio_value(
        "text::orthography::uppercase_character_ratio", uppercase_count, letter_count, "zero_letters"
    )
    values["text::orthography::all_caps_word_count"] = defined_value(
        "text::orthography::all_caps_word_count", float(len(re.findall(r"\b[A-Z]{2,}\b", view.raw))), ()
    )
    values["text::orthography::titlecase_line_count"] = defined_value(
        "text::orthography::titlecase_line_count", float(sum(1 for line in view.raw.splitlines() if _is_titlecase_line(line))), ()
    )
    variant_count = sum(lower_raw.count(variant) for pair in _spelling_variant_pairs() for variant in pair)
    values["text::orthography::spelling_variant_count"] = defined_value(
        "text::orthography::spelling_variant_count", float(variant_count), ()
    )
    values["text::orthography::hyphenated_token_count"] = defined_value(
        "text::orthography::hyphenated_token_count", float(len(re.findall(r"\b[A-Za-z]+(?:-[A-Za-z]+)+\b", view.raw))), ()
    )
    values["text::orthography::abbreviation_count"] = defined_value(
        "text::orthography::abbreviation_count", float(len(re.findall(r"\b(?:e\.g|i\.e|etc|vs|mr|mrs|dr)\.?", lower_raw))), ()
    )
    values["text::orthography::acronym_count"] = defined_value(
        "text::orthography::acronym_count", float(len(re.findall(r"\b[A-Z]{2,}\b", view.raw))), ()
    )


def _add_discourse_layout_features(values: dict[str, FeatureValue], raw: str, lower_raw: str) -> None:
    lines = raw.splitlines()
    values["text::discourse::quote_marker_count"] = defined_value(
        "text::discourse::quote_marker_count", float(raw.count('"') + raw.count("'") + raw.count("“") + raw.count("”")), ()
    )
    values["text::discourse::dialogue_dash_count"] = defined_value(
        "text::discourse::dialogue_dash_count", float(sum(1 for line in lines if line.strip().startswith("-"))), ()
    )
    values["text::discourse::discourse_marker_count"] = defined_value(
        "text::discourse::discourse_marker_count", float(sum(lower_raw.count(marker) for marker in _discourse_markers())), ()
    )
    values["text::discourse::transition_phrase_count"] = defined_value(
        "text::discourse::transition_phrase_count", float(sum(lower_raw.count(phrase) for phrase in _transition_phrases())), ()
    )
    values["text::layout::heading_line_count"] = defined_value(
        "text::layout::heading_line_count", float(sum(1 for line in lines if _is_heading_line(line))), ()
    )
    values["text::layout::bullet_line_count"] = defined_value(
        "text::layout::bullet_line_count", float(sum(1 for line in lines if re.match(r"^\s*[-*]\s+", line) is not None)), ()
    )
    values["text::layout::numbered_line_count"] = defined_value(
        "text::layout::numbered_line_count", float(sum(1 for line in lines if re.match(r"^\s*\d+[.)]\s+", line) is not None)), ()
    )
    values["text::layout::code_fence_count"] = defined_value("text::layout::code_fence_count", float(raw.count("```")), ())
    values["text::layout::table_line_count"] = defined_value(
        "text::layout::table_line_count", float(sum(1 for line in lines if "|" in line and line.count("|") >= 2)), ()
    )
    values["text::layout::greeting_count"] = defined_value(
        "text::layout::greeting_count", float(len(re.findall(r"\b(dear|hello|hi)\b", lower_raw))), ()
    )
    values["text::layout::signoff_count"] = defined_value(
        "text::layout::signoff_count", float(len(re.findall(r"\b(regards|sincerely|cheers|best)\b", lower_raw))), ()
    )
    values["text::whitespace::blank_line_count"] = defined_value(
        "text::whitespace::blank_line_count", float(len(re.findall(r"\n\s*\n", raw))), ()
    )
    values["text::whitespace::line_break_count"] = defined_value("text::whitespace::line_break_count", float(raw.count("\n")), ())


def _add_readability_features(
    values: dict[str, FeatureValue],
    tokens: tuple[str, ...],
    sentence_token_lengths: list[int],
    syllable_counts: list[int],
    word_lengths: list[int],
) -> None:
    token_count = len(tokens)
    sentence_count = len(sentence_token_lengths)
    syllables = sum(syllable_counts)
    characters = sum(word_lengths)
    complex_words = sum(1 for count in syllable_counts if count >= 3)
    single_syllable_words = sum(1 for count in syllable_counts if count == 1)
    long_words = sum(1 for length in word_lengths if length > 6)
    values["text::readability::flesch_reading_ease"] = _readability_value(
        "text::readability::flesch_reading_ease", 206.835, -1.015, -84.6, token_count, sentence_count, syllables
    )
    values["text::readability::flesch_kincaid_grade"] = _readability_value(
        "text::readability::flesch_kincaid_grade", -15.59, 0.39, 11.8, token_count, sentence_count, syllables
    )
    values["text::readability::gunning_fog"] = _gunning_fog(token_count, sentence_count, complex_words)
    values["text::readability::coleman_liau"] = _coleman_liau(token_count, sentence_count, characters)
    values["text::readability::smog"] = _smog(sentence_count, complex_words)
    values["text::readability::ari"] = _ari(token_count, sentence_count, characters)
    values["text::readability::dale_chall"] = _dale_chall(tokens, token_count, sentence_count)
    values["text::readability::forcast"] = _forcast(token_count, single_syllable_words)
    values["text::readability::linsear_write"] = _linsear_write(syllable_counts, sentence_token_lengths)
    values["text::readability::lix"] = _lix(token_count, sentence_count, long_words)


def _ratio_value(name: str, numerator: int, denominator: int, zero_reason: str) -> FeatureValue:
    if denominator == 0:
        return undefined_value(name, zero_reason, ())
    return defined_value(name, float(numerator) / float(denominator), _short_warning(denominator))


def _cttr(token_count: int, type_count: int) -> FeatureValue:
    name = "text::lexical_richness::cttr"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    return defined_value(name, float(type_count) / math.sqrt(2.0 * float(token_count)), _short_warning(token_count))


def _herdan_c(token_count: int, type_count: int) -> FeatureValue:
    name = "text::lexical_richness::herdan_c"
    if token_count <= 1:
        return undefined_value(name, "insufficient_tokens_for_log", _short_warning(token_count))
    if type_count == 0:
        return undefined_value(name, "zero_types", _short_warning(token_count))
    return defined_value(name, math.log(float(type_count)) / math.log(float(token_count)), _short_warning(token_count))


def _msttr(tokens: tuple[str, ...]) -> FeatureValue:
    name = "text::lexical_richness::msttr"
    if len(tokens) == 0:
        return undefined_value(name, "zero_tokens", ())
    segment_size = 50
    segments = [tokens[index : index + segment_size] for index in range(0, len(tokens), segment_size)]
    ratios = [len(set(segment)) / len(segment) for segment in segments if len(segment) > 0]
    return defined_value(name, float(sum(ratios)) / float(len(ratios)), _short_warning(len(tokens)))


def _mattr(tokens: tuple[str, ...]) -> FeatureValue:
    name = "text::lexical_richness::mattr"
    if len(tokens) == 0:
        return undefined_value(name, "zero_tokens", ())
    window_size = min(50, len(tokens))
    windows = [tokens[index : index + window_size] for index in range(0, len(tokens) - window_size + 1)]
    ratios = [len(set(window)) / len(window) for window in windows]
    return defined_value(name, float(sum(ratios)) / float(len(ratios)), _short_warning(len(tokens)))


def _mtld(tokens: tuple[str, ...]) -> FeatureValue:
    name = "text::lexical_richness::mtld"
    if len(tokens) == 0:
        return undefined_value(name, "zero_tokens", ())
    forward = _mtld_direction(tokens)
    backward = _mtld_direction(tuple(reversed(tokens)))
    return defined_value(name, (forward + backward) / 2.0, _short_warning(len(tokens)))


def _mtld_direction(tokens: tuple[str, ...]) -> float:
    factors = 0.0
    seen: set[str] = set()
    token_counter = 0
    threshold = 0.72
    for token in tokens:
        token_counter += 1
        seen.add(token)
        ttr = float(len(seen)) / float(token_counter)
        if ttr <= threshold:
            factors += 1.0
            seen = set()
            token_counter = 0
    if token_counter > 0:
        current_ttr = float(len(seen)) / float(token_counter)
        denominator = 1.0 - threshold
        factors += (1.0 - current_ttr) / denominator
    if factors == 0.0:
        return float(len(tokens))
    return float(len(tokens)) / factors


def _hdd(tokens: tuple[str, ...], token_counts: Counter[str]) -> FeatureValue:
    name = "text::lexical_richness::hdd"
    token_count = len(tokens)
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    sample_size = min(42, token_count)
    denominator = math.comb(token_count, sample_size)
    probability_sum = 0.0
    for frequency in token_counts.values():
        if token_count - frequency >= sample_size:
            probability_sum += 1.0 - (float(math.comb(token_count - frequency, sample_size)) / float(denominator))
        else:
            probability_sum += 1.0
    return defined_value(name, probability_sum / float(sample_size), _short_warning(token_count))


def _vocd_d(tokens: tuple[str, ...]) -> FeatureValue:
    name = "text::lexical_richness::vocd_d"
    token_count = len(tokens)
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    if token_count < _VOCD_MAX_SAMPLE_SIZE:
        return undefined_value(name, "below_vocd_50_token_threshold", _short_warning(token_count))
    ttr_points = _vocd_empirical_ttr_points(tokens)
    if all(point[1] >= 1.0 for point in ttr_points):
        return undefined_value(name, "all_samples_unique_vocd_singularity", _short_warning(token_count))
    return defined_value(name, _fit_vocd_d(ttr_points), _short_warning(token_count))


def _vocd_empirical_ttr_points(tokens: tuple[str, ...]) -> tuple[tuple[int, float], ...]:
    token_count = len(tokens)
    points: list[tuple[int, float]] = []
    for sample_size in range(_VOCD_MIN_SAMPLE_SIZE, _VOCD_MAX_SAMPLE_SIZE + 1):
        ttr_total = 0.0
        for sample_index in range(_VOCD_SAMPLES_PER_SIZE):
            sample_indices = _vocd_sample_indices(token_count, sample_size, sample_index)
            sample_types = {tokens[index] for index in sample_indices}
            ttr_total += float(len(sample_types)) / float(sample_size)
        points.append((sample_size, ttr_total / float(_VOCD_SAMPLES_PER_SIZE)))
    return tuple(points)


def _vocd_sample_indices(token_count: int, sample_size: int, sample_index: int) -> tuple[int, ...]:
    scored_indices = [
        (_vocd_sample_score(token_count, sample_size, sample_index, token_index), token_index) for token_index in range(token_count)
    ]
    scored_indices.sort()
    return tuple(token_index for _score, token_index in scored_indices[:sample_size])


def _vocd_sample_score(token_count: int, sample_size: int, sample_index: int, token_index: int) -> int:
    payload = f"{_VOCD_RANDOM_SEED}:{token_count}:{sample_size}:{sample_index}:{token_index}".encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), byteorder="big", signed=False)


def _fit_vocd_d(ttr_points: tuple[tuple[int, float], ...]) -> float:
    lower = _VOCD_D_LOWER_BOUND
    upper = _VOCD_D_UPPER_BOUND
    inverse_golden_ratio = (math.sqrt(5.0) - 1.0) / 2.0
    left = upper - inverse_golden_ratio * (upper - lower)
    right = lower + inverse_golden_ratio * (upper - lower)
    left_error = _vocd_squared_error(left, ttr_points)
    right_error = _vocd_squared_error(right, ttr_points)
    for _iteration in range(_VOCD_FIT_ITERATIONS):
        if left_error > right_error:
            lower = left
            left = right
            left_error = right_error
            right = lower + inverse_golden_ratio * (upper - lower)
            right_error = _vocd_squared_error(right, ttr_points)
        else:
            upper = right
            right = left
            right_error = left_error
            left = upper - inverse_golden_ratio * (upper - lower)
            left_error = _vocd_squared_error(left, ttr_points)
    return (lower + upper) / 2.0


def _vocd_squared_error(diversity: float, ttr_points: tuple[tuple[int, float], ...]) -> float:
    total = 0.0
    for sample_size, empirical_ttr in ttr_points:
        total += (_vocd_model_ttr(sample_size, diversity) - empirical_ttr) ** 2.0
    return total


def _vocd_model_ttr(sample_size: int, diversity: float) -> float:
    return (diversity / float(sample_size)) * (math.sqrt(1.0 + (2.0 * float(sample_size) / diversity)) - 1.0)


def _yules_k(token_count: int, token_counts: Counter[str]) -> FeatureValue:
    name = "text::lexical_richness::yules_k"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    frequency_spectrum = Counter(token_counts.values())
    m2 = sum((frequency * frequency) * count for frequency, count in frequency_spectrum.items())
    value = 10000.0 * (float(m2) - float(token_count)) / (float(token_count) * float(token_count))
    return defined_value(name, value, _short_warning(token_count))


def _honore_r(token_count: int, type_count: int, hapax_count: int) -> FeatureValue:
    name = "text::lexical_richness::honore_r"
    if token_count <= 1:
        return undefined_value(name, "insufficient_tokens_for_log", _short_warning(token_count))
    if type_count == 0:
        return undefined_value(name, "zero_types", _short_warning(token_count))
    denominator = 1.0 - (float(hapax_count) / float(type_count))
    if denominator == 0.0:
        return undefined_value(name, "all_types_are_hapax", _short_warning(token_count))
    return defined_value(name, 100.0 * math.log(float(token_count)) / denominator, _short_warning(token_count))


def _guiraud_r(token_count: int, type_count: int) -> FeatureValue:
    name = "text::lexical_richness::guiraud_r"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    return defined_value(name, float(type_count) / math.sqrt(float(token_count)), _short_warning(token_count))


def _simpson_d(token_count: int, token_counts: Counter[str]) -> FeatureValue:
    name = "text::lexical_richness::simpson_d"
    if token_count <= 1:
        return undefined_value(name, "insufficient_tokens_for_pairs", _short_warning(token_count))
    numerator = sum(frequency * (frequency - 1) for frequency in token_counts.values())
    denominator = token_count * (token_count - 1)
    return defined_value(name, float(numerator) / float(denominator), _short_warning(token_count))


def _renyi_entropy(token_count: int, token_counts: Counter[str]) -> FeatureValue:
    name = "text::lexical_richness::renyi_entropy_alpha_2"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    probability_square_sum = sum((float(count) / float(token_count)) ** 2.0 for count in token_counts.values())
    if probability_square_sum == 0.0:
        return undefined_value(name, "zero_probability_mass", _short_warning(token_count))
    return defined_value(name, -math.log(probability_square_sum), _short_warning(token_count))


def _mean_value(name: str, values: list[int], zero_reason: str) -> FeatureValue:
    if len(values) == 0:
        return undefined_value(name, zero_reason, ())
    return defined_value(name, float(sum(values)) / float(len(values)), _short_warning(len(values)))


def _median_value(name: str, values: list[int], zero_reason: str) -> FeatureValue:
    if len(values) == 0:
        return undefined_value(name, zero_reason, ())
    array = np.asarray(values, dtype=float)
    return defined_value(name, float(np.median(array)), _short_warning(len(values)))


def _std_value(name: str, values: list[int], zero_reason: str) -> FeatureValue:
    if len(values) == 0:
        return undefined_value(name, zero_reason, ())
    array = np.asarray(values, dtype=float)
    return defined_value(name, float(np.std(array)), _short_warning(len(values)))


def _readability_value(
    name: str, intercept: float, sentence_weight: float, syllable_weight: float, token_count: int, sentence_count: int, syllables: int
) -> FeatureValue:
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    if sentence_count == 0:
        return undefined_value(name, "zero_sentences", ())
    sentence_component = sentence_weight * (float(token_count) / float(sentence_count))
    syllable_component = syllable_weight * (float(syllables) / float(token_count))
    value = intercept + sentence_component + syllable_component
    return defined_value(name, value, _short_warning(token_count))


def _gunning_fog(token_count: int, sentence_count: int, complex_words: int) -> FeatureValue:
    name = "text::readability::gunning_fog"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    if sentence_count == 0:
        return undefined_value(name, "zero_sentences", ())
    value = 0.4 * ((float(token_count) / float(sentence_count)) + 100.0 * (float(complex_words) / float(token_count)))
    return defined_value(name, value, _short_warning(token_count))


def _coleman_liau(token_count: int, sentence_count: int, characters: int) -> FeatureValue:
    name = "text::readability::coleman_liau"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    letters_per_100 = float(characters) / float(token_count) * 100.0
    sentences_per_100 = float(sentence_count) / float(token_count) * 100.0
    return defined_value(name, (0.0588 * letters_per_100) - (0.296 * sentences_per_100) - 15.8, _short_warning(token_count))


def _smog(sentence_count: int, complex_words: int) -> FeatureValue:
    name = "text::readability::smog"
    if sentence_count == 0:
        return undefined_value(name, "zero_sentences", ())
    if sentence_count < 3:
        return undefined_value(name, "below_smog_3_sentence_threshold", _short_warning(sentence_count))
    score = 1.043 * math.sqrt(float(complex_words) * (30.0 / float(sentence_count))) + 3.1291
    return defined_value(name, score, _short_warning(sentence_count))


def _ari(token_count: int, sentence_count: int, characters: int) -> FeatureValue:
    name = "text::readability::ari"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    if sentence_count == 0:
        return undefined_value(name, "zero_sentences", ())
    value = 4.71 * (float(characters) / float(token_count)) + 0.5 * (float(token_count) / float(sentence_count)) - 21.43
    return defined_value(name, value, _short_warning(token_count))


def _dale_chall(tokens: tuple[str, ...], token_count: int, sentence_count: int) -> FeatureValue:
    name = "text::readability::dale_chall"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    if sentence_count == 0:
        return undefined_value(name, "zero_sentences", ())
    easy_words = _dale_chall_easy_words()
    difficult_words = sum(1 for token in tokens if token not in easy_words)
    difficult_percentage = 100.0 * (float(difficult_words) / float(token_count))
    average_sentence_length = float(token_count) / float(sentence_count)
    score = (0.1579 * difficult_percentage) + (0.0496 * average_sentence_length)
    if difficult_percentage > 5.0:
        score += 3.6365
    return defined_value(name, score, _short_warning(token_count))


def _forcast(token_count: int, single_syllable_words: int) -> FeatureValue:
    name = "text::readability::forcast"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    if token_count < 150:
        return undefined_value(name, "below_forcast_150_word_threshold", _short_warning(token_count))
    scaled_single_syllable_words = float(single_syllable_words) * (150.0 / float(token_count))
    return defined_value(name, 20.0 - (scaled_single_syllable_words / 10.0), _short_warning(token_count))


def _linsear_write(syllable_counts: list[int], sentence_lengths: list[int]) -> FeatureValue:
    name = "text::readability::linsear_write"
    token_count = len(syllable_counts)
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    if token_count < 100:
        return undefined_value(name, "below_linsear_write_100_word_threshold", _short_warning(token_count))
    sample = syllable_counts[:100]
    sample_sentence_count = _sentence_count_covering_sample(sentence_lengths, 100)
    if sample_sentence_count == 0:
        return undefined_value(name, "zero_sentences", _short_warning(token_count))
    easy_word_score = sum(1 for count in sample if count <= 2)
    hard_word_score = sum(3 for count in sample if count >= 3)
    raw_score = float(easy_word_score + hard_word_score) / float(sample_sentence_count)
    if raw_score > 20.0:
        return defined_value(name, raw_score / 2.0, _short_warning(token_count))
    return defined_value(name, (raw_score - 2.0) / 2.0, _short_warning(token_count))


def _sentence_count_covering_sample(sentence_lengths: list[int], sample_size: int) -> int:
    covered_tokens = 0
    covered_sentences = 0
    for sentence_length in sentence_lengths:
        if covered_tokens >= sample_size:
            break
        if sentence_length > 0:
            covered_sentences += 1
            covered_tokens += sentence_length
    return covered_sentences


def _lix(token_count: int, sentence_count: int, long_words: int) -> FeatureValue:
    name = "text::readability::lix"
    if token_count == 0:
        return undefined_value(name, "zero_tokens", ())
    if sentence_count == 0:
        return undefined_value(name, "zero_sentences", ())
    value = (float(token_count) / float(sentence_count)) + (100.0 * (float(long_words) / float(token_count)))
    return defined_value(name, value, _short_warning(token_count))


def _sentence_token_lengths(view: DocumentView) -> list[int]:
    lengths: list[int] = []
    for sentence_index, sentence in enumerate(view.sentences):
        sentence_view = DocumentView.from_text(sentence, view.config, document_id=f"{view.document_id}:sentence:{sentence_index}")
        lengths.append(len(sentence_view.tokens))
    return lengths


def _paragraph_token_lengths(view: DocumentView) -> list[int]:
    lengths: list[int] = []
    for paragraph_index, paragraph in enumerate(view.paragraphs):
        paragraph_view = DocumentView.from_text(paragraph, view.config, document_id=f"{view.document_id}:paragraph:{paragraph_index}")
        lengths.append(len(paragraph_view.tokens))
    return lengths


def _short_warning(size: int) -> tuple[str, ...]:
    if size < 50:
        return ("short_text_unstable",)
    return ()


def _diagnostics(values: dict[str, FeatureValue]) -> tuple[FeatureDiagnostic, ...]:
    diagnostics = [
        FeatureDiagnostic(feature_name=value.name, status=value.status, reason=value.reason, warnings=value.warnings)
        for value in values.values()
        if value.status != FeatureStatus.DEFINED
    ]
    return tuple(diagnostics)


def _family_from_name(name: str) -> str:
    parts = name.split("::")
    if len(parts) < 2:
        raise ValueError(f"Invalid feature name: {name}")
    return parts[1]


def _formula_for_name(name: str) -> str:
    if name in _LEXICAL_RICHNESS_FORMULAS:
        return _LEXICAL_RICHNESS_FORMULAS[name]
    if "readability" in name:
        return _readability_formula_for_name(name)
    if "syllables" in name:
        return "hybrid dictionary-backed syllable count with deterministic heuristic fallback"
    return "built-in deterministic count, ratio, or distribution statistic"


def _readability_formula_for_name(name: str) -> str:
    if name.endswith("dale_chall"):
        return "Dale-Chall score using project-owned seed easy-word list, difficult-word percentage, and average sentence length"
    if name.endswith("forcast"):
        return "FORCAST grade: 20 - scaled single-syllable words in a 150-word sample / 10"
    if name.endswith("linsear_write"):
        return "Linsear Write grade over the first 100 words using easy and hard word weights"
    if name.endswith("lix"):
        return "LIX: words per sentence plus percentage of words longer than six characters"
    return "published readability formula using sentence, token, character, and hybrid dictionary-backed syllable counts"


def _provenance_for_name(name: str) -> str:
    if name.endswith("::vocd_d"):
        return (
            "built_in_rules:v1; vocd_sampling=v1; sample_sizes=35-50; samples_per_size=100; "
            "hash_seed=stylometry_python_lib_vocd_d_v1; fit=golden_section_curve_fit_v1; preprocessing_config"
        )
    if name.endswith("dale_chall"):
        return (
            "built_in_rules:v1; dale_chall_easy_words=dale_chall_easy_words_en_seed_v1; "
            "syllable_dictionary=syllable_counts_en_v1; syllable_fallback=heuristic_v1; preprocessing_config"
        )
    if "syllables" in name or "readability" in name:
        return "built_in_rules:v1; syllable_dictionary=syllable_counts_en_v1; syllable_fallback=heuristic_v1; preprocessing_config"
    return "built_in_rules:v1; preprocessing_config"


def _topic_dependence_for_name(name: str) -> TopicDependence:
    if "function_word" in name or "stopword" in name or "punctuation" in name or "whitespace" in name:
        return TopicDependence.MOSTLY_TOPIC_INDEPENDENT
    if "spelling_variant" in name or "acronym" in name or "abbreviation" in name:
        return TopicDependence.TOPIC_SENSITIVE
    return TopicDependence.MIXED


def _length_policy_for_name(name: str) -> str:
    if name.endswith("forcast"):
        return "undefined below the 150-word FORCAST sample threshold; warns below 50 relevant units"
    if name.endswith("linsear_write"):
        return "undefined below the 100-word Linsear Write sample threshold; warns below 50 relevant units"
    if name.endswith("smog"):
        return "undefined below the 3-sentence SMOG threshold; warns below 50 relevant units"
    if "lexical_richness" in name or "discourse" in name or "readability" in name:
        return "defined when denominators exist; warn below 50 relevant units"
    return "defined for empty text only when the raw count itself is meaningful"


def _normalization_for_name(name: str) -> str:
    if name.endswith("_count"):
        return "raw_count"
    if "ratio" in name:
        return "ratio"
    return "formula"


def function_words_lexicon() -> frozenset[str]:
    """Return the built-in closed-class function word lexicon."""
    return frozenset(load_lexicon("function_words").tokens())


def _stopwords() -> frozenset[str]:
    return frozenset(load_lexicon("stopwords").tokens())


def _pronouns() -> frozenset[str]:
    return _first_person_pronouns().union(_second_person_pronouns()).union(_third_person_pronouns())


def _first_person_pronouns() -> frozenset[str]:
    return frozenset(entry.token for entry in load_lexicon("pronouns").entries if "first_person" in entry.groups)


def _second_person_pronouns() -> frozenset[str]:
    return frozenset(entry.token for entry in load_lexicon("pronouns").entries if "second_person" in entry.groups)


def _third_person_pronouns() -> frozenset[str]:
    return frozenset(entry.token for entry in load_lexicon("pronouns").entries if "third_person" in entry.groups)


def _modal_verbs() -> frozenset[str]:
    return frozenset(load_lexicon("modals").tokens())


def _auxiliary_verbs() -> frozenset[str]:
    return frozenset(load_lexicon("auxiliaries").tokens())


def _dale_chall_easy_words() -> frozenset[str]:
    return frozenset(load_lexicon("dale_chall_easy_words").tokens())


def _punctuation_characters() -> frozenset[str]:
    return frozenset({".", ",", ";", ":", "!", "?", "-", "—", "(", ")", "[", "]", "{", "}", "'", '"', "…"})


def _spelling_variant_pairs() -> tuple[tuple[str, str], ...]:
    resource = load_spelling_variants("spelling_variants")
    return tuple((pair.variant_a.token, pair.variant_b.token) for pair in resource.pairs)


def _discourse_markers() -> tuple[str, ...]:
    return load_lexicon("discourse_markers").tokens()


def _transition_phrases() -> tuple[str, ...]:
    return load_lexicon("transition_phrases").tokens()


def _is_titlecase_line(line: str) -> bool:
    words = re.findall(r"[A-Za-z]+", line)
    if len(words) == 0:
        return False
    return all(word[0].isupper() for word in words)


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if stripped == "":
        return False
    if stripped.startswith("#"):
        return True
    return _is_titlecase_line(stripped) and len(stripped) <= 80
