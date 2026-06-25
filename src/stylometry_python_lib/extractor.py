"""Public extraction helpers."""

from __future__ import annotations

from stylometry_python_lib.document import english_preprocessing_config
from stylometry_python_lib.features.abbreviation_acronyms import abbreviation_acronym_profile_transformer
from stylometry_python_lib.features.capitalization import CapitalizationProfileTransformer
from stylometry_python_lib.features.deterministic import DeterministicStylometryTransformer
from stylometry_python_lib.features.discourse import discourse_marker_profile_transformer, transition_phrase_profile_transformer
from stylometry_python_lib.features.distributions import DistributionStatisticsTransformer
from stylometry_python_lib.features.frequencies import (
    MostFrequentWordsTransformer,
    auxiliary_lexicon_transformer,
    contraction_lexicon_transformer,
    function_word_frequency_transformer,
    function_word_lexicon_transformer,
    letter_frequency_transformer,
    modal_lexicon_transformer,
    pronoun_lexicon_transformer,
    punctuation_frequency_transformer,
    stopword_lexicon_transformer,
)
from stylometry_python_lib.features.hyphenation import HyphenationProfileTransformer
from stylometry_python_lib.features.layout_whitespace import LayoutWhitespaceProfileTransformer
from stylometry_python_lib.features.lexical_richness import LexicalRichnessSpectrumTransformer
from stylometry_python_lib.features.lexical_sophistication import lexical_sophistication_profile_transformer
from stylometry_python_lib.features.ngrams import NGramStylometryTransformer
from stylometry_python_lib.features.orthography import OrthographyProfileTransformer
from stylometry_python_lib.features.punctuation import PunctuationProfileTransformer
from stylometry_python_lib.features.quote_dialogue import QuoteDialogueProfileTransformer
from stylometry_python_lib.features.special_tokens import SpecialTokenProfileTransformer
from stylometry_python_lib.features.spelling_variants import spelling_variant_profile_transformer
from stylometry_python_lib.sklearn import FeatureExtractor


def default_deterministic_extractor(text_column: str, output: str) -> FeatureExtractor:
    """Build a deterministic stylometry extractor with scalar and n-gram blocks."""
    config = english_preprocessing_config()
    scalar_block = DeterministicStylometryTransformer(text_column=text_column, config=config, output="pandas")
    capitalization_profile_block = CapitalizationProfileTransformer(text_column=text_column, config=config, output="pandas")
    hyphenation_profile_block = HyphenationProfileTransformer(text_column=text_column, config=config, output="pandas")
    abbreviation_acronym_profile_block = abbreviation_acronym_profile_transformer(
        text_column=text_column,
        config=config,
        lexicon_name="abbreviations",
        output="pandas",
    )
    discourse_marker_profile_block = discourse_marker_profile_transformer(text_column=text_column, config=config, output="pandas")
    transition_phrase_profile_block = transition_phrase_profile_transformer(text_column=text_column, config=config, output="pandas")
    quote_dialogue_profile_block = QuoteDialogueProfileTransformer(text_column=text_column, config=config, output="pandas")
    layout_whitespace_profile_block = LayoutWhitespaceProfileTransformer(text_column=text_column, config=config, output="pandas")
    special_token_profile_block = SpecialTokenProfileTransformer(text_column=text_column, config=config, output="pandas")
    spelling_variant_profile_block = spelling_variant_profile_transformer(
        text_column=text_column,
        config=config,
        resource_name="spelling_variants",
        output="pandas",
    )
    lexical_spectrum_block = LexicalRichnessSpectrumTransformer(
        text_column=text_column,
        config=config,
        max_frequency_bin=20,
        output="pandas",
    )
    lexical_sophistication_profile_block = lexical_sophistication_profile_transformer(
        text_column=text_column,
        config=config,
        resource_name="frequency_bands",
        output="pandas",
    )
    distribution_block = DistributionStatisticsTransformer(text_column=text_column, config=config, output="pandas")
    orthography_profile_block = OrthographyProfileTransformer(text_column=text_column, config=config, output="pandas")
    punctuation_profile_block = PunctuationProfileTransformer(text_column=text_column, config=config, output="pandas")
    function_word_frequency_block = function_word_frequency_transformer(text_column=text_column, config=config, output="pandas")
    function_word_lexicon_block = function_word_lexicon_transformer(text_column=text_column, config=config, output="pandas")
    stopword_block = stopword_lexicon_transformer(text_column=text_column, config=config, output="pandas")
    pronoun_block = pronoun_lexicon_transformer(text_column=text_column, config=config, output="pandas")
    modal_block = modal_lexicon_transformer(text_column=text_column, config=config, output="pandas")
    auxiliary_block = auxiliary_lexicon_transformer(text_column=text_column, config=config, output="pandas")
    contraction_block = contraction_lexicon_transformer(text_column=text_column, config=config, output="pandas")
    letter_frequency_block = letter_frequency_transformer(text_column=text_column, config=config, output="pandas")
    punctuation_frequency_block = punctuation_frequency_transformer(text_column=text_column, config=config, output="pandas")
    most_frequent_word_block = MostFrequentWordsTransformer(text_column=text_column, config=config, max_features=100, output="pandas")
    char_block = NGramStylometryTransformer(
        text_column=text_column,
        config=config,
        analyzer="char",
        ngram_range=(3, 3),
        max_features=100,
        output="sparse",
    )
    function_word_block = NGramStylometryTransformer(
        text_column=text_column,
        config=config,
        analyzer="function_word",
        ngram_range=(1, 2),
        max_features=100,
        output="sparse",
    )
    punctuation_ngram_block = NGramStylometryTransformer(
        text_column=text_column,
        config=config,
        analyzer="punctuation",
        ngram_range=(1, 3),
        max_features=50,
        output="sparse",
    )
    return FeatureExtractor(
        blocks=(
            scalar_block,
            capitalization_profile_block,
            hyphenation_profile_block,
            abbreviation_acronym_profile_block,
            discourse_marker_profile_block,
            transition_phrase_profile_block,
            quote_dialogue_profile_block,
            layout_whitespace_profile_block,
            special_token_profile_block,
            spelling_variant_profile_block,
            lexical_spectrum_block,
            lexical_sophistication_profile_block,
            distribution_block,
            orthography_profile_block,
            punctuation_profile_block,
            function_word_frequency_block,
            function_word_lexicon_block,
            stopword_block,
            pronoun_block,
            modal_block,
            auxiliary_block,
            contraction_block,
            letter_frequency_block,
            punctuation_frequency_block,
            most_frequent_word_block,
            char_block,
            function_word_block,
            punctuation_ngram_block,
        ),
        output=output,
    )
