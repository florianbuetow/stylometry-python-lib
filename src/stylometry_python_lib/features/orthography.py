"""Unicode and raw-codepoint orthography profile features."""

from __future__ import annotations

import unicodedata
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
from stylometry_python_lib.registry import FeatureRegistry
from stylometry_python_lib.specs import FeatureSpec, InputLayer, StabilityStatus, TopicDependence
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus


@dataclass(frozen=True)
class CharacterClass:
    """Named character class for deterministic orthography features."""

    class_id: str
    description: str


@dataclass(frozen=True)
class UnicodeScript:
    """Named Unicode script inventory item backed by explicit codepoint ranges."""

    script_id: str
    description: str
    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class CodepointRecord:
    """One raw-codepoint observation for orthography sidecars."""

    character_index: int
    character: str
    codepoint: int
    unicode_category: str
    unicode_name: str | None


@dataclass(frozen=True)
class OrthographyCodepointSidecar:
    """Raw-codepoint sidecar for one document."""

    document_id: str
    schema_version: str
    normalization: str
    codepoint_count: int
    unique_codepoint_count: int
    records: tuple[CodepointRecord, ...]


class OrthographyProfileTransformer(BaseEstimator):
    """Sklearn-compatible Unicode orthography profile transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze orthography profile metadata."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.feature_names_out_ = np.asarray(orthography_profile_feature_names(), dtype=object)
        self.registry_ = FeatureRegistry(specs=orthography_profile_feature_specs())
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute Unicode orthography profile features without changing rows."""
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[OrthographyCodepointSidecar] = []
        for row_index, text in enumerate(series.tolist()):
            view = DocumentView.from_text(str(text), self.config, document_id=str(series.index[row_index]))
            row, row_diagnostics = _orthography_row(view.raw)
            rows.append(row)
            diagnostics.append(row_diagnostics)
            sidecars.append(_codepoint_sidecar(view.raw, view.document_id))
        self.last_diagnostics_ = tuple(diagnostics)
        self.last_sidecars_ = tuple(sidecars)
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
        """Return stable orthography profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def orthography_profile_feature_names() -> tuple[str, ...]:
    """Return stable orthography profile feature names in output order."""
    names = [
        "text::orthography_profile::codepoint_count",
        "text::orthography_profile::unique_codepoint_count",
        "text::orthography_profile::codepoint_min",
        "text::orthography_profile::codepoint_max",
        "text::orthography_profile::codepoint_mean",
    ]
    for character_class in _character_classes():
        names.append(_class_name(character_class.class_id, "count"))
        names.append(_class_name(character_class.class_id, "per_character"))
    for category in _unicode_categories():
        names.append(_category_name(category, "count"))
        names.append(_category_name(category, "per_character"))
    for script in _unicode_scripts():
        names.append(_script_name(script.script_id, "count"))
        names.append(_script_name(script.script_id, "per_character"))
        names.append(_script_name(script.script_id, "per_alpha"))
    for letter in _latin_letters():
        names.append(_letter_name(letter, "per_alpha"))
        names.append(_letter_name(letter, "per_character"))
    return tuple(names)


def orthography_profile_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return metadata for Unicode orthography profile features."""
    return tuple(_spec_for_name(name) for name in orthography_profile_feature_names())


def _orthography_row(text: str) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    character_count = len(text)
    alphabetic_count = sum(1 for character in text if character.isalpha())
    categories = Counter(unicodedata.category(character) for character in text)
    lower_text = text.lower()
    values: list[float] = []
    diagnostics: list[FeatureDiagnostic] = []

    values.extend(_codepoint_statistics(text, diagnostics))
    for character_class in _character_classes():
        count = float(_class_count(text, character_class.class_id))
        values.append(count)
        values.append(
            _rate_value(_class_name(character_class.class_id, "per_character"), count, character_count, "empty_text", diagnostics)
        )
    for category in _unicode_categories():
        count = float(categories[category])
        values.append(count)
        values.append(_rate_value(_category_name(category, "per_character"), count, character_count, "empty_text", diagnostics))
    for script in _unicode_scripts():
        count = float(_script_count(text, script))
        values.append(count)
        values.append(_rate_value(_script_name(script.script_id, "per_character"), count, character_count, "empty_text", diagnostics))
        values.append(
            _rate_value(_script_name(script.script_id, "per_alpha"), count, alphabetic_count, "zero_alphabetic_characters", diagnostics)
        )
    for letter in _latin_letters():
        count = float(lower_text.count(letter))
        values.append(_rate_value(_letter_name(letter, "per_alpha"), count, alphabetic_count, "zero_alphabetic_characters", diagnostics))
        values.append(_rate_value(_letter_name(letter, "per_character"), count, character_count, "empty_text", diagnostics))
    return values, tuple(diagnostics)


def _codepoint_statistics(text: str, diagnostics: list[FeatureDiagnostic]) -> list[float]:
    codepoints = [ord(character) for character in text]
    if len(codepoints) == 0:
        diagnostics.extend(
            _undefined(name, "empty_text")
            for name in (
                "text::orthography_profile::codepoint_min",
                "text::orthography_profile::codepoint_max",
                "text::orthography_profile::codepoint_mean",
            )
        )
        return [0.0, 0.0, float("nan"), float("nan"), float("nan")]
    return [
        float(len(codepoints)),
        float(len(set(codepoints))),
        float(min(codepoints)),
        float(max(codepoints)),
        float(sum(codepoints)) / float(len(codepoints)),
    ]


def _codepoint_sidecar(text: str, document_id: str) -> OrthographyCodepointSidecar:
    records = tuple(_codepoint_record(character_index, character) for character_index, character in enumerate(text))
    return OrthographyCodepointSidecar(
        document_id=document_id,
        schema_version="orthography_codepoint_sidecar_v1",
        normalization="raw_unicode_codepoints",
        codepoint_count=len(records),
        unique_codepoint_count=len({record.codepoint for record in records}),
        records=records,
    )


def _codepoint_record(character_index: int, character: str) -> CodepointRecord:
    return CodepointRecord(
        character_index=character_index,
        character=character,
        codepoint=ord(character),
        unicode_category=unicodedata.category(character),
        unicode_name=_unicode_name(character),
    )


def _unicode_name(character: str) -> str | None:
    try:
        return unicodedata.name(character)
    except ValueError:
        return None


def _rate_value(feature_name: str, count: float, denominator: int, zero_reason: str, diagnostics: list[FeatureDiagnostic]) -> float:
    if denominator == 0:
        diagnostics.append(_undefined(feature_name, zero_reason))
        return float("nan")
    return count / float(denominator)


def _class_count(text: str, class_id: str) -> int:
    if class_id == "unicode_letter":
        return sum(1 for character in text if character.isalpha())
    if class_id == "ascii_letter":
        return sum(1 for character in text if character.isascii() and character.isalpha())
    if class_id == "non_ascii_letter":
        return sum(1 for character in text if not character.isascii() and character.isalpha())
    if class_id == "digit":
        return sum(1 for character in text if character.isdigit())
    if class_id == "punctuation":
        return sum(1 for character in text if unicodedata.category(character).startswith("P"))
    if class_id == "symbol":
        return sum(1 for character in text if unicodedata.category(character).startswith("S"))
    if class_id == "whitespace":
        return sum(1 for character in text if character.isspace())
    if class_id == "mark":
        return sum(1 for character in text if unicodedata.category(character).startswith("M"))
    if class_id == "control_format_other":
        return sum(1 for character in text if unicodedata.category(character).startswith("C"))
    raise ValueError(f"Unsupported character class: {class_id}")


def _character_classes() -> tuple[CharacterClass, ...]:
    return (
        CharacterClass("unicode_letter", "Any Unicode alphabetic character"),
        CharacterClass("ascii_letter", "ASCII alphabetic character"),
        CharacterClass("non_ascii_letter", "Unicode alphabetic character outside ASCII"),
        CharacterClass("digit", "Unicode digit character"),
        CharacterClass("punctuation", "Unicode punctuation category"),
        CharacterClass("symbol", "Unicode symbol category"),
        CharacterClass("whitespace", "Whitespace character"),
        CharacterClass("mark", "Unicode combining mark category"),
        CharacterClass("control_format_other", "Unicode control, format, surrogate, private-use, or unassigned category"),
    )


def _script_count(text: str, script: UnicodeScript) -> int:
    return sum(1 for character in text if character.isalpha() and _character_in_script(character, script))


def _character_in_script(character: str, script: UnicodeScript) -> bool:
    codepoint = ord(character)
    return any(start <= codepoint <= end for start, end in script.ranges)


def _unicode_scripts() -> tuple[UnicodeScript, ...]:
    return (
        UnicodeScript(
            "latin",
            "Latin letters including common extended Latin blocks",
            ((0x0041, 0x005A), (0x0061, 0x007A), (0x00C0, 0x024F), (0x1E00, 0x1EFF)),
        ),
        UnicodeScript("greek", "Greek and Greek Extended letters", ((0x0370, 0x03FF), (0x1F00, 0x1FFF))),
        UnicodeScript(
            "cyrillic",
            "Cyrillic letters and common Cyrillic supplements",
            ((0x0400, 0x052F), (0x2DE0, 0x2DFF), (0xA640, 0xA69F)),
        ),
        UnicodeScript("hebrew", "Hebrew letters", ((0x0590, 0x05FF),)),
        UnicodeScript(
            "arabic",
            "Arabic letters and common Arabic supplements",
            ((0x0600, 0x06FF), (0x0750, 0x077F), (0x08A0, 0x08FF)),
        ),
        UnicodeScript("devanagari", "Devanagari letters", ((0x0900, 0x097F),)),
        UnicodeScript("hiragana", "Hiragana letters", ((0x3040, 0x309F),)),
        UnicodeScript("katakana", "Katakana letters", ((0x30A0, 0x30FF), (0x31F0, 0x31FF))),
        UnicodeScript("hangul", "Hangul syllables and jamo letters", ((0x1100, 0x11FF), (0x3130, 0x318F), (0xAC00, 0xD7AF))),
        UnicodeScript(
            "cjk_unified_ideograph",
            "CJK unified ideographs and compatibility ideographs",
            ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF)),
        ),
        UnicodeScript("thai", "Thai letters", ((0x0E00, 0x0E7F),)),
        UnicodeScript("armenian", "Armenian letters", ((0x0530, 0x058F),)),
        UnicodeScript("georgian", "Georgian letters and supplement", ((0x10A0, 0x10FF), (0x1C90, 0x1CBF))),
        UnicodeScript("bengali", "Bengali letters", ((0x0980, 0x09FF),)),
        UnicodeScript("tamil", "Tamil letters", ((0x0B80, 0x0BFF),)),
        UnicodeScript("ethiopic", "Ethiopic letters", ((0x1200, 0x137F),)),
    )


def _unicode_categories() -> tuple[str, ...]:
    return (
        "Lu",
        "Ll",
        "Lt",
        "Lm",
        "Lo",
        "Mn",
        "Mc",
        "Me",
        "N" + "d",
        "Nl",
        "No",
        "Pc",
        "Pd",
        "Ps",
        "Pe",
        "Pi",
        "Pf",
        "Po",
        "Sm",
        "Sc",
        "Sk",
        "So",
        "Zs",
        "Zl",
        "Zp",
        "Cc",
        "Cf",
        "Cs",
        "Co",
        "Cn",
    )


def _latin_letters() -> tuple[str, ...]:
    return tuple("abcdefghijklmnopqrstuvwxyz")


def _class_name(class_id: str, measure: str) -> str:
    return f"text::orthography_profile::character_class={class_id}::{measure}"


def _category_name(category: str, measure: str) -> str:
    return f"text::orthography_profile::unicode_category={category}::{measure}"


def _script_name(script_id: str, measure: str) -> str:
    return f"text::orthography_profile::script={script_id}::{measure}"


def _letter_name(letter: str, measure: str) -> str:
    return f"text::orthography_profile::latin_letter={letter}::{measure}"


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _spec_for_name(name: str) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the phenomenon is absent"
    formula_or_rule = "count raw Unicode codepoints matching the named class, category, script, or Latin letter"
    if name.endswith("per_character"):
        normalization = "per_character_ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason empty_text when character count is zero"
        formula_or_rule = "count divided by raw Unicode codepoint count"
    if name.endswith("per_alpha"):
        normalization = "per_alphabetic_character_ratio"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_alphabetic_characters when alphabetic count is zero"
        formula_or_rule = "letter or script count divided by total alphabetic character count"
    if "::script=" in name and name.endswith("count"):
        normalization = "raw_count"
        undefined_behavior = "defined as zero when no alphabetic character from the script is present"
        formula_or_rule = "count alphabetic codepoints in the configured Unicode script ranges"
    if name.endswith("codepoint_min") or name.endswith("codepoint_max") or name.endswith("codepoint_mean"):
        normalization = "codepoint_statistic"
        undefined_behavior = "NaN with FeatureDiagnostic reason empty_text when character count is zero"
        formula_or_rule = "minimum, maximum, or arithmetic mean of raw Unicode codepoint integers"
    if name.endswith("codepoint_count") or name.endswith("unique_codepoint_count"):
        normalization = "raw_count"
        undefined_behavior = "defined as zero for empty text"
        formula_or_rule = "count raw Unicode codepoints or unique raw Unicode codepoints"
    return FeatureSpec(
        name=name,
        family="orthography_profile",
        description=f"Unicode orthography profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.RAW,
        topic_dependence=TopicDependence.MOSTLY_TOPIC_INDEPENDENT,
        text_length_policy="counts are defined for empty text; ratios and codepoint summary statistics require a denominator",
        provenance="built_in_orthography_profile_rules:v1; unicode_category=python_unicodedata; preprocessing_config",
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
