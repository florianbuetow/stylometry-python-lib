"""Abbreviation and acronym profile feature block."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Self

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator

from stylometry_python_lib._fitted import require_fitted
from stylometry_python_lib._tabular import text_series, validate_output_mode
from stylometry_python_lib.document import DocumentView, PreprocessingConfig
from stylometry_python_lib.lexicons import VersionedLexicon, load_lexicon
from stylometry_python_lib.registry import FeatureRegistry
from stylometry_python_lib.specs import FeatureSpec, InputLayer, StabilityStatus, TopicDependence
from stylometry_python_lib.undefined import FeatureDiagnostic, FeatureStatus

_NAMED_ACRONYM_RESOURCE_NAME = "acronyms"


@dataclass(frozen=True)
class AcronymClass:
    """One deterministic acronym class."""

    class_id: str
    pattern: str
    description: str


@dataclass(frozen=True)
class AbbreviationAcronymMatch:
    """One matched abbreviation or acronym span."""

    match_kind: str
    match_id: str
    text: str
    start: int
    end: int
    groups: tuple[str, ...]


@dataclass(frozen=True)
class AbbreviationAcronymSidecar:
    """Matched abbreviation and acronym spans for one document."""

    document_id: str
    schema_version: str
    lexicon_id: str
    language: str
    version: str
    normalization: str
    acronym_lexicon_id: str
    acronym_language: str
    acronym_version: str
    acronym_normalization: str
    abbreviation_match_count: int
    named_acronym_match_count: int
    acronym_match_count: int
    matches: tuple[AbbreviationAcronymMatch, ...]


class AbbreviationAcronymProfileTransformer(BaseEstimator):
    """Sklearn-compatible abbreviation and acronym profile transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, lexicon_name: str, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.lexicon_name = lexicon_name
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Load abbreviation resource metadata and freeze output names."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        lexicon = load_lexicon(self.lexicon_name)
        acronym_lexicon = _load_named_acronym_lexicon()
        self.lexicon_ = lexicon
        self.acronym_lexicon_ = acronym_lexicon
        self.feature_names_out_ = np.asarray(abbreviation_acronym_profile_feature_names(lexicon), dtype=object)
        self.registry_ = FeatureRegistry(specs=abbreviation_acronym_profile_feature_specs(lexicon))
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute abbreviation and acronym profile features without changing rows."""
        require_fitted(self, "lexicon_")
        require_fitted(self, "acronym_lexicon_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[AbbreviationAcronymSidecar] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics, sidecar = _abbreviation_acronym_row(view, self.lexicon_, self.acronym_lexicon_)
            rows.append(row)
            diagnostics.append(row_diagnostics)
            sidecars.append(sidecar)
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
        """Return stable abbreviation and acronym profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def abbreviation_acronym_profile_feature_names(lexicon: VersionedLexicon) -> tuple[str, ...]:
    """Return stable abbreviation and acronym profile feature names in output order."""
    acronym_lexicon = _load_named_acronym_lexicon()
    names: list[str] = []
    for token in lexicon.tokens():
        names.extend(
            (
                _abbreviation_item_name(token, "count"),
                _abbreviation_item_name(token, "per_1000_tokens"),
            )
        )
    for group in lexicon.groups():
        names.extend(
            (
                _abbreviation_group_name(group, "count"),
                _abbreviation_group_name(group, "per_1000_tokens"),
            )
        )
    for token in acronym_lexicon.tokens():
        names.extend(
            (
                _named_acronym_item_name(token, "count"),
                _named_acronym_item_name(token, "per_1000_tokens"),
            )
        )
    for group in acronym_lexicon.groups():
        names.extend(
            (
                _named_acronym_group_name(group, "count"),
                _named_acronym_group_name(group, "per_1000_tokens"),
            )
        )
    for acronym_class in _acronym_classes():
        names.extend(
            (
                _acronym_class_name(acronym_class.class_id, "count"),
                _acronym_class_name(acronym_class.class_id, "per_1000_tokens"),
            )
        )
    return tuple(names)


def abbreviation_acronym_profile_feature_specs(lexicon: VersionedLexicon) -> tuple[FeatureSpec, ...]:
    """Return metadata for abbreviation and acronym profile features."""
    acronym_lexicon = _load_named_acronym_lexicon()
    return tuple(_spec_for_name(name, lexicon, acronym_lexicon) for name in abbreviation_acronym_profile_feature_names(lexicon))


def abbreviation_acronym_profile_transformer(
    text_column: str, config: PreprocessingConfig, lexicon_name: str, output: str
) -> AbbreviationAcronymProfileTransformer:
    """Build an abbreviation and acronym profile transformer for an explicit resource."""
    return AbbreviationAcronymProfileTransformer(
        text_column=text_column,
        config=config,
        lexicon_name=lexicon_name,
        output=output,
    )


def _abbreviation_acronym_row(
    view: DocumentView, lexicon: VersionedLexicon, acronym_lexicon: VersionedLexicon
) -> tuple[list[float], tuple[FeatureDiagnostic, ...], AbbreviationAcronymSidecar]:
    token_count = len(view.tokens)
    abbreviation_matches = _abbreviation_matches(view.raw, lexicon)
    named_acronym_matches = _named_acronym_matches(view.raw, acronym_lexicon)
    acronym_matches = _acronym_matches(view.raw)
    abbreviation_counts = _abbreviation_counts(abbreviation_matches, lexicon)
    named_acronym_counts = _named_acronym_counts(named_acronym_matches, acronym_lexicon)
    acronym_counts = _acronym_counts(acronym_matches)
    diagnostics: list[FeatureDiagnostic] = []
    values: list[float] = []

    for entry in lexicon.entries:
        count = float(abbreviation_counts[entry.token])
        values.append(count)
        values.append(_per_1000(_abbreviation_item_name(entry.token, "per_1000_tokens"), count, token_count, diagnostics))
    group_counts = _abbreviation_group_counts(abbreviation_counts, lexicon)
    for group in lexicon.groups():
        count = float(group_counts[group])
        values.append(count)
        values.append(_per_1000(_abbreviation_group_name(group, "per_1000_tokens"), count, token_count, diagnostics))
    named_acronym_group_counts = _named_acronym_group_counts(named_acronym_counts, acronym_lexicon)
    for entry in acronym_lexicon.entries:
        count = float(named_acronym_counts[entry.token])
        values.append(count)
        values.append(_per_1000(_named_acronym_item_name(entry.token, "per_1000_tokens"), count, token_count, diagnostics))
    for group in acronym_lexicon.groups():
        count = float(named_acronym_group_counts[group])
        values.append(count)
        values.append(_per_1000(_named_acronym_group_name(group, "per_1000_tokens"), count, token_count, diagnostics))
    for acronym_class in _acronym_classes():
        count = float(acronym_counts[acronym_class.class_id])
        values.append(count)
        values.append(_per_1000(_acronym_class_name(acronym_class.class_id, "per_1000_tokens"), count, token_count, diagnostics))
    sidecar = _abbreviation_acronym_sidecar(view, lexicon, acronym_lexicon, abbreviation_matches, named_acronym_matches, acronym_matches)
    return values, tuple(diagnostics), sidecar


def _abbreviation_matches(raw_text: str, lexicon: VersionedLexicon) -> tuple[AbbreviationAcronymMatch, ...]:
    lower_raw = raw_text.lower()
    matches: list[AbbreviationAcronymMatch] = []
    for entry in lexicon.entries:
        matches.extend(
            (
                AbbreviationAcronymMatch(
                    match_kind="abbreviation",
                    match_id=entry.token,
                    text=raw_text[match.start() : match.end()],
                    start=match.start(),
                    end=match.end(),
                    groups=entry.groups,
                )
            )
            for match in re.finditer(_abbreviation_pattern(entry.token), lower_raw)
        )
    return _sorted_matches(matches)


def _abbreviation_pattern(token: str) -> str:
    escaped = re.escape(token.lower())
    boundary = r"[A-Za-z0-9_]"
    if "." in token:
        return rf"(?<!{boundary}){escaped}(?!{boundary})"
    return rf"(?<![A-Za-z0-9_.]){escaped}(?![A-Za-z0-9_.])"


def _abbreviation_counts(matches: tuple[AbbreviationAcronymMatch, ...], lexicon: VersionedLexicon) -> Counter[str]:
    counts: Counter[str] = Counter({token: 0 for token in lexicon.tokens()})
    for match in matches:
        counts[match.match_id] += 1
    return counts


def _abbreviation_group_counts(counts: Counter[str], lexicon: VersionedLexicon) -> dict[str, int]:
    group_counts = {group: 0 for group in lexicon.groups()}
    for entry in lexicon.entries:
        count = counts[entry.token]
        for group in entry.groups:
            group_counts[group] += count
    return group_counts


def _named_acronym_matches(raw_text: str, acronym_lexicon: VersionedLexicon) -> tuple[AbbreviationAcronymMatch, ...]:
    matches: list[AbbreviationAcronymMatch] = []
    for entry in acronym_lexicon.entries:
        matches.extend(
            (
                AbbreviationAcronymMatch(
                    match_kind="named_acronym",
                    match_id=entry.token,
                    text=raw_text[match.start() : match.end()],
                    start=match.start(),
                    end=match.end(),
                    groups=entry.groups,
                )
            )
            for match in re.finditer(_named_acronym_pattern(entry.token), raw_text)
        )
    return _sorted_matches(matches)


def _named_acronym_pattern(token: str) -> str:
    escaped = re.escape(token)
    boundary = r"[A-Za-z0-9_]"
    if "." in token:
        return rf"(?<!{boundary}){escaped}(?!{boundary})"
    return rf"(?<!{boundary}){escaped}(?!{boundary})"


def _named_acronym_counts(matches: tuple[AbbreviationAcronymMatch, ...], acronym_lexicon: VersionedLexicon) -> Counter[str]:
    counts: Counter[str] = Counter({token: 0 for token in acronym_lexicon.tokens()})
    for match in matches:
        counts[match.match_id] += 1
    return counts


def _named_acronym_group_counts(counts: Counter[str], acronym_lexicon: VersionedLexicon) -> dict[str, int]:
    group_counts = {group: 0 for group in acronym_lexicon.groups()}
    for entry in acronym_lexicon.entries:
        count = counts[entry.token]
        for group in entry.groups:
            group_counts[group] += count
    return group_counts


def _acronym_matches(raw_text: str) -> tuple[AbbreviationAcronymMatch, ...]:
    matches: list[AbbreviationAcronymMatch] = []
    for acronym_class in _acronym_classes():
        matches.extend(
            (
                AbbreviationAcronymMatch(
                    match_kind="acronym",
                    match_id=acronym_class.class_id,
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    groups=(),
                )
            )
            for match in re.finditer(acronym_class.pattern, raw_text)
        )
    return _sorted_matches(matches)


def _acronym_counts(matches: tuple[AbbreviationAcronymMatch, ...]) -> Counter[str]:
    counts: Counter[str] = Counter({acronym_class.class_id: 0 for acronym_class in _acronym_classes()})
    for match in matches:
        counts[match.match_id] += 1
    return counts


def _abbreviation_acronym_sidecar(
    view: DocumentView,
    lexicon: VersionedLexicon,
    acronym_lexicon: VersionedLexicon,
    abbreviation_matches: tuple[AbbreviationAcronymMatch, ...],
    named_acronym_matches: tuple[AbbreviationAcronymMatch, ...],
    acronym_matches: tuple[AbbreviationAcronymMatch, ...],
) -> AbbreviationAcronymSidecar:
    return AbbreviationAcronymSidecar(
        document_id=view.document_id,
        schema_version="abbreviation_acronym_matches_v2",
        lexicon_id=lexicon.lexicon_id,
        language=lexicon.language,
        version=lexicon.version,
        normalization=lexicon.normalization,
        acronym_lexicon_id=acronym_lexicon.lexicon_id,
        acronym_language=acronym_lexicon.language,
        acronym_version=acronym_lexicon.version,
        acronym_normalization=acronym_lexicon.normalization,
        abbreviation_match_count=len(abbreviation_matches),
        named_acronym_match_count=len(named_acronym_matches),
        acronym_match_count=len(acronym_matches),
        matches=_sorted_matches(abbreviation_matches + named_acronym_matches + acronym_matches),
    )


def _sorted_matches(matches: Iterable[AbbreviationAcronymMatch]) -> tuple[AbbreviationAcronymMatch, ...]:
    return tuple(sorted(matches, key=lambda item: (item.start, item.end, item.match_kind, item.match_id)))


def _acronym_classes() -> tuple[AcronymClass, ...]:
    return (
        AcronymClass("all_caps", r"\b[A-Z]{2,}\b", "Consecutive uppercase letter token"),
        AcronymClass("dotted", r"(?<![A-Za-z0-9_])(?:[A-Z]\.){2,}(?![A-Za-z0-9_])", "Dotted uppercase acronym"),
        AcronymClass("mixed_alnum", r"\b(?:[A-Z]+[0-9][A-Z0-9]*|[0-9]+[A-Z][A-Z0-9]*)\b", "Uppercase alphanumeric acronym"),
    )


def _per_1000(feature_name: str, count: float, token_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if token_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_tokens"))
        return float("nan")
    return count * 1000.0 / float(token_count)


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _abbreviation_item_name(token: str, measure: str) -> str:
    return f"text::abbreviation_acronym_profile::abbreviation_item={token}::{measure}"


def _abbreviation_group_name(group: str, measure: str) -> str:
    return f"text::abbreviation_acronym_profile::abbreviation_group={group}::{measure}"


def _acronym_class_name(class_id: str, measure: str) -> str:
    return f"text::abbreviation_acronym_profile::acronym_class={class_id}::{measure}"


def _named_acronym_item_name(token: str, measure: str) -> str:
    return f"text::abbreviation_acronym_profile::named_acronym_item={token}::{measure}"


def _named_acronym_group_name(group: str, measure: str) -> str:
    return f"text::abbreviation_acronym_profile::named_acronym_group={group}::{measure}"


def _load_named_acronym_lexicon() -> VersionedLexicon:
    return load_lexicon(_NAMED_ACRONYM_RESOURCE_NAME)


def _spec_for_name(name: str, lexicon: VersionedLexicon, acronym_lexicon: VersionedLexicon) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the abbreviation or acronym class is absent"
    formula_or_rule = "count boundary-guarded abbreviation entry, named acronym entry, or deterministic acronym class"
    if name.endswith("per_1000_tokens"):
        normalization = "per_1000_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_tokens when no tokens exist"
        formula_or_rule = "abbreviation, named acronym, or acronym class count * 1000 / token count"
    return FeatureSpec(
        name=name,
        family="abbreviation_acronym_profile",
        description=f"Abbreviation and acronym profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.RAW,
        topic_dependence=TopicDependence.MIXED,
        text_length_policy="counts are always defined; per-1,000-token rates require token denominator",
        provenance=(
            f"lexicon_id={lexicon.lexicon_id}; language={lexicon.language}; version={lexicon.version}; "
            f"source={lexicon.source}; license_note={lexicon.license_note}; normalization={lexicon.normalization}; "
            f"acronym_lexicon_id={acronym_lexicon.lexicon_id}; acronym_language={acronym_lexicon.language}; "
            f"acronym_version={acronym_lexicon.version}; acronym_source={acronym_lexicon.source}; "
            f"acronym_license_note={acronym_lexicon.license_note}; acronym_normalization={acronym_lexicon.normalization}; "
            "built_in_acronym_regex_rules:v1"
        ),
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
