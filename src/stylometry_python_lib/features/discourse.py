"""Discourse marker and transition phrase profile feature blocks."""

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


@dataclass(frozen=True)
class DiscourseLexiconMatch:
    """One matched discourse marker or transition phrase span."""

    item_id: str
    token: str
    text: str
    start: int
    end: int
    groups: tuple[str, ...]
    polyfunctional: bool


@dataclass(frozen=True)
class DiscourseLexiconSidecar:
    """Matched raw phrase lexicon spans for one document."""

    document_id: str
    schema_version: str
    family: str
    lexicon_id: str
    language: str
    version: str
    normalization: str
    match_count: int
    matches: tuple[DiscourseLexiconMatch, ...]


class DiscourseLexiconProfileTransformer(BaseEstimator):
    """Sklearn-compatible raw phrase lexicon profile transformer for discourse features."""

    def __init__(self, text_column: str, config: PreprocessingConfig, lexicon_name: str, family: str, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.lexicon_name = lexicon_name
        self.family = family
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Load lexicon metadata and freeze discourse profile feature names."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        lexicon = load_lexicon(self.lexicon_name)
        _validate_unique_item_ids(lexicon)
        self.lexicon_ = lexicon
        self.feature_names_out_ = np.asarray(discourse_lexicon_profile_feature_names(self.family, lexicon), dtype=object)
        self.registry_ = FeatureRegistry(specs=discourse_lexicon_profile_feature_specs(self.family, lexicon))
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute discourse lexicon profile features without changing rows."""
        require_fitted(self, "lexicon_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[DiscourseLexiconSidecar] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics, sidecar = _discourse_lexicon_row(view, self.lexicon_, self.family)
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
        """Return stable discourse profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def discourse_marker_profile_transformer(text_column: str, config: PreprocessingConfig, output: str) -> DiscourseLexiconProfileTransformer:
    """Build a discourse marker profile transformer."""
    return DiscourseLexiconProfileTransformer(
        text_column=text_column,
        config=config,
        lexicon_name="discourse_markers",
        family="discourse_marker_profile",
        output=output,
    )


def transition_phrase_profile_transformer(text_column: str, config: PreprocessingConfig, output: str) -> DiscourseLexiconProfileTransformer:
    """Build a transition phrase profile transformer."""
    return DiscourseLexiconProfileTransformer(
        text_column=text_column,
        config=config,
        lexicon_name="transition_phrases",
        family="transition_phrase_profile",
        output=output,
    )


def discourse_marker_profile_feature_names(lexicon: VersionedLexicon) -> tuple[str, ...]:
    """Return stable discourse marker profile feature names."""
    return discourse_lexicon_profile_feature_names("discourse_marker_profile", lexicon)


def transition_phrase_profile_feature_names(lexicon: VersionedLexicon) -> tuple[str, ...]:
    """Return stable transition phrase profile feature names."""
    return discourse_lexicon_profile_feature_names("transition_phrase_profile", lexicon)


def discourse_marker_profile_feature_specs(lexicon: VersionedLexicon) -> tuple[FeatureSpec, ...]:
    """Return metadata for discourse marker profile features."""
    return discourse_lexicon_profile_feature_specs("discourse_marker_profile", lexicon)


def transition_phrase_profile_feature_specs(lexicon: VersionedLexicon) -> tuple[FeatureSpec, ...]:
    """Return metadata for transition phrase profile features."""
    return discourse_lexicon_profile_feature_specs("transition_phrase_profile", lexicon)


def discourse_lexicon_profile_feature_names(family: str, lexicon: VersionedLexicon) -> tuple[str, ...]:
    """Return stable raw phrase lexicon feature names for a family."""
    names: list[str] = []
    for entry in lexicon.entries:
        item_id = _item_id(entry.token)
        names.extend((_item_name(family, item_id, "count"), _item_name(family, item_id, "per_1000_tokens")))
    for group in lexicon.groups():
        names.extend((_group_name(family, group, "count"), _group_name(family, group, "per_1000_tokens")))
    return tuple(names)


def discourse_lexicon_profile_feature_specs(family: str, lexicon: VersionedLexicon) -> tuple[FeatureSpec, ...]:
    """Return metadata for raw phrase lexicon profile features."""
    return tuple(_spec_for_name(name, family, lexicon) for name in discourse_lexicon_profile_feature_names(family, lexicon))


def _discourse_lexicon_row(
    view: DocumentView, lexicon: VersionedLexicon, family: str
) -> tuple[list[float], tuple[FeatureDiagnostic, ...], DiscourseLexiconSidecar]:
    token_count = len(view.tokens)
    matches = _phrase_matches(view.normalized, lexicon)
    counts = _phrase_counts(matches, lexicon)
    diagnostics: list[FeatureDiagnostic] = []
    values: list[float] = []

    for entry in lexicon.entries:
        item_id = _item_id(entry.token)
        count = float(counts[entry.token])
        values.append(count)
        values.append(_per_1000(_item_name(family, item_id, "per_1000_tokens"), count, token_count, diagnostics))
        if count > 0.0 and len(entry.groups) > 1:
            diagnostics.append(_polyfunctional_warning(_item_name(family, item_id, "count"), entry.groups))
    group_counts = _group_counts(counts, lexicon)
    for group in lexicon.groups():
        count = float(group_counts[group])
        values.append(count)
        values.append(_per_1000(_group_name(family, group, "per_1000_tokens"), count, token_count, diagnostics))
    sidecar = _discourse_lexicon_sidecar(view, lexicon, family, matches)
    return values, tuple(diagnostics), sidecar


def _phrase_matches(text: str, lexicon: VersionedLexicon) -> tuple[DiscourseLexiconMatch, ...]:
    lower_text = text.lower()
    matches: list[DiscourseLexiconMatch] = []
    for entry in lexicon.entries:
        item_id = _item_id(entry.token)
        matches.extend(
            (
                DiscourseLexiconMatch(
                    item_id=item_id,
                    token=entry.token,
                    text=text[match.start() : match.end()],
                    start=match.start(),
                    end=match.end(),
                    groups=entry.groups,
                    polyfunctional=len(entry.groups) > 1,
                )
            )
            for match in re.finditer(_phrase_pattern(entry.token), lower_text)
        )
    return _sorted_matches(matches)


def _phrase_pattern(phrase: str) -> str:
    escaped_parts = tuple(re.escape(part) for part in phrase.lower().split(" "))
    escaped_phrase = r"\s+".join(escaped_parts)
    boundary = r"[A-Za-z0-9_-]"
    return rf"(?<!{boundary}){escaped_phrase}(?!{boundary})"


def _phrase_counts(matches: tuple[DiscourseLexiconMatch, ...], lexicon: VersionedLexicon) -> Counter[str]:
    counts: Counter[str] = Counter({token: 0 for token in lexicon.tokens()})
    for match in matches:
        counts[match.token] += 1
    return counts


def _group_counts(counts: Counter[str], lexicon: VersionedLexicon) -> dict[str, int]:
    group_counts = {group: 0 for group in lexicon.groups()}
    for entry in lexicon.entries:
        count = counts[entry.token]
        for group in entry.groups:
            group_counts[group] += count
    return group_counts


def _discourse_lexicon_sidecar(
    view: DocumentView, lexicon: VersionedLexicon, family: str, matches: tuple[DiscourseLexiconMatch, ...]
) -> DiscourseLexiconSidecar:
    return DiscourseLexiconSidecar(
        document_id=view.document_id,
        schema_version=f"{family}_matches_v1",
        family=family,
        lexicon_id=lexicon.lexicon_id,
        language=lexicon.language,
        version=lexicon.version,
        normalization=lexicon.normalization,
        match_count=len(matches),
        matches=matches,
    )


def _sorted_matches(matches: Iterable[DiscourseLexiconMatch]) -> tuple[DiscourseLexiconMatch, ...]:
    return tuple(sorted(matches, key=lambda item: (item.start, item.end, item.item_id)))


def _per_1000(feature_name: str, count: float, token_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if token_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_tokens"))
        return float("nan")
    return count * 1000.0 / float(token_count)


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _polyfunctional_warning(feature_name: str, groups: tuple[str, ...]) -> FeatureDiagnostic:
    return FeatureDiagnostic(
        feature_name=feature_name,
        status=FeatureStatus.WARNING,
        reason="polyfunctional_marker",
        warnings=(f"groups={','.join(groups)}",),
    )


def _item_id(token: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", token.lower()).strip("_")
    if normalized == "":
        raise ValueError(f"Cannot derive feature item id for lexicon token: {token}")
    return normalized


def _validate_unique_item_ids(lexicon: VersionedLexicon) -> None:
    seen: dict[str, str] = {}
    for token in lexicon.tokens():
        item_id = _item_id(token)
        if item_id in seen:
            raise ValueError(f"Lexicon resource {lexicon.name} item id collision: {seen[item_id]} and {token}")
        seen[item_id] = token


def _item_name(family: str, item_id: str, measure: str) -> str:
    return f"text::{family}::item={item_id}::{measure}"


def _group_name(family: str, group: str, measure: str) -> str:
    return f"text::{family}::group={group}::{measure}"


def _spec_for_name(name: str, family: str, lexicon: VersionedLexicon) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the marker or phrase is absent"
    formula_or_rule = "count boundary-guarded lowercase resource phrase in normalized raw text"
    if name.endswith("per_1000_tokens"):
        normalization = "per_1000_tokens"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_tokens when no tokens exist"
        formula_or_rule = "boundary-guarded lowercase resource phrase count * 1000 / token count"
    return FeatureSpec(
        name=name,
        family=family,
        description=f"Raw phrase lexicon profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.RAW,
        topic_dependence=TopicDependence.MIXED,
        text_length_policy="counts are always defined; per-1,000-token rates require token denominator",
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
