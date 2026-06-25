"""Layout and whitespace profile feature block."""

from __future__ import annotations

import re
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
class LayoutMarkerClass:
    """One deterministic layout marker class."""

    marker_id: str
    description: str


@dataclass(frozen=True)
class WhitespaceSignal:
    """One deterministic whitespace signal."""

    signal_id: str
    description: str


@dataclass(frozen=True)
class LayoutMarkerMatch:
    """One matched layout marker line for sidecar output."""

    line_index: int
    marker_id: str
    line_text: str
    markdown_heading_depth: int | None


@dataclass(frozen=True)
class LayoutMarkerSidecar:
    """Matched layout marker sidecar for one document."""

    document_id: str
    schema_version: str
    line_count: int
    matches: tuple[LayoutMarkerMatch, ...]


class LayoutWhitespaceProfileTransformer(BaseEstimator):
    """Sklearn-compatible layout and whitespace profile transformer."""

    def __init__(self, text_column: str, config: PreprocessingConfig, output: str) -> None:
        self.text_column = text_column
        self.config = config
        self.output = output

    def fit(self, x: object, y: object) -> Self:
        """Validate configuration and freeze layout/whitespace metadata."""
        del y
        validate_output_mode(self.output)
        _ = text_series(x, self.text_column)
        self.feature_names_out_ = np.asarray(layout_whitespace_profile_feature_names(), dtype=object)
        self.registry_ = FeatureRegistry(specs=layout_whitespace_profile_feature_specs())
        self.registry_.require_complete()
        self.n_features_in_ = 1
        self.feature_names_in_ = np.asarray([self.text_column], dtype=object)
        return self

    def transform(self, x: object) -> pd.DataFrame | np.ndarray | sparse.csr_matrix:
        """Compute layout/whitespace profile features without changing rows."""
        require_fitted(self, "feature_names_out_")
        series = text_series(x, self.text_column)
        rows: list[list[float]] = []
        diagnostics: list[tuple[FeatureDiagnostic, ...]] = []
        sidecars: list[LayoutMarkerSidecar] = []
        for row_index, text in enumerate(series.tolist()):
            document_id = str(series.index[row_index])
            view = DocumentView.from_text(str(text), self.config, document_id=document_id)
            row, row_diagnostics = _layout_whitespace_row(view)
            rows.append(row)
            diagnostics.append(row_diagnostics)
            sidecars.append(_layout_marker_sidecar(view))
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
        """Return stable layout/whitespace profile feature names."""
        del input_features
        require_fitted(self, "feature_names_out_")
        return self.feature_names_out_


def layout_whitespace_profile_feature_names() -> tuple[str, ...]:
    """Return stable layout/whitespace feature names in output order."""
    names: list[str] = []
    for marker_class in _layout_marker_classes():
        names.extend((_layout_marker_name(marker_class.marker_id, "count"), _layout_marker_name(marker_class.marker_id, "per_100_lines")))
    names.extend((_section_depth_name("max_markdown_heading_depth"), _section_depth_name("mean_markdown_heading_depth")))
    for signal in _whitespace_signals():
        if signal.signal_id in {"max_indentation_spaces", "mean_indentation_spaces"}:
            names.append(_whitespace_name(signal.signal_id, None))
        else:
            names.extend((_whitespace_name(signal.signal_id, "count"), _whitespace_name(signal.signal_id, "per_100_lines")))
    return tuple(names)


def layout_whitespace_profile_feature_specs() -> tuple[FeatureSpec, ...]:
    """Return metadata for layout/whitespace profile features."""
    return tuple(_spec_for_name(name) for name in layout_whitespace_profile_feature_names())


def _layout_whitespace_row(view: DocumentView) -> tuple[list[float], tuple[FeatureDiagnostic, ...]]:
    raw = view.raw
    lines = raw.splitlines()
    line_count = len(lines)
    nonblank_line_count = sum(1 for line in lines if line.strip() != "")
    marker_counts = _layout_marker_counts(lines)
    heading_depths = _markdown_heading_depths(lines)
    whitespace_counts = _whitespace_counts(raw, lines)
    diagnostics: list[FeatureDiagnostic] = []
    values: list[float] = []

    for marker_class in _layout_marker_classes():
        count = float(marker_counts[marker_class.marker_id])
        values.append(count)
        values.append(_per_100_lines(_layout_marker_name(marker_class.marker_id, "per_100_lines"), count, line_count, diagnostics))

    max_depth = float(max(heading_depths, default=0))
    values.append(max_depth)
    values.append(_mean_heading_depth(heading_depths, diagnostics))

    for signal in _whitespace_signals():
        if signal.signal_id == "max_indentation_spaces":
            values.append(float(_max_indentation_spaces(lines)))
        elif signal.signal_id == "mean_indentation_spaces":
            values.append(_mean_indentation_spaces(lines, nonblank_line_count, diagnostics))
        else:
            count = float(whitespace_counts[signal.signal_id])
            values.append(count)
            values.append(_per_100_lines(_whitespace_name(signal.signal_id, "per_100_lines"), count, line_count, diagnostics))

    return values, tuple(diagnostics)


def _layout_marker_counts(lines: list[str]) -> dict[str, int]:
    matches = _layout_marker_matches(lines)
    counts = {marker_class.marker_id: 0 for marker_class in _layout_marker_classes()}
    for match in matches:
        counts[match.marker_id] += 1
    return counts


def _layout_marker_sidecar(view: DocumentView) -> LayoutMarkerSidecar:
    lines = view.raw.splitlines()
    return LayoutMarkerSidecar(
        document_id=view.document_id,
        schema_version="layout_marker_sidecar_v1",
        line_count=len(lines),
        matches=_layout_marker_matches(lines),
    )


def _layout_marker_matches(lines: list[str]) -> tuple[LayoutMarkerMatch, ...]:
    matches: list[LayoutMarkerMatch] = []
    for line_index, line in enumerate(lines):
        for marker_class in _layout_marker_classes():
            if _line_matches_marker(line, marker_class.marker_id):
                heading_depth = None
                if marker_class.marker_id == "markdown_heading_line":
                    heading_depth = _markdown_heading_depth(line)
                matches.append(
                    LayoutMarkerMatch(
                        line_index=line_index,
                        marker_id=marker_class.marker_id,
                        line_text=line,
                        markdown_heading_depth=heading_depth,
                    )
                )
    return tuple(matches)


def _line_matches_marker(line: str, marker_id: str) -> bool:
    if marker_id == "markdown_heading_line":
        return _markdown_heading_depth(line) > 0
    if marker_id == "bullet_line":
        return re.match(r"^\s*[-*]\s+", line) is not None
    if marker_id == "numbered_line":
        return re.match(r"^\s*\d+[.)]\s+", line) is not None
    if marker_id == "block_quote_line":
        return line.lstrip().startswith(">")
    if marker_id == "code_fence_line":
        return line.strip().startswith("```")
    if marker_id == "table_line":
        return "|" in line and line.count("|") >= 2
    if marker_id == "email_header_line":
        return re.match(r"^\s*(from|to|subject|date):", line, flags=re.IGNORECASE) is not None
    raise ValueError(f"Unsupported layout marker id: {marker_id}")


def _markdown_heading_depths(lines: list[str]) -> tuple[int, ...]:
    return tuple(depth for line in lines if (depth := _markdown_heading_depth(line)) > 0)


def _markdown_heading_depth(line: str) -> int:
    match = re.match(r"^\s{0,3}(#{1,6})\s+", line)
    if match is None:
        return 0
    return len(match.group(1))


def _whitespace_counts(raw: str, lines: list[str]) -> dict[str, int]:
    repeated_space_runs = tuple(re.finditer(r" {2,}", raw))
    return {
        "tab_character": raw.count("\t"),
        "tab_indented_line": sum(1 for line in lines if line.startswith("\t")),
        "space_indented_line": sum(1 for line in lines if re.match(r"^ +\S", line) is not None),
        "repeated_space_run": len(repeated_space_runs),
        "repeated_space_character": sum(len(match.group(0)) for match in repeated_space_runs),
        "leading_whitespace_line": sum(1 for line in lines if line != "" and line[0].isspace()),
        "trailing_whitespace_line": sum(1 for line in lines if line != "" and line[-1] in {" ", "\t"}),
        "hard_wrap_candidate_line": _hard_wrap_candidate_count(lines),
    }


def _max_indentation_spaces(lines: list[str]) -> int:
    return max((_leading_space_count(line) for line in lines if line.strip() != ""), default=0)


def _mean_indentation_spaces(lines: list[str], nonblank_line_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    feature_name = _whitespace_name("mean_indentation_spaces", None)
    if nonblank_line_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_nonblank_lines"))
        return float("nan")
    return float(sum(_leading_space_count(line) for line in lines if line.strip() != "")) / float(nonblank_line_count)


def _leading_space_count(line: str) -> int:
    match = re.match(r"^ *", line)
    if match is None:
        return 0
    return len(match.group(0))


def _hard_wrap_candidate_count(lines: list[str]) -> int:
    count = 0
    for index, line in enumerate(lines[:-1]):
        stripped = line.strip()
        next_stripped = lines[index + 1].strip()
        if stripped == "" or next_stripped == "":
            continue
        if len(line.rstrip(" \t")) < 72:
            continue
        if stripped.endswith((".", "!", "?", ":", ";")):
            continue
        if next_stripped[0].islower():
            count += 1
    return count


def _mean_heading_depth(heading_depths: tuple[int, ...], diagnostics: list[FeatureDiagnostic]) -> float:
    feature_name = _section_depth_name("mean_markdown_heading_depth")
    if len(heading_depths) == 0:
        diagnostics.append(_undefined(feature_name, "zero_heading_lines"))
        return float("nan")
    return float(sum(heading_depths)) / float(len(heading_depths))


def _per_100_lines(feature_name: str, count: float, line_count: int, diagnostics: list[FeatureDiagnostic]) -> float:
    if line_count == 0:
        diagnostics.append(_undefined(feature_name, "zero_lines"))
        return float("nan")
    return count * 100.0 / float(line_count)


def _undefined(feature_name: str, reason: str) -> FeatureDiagnostic:
    return FeatureDiagnostic(feature_name=feature_name, status=FeatureStatus.UNDEFINED, reason=reason, warnings=())


def _layout_marker_classes() -> tuple[LayoutMarkerClass, ...]:
    return (
        LayoutMarkerClass("markdown_heading_line", "Markdown ATX heading line"),
        LayoutMarkerClass("bullet_line", "Markdown-like unordered bullet line"),
        LayoutMarkerClass("numbered_line", "Markdown-like ordered list line"),
        LayoutMarkerClass("block_quote_line", "Line beginning with a block quote marker"),
        LayoutMarkerClass("code_fence_line", "Line beginning a fenced code marker"),
        LayoutMarkerClass("table_line", "Line containing at least two pipe delimiters"),
        LayoutMarkerClass("email_header_line", "Email-style header line"),
    )


def _whitespace_signals() -> tuple[WhitespaceSignal, ...]:
    return (
        WhitespaceSignal("tab_character", "Raw tab character count"),
        WhitespaceSignal("tab_indented_line", "Line beginning with a tab"),
        WhitespaceSignal("space_indented_line", "Line beginning with one or more spaces before text"),
        WhitespaceSignal("max_indentation_spaces", "Maximum leading spaces on a nonblank line"),
        WhitespaceSignal("mean_indentation_spaces", "Mean leading spaces across nonblank lines"),
        WhitespaceSignal("repeated_space_run", "Run of two or more consecutive spaces"),
        WhitespaceSignal("repeated_space_character", "Characters belonging to runs of two or more consecutive spaces"),
        WhitespaceSignal("leading_whitespace_line", "Line beginning with whitespace"),
        WhitespaceSignal("trailing_whitespace_line", "Line ending with space or tab"),
        WhitespaceSignal("hard_wrap_candidate_line", "Long nonterminal line followed by a lowercase continuation line"),
    )


def _layout_marker_name(marker_id: str, measure: str) -> str:
    return f"text::layout_whitespace_profile::layout_marker={marker_id}::{measure}"


def _section_depth_name(measure: str) -> str:
    return f"text::layout_whitespace_profile::section_depth::{measure}"


def _whitespace_name(signal_id: str, measure: str | None) -> str:
    if measure is None:
        return f"text::layout_whitespace_profile::whitespace={signal_id}"
    return f"text::layout_whitespace_profile::whitespace={signal_id}::{measure}"


def _spec_for_name(name: str) -> FeatureSpec:
    normalization = "raw_count"
    undefined_behavior = "defined as zero when the layout or whitespace signal is absent"
    formula_or_rule = "count deterministic raw layout marker or whitespace signal"
    topic_dependence = TopicDependence.MIXED
    if "::whitespace=" in name:
        topic_dependence = TopicDependence.MOSTLY_TOPIC_INDEPENDENT
    if name.endswith("per_100_lines"):
        normalization = "per_100_lines"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_lines when no physical lines exist"
        formula_or_rule = "layout or whitespace signal count * 100 / physical line count"
    if name.endswith("mean_markdown_heading_depth"):
        normalization = "heading_depth_mean"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_heading_lines when no Markdown headings exist"
        formula_or_rule = "mean Markdown ATX heading marker depth"
    if name.endswith("max_markdown_heading_depth"):
        normalization = "heading_depth_max"
        formula_or_rule = "maximum Markdown ATX heading marker depth, or zero when no Markdown headings exist"
    if name.endswith("mean_indentation_spaces"):
        normalization = "indentation_mean"
        undefined_behavior = "NaN with FeatureDiagnostic reason zero_nonblank_lines when no nonblank lines exist"
        formula_or_rule = "mean leading space characters across nonblank physical lines; tabs are counted separately"
    if name.endswith("max_indentation_spaces"):
        normalization = "indentation_max"
        formula_or_rule = "maximum leading space characters on a nonblank physical line; tabs are counted separately"
    return FeatureSpec(
        name=name,
        family="layout_whitespace_profile",
        description=f"Layout and whitespace profile feature for {name}",
        formula_or_rule=formula_or_rule,
        input_layer=InputLayer.RAW,
        topic_dependence=topic_dependence,
        text_length_policy="counts are always defined; line-normalized rates and means require non-empty line denominators",
        provenance="built_in_layout_whitespace_profile_rules:v1; preprocessing_config",
        output_dtype="float64",
        undefined_behavior=undefined_behavior,
        normalization=normalization,
        sparsity="dense_scalar",
        stability_status=StabilityStatus.DETERMINISTIC,
    )
