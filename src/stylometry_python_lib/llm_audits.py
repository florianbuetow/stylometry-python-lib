"""Stability audit summaries for LLM-backed stylometry features."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from stylometry_python_lib.llm_features import RenderedLLMPrompt


@dataclass(frozen=True)
class LLMRepeatedRunAudit:
    """Variance/agreement summary for repeated LLM runs."""

    feature_names: tuple[str, ...]
    run_count: int
    variances: tuple[float, ...]
    exact_agreement: tuple[float, ...]
    model: str
    prompt_versions: tuple[str, ...]


@dataclass(frozen=True)
class LLMPromptParaphraseAudit:
    """Variance summary for prompt paraphrase variants."""

    feature_name: str
    variant_count: int
    prompt_hashes: tuple[str, ...]
    variance: float
    model: str
    prompt_versions: tuple[str, ...]


def summarize_llm_repeated_runs(
    feature_names: tuple[str, ...],
    run_values: tuple[tuple[float, ...], ...],
    run_count: int,
    model: str,
    prompt_versions: tuple[str, ...],
) -> LLMRepeatedRunAudit:
    """Summarize repeated run variance and exact agreement for each feature."""
    _validate_audit_identity(feature_names, model, prompt_versions)
    if run_count <= 0:
        raise ValueError("run_count must be greater than 0")
    if len(run_values) != run_count:
        raise ValueError("run_values length must equal run_count")
    if len(run_values) == 0:
        raise ValueError("run_values must not be empty")
    for row in run_values:
        if len(row) != len(feature_names):
            raise ValueError("Each repeated-run row must match feature_names length")
        _validate_finite_values(row)
    matrix = np.asarray(run_values, dtype=np.float64)
    variances = tuple(float(value) for value in np.var(matrix, axis=0))
    exact_agreement = tuple(1.0 if len(set(matrix[:, index].tolist())) == 1 else 0.0 for index in range(matrix.shape[1]))
    return LLMRepeatedRunAudit(
        feature_names=feature_names,
        run_count=run_count,
        variances=variances,
        exact_agreement=exact_agreement,
        model=model,
        prompt_versions=prompt_versions,
    )


def summarize_llm_prompt_paraphrases(
    feature_name: str,
    rendered_prompts: tuple[RenderedLLMPrompt, ...],
    values: tuple[float, ...],
    model: str,
) -> LLMPromptParaphraseAudit:
    """Summarize variance across prompt paraphrase variants for one feature."""
    if feature_name == "":
        raise ValueError("feature_name must not be empty")
    if model == "":
        raise ValueError("model must not be empty")
    if len(rendered_prompts) == 0:
        raise ValueError("rendered_prompts must not be empty")
    if len(rendered_prompts) != len(values):
        raise ValueError("rendered_prompts length must equal values length")
    _validate_finite_values(values)
    prompt_hashes = tuple(prompt.prompt_hash for prompt in rendered_prompts)
    prompt_versions = tuple(prompt.prompt_version for prompt in rendered_prompts)
    for prompt in rendered_prompts:
        if prompt.feature_name != feature_name:
            raise ValueError("All rendered prompts must match feature_name")
    return LLMPromptParaphraseAudit(
        feature_name=feature_name,
        variant_count=len(rendered_prompts),
        prompt_hashes=prompt_hashes,
        variance=float(np.var(np.asarray(values, dtype=np.float64))),
        model=model,
        prompt_versions=prompt_versions,
    )


def _validate_audit_identity(feature_names: tuple[str, ...], model: str, prompt_versions: tuple[str, ...]) -> None:
    if len(feature_names) == 0:
        raise ValueError("feature_names must not be empty")
    if len(prompt_versions) == 0:
        raise ValueError("prompt_versions must not be empty")
    if model == "":
        raise ValueError("model must not be empty")
    for feature_name in feature_names:
        if feature_name == "":
            raise ValueError("feature_names must not contain empty values")
    for prompt_version in prompt_versions:
        if prompt_version == "":
            raise ValueError("prompt_versions must not contain empty values")


def _validate_finite_values(values: tuple[float, ...]) -> None:
    for value in values:
        if not math.isfinite(value):
            raise ValueError("audit values must be finite")
