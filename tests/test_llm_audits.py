"""Tests for LLM stability audit summaries."""

from __future__ import annotations

import pytest

from stylometry_python_lib import (
    render_llm_prompt,
    summarize_llm_prompt_paraphrases,
    summarize_llm_repeated_runs,
)


def test_summarize_llm_repeated_runs_records_variance_and_agreement() -> None:
    audit = summarize_llm_repeated_runs(
        feature_names=("text::llm::tone", "text::llm::cohesion_judgment"),
        run_values=((1.0, 0.5), (1.0, 0.7), (1.0, 0.6)),
        run_count=3,
        model="fixture-model",
        prompt_versions=("tone_prompt_v1", "cohesion_judgment_prompt_v1"),
    )

    assert audit.run_count == 3
    assert audit.variances[0] == 0.0
    assert audit.variances[1] > 0.0
    assert audit.exact_agreement == (1.0, 0.0)
    assert audit.model == "fixture-model"


def test_summarize_llm_prompt_paraphrases_records_prompt_hashes_and_variance() -> None:
    prompts = (
        render_llm_prompt("text::llm::tone", document_id="doc-a", text="A formal sentence."),
        render_llm_prompt("text::llm::tone", document_id="doc-a", text="A restrained formal sentence."),
    )

    audit = summarize_llm_prompt_paraphrases(
        feature_name="text::llm::tone",
        rendered_prompts=prompts,
        values=(0.0, 1.0),
        model="fixture-model",
    )

    assert audit.variant_count == 2
    assert audit.prompt_hashes[0] != audit.prompt_hashes[1]
    assert audit.variance == 0.25
    assert audit.prompt_versions == ("tone_prompt_v1", "tone_prompt_v1")


def test_llm_audits_fail_fast_on_missing_or_inconsistent_inputs() -> None:
    with pytest.raises(ValueError, match="run_count"):
        summarize_llm_repeated_runs(
            feature_names=("text::llm::tone",),
            run_values=((1.0,),),
            run_count=0,
            model="fixture-model",
            prompt_versions=("tone_prompt_v1",),
        )
    with pytest.raises(ValueError, match="match feature_names"):
        summarize_llm_repeated_runs(
            feature_names=("text::llm::tone",),
            run_values=((1.0, 2.0),),
            run_count=1,
            model="fixture-model",
            prompt_versions=("tone_prompt_v1",),
        )
    with pytest.raises(ValueError, match="feature_name"):
        summarize_llm_prompt_paraphrases(
            feature_name="text::llm::register",
            rendered_prompts=(render_llm_prompt("text::llm::tone", document_id="doc-a", text="A formal sentence."),),
            values=(0.0,),
            model="fixture-model",
        )
