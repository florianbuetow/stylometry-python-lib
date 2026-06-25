"""Tests for public formula documentation pages."""

from __future__ import annotations

from pathlib import Path

from stylometry_python_lib import built_in_research_registry
from stylometry_python_lib.features.deterministic import deterministic_feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _read_doc(relative_path: str) -> str:
    path = PROJECT_ROOT / relative_path
    assert path.exists(), f"Missing formula documentation page: {relative_path}"
    return path.read_text(encoding="utf-8")


def test_formula_docs_index_links_implemented_pages_and_v3_research_gap_policy() -> None:
    text = _read_doc("docs/formulas/README.md")

    assert "(lexical-richness.md)" in text
    assert "(readability.md)" in text
    assert "docs/RESEARCH-v2/" in text
    assert "docs/REASEARCH-v3.md" in text


def test_lexical_richness_formula_doc_covers_names_formulas_thresholds_and_provenance() -> None:
    text = _read_doc("docs/formulas/lexical-richness.md")
    scalar_names = tuple(name for name in deterministic_feature_names() if "::lexical_richness::" in name)
    required_markers = (
        "docs/RESEARCH-v2/",
        "src/stylometry_python_lib/features/deterministic.py",
        "src/stylometry_python_lib/features/lexical_richness.py",
        "N",
        "V",
        "V_i",
        "f_t",
        "text::lexical_richness_spectrum::frequency_bin=i::type_count",
        "text::lexical_richness_spectrum::frequency_bin=i::types_ratio",
        "text::lexical_richness_spectrum::frequency_bin=i::tokens_ratio",
        "frequency_spectrum_v1",
        "segment_size=50",
        "window_size=min(50,N)",
        "threshold=0.72",
        "sample_size=min(42,N)",
        "sample_sizes=35..50",
        "samples_per_size=100",
        "fit_iterations=80",
        "TTR(s,D)=D/s*(sqrt(1+2s/D)-1)",
        "below_vocd_50_token_threshold",
        "all_samples_unique_vocd_singularity",
        "short_text_unstable",
        "vocd_sampling=v1",
        "hash_seed=stylometry_python_lib_vocd_d_v1",
        "fit=golden_section_curve_fit_v1",
    )

    for name in scalar_names:
        assert name in text
    for marker in required_markers:
        assert marker in text


def test_readability_formula_doc_covers_names_formulas_thresholds_and_provenance() -> None:
    text = _read_doc("docs/formulas/readability.md")
    scalar_names = tuple(name for name in deterministic_feature_names() if "::readability::" in name)
    required_markers = (
        "docs/RESEARCH-v2/",
        "src/stylometry_python_lib/features/deterministic.py",
        "W",
        "S",
        "SYL",
        "CW",
        "SSW",
        "LW",
        "D",
        "206.835 - 1.015 * (W / S) - 84.6 * (SYL / W)",
        "0.39 * (W / S) + 11.8 * (SYL / W) - 15.59",
        "0.4 * ((W / S) + 100 * (CW / W))",
        "0.0588 * (C / W * 100) - 0.296 * (S / W * 100) - 15.8",
        "1.043 * sqrt(CW * (30 / S)) + 3.1291",
        "4.71 * (C / W) + 0.5 * (W / S) - 21.43",
        "3.6365",
        "below_smog_3_sentence_threshold",
        "below_forcast_150_word_threshold",
        "below_linsear_write_100_word_threshold",
        "short_text_unstable",
        "syllable_dictionary=syllable_counts_en_v1",
        "syllable_fallback=heuristic_v1",
        "dale_chall_easy_words=dale_chall_easy_words_en_seed_v1",
    )

    for name in scalar_names:
        assert name in text
    for marker in required_markers:
        assert marker in text


def test_research_registry_links_formula_docs_for_stable_implemented_formula_families() -> None:
    registry = built_in_research_registry()

    assert registry.by_taxonomy_id("det.type_token_ratio").documentation_link == "docs/formulas/lexical-richness.md"
    assert registry.by_taxonomy_id("det.ttr_variants").documentation_link == "docs/formulas/lexical-richness.md"
    assert registry.by_taxonomy_id("det.dis_legomena_frequency_spectrum").documentation_link == "docs/formulas/lexical-richness.md"
    assert registry.by_taxonomy_id("det.readability_explicit_formula").documentation_link == "docs/formulas/readability.md"
    assert "formula page tests present" in registry.by_taxonomy_id("det.ttr_variants").test_status
    assert "formula page tests present" in registry.by_taxonomy_id("det.readability_explicit_formula").test_status
