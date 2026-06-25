"""Tests for the DocumentView preprocessing contract."""

from __future__ import annotations

from dataclasses import replace

from stylometry_python_lib import AnnotationLayerMetadata, DocumentView, SpecialTokenPolicy, english_preprocessing_config


def test_document_view_preserves_raw_normalization_policy_and_parallel_token_layers() -> None:
    config = english_preprocessing_config()
    raw = "Cafe\u0301 can't re-enter parseHTTPValue.\n\nNext line."

    view = DocumentView.from_text(raw, config, document_id="row-1")

    assert view.document_id == "row-1"
    assert view.raw == raw
    assert view.normalized.startswith("Café can't")
    assert view.config.unicode_normalization == "NFC"
    assert view.config_hash != ""
    assert "can't" in view.tokens
    assert "can't" in view.orthographic_tokens
    assert "can" in view.expanded_tokens
    assert "not" in view.expanded_tokens
    assert "re-enter" in view.hyphenated_tokens
    assert "re" in view.hyphen_split_tokens
    assert "enter" in view.hyphen_split_tokens


def test_document_view_tracks_special_tokens_and_class_normalization() -> None:
    retain_config = english_preprocessing_config()
    normalized_config = replace(retain_config, special_token_policy=SpecialTokenPolicy.CLASS_NORMALIZE)
    raw = "Email me@example.com, visit https://example.test, ping @writer about #Style parseHTTPValue snake_case 12.5 😀"

    retained = DocumentView.from_text(raw, retain_config, document_id="row-special")
    normalized = DocumentView.from_text(raw, normalized_config, document_id="row-special-normalized")

    retained_kinds = tuple(token.kind for token in retained.special_tokens)
    assert retained_kinds == (
        "email",
        "url",
        "mention",
        "hashtag",
        "code_identifier",
        "code_identifier",
        "number",
        "emoji",
    )
    assert all(token.document_id == "row-special" for token in retained.special_tokens)
    assert retained.special_tokens[0].raw == "me@example.com"
    assert retained.special_tokens[0].normalized == "me@example.com"
    assert "<EMAIL>" in normalized.orthographic_tokens
    assert "<URL>" in normalized.orthographic_tokens
    assert "<HASHTAG>" in normalized.orthographic_tokens
    assert "<MENTION>" in normalized.orthographic_tokens
    assert "<NUMBER>" in normalized.orthographic_tokens
    assert "<CODE_IDENTIFIER>" in normalized.orthographic_tokens
    assert "<EMOJI>" in normalized.orthographic_tokens
    assert normalized.special_tokens[0].normalized == "<EMAIL>"


def test_document_view_quote_authorial_and_format_layers_preserve_document_id() -> None:
    config = english_preprocessing_config()
    raw = 'Subject: Hi\n\nAlice said "quoted words".\n> previous message\n```python\nx = 1\n```'

    view = DocumentView.from_text(raw, config, document_id="email-row")
    layers = {layer.name: layer for layer in view.format_layers}

    assert tuple(span.text for span in view.quoted_spans) == ('"quoted words"',)
    assert all(span.document_id == "email-row" for span in view.quoted_spans)
    assert all(span.document_id == "email-row" for span in view.authorial_spans)
    assert "quoted words" not in "".join(span.text for span in view.authorial_spans)
    assert set(layers) == {"markdown", "html", "email", "code_fence"}
    assert all(layer.document_id == "email-row" for layer in layers.values())
    assert all(layer.preprocessing_settings != "" for layer in layers.values())
    assert all(layer.config_hash == view.config_hash for layer in layers.values())
    assert "Subject:" not in layers["email"].cleaned_text
    assert "> previous message" not in layers["email"].cleaned_text
    assert "<CODE_FENCE>" in layers["code_fence"].cleaned_text


def test_document_view_attaches_optional_annotation_layer_metadata_without_mutation() -> None:
    config = english_preprocessing_config()
    view = DocumentView.from_text("Parser and LLM metadata.", config, document_id="row-meta")
    parser_layer = AnnotationLayerMetadata(
        layer_name="parser",
        provider="fake-parser",
        model="fixture-model",
        version="1",
        language="en",
        tokenizer_settings="regex-v1",
        preprocessing_settings="fixture settings",
        config_hash=view.config_hash,
        diagnostics=("fixture",),
    )
    llm_layer = AnnotationLayerMetadata(
        layer_name="llm",
        provider="fake-llm",
        model="fixture-judge",
        version="1",
        language="en",
        tokenizer_settings="not_applicable",
        preprocessing_settings="fixture settings",
        config_hash=view.config_hash,
        diagnostics=("schema_valid",),
    )

    enriched = view.with_annotation_layers(parser_layer=parser_layer, llm_layer=llm_layer)

    assert view.parser_layer is None
    assert view.llm_layer is None
    assert enriched.parser_layer == parser_layer
    assert enriched.llm_layer == llm_layer
    assert enriched.raw == view.raw
