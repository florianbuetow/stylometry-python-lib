"""Preprocessing layers for stylometry extraction."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Literal, Protocol

UnicodeNormalization = Literal["NFC", "NFD", "NFKC", "NFKD"]
SpecialTokenKind = Literal["url", "email", "hashtag", "mention", "number", "code_identifier", "emoji"]
FormatLayerName = Literal["markdown", "html", "email", "code_fence"]


class SpecialTokenPolicy(StrEnum):
    """Special-token normalization policies."""

    RAW_TEXT = "raw_text"
    CLASS_NORMALIZE = "class_normalize"


@dataclass(frozen=True)
class PreprocessingConfig:
    """Explicit preprocessing policy used to build a DocumentView."""

    unicode_normalization: UnicodeNormalization
    lowercase_tokens: bool
    preserve_punctuation: bool
    normalize_whitespace: bool
    sentence_pattern: str
    token_pattern: str
    paragraph_separator_pattern: str
    retain_urls: bool
    retain_numbers: bool
    retain_quotes: bool
    special_token_policy: SpecialTokenPolicy
    enabled_format_layers: tuple[FormatLayerName, ...]


@dataclass(frozen=True)
class SpecialToken:
    """A special token span retained from raw text."""

    kind: SpecialTokenKind
    raw: str
    normalized: str
    start: int
    end: int
    document_id: str


@dataclass(frozen=True)
class TextSpan:
    """A raw-text span belonging to one logical text layer."""

    layer: Literal["quoted", "authorial"]
    text: str
    start: int
    end: int
    document_id: str


@dataclass(frozen=True)
class FormatLayer:
    """Optional format-aware text layer and its preprocessing metadata."""

    name: FormatLayerName
    cleaned_text: str
    preprocessing_settings: str
    config_hash: str
    document_id: str
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class AnnotationLayerMetadata:
    """Optional parser or LLM annotation layer metadata."""

    layer_name: Literal["parser", "llm"]
    provider: str
    model: str
    version: str
    language: str
    tokenizer_settings: str
    preprocessing_settings: str
    config_hash: str
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class TokenizationResult:
    """Parallel token layers produced by a tokenizer."""

    tokens: tuple[str, ...]
    orthographic_tokens: tuple[str, ...]
    expanded_tokens: tuple[str, ...]
    hyphenated_tokens: tuple[str, ...]
    hyphen_split_tokens: tuple[str, ...]
    special_tokens: tuple[SpecialToken, ...]


class TokenizerProtocol(Protocol):
    """Protocol for deterministic or provider-backed tokenizers."""

    def tokenize(self, text: str, config: PreprocessingConfig, document_id: str) -> TokenizationResult:
        """Tokenize text into the DocumentView token layers."""
        ...


@dataclass(frozen=True)
class DocumentView:
    """Layered representation of one text sample."""

    document_id: str
    raw: str
    normalized: str
    tokens: tuple[str, ...]
    orthographic_tokens: tuple[str, ...]
    expanded_tokens: tuple[str, ...]
    hyphenated_tokens: tuple[str, ...]
    hyphen_split_tokens: tuple[str, ...]
    special_tokens: tuple[SpecialToken, ...]
    sentences: tuple[str, ...]
    paragraphs: tuple[str, ...]
    quoted_spans: tuple[TextSpan, ...]
    authorial_spans: tuple[TextSpan, ...]
    format_layers: tuple[FormatLayer, ...]
    parser_layer: AnnotationLayerMetadata | None
    llm_layer: AnnotationLayerMetadata | None
    config: PreprocessingConfig
    config_hash: str

    @classmethod
    def from_text(cls, text: str, config: PreprocessingConfig, document_id: str) -> DocumentView:
        """Build a DocumentView from raw text and explicit preprocessing config."""
        normalized = _normalize_text(text, config)
        tokenizer = RegexTokenizer()
        tokenization = tokenizer.tokenize(normalized, config, document_id)
        sentences = tuple(_split_sentences(normalized, config))
        paragraphs = tuple(_split_paragraphs(text, config))
        quoted_spans = _extract_quoted_spans(text, document_id)
        config_hash = _config_hash(config)
        return cls(
            document_id=document_id,
            raw=text,
            normalized=normalized,
            tokens=tokenization.tokens,
            orthographic_tokens=tokenization.orthographic_tokens,
            expanded_tokens=tokenization.expanded_tokens,
            hyphenated_tokens=tokenization.hyphenated_tokens,
            hyphen_split_tokens=tokenization.hyphen_split_tokens,
            special_tokens=tokenization.special_tokens,
            sentences=sentences,
            paragraphs=paragraphs,
            quoted_spans=quoted_spans,
            authorial_spans=_extract_authorial_spans(text, quoted_spans, document_id),
            format_layers=_build_format_layers(text, config, config_hash, document_id),
            parser_layer=None,
            llm_layer=None,
            config=config,
            config_hash=config_hash,
        )

    def token_counts(self) -> Counter[str]:
        """Return token frequency counts."""
        return Counter(self.tokens)

    def with_annotation_layers(
        self, parser_layer: AnnotationLayerMetadata | None, llm_layer: AnnotationLayerMetadata | None
    ) -> DocumentView:
        """Return a copy with optional parser and LLM annotation metadata attached."""
        return replace(self, parser_layer=parser_layer, llm_layer=llm_layer)


class RegexTokenizer:
    """Default deterministic tokenizer with parallel stylometry token layers."""

    def tokenize(self, text: str, config: PreprocessingConfig, document_id: str) -> TokenizationResult:
        """Tokenize text with explicit special-token and orthographic layers."""
        tokens = tuple(_tokenize(text, config))
        orthographic_tokens = tuple(_orthographic_tokenize(text, config))
        expanded_tokens = tuple(_expand_tokens(orthographic_tokens, config))
        hyphenated_tokens = tuple(
            token for token in orthographic_tokens if "-" in token and any(character.isalpha() for character in token)
        )
        hyphen_split_tokens = tuple(_split_hyphenated_tokens(orthographic_tokens))
        special_tokens = tuple(_extract_special_tokens(text, config, document_id))
        return TokenizationResult(
            tokens=tokens,
            orthographic_tokens=orthographic_tokens,
            expanded_tokens=expanded_tokens,
            hyphenated_tokens=hyphenated_tokens,
            hyphen_split_tokens=hyphen_split_tokens,
            special_tokens=special_tokens,
        )


def english_preprocessing_config() -> PreprocessingConfig:
    """Return the explicit English preprocessing policy used by built-in extractors."""
    return PreprocessingConfig(
        unicode_normalization="NFC",
        lowercase_tokens=True,
        preserve_punctuation=True,
        normalize_whitespace=True,
        sentence_pattern=r"(?<=[.!?])\s+",
        token_pattern=_english_token_pattern(),
        paragraph_separator_pattern=r"\n\s*\n+",
        retain_urls=True,
        retain_numbers=True,
        retain_quotes=True,
        special_token_policy=SpecialTokenPolicy.RAW_TEXT,
        enabled_format_layers=("markdown", "html", "email", "code_fence"),
    )


def _english_token_pattern() -> str:
    return r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+(?:\.\d+)?"


def _normalize_text(text: str, config: PreprocessingConfig) -> str:
    normalized = unicodedata.normalize(config.unicode_normalization, text)
    if config.normalize_whitespace:
        normalized = re.sub(r"[ \t\r\f\v]+", " ", normalized)
    return normalized


def _tokenize(text: str, config: PreprocessingConfig) -> list[str]:
    raw_tokens = re.findall(config.token_pattern, text)
    tokens: list[str] = []
    for token in raw_tokens:
        if not config.retain_numbers and any(character.isdigit() for character in token):
            continue
        if config.lowercase_tokens:
            tokens.append(token.lower())
        else:
            tokens.append(token)
    return tokens


def _orthographic_tokenize(text: str, config: PreprocessingConfig) -> list[str]:
    pattern = (
        r"https?://[^\s]+|www\.[^\s]+|[\w.+-]+@[\w-]+(?:\.[\w-]+)+|"
        r"#[A-Za-z0-9_]+|@[A-Za-z0-9_]+|"
        r"[A-Za-z]+(?:'[A-Za-z]+)?(?:-[A-Za-z]+(?:'[A-Za-z]+)?)*|"
        r"\d+(?:\.\d+)?|[A-Za-z_][A-Za-z0-9_]*|"
        r"[\U0001F300-\U0001FAFF]"
    )
    raw_tokens = re.findall(pattern, text)
    tokens: list[str] = []
    for token in raw_tokens:
        if not config.retain_numbers and _is_number_token(token):
            continue
        if not config.retain_urls and (_is_url_token(token) or _is_email_token(token)):
            continue
        normalized = _class_normalized_token(token, config)
        if config.lowercase_tokens and not normalized.startswith("<"):
            tokens.append(normalized.lower())
        else:
            tokens.append(normalized)
    return tokens


def _class_normalized_token(token: str, config: PreprocessingConfig) -> str:
    if config.special_token_policy == SpecialTokenPolicy.RAW_TEXT:
        return token
    kind = _classify_special_token(token)
    if kind is None:
        return token
    return f"<{kind.upper()}>"


def _extract_special_tokens(text: str, config: PreprocessingConfig, document_id: str) -> list[SpecialToken]:
    patterns: tuple[tuple[SpecialTokenKind, str], ...] = (
        ("url", r"https?://[^\s]+|www\.[^\s]+"),
        ("email", r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"),
        ("hashtag", r"#[A-Za-z0-9_]+"),
        ("mention", r"@[A-Za-z0-9_]+"),
        ("number", r"\b\d+(?:\.\d+)?\b"),
        ("code_identifier", r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b"),
        ("emoji", r"[\U0001F300-\U0001FAFF]"),
    )
    matches: list[tuple[int, int, SpecialTokenKind, str]] = []
    occupied: list[tuple[int, int]] = []
    for kind, pattern in patterns:
        for match in re.finditer(pattern, text):
            start, end = match.span()
            if _overlaps(start, end, occupied):
                continue
            raw = match.group(0)
            if kind == "number" and not config.retain_numbers:
                continue
            if kind in {"url", "email"} and not config.retain_urls:
                continue
            occupied.append((start, end))
            matches.append((start, end, kind, raw))
    matches.sort(key=lambda item: (item[0], item[1]))
    return [
        SpecialToken(
            kind=kind,
            raw=raw,
            normalized=raw if config.special_token_policy == SpecialTokenPolicy.RAW_TEXT else f"<{kind.upper()}>",
            start=start,
            end=end,
            document_id=document_id,
        )
        for start, end, kind, raw in matches
    ]


def _overlaps(start: int, end: int, occupied: list[tuple[int, int]]) -> bool:
    return any(start < occupied_end and end > occupied_start for occupied_start, occupied_end in occupied)


def _classify_special_token(token: str) -> SpecialTokenKind | None:
    if _is_url_token(token):
        return "url"
    if _is_email_token(token):
        return "email"
    if re.fullmatch(r"#[A-Za-z0-9_]+", token) is not None:
        return "hashtag"
    if re.fullmatch(r"@[A-Za-z0-9_]+", token) is not None:
        return "mention"
    if _is_number_token(token):
        return "number"
    if re.fullmatch(r"[\U0001F300-\U0001FAFF]", token) is not None:
        return "emoji"
    if re.fullmatch(r"(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Za-z]+_[A-Za-z0-9_]+|[A-Z][a-z]+[A-Z][A-Za-z0-9]*)", token) is not None:
        return "code_identifier"
    return None


def _is_url_token(token: str) -> bool:
    return re.fullmatch(r"https?://[^\s]+|www\.[^\s]+", token) is not None


def _is_email_token(token: str) -> bool:
    return re.fullmatch(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+", token) is not None


def _is_number_token(token: str) -> bool:
    return re.fullmatch(r"\d+(?:\.\d+)?", token) is not None


def _expand_tokens(tokens: tuple[str, ...], config: PreprocessingConfig) -> list[str]:
    expanded: list[str] = []
    for token in tokens:
        expanded.extend(_expand_one_token(token))
    if config.lowercase_tokens:
        return [token.lower() if not token.startswith("<") else token for token in expanded]
    return expanded


def _expand_one_token(token: str) -> tuple[str, ...]:
    lower = token.lower()
    contraction_map: dict[str, tuple[str, ...]] = {
        "can't": ("can", "not"),
        "cannot": ("can", "not"),
        "won't": ("will", "not"),
        "shan't": ("shall", "not"),
        "ain't": ("am", "not"),
        "let's": ("let", "us"),
    }
    if lower in contraction_map:
        return contraction_map[lower]
    suffix_map: tuple[tuple[str, tuple[str, ...]], ...] = (
        ("n't", ("not",)),
        ("'re", ("are",)),
        ("'ve", ("have",)),
        ("'ll", ("will",)),
        ("'d", ("would",)),
        ("'m", ("am",)),
        ("'s", ("is",)),
    )
    for suffix, expansion in suffix_map:
        if lower.endswith(suffix) and len(lower) > len(suffix):
            return (token[: -len(suffix)], *expansion)
    return (token,)


def _split_hyphenated_tokens(tokens: tuple[str, ...]) -> list[str]:
    split_tokens: list[str] = []
    for token in tokens:
        if "-" in token and any(character.isalpha() for character in token):
            split_tokens.extend(part for part in token.split("-") if part != "")
        else:
            split_tokens.append(token)
    return split_tokens


def _split_sentences(text: str, config: PreprocessingConfig) -> list[str]:
    stripped = text.strip()
    if stripped == "":
        return []
    sentence_candidates = re.split(config.sentence_pattern, stripped)
    sentences: list[str] = []
    for sentence in sentence_candidates:
        cleaned = sentence.strip()
        if cleaned != "":
            sentences.append(cleaned)
    return sentences


def _split_paragraphs(text: str, config: PreprocessingConfig) -> list[str]:
    stripped = text.strip()
    if stripped == "":
        return []
    paragraph_candidates = re.split(config.paragraph_separator_pattern, stripped)
    paragraphs: list[str] = []
    for paragraph in paragraph_candidates:
        cleaned = paragraph.strip()
        if cleaned != "":
            paragraphs.append(cleaned)
    return paragraphs


def _extract_quoted_spans(text: str, document_id: str) -> tuple[TextSpan, ...]:
    spans: list[TextSpan] = []
    quote_pairs = (('"', '"'), ("“", "”"))
    for open_quote, close_quote in quote_pairs:
        search_start = 0
        while True:
            start = text.find(open_quote, search_start)
            if start == -1:
                break
            end = text.find(close_quote, start + len(open_quote))
            if end == -1:
                break
            span_end = end + len(close_quote)
            spans.append(TextSpan(layer="quoted", text=text[start:span_end], start=start, end=span_end, document_id=document_id))
            search_start = span_end
    spans.sort(key=lambda span: (span.start, span.end))
    return tuple(spans)


def _extract_authorial_spans(text: str, quoted_spans: tuple[TextSpan, ...], document_id: str) -> tuple[TextSpan, ...]:
    spans: list[TextSpan] = []
    cursor = 0
    for quoted_span in quoted_spans:
        if cursor < quoted_span.start:
            spans.append(
                TextSpan(
                    layer="authorial",
                    text=text[cursor : quoted_span.start],
                    start=cursor,
                    end=quoted_span.start,
                    document_id=document_id,
                )
            )
        cursor = max(cursor, quoted_span.end)
    if cursor < len(text):
        spans.append(TextSpan(layer="authorial", text=text[cursor:], start=cursor, end=len(text), document_id=document_id))
    return tuple(span for span in spans if span.text != "")


def _build_format_layers(text: str, config: PreprocessingConfig, config_hash: str, document_id: str) -> tuple[FormatLayer, ...]:
    return tuple(_build_format_layer(text, layer_name, config, config_hash, document_id) for layer_name in config.enabled_format_layers)


def _build_format_layer(
    text: str, layer_name: FormatLayerName, config: PreprocessingConfig, config_hash: str, document_id: str
) -> FormatLayer:
    settings = f"unicode={config.unicode_normalization}; normalize_whitespace={config.normalize_whitespace}; layer={layer_name}"
    if layer_name == "markdown":
        cleaned = _clean_markdown(text)
        diagnostics = ("markdown_rules_regex_v1",)
    elif layer_name == "html":
        cleaned = re.sub(r"<[^>]+>", " ", text)
        diagnostics = ("html_tag_regex_v1",)
    elif layer_name == "email":
        cleaned = _clean_email(text)
        diagnostics = ("email_quote_header_regex_v1",)
    elif layer_name == "code_fence":
        cleaned = re.sub(r"```.*?```", "<CODE_FENCE>", text, flags=re.DOTALL)
        diagnostics = ("code_fence_regex_v1",)
    else:
        raise ValueError(f"Unsupported format layer: {layer_name}")
    return FormatLayer(
        name=layer_name,
        cleaned_text=_normalize_text(cleaned, config),
        preprocessing_settings=settings,
        config_hash=config_hash,
        document_id=document_id,
        diagnostics=diagnostics,
    )


def _clean_markdown(text: str) -> str:
    cleaned_lines: list[str] = []
    for line in text.splitlines():
        without_heading = re.sub(r"^\s{0,3}#{1,6}\s+", "", line)
        without_bullet = re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", without_heading)
        cleaned_lines.append(without_bullet)
    return "\n".join(cleaned_lines)


def _clean_email(text: str) -> str:
    cleaned_lines = [
        line
        for line in text.splitlines()
        if re.match(r"^\s*>", line) is None and re.match(r"^\s*(from|to|subject|date):", line, flags=re.IGNORECASE) is None
    ]
    return "\n".join(cleaned_lines)


def _config_hash(config: PreprocessingConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
