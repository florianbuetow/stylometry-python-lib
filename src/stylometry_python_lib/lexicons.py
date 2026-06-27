"""Versioned closed-class lexicon resources."""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import resources
from typing import Any, cast


@dataclass(frozen=True)
class LexiconEntry:
    """One normalized lexicon entry."""

    token: str
    groups: tuple[str, ...]
    expansion: tuple[str, ...]
    expansion_alternatives: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class VersionedLexicon:
    """Project-owned lexicon with required provenance metadata."""

    name: str
    lexicon_id: str
    family: str
    language: str
    version: str
    source: str
    license_note: str
    normalization: str
    entries: tuple[LexiconEntry, ...]

    def tokens(self) -> tuple[str, ...]:
        """Return tokens in deterministic resource order."""
        return tuple(entry.token for entry in self.entries)

    def groups(self) -> tuple[str, ...]:
        """Return all group ids in deterministic first-seen order."""
        seen: set[str] = set()
        groups: list[str] = []
        for entry in self.entries:
            for group in entry.groups:
                if group not in seen:
                    seen.add(group)
                    groups.append(group)
        return tuple(groups)


@dataclass(frozen=True)
class SpellingVariantForm:
    """One side of a spelling-variant pair."""

    label: str
    token: str


@dataclass(frozen=True)
class SpellingVariantPair:
    """One pair of spelling variants with stable labels."""

    pair_id: str
    variant_a: SpellingVariantForm
    variant_b: SpellingVariantForm
    groups: tuple[str, ...]


@dataclass(frozen=True)
class VersionedSpellingVariantResource:
    """Project-owned spelling variant pairs with required provenance metadata."""

    name: str
    lexicon_id: str
    family: str
    language: str
    version: str
    source: str
    license_note: str
    normalization: str
    pairs: tuple[SpellingVariantPair, ...]

    def groups(self) -> tuple[str, ...]:
        """Return all group ids in deterministic first-seen order."""
        seen: set[str] = set()
        groups: list[str] = []
        for pair in self.pairs:
            for group in pair.groups:
                if group not in seen:
                    seen.add(group)
                    groups.append(group)
        return tuple(groups)


@dataclass(frozen=True)
class SyllableCountEntry:
    """One dictionary syllable count entry."""

    token: str
    syllables: int


@dataclass(frozen=True)
class PronunciationEntry:
    """One dictionary pronunciation entry."""

    token: str
    phonemes: tuple[str, ...]


@dataclass(frozen=True)
class VersionedPronunciationResource:
    """Project-owned pronunciations with required provenance metadata."""

    name: str
    lexicon_id: str
    family: str
    language: str
    version: str
    source: str
    license_note: str
    normalization: str
    entries: tuple[PronunciationEntry, ...]

    def phonemes_by_token(self) -> dict[str, tuple[str, ...]]:
        """Return phoneme sequences keyed by normalized token."""
        return {entry.token: entry.phonemes for entry in self.entries}


@dataclass(frozen=True)
class VersionedSyllableDictionary:
    """Project-owned syllable counts with required provenance metadata."""

    name: str
    lexicon_id: str
    family: str
    language: str
    version: str
    source: str
    license_note: str
    normalization: str
    entries: tuple[SyllableCountEntry, ...]

    def tokens(self) -> tuple[str, ...]:
        """Return tokens in deterministic resource order."""
        return tuple(entry.token for entry in self.entries)

    def count_by_token(self) -> dict[str, int]:
        """Return dictionary counts keyed by normalized token."""
        return {entry.token: entry.syllables for entry in self.entries}


@dataclass(frozen=True)
class FrequencyBandEntry:
    """One reference-frequency entry."""

    token: str
    band: str
    rank: int
    frequency_per_million: float


@dataclass(frozen=True)
class VersionedFrequencyBandResource:
    """Project-owned reference frequency bands with required provenance metadata."""

    name: str
    lexicon_id: str
    family: str
    language: str
    version: str
    source: str
    license_note: str
    normalization: str
    entries: tuple[FrequencyBandEntry, ...]

    def tokens(self) -> tuple[str, ...]:
        """Return tokens in deterministic resource order."""
        return tuple(entry.token for entry in self.entries)

    def bands(self) -> tuple[str, ...]:
        """Return frequency bands in deterministic first-seen order."""
        seen: set[str] = set()
        bands: list[str] = []
        for entry in self.entries:
            if entry.band not in seen:
                seen.add(entry.band)
                bands.append(entry.band)
        return tuple(bands)

    def entry_by_token(self) -> dict[str, FrequencyBandEntry]:
        """Return frequency entries keyed by normalized token."""
        return {entry.token: entry for entry in self.entries}


def load_lexicon(name: str) -> VersionedLexicon:
    """Load and validate one built-in lexicon by resource name."""
    if not _is_valid_resource_name(name):
        raise ValueError(f"Invalid lexicon resource name: {name}")
    resource = resources.files("stylometry_python_lib").joinpath("data", "lexicons", f"{name}.json")
    if not resource.is_file():
        raise FileNotFoundError(f"Missing lexicon resource: {name}")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Lexicon resource {name} must contain a JSON object")
    return _parse_lexicon(name, cast(dict[str, Any], payload))


def load_spelling_variants(name: str) -> VersionedSpellingVariantResource:
    """Load and validate one built-in spelling-variant resource by name."""
    if not _is_valid_resource_name(name):
        raise ValueError(f"Invalid spelling variant resource name: {name}")
    resource = resources.files("stylometry_python_lib").joinpath("data", "lexicons", f"{name}.json")
    if not resource.is_file():
        raise FileNotFoundError(f"Missing spelling variant resource: {name}")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Spelling variant resource {name} must contain a JSON object")
    return _parse_spelling_variants(name, cast(dict[str, Any], payload))


def load_syllable_dictionary(name: str) -> VersionedSyllableDictionary:
    """Load and validate one built-in syllable-count dictionary by name."""
    if not _is_valid_resource_name(name):
        raise ValueError(f"Invalid syllable dictionary resource name: {name}")
    resource = resources.files("stylometry_python_lib").joinpath("data", "lexicons", f"{name}.json")
    if not resource.is_file():
        raise FileNotFoundError(f"Missing syllable dictionary resource: {name}")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Syllable dictionary resource {name} must contain a JSON object")
    return _parse_syllable_dictionary(name, cast(dict[str, Any], payload))


def load_pronunciations(name: str) -> VersionedPronunciationResource:
    """Load and validate one built-in pronunciation dictionary by name."""
    if not _is_valid_resource_name(name):
        raise ValueError(f"Invalid pronunciation resource name: {name}")
    resource = resources.files("stylometry_python_lib").joinpath("data", "lexicons", f"{name}.json")
    if not resource.is_file():
        raise FileNotFoundError(f"Missing pronunciation resource: {name}")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Pronunciation resource {name} must contain a JSON object")
    return _parse_pronunciations(name, cast(dict[str, Any], payload))


def load_frequency_bands(name: str) -> VersionedFrequencyBandResource:
    """Load and validate one built-in reference-frequency resource by name."""
    if not _is_valid_resource_name(name):
        raise ValueError(f"Invalid frequency-band resource name: {name}")
    resource = resources.files("stylometry_python_lib").joinpath("data", "lexicons", f"{name}.json")
    if not resource.is_file():
        raise FileNotFoundError(f"Missing frequency-band resource: {name}")
    payload = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Frequency-band resource {name} must contain a JSON object")
    return _parse_frequency_bands(name, cast(dict[str, Any], payload))


def _parse_lexicon(name: str, payload: dict[str, Any]) -> VersionedLexicon:
    required_metadata = ("lexicon_id", "family", "language", "version", "source", "license_note", "normalization")
    metadata = {field_name: _required_string(payload, field_name, name) for field_name in required_metadata}
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Lexicon resource {name} requires a non-empty entries list")
    entries_payload = cast(list[object], raw_entries)
    if len(entries_payload) == 0:
        raise ValueError(f"Lexicon resource {name} requires a non-empty entries list")
    entries = tuple(_parse_entry(name, raw_entry) for raw_entry in entries_payload)
    _validate_unique_tokens(name, entries)
    return VersionedLexicon(
        name=name,
        lexicon_id=metadata["lexicon_id"],
        family=metadata["family"],
        language=metadata["language"],
        version=metadata["version"],
        source=metadata["source"],
        license_note=metadata["license_note"],
        normalization=metadata["normalization"],
        entries=entries,
    )


def _parse_spelling_variants(name: str, payload: dict[str, Any]) -> VersionedSpellingVariantResource:
    required_metadata = ("lexicon_id", "family", "language", "version", "source", "license_note", "normalization")
    metadata = {field_name: _required_string(payload, field_name, name) for field_name in required_metadata}
    raw_pairs = payload.get("pairs")
    if not isinstance(raw_pairs, list):
        raise ValueError(f"Spelling variant resource {name} requires a non-empty pairs list")
    pairs_payload = cast(list[object], raw_pairs)
    if len(pairs_payload) == 0:
        raise ValueError(f"Spelling variant resource {name} requires a non-empty pairs list")
    pairs = tuple(_parse_spelling_pair(name, raw_pair) for raw_pair in pairs_payload)
    _validate_unique_spelling_pairs(name, pairs)
    return VersionedSpellingVariantResource(
        name=name,
        lexicon_id=metadata["lexicon_id"],
        family=metadata["family"],
        language=metadata["language"],
        version=metadata["version"],
        source=metadata["source"],
        license_note=metadata["license_note"],
        normalization=metadata["normalization"],
        pairs=pairs,
    )


def _parse_syllable_dictionary(name: str, payload: dict[str, Any]) -> VersionedSyllableDictionary:
    required_metadata = ("lexicon_id", "family", "language", "version", "source", "license_note", "normalization")
    metadata = {field_name: _required_string(payload, field_name, name) for field_name in required_metadata}
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Syllable dictionary resource {name} requires a non-empty entries list")
    entries_payload = cast(list[object], raw_entries)
    if len(entries_payload) == 0:
        raise ValueError(f"Syllable dictionary resource {name} requires a non-empty entries list")
    entries = tuple(_parse_syllable_entry(name, raw_entry) for raw_entry in entries_payload)
    _validate_unique_syllable_tokens(name, entries)
    return VersionedSyllableDictionary(
        name=name,
        lexicon_id=metadata["lexicon_id"],
        family=metadata["family"],
        language=metadata["language"],
        version=metadata["version"],
        source=metadata["source"],
        license_note=metadata["license_note"],
        normalization=metadata["normalization"],
        entries=entries,
    )


def _parse_frequency_bands(name: str, payload: dict[str, Any]) -> VersionedFrequencyBandResource:
    required_metadata = ("lexicon_id", "family", "language", "version", "source", "license_note", "normalization")
    metadata = {field_name: _required_string(payload, field_name, name) for field_name in required_metadata}
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Frequency-band resource {name} requires a non-empty entries list")
    entries_payload = cast(list[object], raw_entries)
    if len(entries_payload) == 0:
        raise ValueError(f"Frequency-band resource {name} requires a non-empty entries list")
    entries = tuple(_parse_frequency_band_entry(name, raw_entry) for raw_entry in entries_payload)
    _validate_unique_frequency_tokens(name, entries)
    return VersionedFrequencyBandResource(
        name=name,
        lexicon_id=metadata["lexicon_id"],
        family=metadata["family"],
        language=metadata["language"],
        version=metadata["version"],
        source=metadata["source"],
        license_note=metadata["license_note"],
        normalization=metadata["normalization"],
        entries=entries,
    )


def _parse_entry(name: str, raw_entry: object) -> LexiconEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError(f"Lexicon resource {name} has a non-object entry")
    entry = cast(dict[str, Any], raw_entry)
    token = _required_string(entry, "token", name)
    raw_groups = entry.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError(f"Lexicon resource {name} entry {token} requires non-empty groups")
    groups_payload = cast(list[object], raw_groups)
    if len(groups_payload) == 0:
        raise ValueError(f"Lexicon resource {name} entry {token} requires non-empty groups")
    groups = tuple(_string_sequence(groups_payload, f"{name}:{token}:groups"))
    expansion: tuple[str, ...] = ()
    if "expansion" in entry:
        raw_expansion = entry["expansion"]
        if not isinstance(raw_expansion, list):
            raise ValueError(f"Lexicon resource {name} entry {token} expansion must be a list")
        expansion = tuple(_string_sequence(cast(list[object], raw_expansion), f"{name}:{token}:expansion"))
    expansion_alternatives = _parse_expansion_alternatives(name, token, entry, expansion)
    return LexiconEntry(token=token, groups=groups, expansion=expansion, expansion_alternatives=expansion_alternatives)


def _parse_expansion_alternatives(name: str, token: str, entry: dict[str, Any], expansion: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    if "expansion_alternatives" not in entry:
        if len(expansion) == 0:
            return ()
        return (expansion,)
    raw_alternatives = entry["expansion_alternatives"]
    if not isinstance(raw_alternatives, list):
        raise ValueError(f"Lexicon resource {name} entry {token} expansion_alternatives must be a list")
    alternatives_payload = cast(list[object], raw_alternatives)
    if len(alternatives_payload) == 0:
        raise ValueError(f"Lexicon resource {name} entry {token} expansion_alternatives must be non-empty")
    alternatives: list[tuple[str, ...]] = []
    for alternative_index, raw_alternative in enumerate(alternatives_payload):
        if not isinstance(raw_alternative, list):
            raise ValueError(f"Lexicon resource {name} entry {token} expansion_alternatives[{alternative_index}] must be a list")
        alternative = tuple(
            _string_sequence(cast(list[object], raw_alternative), f"{name}:{token}:expansion_alternatives:{alternative_index}")
        )
        if len(alternative) == 0:
            raise ValueError(f"Lexicon resource {name} entry {token} expansion_alternatives[{alternative_index}] must be non-empty")
        alternatives.append(alternative)
    if len(expansion) > 0 and expansion not in alternatives:
        raise ValueError(f"Lexicon resource {name} entry {token} expansion_alternatives must include default expansion")
    return tuple(alternatives)


def _parse_spelling_pair(name: str, raw_pair: object) -> SpellingVariantPair:
    if not isinstance(raw_pair, dict):
        raise ValueError(f"Spelling variant resource {name} has a non-object pair")
    pair = cast(dict[str, Any], raw_pair)
    pair_id = _required_string(pair, "pair_id", name)
    variant_a = _parse_spelling_form(name, pair_id, pair, "variant_a")
    variant_b = _parse_spelling_form(name, pair_id, pair, "variant_b")
    raw_groups = pair.get("groups")
    if not isinstance(raw_groups, list):
        raise ValueError(f"Spelling variant resource {name} pair {pair_id} requires non-empty groups")
    groups_payload = cast(list[object], raw_groups)
    if len(groups_payload) == 0:
        raise ValueError(f"Spelling variant resource {name} pair {pair_id} requires non-empty groups")
    groups = tuple(_string_sequence(groups_payload, f"{name}:{pair_id}:groups"))
    return SpellingVariantPair(pair_id=pair_id, variant_a=variant_a, variant_b=variant_b, groups=groups)


def _parse_spelling_form(name: str, pair_id: str, pair: dict[str, Any], field_name: str) -> SpellingVariantForm:
    raw_form = pair.get(field_name)
    if not isinstance(raw_form, dict):
        raise ValueError(f"Spelling variant resource {name} pair {pair_id} requires object {field_name}")
    form = cast(dict[str, Any], raw_form)
    return SpellingVariantForm(
        label=_required_string(form, "label", name),
        token=_required_string(form, "token", name),
    )


def _parse_pronunciations(name: str, payload: dict[str, Any]) -> VersionedPronunciationResource:
    required_metadata = ("lexicon_id", "family", "language", "version", "source", "license_note", "normalization")
    metadata = {field_name: _required_string(payload, field_name, name) for field_name in required_metadata}
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list):
        raise ValueError(f"Pronunciation resource {name} requires a non-empty entries list")
    entries_payload = cast(list[object], raw_entries)
    if len(entries_payload) == 0:
        raise ValueError(f"Pronunciation resource {name} requires a non-empty entries list")
    entries = tuple(_parse_pronunciation_entry(name, raw_entry) for raw_entry in entries_payload)
    _validate_unique_pronunciation_tokens(name, entries)
    return VersionedPronunciationResource(
        name=name,
        lexicon_id=metadata["lexicon_id"],
        family=metadata["family"],
        language=metadata["language"],
        version=metadata["version"],
        source=metadata["source"],
        license_note=metadata["license_note"],
        normalization=metadata["normalization"],
        entries=entries,
    )


def _parse_pronunciation_entry(name: str, raw_entry: object) -> PronunciationEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError(f"Pronunciation resource {name} has a non-object entry")
    entry = cast(dict[str, Any], raw_entry)
    token = _required_string(entry, "token", name)
    if "phonemes" not in entry:
        raise ValueError(f"Pronunciation resource {name} entry {token} requires a non-empty phonemes list")
    raw_phonemes = entry["phonemes"]
    if not isinstance(raw_phonemes, list):
        raise ValueError(f"Pronunciation resource {name} entry {token} requires a phonemes list")
    phonemes = tuple(_string_sequence(cast(list[object], raw_phonemes), f"{name}:{token}:phonemes"))
    if len(phonemes) == 0:
        raise ValueError(f"Pronunciation resource {name} entry {token} requires a non-empty phonemes list")
    return PronunciationEntry(token=token, phonemes=phonemes)


def _validate_unique_pronunciation_tokens(name: str, entries: tuple[PronunciationEntry, ...]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.token in seen:
            raise ValueError(f"Pronunciation resource {name} has duplicate token: {entry.token}")
        seen.add(entry.token)


def _parse_syllable_entry(name: str, raw_entry: object) -> SyllableCountEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError(f"Syllable dictionary resource {name} has a non-object entry")
    entry = cast(dict[str, Any], raw_entry)
    token = _required_string(entry, "token", name)
    syllables = entry.get("syllables")
    if not isinstance(syllables, int) or syllables <= 0:
        raise ValueError(f"Syllable dictionary resource {name} entry {token} requires positive integer syllables")
    return SyllableCountEntry(token=token, syllables=syllables)


def _parse_frequency_band_entry(name: str, raw_entry: object) -> FrequencyBandEntry:
    if not isinstance(raw_entry, dict):
        raise ValueError(f"Frequency-band resource {name} has a non-object entry")
    entry = cast(dict[str, Any], raw_entry)
    token = _required_string(entry, "token", name)
    band = _required_string(entry, "band", name)
    rank = entry.get("rank")
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"Frequency-band resource {name} entry {token} requires positive integer rank")
    frequency_per_million = entry.get("frequency_per_million")
    if not isinstance(frequency_per_million, int | float) or frequency_per_million < 0.0:
        raise ValueError(f"Frequency-band resource {name} entry {token} requires non-negative numeric frequency_per_million")
    return FrequencyBandEntry(
        token=token,
        band=band,
        rank=rank,
        frequency_per_million=float(frequency_per_million),
    )


def _required_string(payload: dict[str, Any], field_name: str, lexicon_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str) or value == "":
        raise ValueError(f"Lexicon resource {lexicon_name} requires non-empty string field {field_name}")
    return value


def _string_sequence(values: list[object], field_name: str) -> tuple[str, ...]:
    strings: list[str] = []
    for value in values:
        if not isinstance(value, str) or value == "":
            raise ValueError(f"Lexicon field {field_name} must contain only non-empty strings")
        strings.append(value)
    return tuple(strings)


def _validate_unique_tokens(name: str, entries: tuple[LexiconEntry, ...]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.token in seen:
            raise ValueError(f"Lexicon resource {name} has duplicate token: {entry.token}")
        seen.add(entry.token)


def _validate_unique_spelling_pairs(name: str, pairs: tuple[SpellingVariantPair, ...]) -> None:
    seen_pair_ids: set[str] = set()
    for pair in pairs:
        if pair.pair_id in seen_pair_ids:
            raise ValueError(f"Spelling variant resource {name} has duplicate pair: {pair.pair_id}")
        seen_pair_ids.add(pair.pair_id)
        if pair.variant_a.label == pair.variant_b.label:
            raise ValueError(f"Spelling variant resource {name} pair {pair.pair_id} has duplicate labels")
        if pair.variant_a.token == pair.variant_b.token:
            raise ValueError(f"Spelling variant resource {name} pair {pair.pair_id} has duplicate tokens")


def _validate_unique_syllable_tokens(name: str, entries: tuple[SyllableCountEntry, ...]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.token in seen:
            raise ValueError(f"Syllable dictionary resource {name} has duplicate token: {entry.token}")
        seen.add(entry.token)


def _validate_unique_frequency_tokens(name: str, entries: tuple[FrequencyBandEntry, ...]) -> None:
    seen: set[str] = set()
    for entry in entries:
        if entry.token in seen:
            raise ValueError(f"Frequency-band resource {name} has duplicate token: {entry.token}")
        seen.add(entry.token)


def _is_valid_resource_name(name: str) -> bool:
    return name.replace("_", "").isalnum()
