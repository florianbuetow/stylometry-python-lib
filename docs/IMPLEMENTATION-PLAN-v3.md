# Stylometry v3 LLM Implementation Plan

This plan supersedes the LLM sections of `docs/IMPLEMENTATION-PLAN-v2.md` while preserving the original LLM feature scope from `docs/RESEARCH.md`, `docs/RESEARCH-QUESTIONS-v2.md`, and `docs/FEATURE-TEST-PLAN-v2.md`.

The v2 framework already has registry coverage and offline fake-provider contracts for all built-in LLM rows. v3 completes the real LLM feature layer by adding configured LM Studio/OpenAI-compatible execution, prompt/schema validation, diagnostics, pairwise estimators, embeddings, and configured integration tests.

## Scope Rules

- Include every LLM feature family from the original implementation plans and research.
- Preserve the deterministic and parser-backed core; LLM features remain optional feature layers, not replacements for deterministic measurements.
- Default offline tests continue to use fake providers and must never require a running LLM.
- Configured LLM tests are required whenever `config.yaml` contains a valid `llm` block and the endpoint is reachable.
- Tests that need a real LLM must be gated by configuration, not by ad hoc environment-variable opt-in.

## Config Contract

The project uses root-level `config.yaml` for configured LLM tests.

Required `llm` fields:

- `model`
- `api_base`
- `api_key`
- `context_window`
- `max_tokens`
- `temperature`
- `context_window_threshold`
- `max_retries`
- `retry_delay`
- `request_timeout_seconds`

The configured test target is LM Studio's OpenAI-compatible local endpoint with a qwen model:

```yaml
llm:
  model: qwen/qwen3.6-35b-a3b
  api_base: http://127.0.0.1:1234/v1
  api_key: lm-studio
  context_window: 32786
  max_tokens: 32768
  temperature: 0.3
  context_window_threshold: 90
  max_retries: 3
  retry_delay: 2.0
  request_timeout_seconds: 1800.0
```

Behavior:

- Missing `config.yaml` means configured LLM tests are skipped with a clear `llm config missing` reason.
- Missing `llm` keys means configured LLM tests fail fast with a validation error.
- An unreachable configured endpoint means configured LLM tests fail fast with a provider connectivity error.
- Default CI may run without `config.yaml`; configured LLM CI must provide `config.yaml` and run the same integration tests.

## LLM Feature Catalog

All of these feature families must have fake-provider tests, configured LLM tests, schema validation, provenance, diagnostics, and stable feature names.

1. `text::llm::tone`
2. `text::llm::register`
3. `text::llm::persona`
4. `text::llm::narrative_perspective`
5. `text::llm::sentence_intent`
6. `text::llm::discourse_function`
7. `text::llm::rhetorical_structure`
8. `text::llm::argumentation_style`
9. `text::llm::cohesion_judgment`
10. `text::llm::style_topic_separation`
11. `text::llm::stylistic_similarity`
12. `text::llm::pairwise_style_comparison`
13. `text::llm::style_difference_explanation`
14. `text::llm::style_transfer_descriptor`
15. `text::llm::authorial_habit_summary`
16. `text::llm::prompt_derived_vector`
17. `text::llm::embedding`
18. `text::llm::style_tuned_embedding`
19. `text::llm::same_author_prediction`
20. `text::llm::generated_feature_extraction`

## Milestone 1: LLM Configuration And Client Protocol

Implement:

- `config.yaml` loader with strict validation and no fallback defaults.
- Typed LLM config object.
- Provider-neutral request/response protocol.
- OpenAI-compatible HTTP adapter for LM Studio.
- Retry handling using the configured retry values.
- Request timeout using the configured timeout value.
- Context-window checks using `context_window` and `context_window_threshold`.
- Provider errors mapped to explicit diagnostics.

Acceptance tests:

- Valid `config.yaml` loads exactly.
- Missing file skips configured LLM tests.
- Missing key fails validation.
- Invalid numeric bounds fail validation.
- LM Studio adapter builds OpenAI-compatible request payloads.
- Fake transport tests retries, timeout handling, provider errors, and malformed responses.

## Milestone 2: Prompt Templates And Schema Definitions

Implement:

- Versioned prompt templates for all 20 LLM feature families.
- Prompt renderer that records prompt version and rendered prompt hash.
- JSON schema for each feature family.
- Closed-label taxonomies for label-based features.
- Numeric projection rules for labels, ratings, probabilities, pair scores, and vectors.
- Structured sidecars for non-scalar details.

Acceptance tests:

- Every catalog feature has a prompt template.
- Every catalog feature has a schema.
- Prompt rendering is deterministic.
- Prompt provenance includes prompt version and schema id.
- Closed-label schemas reject unknown labels.
- Numeric projection is stable and documented in metadata.

## Milestone 3: Row-Wise LLM Annotation Transformer

Implement row-wise configured-provider execution for:

- tone;
- register;
- persona;
- narrative perspective;
- sentence intent;
- discourse function;
- rhetorical structure;
- argumentation style;
- cohesion judgment;
- style-transfer descriptor;
- authorial habit summary;
- generated feature extraction.

Required behavior:

- Preserve row identity.
- Never mutate input.
- Return stable feature-name order.
- Emit structured sidecars with raw response reference, validation status, prompt metadata, schema metadata, diagnostics, and provider metadata.
- Support serialization of fitted transformer state.
- Fail fast on missing configured provider.

Acceptance tests:

- Fake provider remains always available in default tests.
- Configured LM Studio tests run when `config.yaml` is valid and endpoint is reachable.
- Valid configured responses project into numeric output or sidecars.
- Invalid configured responses produce explicit diagnostics.
- pandas row order is preserved.
- Pickle round trip preserves fitted metadata.

## Milestone 4: Pairwise LLM Estimator

Implement a separate pairwise API for:

- style/topic separation;
- stylistic similarity;
- pairwise style comparison;
- stylistic difference explanation;
- same-author prediction.

Required behavior:

- Accept explicit `(doc_i, doc_j)` pairs.
- Preserve pair identity in outputs and sidecars.
- Do not force pairwise features into normal row-wise transformers.
- Support reversed-pair audits for pair-order sensitivity.
- Record prompt order, input document ids, and pair id.

Acceptance tests:

- Pair ids are preserved.
- Reversed-pair audit records both directions.
- Same configured model and prompt version are recorded for every pair.
- Invalid pair ids fail fast.
- Configured LM Studio tests run under valid `config.yaml`.

## Milestone 5: Embeddings And Vector Outputs

Implement:

- OpenAI-compatible embedding adapter if LM Studio exposes embeddings for the configured model.
- User-provided embedding matrix acceptance when provider embeddings are unavailable.
- `text::llm::embedding` metadata marked `topic_sensitive`.
- `text::llm::style_tuned_embedding` metadata marked according to training/evaluation provenance.
- Vector shape validation.
- Vector sidecars with provider/model/config metadata.

Acceptance tests:

- Fake embedding provider emits deterministic vectors.
- Configured embedding test runs only when the configured endpoint supports embeddings.
- User-provided embeddings preserve row identity.
- Vector width is stable after fit.
- Shape mismatches fail fast.

## Milestone 6: Diagnostics And Undefined Results

Implement explicit diagnostics for:

- invalid JSON;
- schema mismatch;
- missing required field;
- refusal;
- timeout;
- provider error;
- truncation;
- context-window overflow;
- empty response;
- unsupported endpoint capability.

Acceptance tests:

- Each diagnostic is covered by fake transport tests.
- Diagnostics include feature name, row or pair id, provider, model, prompt version, and schema id.
- Undefined numeric values use the existing `FeatureDiagnostic` contract.
- Sidecars retain enough structured data to audit the failure without rerunning the model.

## Milestone 7: Stability Audits

Implement:

- repeated-run audit;
- prompt paraphrase audit;
- reversed pair-order audit for pairwise features;
- variance/agreement summary.

Required behavior:

- Audit run count is explicit configuration, not a hidden default.
- Audit results are sidecars or report objects.
- Regular feature extraction can run without stability audit when audit count is not configured.

Acceptance tests:

- Fake provider produces deterministic repeated-run summaries.
- Configured LM Studio audit tests run under valid `config.yaml`.
- Variance/agreement summaries preserve row or pair identity.
- Audit metadata records prompt versions and model config.

## Milestone 8: Registry And Metadata Completion

For every LLM feature:

- status remains `partial` until configured-provider tests pass for that feature family;
- metadata records provider, model, model version or resolved model id, prompt version, schema id, decoding settings, preprocessing settings, validation status, and sidecar schema;
- topic-dependence label matches the research taxonomy;
- output dtype and sidecar schema are explicit.

Acceptance tests:

- Registry coverage still reports 90 planned families.
- LLM rows expose configured-provider test status after implementation.
- No LLM row is catalog-only.
- Every LLM feature has metadata and a configured-provider test or a documented endpoint capability failure.

## Milestone 9: Test Gating

Test layers:

- Unit tests: fake providers, schema validation, prompt rendering, diagnostics, projection, pair identity.
- Configured LLM tests: run against LM Studio when `config.yaml` is valid and reachable.
- Full validation: `just run` and `just ci`.

Rules:

- Tests that do not need a real LLM must not read `config.yaml`.
- Tests that need a real LLM must read `config.yaml`.
- Configured LLM tests must not be deleted or treated as optional extras.
- In environments with valid `config.yaml`, configured LLM tests must run.
- In environments without valid `config.yaml`, configured LLM tests must skip with one explicit reason.

## Completion Checklist

The LLM feature layer is complete when:

- all 20 LLM feature families have prompt templates;
- all 20 LLM feature families have schemas;
- all 20 LLM feature families have fake-provider tests;
- all applicable row-wise families have configured LM Studio tests;
- all applicable pairwise families have configured LM Studio tests;
- embedding features either pass configured embedding tests or document an endpoint capability failure in tests;
- diagnostics cover invalid JSON, schema mismatch, missing fields, refusal, timeout, provider error, truncation, context-window overflow, empty response, and unsupported endpoint capability;
- pairwise estimators preserve pair identity;
- sidecars contain raw response references and validation metadata;
- registry metadata is complete for every LLM row;
- `just run` passes;
- `just ci` passes.
