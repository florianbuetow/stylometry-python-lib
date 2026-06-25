# stylometry-python-lib

A Python CLI application

## Extractable Features

The library can extract **90 feature families**, emitting roughly **1,500 numeric
feature columns** plus **55 structured sidecar annotation types** for a single
document. Features are organized into three buckets by how they are computed:

| Bucket | Families | Dependencies | What it provides |
| --- | --- | --- | --- |
| **Deterministic** | 41 | core install (numpy / pandas / scikit-learn / scipy) | Reproducible counting/statistics over characters, words, punctuation, and layout. Runs out of the box. |
| **Other (non-LLM)** | 29 | core, plus optional spaCy for syntax / parser-backed families | Parser-backed syntactic features and evaluation/analysis utilities. |
| **LLM** | 20 | an LLM provider (a deterministic fake provider ships for testing) | Model-judged stylistic descriptors, embeddings, and pairwise comparisons. |

The fastest way to extract every deterministic feature at once is
`default_deterministic_extractor`:

```python
import pandas as pd
from stylometry_python_lib import default_deterministic_extractor

documents = pd.DataFrame({"text": ["Your document text goes here."]})
extractor = default_deterministic_extractor(text_column="text", output="pandas")
features = extractor.fit_transform(documents, None)  # one row of features per document
```

See `scripts/extract_all_features.py` for a complete, runnable example that reads a
text file and writes its features to CSV.

### Deterministic features (core install)

All 41 families below run with the core dependencies. Counts are the number of
numeric columns emitted; families marked *vocabulary-sized* emit a sparse/dense
block whose width depends on the fitted vocabulary.

| Feature family | Output columns | Description |
| --- | --- | --- |
| Type-token ratio | `text::lexical_richness::ttr` | Vocabulary diversity (types ÷ tokens). |
| TTR variants | `text::lexical_richness::*` | Root TTR/CTTR, Herdan's C, MSTTR, MATTR and related diversity indices. |
| Vocabulary size | `text::counts::type_count` | Number of distinct word types. |
| Guiraud's R | `text::lexical_richness::guiraud_r` | Guiraud lexical richness index. |
| Honoré's R | `text::lexical_richness::honore_r` | Hapax-based richness statistic. |
| Yule's K | `text::lexical_richness::yules_k` | Vocabulary concentration / repetition. |
| Sichel / Simpson / Rényi | `text::lexical_richness::{sichel_s,simpson_d,renyi_entropy_alpha_2}` | Additional richness and entropy measures. |
| Hapax legomena | `text::lexical_richness_spectrum::hapax::*` | Counts and ratios of once-occurring words. |
| Frequency spectrum | `text::lexical_richness_spectrum::*` | Hapax/dis-legomena frequency-spectrum bins. |
| Word-length statistics | `text::length::word_*` | Mean/median/std and shape of word lengths. |
| Syllable distribution | `text::length::syllables_per_word_*` | Syllables-per-word distribution. |
| Sentence-length statistics | `text::length::sentence_tokens_*` | Sentence length (in tokens) distribution. |
| Paragraph-length statistics | `text::length::paragraph_tokens_*` | Paragraph length distribution. |
| Readability formulas | `text::readability::*` | Explicit readability indices (Flesch and others). |
| Function-word frequencies | `text::function_word*` | Frequencies of common function words (107 columns). |
| Function-word n-grams | `text::function_word_ngram::*` | Function-word n-gram block (*vocabulary-sized*). |
| Stopword ratios | `text::stopword::*` | Stopword usage ratios. |
| Pronoun usage | `text::pronoun::*` | Pronoun frequencies by class. |
| Person ratios | `text::stance::{first,second,third}_person_ratio` | First/second/third-person pronoun ratios. |
| Modal verb usage | `text::modal::*` | Modal verb frequencies. |
| Auxiliary verb usage | `text::auxiliary::*` | Auxiliary verb frequencies. |
| Contraction usage | `text::contraction::*` | Contraction frequencies and expansion sidecar. |
| Lexical density | `text::lexical_density::*` | Content-word vs. function-word density. |
| Lexical sophistication | `text::lexical_sophistication_profile::*` | Word distribution across frequency bands. |
| Most frequent words | `text::most_frequent_words::*` | Top word frequencies (*vocabulary-sized*). |
| Punctuation frequencies | `text::punctuation_profile::*` | Per-mark punctuation frequencies (110 columns). |
| Punctuation sequences | `text::punctuation_ngram::*` | Punctuation sequence n-grams (*vocabulary-sized*). |
| Letter / character frequencies | `text::orthography_profile::*` | Letter and character frequencies (194 columns). |
| Capitalization habits | `text::capitalization_profile::*` | Casing patterns (all-caps, title-case, etc.). |
| Hyphenation patterns | `text::hyphenation_profile::*` | Hyphenation and compounding habits. |
| Spelling variants | `text::spelling_variant_profile::*` | US/UK and other spelling-variant usage. |
| Abbreviation & acronym patterns | `text::abbreviation_acronym_profile::*` | Abbreviation and acronym detection profile. |
| Special tokens | `text::special_token_profile::*` | URLs, emails, numbers, emoji and other special tokens. |
| Discourse markers | `text::discourse_marker_profile::*` | Discourse-marker frequencies. |
| Transition phrases | `text::transition_phrase_profile::*` | Transition-phrase frequencies. |
| Quote / dialogue markers | `text::quote_dialogue_profile::*` | Quotation and dialogue-dash signals. |
| Formatting / layout | `text::layout_whitespace_profile::*` | Layout markers and section-depth signals. |
| Whitespace / line-break habits | `text::whitespace::*` | Spacing and line-break patterns. |
| Character n-grams | `text::char_ngram::*` | Character n-gram block (*vocabulary-sized*). |
| Word unigrams | `text::word_ngram::*` | Word unigram block (*vocabulary-sized*). |
| Word n-grams | `text::word_ngram::*` | Word n-gram block (*vocabulary-sized*). |

### Syntactic / parser-backed features (require spaCy)

These 16 families derive from part-of-speech tags and dependency parses. They need
the optional spaCy backend (a deterministic fake provider is used in tests).

| Feature family | Output columns | Description |
| --- | --- | --- |
| POS tag frequencies | `text::syntax::pos_frequency::upos=*` | Universal part-of-speech tag frequencies. |
| POS n-grams | `text::syntax::pos_ngram::*` | Part-of-speech n-grams. |
| POS skip-grams | `text::syntax::pos_skipgram::*` | Part-of-speech skip-grams. |
| Lexical density by POS | `text::syntax::pos_lexical_density` | Content-word ratio derived from POS tags. |
| Clause counts | `text::syntax::{clause_count,t_unit_count}` | Clause and T-unit counts. |
| Coordination metrics | `text::syntax::coordination_ratio` | Coordination ratio. |
| Subordination metrics | `text::syntax::subordination_ratio` | Subordination ratio. |
| Syntactic complexity | `text::syntax::syntactic_complexity::*` | Aggregate syntactic-complexity metrics. |
| Parse-tree depth | `text::syntax::parse_depth_*` | Parse-tree depth statistics. |
| Dependency distance | `text::syntax::dependency_distance_*` | Dependency-distance and root statistics. |
| Dependency relation frequencies | `text::syntax::dependency_relation_frequency::deprel=*` | Dependency-relation label frequencies. |
| Dependency n-grams / subtrees | `text::syntax::dependency_{ngram,path,subtree,dtgram}::*` | Dependency-based n-grams, paths and subtrees. |
| Morphological features | `text::syntax::morphology_frequency::*` | Morphological feature frequencies. |
| Passive-voice frequency | `text::syntax::passive_voice_frequency` | Passive-construction frequency. |
| Named-entity density | `text::*::named_entity_density::entity_type=*` | Named-entity density by type. |
| Content masking / topic-neutral distortion | `text::content_control::*` | Topic-neutralized text and masked-token ratio. |

### Evaluation & analysis utilities (non-LLM)

These 13 families operate over feature matrices or document collections (distances,
classifiers, importance, controls) rather than emitting per-document columns. Some
need optional extras (e.g. SHAP, plotting, ML extensions).

| Feature family | Output | Description |
| --- | --- | --- |
| Burrows's Delta & distance metrics | `evaluation::{burrows_delta,cosine_distance,euclidean_distance}` | Authorship distance metrics. |
| Same-topic hard negatives | `evaluation::same_topic_hard_negative_pairs` | Hard-negative pair construction. |
| Cross-topic evaluation | `evaluation::cross_topic_holdout` | Cross-topic hold-out splits. |
| Standardization (z-scores) | `evaluation::z_score` | Feature standardization. |
| PCA dimensionality reduction | `evaluation::pca_reducer` | Principal-component reduction. |
| Clustering / PCA visualization | `evaluation::{cluster_feature_matrix,clustering_visualization}` | Clustering and visualization helpers. |
| Supervised classifiers | `evaluation::{SupervisedClassifier,classifier_report}` | Classifier training and reporting. |
| Open-world verification models | `evaluation::{thresholded_distance_verification,open_world_verification}` | Authorship-verification models. |
| Permutation / SHAP importance | `evaluation::{permutation_importance_report,feature_importance}` | Feature-importance analysis. |
| Ablation studies | `evaluation::ablation_scores` | Feature-block ablation scoring. |
| Two-way ANOVA | `evaluation::two_way_effect_sizes` | Two-way ANOVA effect sizes. |
| Topic-modeling prediction controls | `evaluation::topic_prediction_*` | Topic-leakage controls. |
| Human expert validation | `evaluation::human_review_packet` | Human-review export packet. |

### LLM-judged features (require an LLM provider)

These 20 families call a language model to judge or describe style. A deterministic
fake provider ships for testing; a real provider is required for production use.

| Feature family | Output column | Description |
| --- | --- | --- |
| Tone classification | `text::llm::tone` | Tone label. |
| Register / formality | `text::llm::register` | Register and formality judgment. |
| Argumentation style | `text::llm::argumentation_style` | Argumentation-style descriptor. |
| Rhetorical structure | `text::llm::rhetorical_structure` | Rhetorical-structure analysis. |
| Narrative perspective | `text::llm::narrative_perspective` | Narrative point-of-view. |
| Discourse function labels | `text::llm::discourse_function` | Discourse-function labels. |
| Sentence-level intent | `text::llm::sentence_intent` | Per-sentence intent labels. |
| Coherence / cohesion judgments | `text::llm::cohesion_judgment` | Coherence and cohesion ratings. |
| Voice / persona description | `text::llm::persona` | Authorial voice / persona description. |
| Authorial habit summaries | `text::llm::authorial_habit_summary` | Summary of authorial habits. |
| Imitability / style-transfer descriptors | `text::llm::style_transfer_descriptor` | Style-transfer descriptors. |
| Generated feature extraction | `text::llm::generated_feature_extraction` | Free-form model-generated features. |
| Embeddings (style similarity) | `text::llm::embedding` | Style embedding vector. |
| Prompt-derived style vectors | `text::llm::prompt_derived_vector` | Prompt-derived style vector. |
| Style-tuned contrastive embeddings | `text::llm::style_tuned_embedding` | Contrastively tuned style embedding. |
| Stylistic similarity judgments | `text::llm::stylistic_similarity` | Pairwise similarity judgment. |
| Stylistic difference explanations | `text::llm::style_difference_explanation` | Explanation of stylistic differences. |
| Pairwise comparison features | `text::llm::pairwise_style_comparison` | Pairwise style comparison. |
| Same-author prediction | `text::llm::same_author_prediction` | Same-author probability. |
| Topic/style separation judgments | `text::llm::style_topic_separation` | Topic-vs-style separation judgment. |

## Repository Structure

```
stylometry-python-lib/
├── pyproject.toml          # Project dependencies and metadata
├── pyrightconfig.json      # Pyright LSP type checker configuration
├── .python-version         # Python version for uv/pyenv/asdf
├── .pre-commit-config.yaml # Pre-commit hooks configuration
├── .gitignore              # Git ignore patterns
├── justfile                # Task runner with build/test/validation commands
├── AGENTS.md               # AI agent development rules
├── CLAUDE.md               # Claude Code compatibility (symlink to AGENTS.md)
├── README.md               # This file
├── .cursor/                # Cursor IDE configuration
│   └── commands/          # Cursor AI commands
│       └── doc-statemachine.md  # Generate state machine diagrams
├── src/                    # Source code
│   └── stylometry_python_lib/   # Main package
│       └── __init__.py     # Package initialization
├── main.py                 # Main entry point
├── tests/                  # Test files
│   └── __init__.py         # Test package marker
├── scripts/                # Utility scripts
├── data/                   # Data files
│   ├── input/             # Input data files
│   └── output/            # Generated output files
├── config/                 # Configuration files
│   ├── semgrep/           # Semgrep static analysis rules
│   │   ├── no-default-values.yml
│   │   ├── no-sneaky-fallbacks.yml
│   │   ├── no_type_suppression.yml
│   │   ├── no-noqa.yml
│   │   └── python-constants.yml
│   └── codespell/         # Spell-check configuration
│       └── ignore.txt      # Spell-check ignore list
└── reports/                # Generated reports (not in git)
    ├── coverage/          # Code coverage reports
    ├── security/          # Security scan reports
    ├── pyright/           # Type checking reports
    └── deptry/            # Dependency reports
```

## Prerequisites

- **Python 3.12+** - Programming language
- **uv** - Python package manager ([installation guide](https://docs.astral.sh/uv/getting-started/installation/))
- **just** - Command runner ([installation guide](https://github.com/casey/just#installation))

## Setup

Initialize the project environment:

```bash
just init
```

This will:
- Create necessary directories (reports/, etc.)
- Install all dependencies via `uv sync --all-extras`

## Usage

Run the main application:

```bash
just run
```

See all available commands:

```bash
just help
```

Or simply:

```bash
just
```

## Development

### Available Commands

- `just init` - Initialize development environment
- `just run` - Run the main application
- `just destroy` - Remove virtual environment
- `just help` - Show available commands

### Code Quality

- `just code-style` - Check code style (read-only)
- `just code-format` - Auto-fix code style
- `just code-typecheck` - Run type checking (mypy)
- `just code-lspchecks` - Run strict type checking (pyright)
- `just code-security` - Run security checks (bandit)
- `just code-deptry` - Check dependency hygiene
- `just code-stats` - Generate code statistics
- `just code-spell` - Check spelling
- `just code-audit` - Scan for vulnerabilities
- `just code-semgrep` - Run custom static analysis

### Testing

- `just test` - Run unit tests
- `just test-coverage` - Run tests with coverage (80% threshold)

### CI

- `just ci` - Run all checks (silent, fail-fast)
- `just ci-verbose` - Run all validation checks (verbose)

The CI pipeline runs the following steps in order:
1. `init` - Initialize environment
2. `code-format` - Auto-format code
3. `code-style` - Verify formatting
4. `code-typecheck` - Type checking (mypy)
5. `code-security` - Security scan (bandit)
6. `code-deptry` - Dependency hygiene
7. `code-spell` - Spell checking
8. `code-semgrep` - Custom static analysis
9. `code-audit` - Vulnerability scanning
10. `test` - Unit tests
11. `code-lspchecks` - Strict type checking (pyright)

## Project Rules

See [AGENTS.md](AGENTS.md) for detailed development guidelines including:
- Python execution rules (use `uv run` exclusively)
- Git commit guidelines (no AI attribution)
- Testing requirements
- Project structure conventions

## License

<!-- Add your license here -->
