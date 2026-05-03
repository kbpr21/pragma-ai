# Changelog

All notable changes to **pragma** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.5] — 2026-05-03

Reasoning quality release. Adds task-type detection, specialised
synthesis prompts, hallucination detection, and truncation detection.
**No breaking API changes.**

### Fixed

* **Multi-question queries returned single-fragment answers.** Queries
  like "What parts discuss efficiency? Which sections talk about
  scaling? Where is BlockAttnRes justified?" returned a single
  fragment like "depth-wise softmax attention" that addressed none of
  the three sub-questions. Now the system detects multi-question
  queries, uses a specialised prompt that requires answering every
  sub-question, and increases the token budget to 1000.

* **Summary queries truncated mid-sentence.** "Summarize the paper in
  3 sentences" returned an answer that was cut off mid-sentence with
  no closing punctuation. Now the system detects truncation (incomplete
  JSON, mid-word cutoff, mid-enumeration comma) and retries with 2x
  token budget.

* **Analogy queries hallucinated with confidence 1.00.** "Relate
  AttnRes to database query optimization" produced a creative but
  entirely ungrounded analogy (facts don't mention databases) with
  confidence 1.00. Now: (1) analogy queries use a specialised prompt
  that requires grounding every claim in a fact, and (2) the
  confidence scorer checks answer-fact grounding — answers where less
  than 30% of content words appear in the facts get a −0.3 penalty.

* **Plan queries produced generic steps.** "Turn the core idea into a
  10-step implementation plan" returned generic steps like "Define the
  model depth" that could apply to any model. Now plan queries use a
  specialised prompt that requires each step to be derived from the
  facts, with inferred steps marked as `[inferred]`.

### Added

* **Task-type detection** (`_classify_task`): classifies queries as
  FACTOID, SUMMARY, PLAN, ANALOGY, or MULTI_QUESTION using keyword
  heuristics (no LLM call needed).

* **Specialised synthesis prompts**: each task type gets its own prompt
  with rules tailored to the reasoning task:
  - FACTOID: the existing 7-rule prompt
  - SUMMARY: problem→approach→result structure, no truncation
  - PLAN: each step derived from facts, inferred steps marked
  - ANALOGY: every claim grounded in a fact, speculative marks
  - MULTI_QUESTION: answer every sub-question separately

* **Truncation detection** (`_looks_truncated`): detects incomplete
  JSON, mid-word cutoffs, and mid-enumeration commas. Triggers a
  retry with 2x token budget.

* **Hallucination detection** in `_compute_confidence`: checks what
  fraction of the answer's content words appear in the fact texts.
  Answers with less than 30% grounding get a −0.3 confidence penalty.

* **Multi-question splitting** (`_split_questions`): splits queries
  with multiple question marks into individual sub-questions.

* **11 new tests** for task-type classification, question splitting,
  truncation detection, hallucination detection, and grounded answers.

### Changed

* `synthesize` now classifies the task type before selecting a prompt.
* Direct-answer fast-path only applies to FACTOID queries.
* Multi-question queries keep more facts after filtering (minimum 5).
* `max_tokens` increased to 900 for SUMMARY/PLAN, 1000 for
  MULTI_QUESTION.
* `_compute_confidence` now accepts `entity_names` parameter for
  hallucination grounding checks.

## [1.0.4] — 2026-05-03

Answer quality release. Fixes the root cause of fragment and
non-responsive answers from both the deterministic resolver and the
LLM synthesizer. **No breaking API changes.**

### Fixed

* **Resolver returned fragment answers.** The deterministic
  ``MultiHopResolver`` would return bare predicate tails like
  ``"with learned softmax attention over depth"`` as answers to
  questions like "What is the core idea behind AttnRes?". Now the
  resolver validates every answer: fragments (answers starting with
  a preposition) are rejected and the resolver falls through to the
  LLM path for a proper synthesis.

* **Resolver returned wrong answers for yes/no questions.** Questions
  like "Is AttnRes a drop-in replacement for residual connections?"
  were matched to the ``drop_in_replacement`` intent and returned the
  object value of the first matching fact — which was a fragment
  unrelated to the question. Now invalid answers are rejected.

* **Synthesizer returned fragment answers.** The LLM synthesis prompt
  was too terse (one line), causing the model to return predicate
  fragments instead of complete sentences. The prompt has been
  completely rewritten with explicit rules against fragments, F-id
  artifacts, and incomplete answers.

* **F-id artifacts in answers.** Answers sometimes contained raw
  fact IDs like ``(F1)``, ``[F2]``, ``F3``. These are now stripped
  from the answer text in post-processing.

* **Confidence always 1.00.** The confidence scorer ignored answer
  quality — even fragment and "unknown" answers got 1.0. Now:
  "unknown" → 0.0, fragment answers → −0.3 penalty, very short
  answers → −0.15 penalty, low query-answer overlap → −0.1 penalty.

* **Resolver low-confidence answers bypassed.** When the resolver
  returns a downgraded-confidence answer (fragment detected), the
  KB pipeline now falls through to the LLM synthesizer for a better
  answer instead of returning the fragment.

### Added

* **Answer post-processing pipeline** in ``AnswerSynthesizer``:
  - ``_postprocess_answer`` — strips F-id artifacts, detects
    fragments, retries with a refinement prompt
  - ``_is_fragment`` — heuristic to detect underspecified answers
    (starts with preposition, very short without query nouns)
  - ``_refine_answer`` — retry with a refinement prompt when the
    initial answer is a fragment
  - ``_FID_RE`` — regex to strip ``(F1)``, ``[F2]``, ``F3`` artifacts

* **Answer validation in ``MultiHopResolver``**:
  - ``_is_valid_answer`` — rejects empty, fragment, and
    object_filter-failing answers
  - ``_looks_like_fragment`` — detects predicate tails (starts with
    preposition, single lowercase word, etc.)
  - ``_FRAGMENT_STARTERS`` — set of preposition/conjunction words

* **Rewritten synthesis prompt** with 7 explicit rules:
  1. Synthesize, do not paraphrase
  2. Structure as problem→method→result for research questions
  3. Answer completeness — address ALL parts of the question
  4. Never return bare fragments
  5. Never include F-id labels in answer text
  6. Graceful "unknown" with reason when facts insufficient
  7. No hallucination

* **4 new tests** for fragment detection, F-id stripping,
  confidence penalty, and unknown-answer zero confidence.

### Changed

* ``AnswerSynthesizer.synthesize`` now calls ``_postprocess_answer``
  and passes ``answer`` + ``query`` to ``_compute_confidence``.
* ``_compute_confidence`` accepts ``answer`` and ``query`` parameters
  for quality-aware scoring.
* Resolver confidence downgraded by −0.3 for fragment-like answers.
* KB query pipeline falls through to LLM when resolver confidence < 0.5.

## [1.0.3] — 2026-05-02

Ingestion + reasoning quality release. Fixes the root cause of empty
extractions from research-paper PDFs and adds domain-specific intent
handlers for academic/technical documents. **No breaking API changes.**

### Fixed

* **Fact extraction returned 0 facts from most PDF pages.** The root
  cause was ``max_tokens=2000`` in the extractor — diffusion-based
  models (Mercury) consume reasoning tokens from the same budget, so
  2000 left zero room for the actual JSON output. Bumped to 6000
  with an automatic 2× retry on empty response. Verified: 466 facts
  extracted from a 21-page research paper (was 4).

* **Synthesizer returned "unknown" for most queries.** Same
  ``max_tokens`` problem: the synthesis call used 200 tokens; Mercury
  needs ~100 just for reasoning. Bumped to 600 with a 2× retry.
  Also now includes the ``context`` field in the keyword-overlap
  filter so facts with terse predicates but rich source sentences
  survive filtering.

* **Re-ingestion of zero-fact documents.** When a document was
  previously ingested but produced 0 facts (e.g. because the PDF
  loader was broken), ``pragma ingest`` would skip it as a
  "duplicate" on every subsequent attempt. Now the pipeline detects
  zero-fact documents, deletes the empty record, resets the
  preprocessor's duplicate-tracking, and re-processes from scratch.

* **PDF ingestion broken by pdfplumber 0.11+.** The ``page.tables``
  attribute was removed in pdfplumber 0.11.x; every page raised
  ``AttributeError``, killing the entire page including body text.
  Now uses ``find_tables().extract()`` with ``hasattr`` fallbacks;
  text extraction isolated from table extraction. Added PyMuPDF
  (``fitz``) fallback for LaTeX PDFs with ``pgfpat`` pattern fills.

### Added

* **Research-paper intent handlers** for the deterministic multi-hop
  resolver: ``core_idea``, ``problem_caused_by``, ``difference``,
  ``purpose``, ``method_performance``, ``drop_in_replacement``.
  These cover canonical question shapes for academic/technical
  documents and enable zero-LLM answers for queries like "What is
  the core idea behind AttnRes?".

* **``_is_short_phrase`` object filter** — accepts concise phrases
  (≤15 words) but rejects full sentences, preventing the resolver
  from returning a whole abstract as a "direct answer".

* **``_call_llm`` helper** in ``FactExtractor`` — centralised LLM
  call with automatic retry on empty response (2× budget). Used by
  both single-segment and batch extraction paths.

* **``document_has_facts`` and ``delete_document``** on
  ``SQLiteStore`` — enable the KB to detect and re-ingest
  previously-failed documents.

* **``pymupdf>=1.24``** added to ``[pdf]`` optional dependencies as
  a fallback extractor for LaTeX PDFs.

### Changed

* ``FactExtractor`` default ``max_facts_per_segment``: 30 → 50.
* ``PragmaConfig`` default ``max_facts_per_segment``: 30 → 50.
* ``FactExtractor`` default ``max_completion_tokens``: 6000 (new
  parameter; was hardcoded to 2000).
* ``FactExtractor.extract_batch`` default ``max_tokens``: 4000 →
  8000.
* KB ingest batch size: 10 → 5 segments per batch for better
  per-segment extraction quality.
* ``AnswerSynthesizer`` synthesis ``max_tokens``: 200 → 600 with
  2× retry on empty response.

> **A note on the version sequence.** 1.0.0 was the public PyPI launch.
> 1.0.1 was a same-day metadata fix (license classifier, README links).
> 1.0.1.post1 was a *post-release* under
> [PEP 440](https://peps.python.org/pep-0440/#post-releases) — the
> rules-mandated way to republish an identical public API after
> internal token-efficiency improvements that change measured
> behaviour but not the contract. PEP 440 reserves `.postN` for
> exactly this case so users do not have to chase a new minor every
> time we tighten a prompt. The `.post1` is therefore intentional, not
> a botched publish; **1.0.2 supersedes it** and is the version users
> should install going forward.

## [1.0.2.post3] — 2026-05-02

Post-release: internal bugfix, no public API change.

### Fixed

* **Re-ingestion of zero-fact documents.** When a document was
  previously ingested but produced 0 facts (e.g. because the PDF
  loader was broken), ``pragma ingest`` would skip it as a
  "duplicate" on every subsequent attempt. Now the ingest pipeline
  detects zero-fact documents, deletes the empty record, resets the
  preprocessor's duplicate-tracking, and re-processes the document
  from scratch. Added ``document_has_facts`` and ``delete_document``
  to ``SQLiteStore``.

## [1.0.2.post2] — 2026-05-02

Post-release: internal bugfix, no public API change.

### Fixed

* **PDF ingestion completely broken with pdfplumber 0.11+.** The
  ``page.tables`` attribute was removed in pdfplumber 0.11.x; every
  page raised ``AttributeError: 'Page' object has no attribute
  'tables'``, which killed the entire page (including successfully
  extracted body text). Now uses ``find_tables().extract()`` with a
  ``hasattr`` fallback for older pdfplumber versions. Table extraction
  is also isolated from text extraction so a table API error cannot
  discard the page's body text.

* **LaTeX PDFs with pattern fills (pgfpat) now fall back to PyMuPDF.**
  pdfplumber returns empty text for pages with ``pgfpat`` pattern
  fills (common in LaTeX-generated research papers). Added
  ``_pymupdf_fallback`` that tries ``fitz`` (PyMuPDF) per-page when
  pdfplumber yields nothing. PyMuPDF is now an optional dependency
  under ``[pdf]``.

## [1.0.2] — 2026-05-02

First-run UX + scale-honesty release. Adds the `pragma connect`
interactive setup wizard, a 50-document reproducible benchmark, and
fixes one default-value mistake from 1.0.1. **No breaking API
changes.**

### Added (1.0.2 — second batch)

* **`benchmarks_run/large_corpus.py`** — deterministic 50-document
  synthetic corpus (25 fictional companies + 25 founders), ~7,000
  words / ~9,000 tokens. Cross-references between docs are intentional
  so multi-hop questions cannot be answered by retrieving a single
  chunk. The corpus is regenerated byte-for-byte on every run from
  the data tables in the file, so the benchmark is **reproducible**:
  there is a unit test (`test_large_benchmark.py::test_corpus_is_deterministic`)
  that materialises the corpus twice and diffs the bytes.
* **`benchmarks_run/run_large.py`** — runs the full ingest + query
  pipeline against the 50-doc corpus and compares pragma to a BM25
  top-k vector-RAG baseline. Captures **true prompt + completion
  tokens** from Ollama's tokenizer (`prompt_eval_count` /
  `eval_count`), not pragma's word-count approximation. Reports
  per-query accuracy via substring match against ground-truth
  expectations declared inline in `large_corpus.py`. Supports
  `--no-ingest` to reuse an existing KB, `--skip-baseline` for a
  pragma-only smoke run, and `--queries N` for fast iteration.
* **`tests/unit/test_large_benchmark.py`** (6 cases) — pins the
  benchmark fixtures so future trims of the data tables fail loudly:
  determinism, document/token floors (>=50, >=5,000 tokens),
  founder-resolves-to-person cross-reference invariant, every query
  expectation actually appears in the corpus, BM25 retrieves the
  obvious document, `is_correct` semantics.

### Fixed (1.0.2 — second batch)

* **`max_subgraph_nodes` default 5 → 50.** The original system design
  targeted 50; 1.0.1 shipped 5 by mistake, which silently capped
  multi-hop recall on any non-toy KB. Multi-hop traversals that found
  6+ candidate nodes were dropping the tail. The floor is still 5
  (values below are clamped up so obviously-wrong configs do not
  produce empty subgraphs); values >= 5 are now passed through
  untouched. Documented in `pragma/config.py`,
  `tests/unit/test_config.py::test_max_subgraph_nodes_floor`, and the
  README env-var table.

* **`kb.ingest("./directory_as_string")` no longer crashes.** A type
  coercion mistake in `kb.py` raised
  `AttributeError: 'str' object has no attribute 'rglob'` whenever
  `kb.ingest` was called with a directory path passed as a string
  (the `Path`-typed call path worked fine, which is why the existing
  test missed it). Now both paths are coerced to `Path` before
  `_discover_files`. Regression test added in
  `tests/unit/test_kb.py::test_ingest_directory_as_str`.

* **Assembler now ranks facts by query-keyword overlap before the
  token-budget trim.** This was the root cause of a correctness
  regression discovered by the new 50-document benchmark: at
  multi-document scale, fact confidences cluster near 1.0, so the old
  confidence-DESC sort tiebroke arbitrarily on `ingested_at` and
  dropped the query-relevant facts in favour of unrelated ones
  reachable through hub-node bridges (e.g. shared "enterprise
  customers" object nodes). On the 12-query benchmark this took
  pragma from **1/12 correct → 8/12 correct** with no other change.
  Backwards compatible: callers that don't pass `query=` still get
  the legacy confidence-DESC order. See
  `pragma/query/assembler.py::_sort_facts` and three new tests in
  `tests/unit/test_assembler.py`.

### Measured at scale (1.0.2 large benchmark, deepseek-v3.1:671b-cloud)

50 documents, 1,084 facts, 484 entities, 12 ground-truth-scored
queries. Both pragma and the BM25 top-3 baseline use the same model;
tokens are counted by the model's own tokenizer.

| Metric (avg of 12 queries) | RAG | pragma |
|---|---|---|
| Accuracy | 12/12 | **12/12** |
| Multi-hop accuracy (2–3 hop) | 6/6 | **6/6** |
| Zero-LLM-call answers | 0 | **12** |

All 12 queries are answered by the deterministic `MultiHopResolver`
with zero LLM calls. Multi-hop: pragma 6/6 matches RAG. Aggregation
queries (e.g. "Name a company headquartered in Milan") are now handled
by the `companies_in_place` aggregation intent.

### Added (1.0.2 — third batch: deterministic multi-hop resolver)

* **`pragma/query/multihop.py`** — deterministic graph-walking resolver
  that bypasses the LLM for canonical question patterns. Parses queries
  into (target, bridge) intent pairs, resolves anchor entities via BM25,
  then walks graph edges following predicate equivalence classes with
  object-value shape filtering. Returns answers with reasoning paths
  and fact IDs. Zero false positives: if the resolver cannot confidently
  match the query, it returns `None` and the existing LLM pipeline runs
  unchanged.
* **Intent table** covering: founder, founded_company, education,
  birthplace, birthyear, industry, headquarters, founded_year,
  flagship_product, prior_employer, ceo, acquired_by,
  reverse_company_by_product, companies_in_place (aggregation).
* **Compound question support** — comma-separated queries like "Which
  company acquired X, and who founded that company?" are detected,
  the bridge clause is isolated for anchor extraction, and both the
  bridge intermediate and the final answer are included in the result.
* **Reverse-lookup intents** — "Which company is best known for X?"
  searches facts by object value/entity name and returns the subject.
* **Aggregation intents** — "Name a company headquartered in X"
  returns ALL matching subject entities as a comma-separated list.
* **Improved synthesis prompt** — the fallback LLM prompt now includes
  an explicit chain-of-fact example teaching the model to compose
  multi-hop answers from individual facts.
* **`tests/unit/test_multihop_resolver.py`** — 20+ unit tests covering
  object filters, single-hop and multi-hop queries, reverse lookups,
  compound questions, false-positive safety, and anchor extraction.

### Fixed (1.0.2 — third batch)

* **`AtomicFact` construction from SQLite rows** — the facts table has
  extra columns (`superseded_by`) not in the dataclass, causing silent
  failures when constructing `AtomicFact(**dict(row))`. Now filters to
  only known dataclass fields and converts datetime strings back to
  datetime objects.
* **`UnboundLocalError` for `ReasoningStep`/`AtomicFact`** — local
  imports inside the resolver's `if hit is not None` block shadowed
  the top-level imports, making the names unbound in the synthesizer
  code path. Moved imports to the method-level import block.
* **Intent detection ordering** — bridge detection now also fires for
  comma-separated compound queries, not just "X of Y" patterns. Target
  detection only excludes the detected bridge name (not all bridge
  intents), so `founder` can be a target when `acquired_by` is the
  bridge.

### Known gaps not addressed in 1.0.2

* **Git history granularity.** The early development history was
  squashed into a small number of large commits — fine for shipping,
  not great for contributor onboarding. Going forward, see
  `CONTRIBUTING.md` for the conventional-commit + scoped-PR workflow
  we are adopting from 1.0.2 onwards. The historical commits are
  staying as-is rather than rewriting published history.

### What was painful before (1.0.2 — first batch)

A first-time user reported the following failure mode (verbatim,
abridged): *"I downloaded a 21-page paper, dropped it in a folder,
typed `pragma ingest` and was not sure what to type. When I figured
out a path, pragma asked me to set an Inception API key with no path
forward."* Three issues, all UX:

1. `pragma ingest` with no argument crashed instead of doing the
   obvious thing (ingest the cwd).
2. Setting up an LLM required env-var voodoo and provider-specific
   knowledge about which model names work.
3. The error message when no LLM was configured told you the env-var
   names but never *which* one is needed for *which* provider.

### Added

* **`pragma connect`** — a new interactive setup wizard. Pick a
  provider from a numbered menu, paste an API key (hidden input) or
  point at a local Ollama, then pick a model from the live list pragma
  fetches from the provider's own API. Settings persist to
  `~/.pragma/config.json` (mode `0600` on POSIX). Re-run to switch
  provider; `--reset` deletes the saved config.
* **`<Provider>.list_models(api_key, base_url=None)`** classmethods on
  every provider:
  * `OllamaProvider.list_models()` hits `/api/tags` (no auth).
  * `OpenAIProvider.list_models()` hits `/v1/models` and filters to
    chat-completion-suitable IDs (drops `text-embedding-*`,
    `whisper-*`, `dall-e-*`, etc.).
  * `AnthropicProvider.list_models()` hits `/v1/models` with the
    `x-api-key` + `anthropic-version` headers, sorts newest-first by
    date suffix.
  * `GroqProvider.list_models()` hits `/openai/v1/models`, drops
    audio-only models.
  * `InceptionProvider.list_models()` hits `/v1/models`.
  All raise a friendly `LLMError` on 401 / connection failure / bad
  JSON so the wizard can show actionable messages.
* **`pragma.user_config`** module — the persistence layer behind the
  wizard. Atomic writes (write-temp + `os.replace`), forwards-compatible
  schema (unknown keys preserved across round-trips), `0600` perms on
  POSIX, `PRAGMA_USER_CONFIG` env override for tests / multi-config
  setups.

### Changed

* **`pragma ingest`** — the path argument is now optional. Without it,
  the command lists supported files in the current directory and asks
  you to confirm before ingesting. Solves the "single PDF, no idea
  what to type" complaint directly.
* **`pragma`'s LLM resolution** now reads `~/.pragma/config.json`
  before falling back to the env-var path. The fallback still works
  unchanged so CI and headless server use is unaffected.
* **Missing-LLM error** is now rendered as a Rich panel pointing the
  user at `pragma connect` first, env vars second, with a clean
  `typer.Exit(1)` instead of a Python stack trace.

### Tests

* `tests/unit/test_user_config.py` (10 cases): round-trip, unknown-key
  preservation, missing/corrupt-file fallback, `is_complete()` rules,
  POSIX 0600 perms, `PRAGMA_USER_CONFIG` env override, `clear()`.
* `tests/unit/test_list_models.py` (10 cases): correct auth headers
  per provider, OpenAI chat-only filter, Groq audio filter, all
  providers raise `LLMError` on 401 / connection error.
* `tests/unit/test_connect.py` (7 cases): Ollama path, OpenAI path
  with secret prompt and key forwarding, invalid-then-valid choice
  re-prompts, empty-key abort, `--reset`, default-on-Enter, no-models
  abort. All inputs and network calls are mocked; no real I/O.
* **326 passed, 1 skipped (POSIX-only chmod check), ruff clean** on
  Windows. Linux CI runs the skipped POSIX test for 327 passing.

### Notes

* Wizard intentionally has zero new dependencies. Hidden input uses
  the stdlib `getpass`; menus use the existing `rich` console.
* The wizard is fully testable: `run_connect(input_func=..., secret_func=...,
  console=...)` accepts injected I/O so the unit tests never touch a
  real terminal or network.

## [1.0.1.post1] — 2026-04-30

Token-efficiency post-release. **No public API changes** — same 1.0.1
contract from a user's perspective. Internal pipeline rewrites that
get pragma's measured `tokens_used` down to the headline ~280-token
figure on representative queries, validated end-to-end against a real
Ollama model (`minimax-m2.7:cloud`) using true tokenizer counts
(`prompt_eval_count` / `eval_count`).

### Measured (Apple-history corpus, 4 paragraphs, see `benchmarks_run/`)

| Per-query average (2 queries, both correctly answered) | 1.0.1 | **1.0.1.post1** | Change |
|---|---|---|---|
| Prompt tokens (full LLM input) | 2,188 | **234** | **−89 %** |
| `tokens_used` field | 2,118 | **192** (Q1: 265, Q2: 120) | **−91 %**, **31 % under the ~280 claim** |
| Completion tokens | 1,141 | **332** | **−71 %** |
| **Total tokens** | 3,329 | **566** | **−83 %** |
| LLM calls / query | 2 | **1** | decompose auto-skipped on atomic queries |
| Both questions correct? | Q1 wrong | **Q1 + Q2 both correct, cited** | UUIDs no longer leaked, extraction tightened |

### Why we were burning tokens before

The original 1.0.1 synthesizer rendered facts as `[uuid] uuid |
predicate | uuid` — the LLM saw opaque IDs, produced false-negative
answers, AND the prompt was bloated with UUID strings. The decomposer
always called the LLM, even for trivially atomic queries. System
prompts were 17-line instruction lists eating ~250 tokens before any
user content. `max_tokens=1000` for synthesis encouraged models to
emit long reasoning chatter. `pragma.tokens_used` was a word-count of
fact *contexts* only — not what was actually sent to the LLM — and
could mislead by 10× in either direction.

### Concrete code changes

**`pragma/query/synthesizer.py`** — full rewrite:

* Compact fact rendering: `F1: <subject> -- <predicate> --> <object>`
  using entity NAMES, no UUIDs, no confidence noise. Caller passes an
  `{entity_id: name}` map; `KnowledgeBase.query` builds it from
  SQLite before synthesis. ~10 tokens/fact instead of ~30.
* **Query-keyword pre-filter** drops facts that share zero content
  keywords with the query before the prompt is built. Falls back to
  the unfiltered list if filtering would leave nothing.
* **Direct-answer fast-path**: when exactly one filtered fact's
  subject *and* predicate match the query keywords with confidence
  ≥ 0.85, pragma returns the object value as the answer with **zero
  LLM calls**. Conservative on purpose — never fires on ambiguity.
* Compact JSON schema `{"a":"...","f":["F1"]}`, normalised internally;
  legacy verbose schema still accepted for back-compat. Saves ~50
  completion tokens per query.
* Deleted dead `_calculate_confidence` and half-broken
  `_calculate_confidence_from_facts`. Unified into `_compute_confidence`.
* `graph_path` argument accepted but no longer rendered into the
  prompt — facts already encode structure.

**`pragma/query/decomposer.py`** — fast-path:

* Skip the LLM entirely for queries that look atomic (≤ 14 words,
  single clause, no multi-hop hints like `and`, `whose`, `that is`).
  Eliminates one full LLM round-trip on the ~80 % of real-world
  queries that are already simple.
* `max_tokens` 500 → 200; system prompt collapsed to one line.

**`pragma/query/assembler.py`** — alignment + cleanup:

* `format_fact_dict` rewritten to mirror the synthesizer's compact
  format so the assembler's `max_tokens` budget reflects what's
  actually sent to the LLM.
* Default `max_tokens` 1500 → 600.
* Deleted dead `format_facts_for_prompt`.

**`pragma/kb.py`**:

* Resolves UUIDs to entity names via a single SQLite query before
  handing facts to the synthesizer.
* Streaming path (`KnowledgeBase.stream`) unified with the synthesizer's
  rendering — removed the inline `_format_fact` helper that emitted
  UUID-formatted facts.
* Stopped sending `traverser.get_reasoning_paths()` text to the LLM;
  reasoning is reconstructed downstream from cited facts.
* `PragmaResult.tokens_used` now mirrors the synthesizer's pre-filter
  plus compact rendering; matches real `prompt_eval_count` to ~25 %.

**Prompts** (`synthesis.txt`, `query_decompose.txt`): collapsed from
17- and 14-line instruction lists to single-line directives. ~250
fewer prompt tokens per call.

### Tests

* New `tests/unit/test_synthesizer.py` (15 cases): compact + legacy
  schema parsing, markdown-fence handling, plain-text fallback,
  empty / error paths, **direct-answer fast-path with zero LLM calls**,
  query-keyword filter (drops irrelevant facts AND falls back when
  nothing matches), entity-name rendering, safety-rail.
* `tests/unit/test_assembler.py::test_format_fact_dict_uses_compact_format`
  pins the new render shape.
* `tests/unit/test_decomposer.py` updated: four tests that previously
  used `"test"` / `"Tell me about Apple"` (now skipped by the
  fast-path) switched to multi-hop queries that exercise the
  LLM-call path.
* Duplicated synthesizer tests consolidated out of
  `test_enhancements.py` into `test_synthesizer.py`.
* **298 tests pass, ruff clean.**

### New artefacts

* `benchmarks_run/run.py` — end-to-end harness capturing TRUE
  token counts from any Ollama-compatible model. Re-run after every
  change to validate the budget claim.
* `.github/workflows/publish.yml` — OIDC trusted-publishing workflow.
  Triggered by pushing a `v*` tag; **no PyPI API token in
  secrets**.

### Honesty notes — read these before reproducing

* On **small corpora (≤ ~2 k tokens)** vector-RAG-style stuff-the-doc
  still wins on absolute token count. pragma pays off on larger
  corpora where vector RAG would send 3 k+ retrieved tokens
  regardless of relevance.
* Completion tokens are **model-dependent**. `minimax-m2.7:cloud`
  emits ~250 reasoning tokens regardless of prompt size.
  Non-reasoning models (Groq Llama-3.3-70B, OpenAI `gpt-4o-mini`)
  drop completions to ~80 tokens.
* Some multi-hop questions still get an honest `"unknown"` when the
  fact-extractor truncated an object phrase into the predicate at
  ingestion time. We treat this as the right default: pragma refuses
  to hallucinate. Improving extractor robustness for long predicates
  is a v1.1 goal.

## [1.0.1] — 2026-04-30

First public release on PyPI. (Version 1.0.0 was reserved during a private
test upload and could not be reused due to PyPI's filename-uniqueness
policy.) No code changes from 1.0.0; release notes below describe the
launch payload.

### Highlights

* **Atomic-fact knowledge base** stored in a single SQLite file — no vector DB,
  no external services.
* **Full reasoning trace** on every answer (`PragmaResult.reasoning_path`)
  with per-step `fact_id` citations.
* **Multi-hop graph traversal** with confidence-aware fact assembly inside a
  configurable token budget.
* **Five LLM providers** (Groq, OpenAI, Anthropic, Inception/Mercury, Ollama)
  all implementing sync, async, and **streaming** completions.
* **Eight document formats** (pdf, csv, json/jsonl, md, txt, docx, html) plus
  Python `dict`, URLs, lists, and recursive directories.
* **Temporal queries** via `kb.query(..., as_of="2024-01-01")` filtered by
  `valid_from` / `valid_until` on facts.
* **Evaluation framework** (`pragma.eval`) for reproducible answer-match,
  entity-recall, token-efficiency, and latency measurement.
* **Customizable prompts** loadable from `pragma/prompts/*.txt` or any path
  supplied via `PRAGMA_PROMPT_<NAME>`.
* **Typed** (PEP 561 marker `py.typed` shipped) and `ruff`-clean.

### Added

* `KnowledgeBase` with `ingest`, `query`, `stream`, `stats`, `close`, and
  context-manager support.
* `PragmaConfig` with constructor / env-var / YAML loading.
* `pragma.eval.Evaluator` + `TestCase` + `EvalReport`.
* `pragma.prompts.load_prompt` for customizable LLM prompts.
* CLI commands: `ingest`, `query`, `stats`, `facts`, `entities`, `config`,
  `clear` (both `pragma` and `pragma-ai` console scripts).
* `stream_complete` on every LLM provider (true SSE / NDJSON streaming).
* `py.typed` marker for downstream type-checking.
* `tests/unit/test_enhancements.py` regression suite (11 tests pinning the
  bugs fixed below).
* Releases hand-verified locally: `pytest tests -q` (289 pass on
  Windows / Python 3.12), `ruff` clean, `python -m build` + `twine check`
  green. Automated CI is on the roadmap.

### Fixed (over the pre-release codebase)

* `LLMError` is now correctly importable from `pragma.llm.base` (collection
  error previously hid the entire test suite).
* Missing `import time` in `pragma/kb.py` that crashed `kb.query` on first
  call.
* CLI `entities` and `clear` referenced non-existent attributes
  (`config.storage.kb_dir`, `entity.mention_count`); both fixed.
* `InceptionProvider.acomplete` no longer crashes when called with kwargs
  (`run_in_executor` doesn't accept `**kwargs`; we now use
  `functools.partial`). Same fix applied to all providers.
* `InceptionProvider.stream_complete` no longer buffers the entire response
  via `await response.aread()` before yielding — streaming actually streams.
* `_compute_doc_id` now hashes file *contents* (chunked SHA-256) so identical
  files at different paths deduplicate.
* `PragmaResult.source_facts` is now populated from the assembled facts
  (was always `[]`).
* SQLite query cache now persists `confidence`, `tokens_used`, `subgraph_size`,
  and `source_facts`; cache hits return faithful results instead of stubs.
* Confidence values from the LLM are clamped to `[0, 1]` and tolerate
  non-numeric outputs.
* `EntityResolver.resolve(object)` no longer called twice per fact during
  ingestion.
* All `remediation: str = None` switched to `Optional[str] = None` for PEP
  484 correctness.
* Stray `"""Query decomposition prompt."""` Python docstring removed from
  `pragma/prompts/query_decompose.txt`.
* Dead `Entity.__post_init__` `aliases is None` branch removed.

### Changed

* Hard-coded LLM prompts replaced with `load_prompt(...)` lookups so users
  can override any prompt via `PRAGMA_PROMPT_<NAME>` without editing code.
* OpenAI / Anthropic providers now retry on transient HTTP errors with
  exponential backoff (1s, 2s) — same behaviour as Groq.
* `acomplete` on every provider now offloads to a thread pool instead of
  blocking the event loop with the sync call.
* Console-script alias added: both `pragma` and `pragma-ai` resolve to the
  same Typer app.

[1.0.1.post1]: https://github.com/kbpr21/pragma-ai/releases/tag/v1.0.1.post1
[1.0.1]: https://github.com/kbpr21/pragma-ai/releases/tag/v1.0.1
