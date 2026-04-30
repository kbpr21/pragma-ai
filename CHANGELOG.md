# Changelog

All notable changes to **pragma** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
