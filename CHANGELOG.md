# Changelog

All notable changes to **pragma** are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
* GitHub Actions matrix CI: 3 OS × 4 Python versions, ruff, mypy, pytest,
  build sanity, twine metadata check.
* GitHub Actions publish workflow using PyPI **Trusted Publisher (OIDC)**.

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

[1.0.1]: https://github.com/kbpr21/pragma-ai/releases/tag/v1.0.1
