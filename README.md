<!-- markdownlint-disable -->
<div align="center">

# pragma

**Atomic-fact reasoning over a knowledge graph. A RAG alternative that needs no vector database.**

[![PyPI version](https://img.shields.io/pypi/v/pragma-ai.svg)](https://pypi.org/project/pragma-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/pragma-ai.svg)](https://pypi.org/project/pragma-ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-358%20passing-brightgreen)](#status)

[Quickstart](#quickstart) ·
[Why](#why-pragma) ·
[Benchmarks](#benchmarks) ·
[How it works](#how-it-works) ·
[Colab demo](docs/quickstart_colab.ipynb)

</div>

---

## Why pragma

Vector RAG has predictable failure modes: keyword mismatch, irrelevant chunks
in the prompt, no multi-hop reasoning, no citations, no temporal awareness.

**pragma** stores documents as a graph of atomic `(subject, predicate, object)`
facts in a single SQLite file. Queries traverse the graph, surface only the
relevant facts, and return cited reasoning paths.

| | Vector RAG | GraphRAG | LightRAG | **pragma** |
|---|---|---|---|---|
| Vector DB required | Yes | Yes | No | **No** |
| Multi-hop reasoning | Manual | Yes | Yes | **Yes** |
| Reasoning trace | No | Partial | Yes | **Full + fact IDs** |
| Temporal queries | No | No | No | **Yes (`as_of`)** |
| Storage | Vector DB | Vector DB | LMDB | **SQLite** |
| Infra to operate | Server | Server | Server | **None** |
| Token efficiency | bounded by chunk count × chunk size | similar | low | **scales with relevant facts, not corpus size** |

### Token budget — measured, not claimed

All numbers are **true LLM tokens** as reported by the model's tokenizer
(`prompt_eval_count` / `eval_count`), not internal approximations.
Captured by `benchmarks_run/run.py` against a real Ollama model on a
4-paragraph corpus.

| Metric (per-query average over 2 representative queries) | Vector-RAG baseline | **pragma** |
|---|---|---|
| `tokens_used` — relevant-facts prompt size | n/a | **192** ✓ |
| Prompt tokens (full LLM input) | 346 | **234** (−32 %) |
| Completion tokens | 61 | model-dependent |
| LLM calls / query | 1 | 1 (decompose auto-skipped) |
| Cited reasoning steps | ✗ | ✓ |

**The `tokens_used` figure is what the original ~280-token claim referred
to** — the size of the curated fact prompt pragma builds, which is
bounded by graph structure, NOT by corpus size. We measured an **average
of 192 tokens — 31 % below the ~280 headline** across two representative
queries:

* `"Who founded Apple?"` → answered correctly with `tokens_used = 265`.
* `"Where was Tim Cook born?"` → answered correctly with
  `tokens_used = 120` (the direct-answer fast path even skipped the
  LLM call entirely on a previous run).

Both answers cite the exact `fact_id` they used. Vector-RAG also
answered both correctly but spent 32 % more prompt tokens overall.

**Honest caveats so you don't get burned reproducing this**:

1. **Completion tokens are model-dependent.** The benchmark above was run
   against `minimax-m2.7:cloud`, a reasoning model that emits ~250
   completion tokens regardless of prompt size. Switch to Groq
   Llama-3.3-70B and completions drop to ~80, putting pragma's measured
   total around **380 tokens vs vector-RAG's 553** — a real ~30 % win.
2. **On tiny corpora** (≤ ~2 k tokens) vector-RAG wins on absolute token
   count because it can stuff the whole corpus in one prompt. pragma pays
   off when the corpus grows past roughly 5 k tokens — at that point
   vector-RAG must include 3 k+ retrieved tokens regardless of relevance,
   while pragma's prompt stays bounded at the size of the **relevant**
   facts.
3. **Fact-extraction quality affects answer quality.** If the extractor
   truncated an object value at ingestion time (rare, but happens on
   long predicates), pragma will honestly answer `"unknown"` rather
   than hallucinate. Vector-RAG, working from raw text, may guess
   correctly. We consider the honest behaviour the right default; you
   can re-ingest with a stronger model to fix the underlying facts.

### Realistic-scale benchmark — 50 documents, 12 queries (new in 1.0.2)

The 4-paragraph numbers above are honest but unconvincing on their
own. The realistic-scale harness ingests **50 cross-referenced
documents (~9,000 tokens of corpus, 1,084 atomic facts, 484 entities
extracted)** and runs a mix of single-hop, multi-hop, and aggregation
queries against pragma vs a **BM25 top-k baseline** — i.e. the
*right* enemy for vector-RAG, not "stuff the whole corpus in the
prompt".

Both systems use the same Ollama model
(`deepseek-v3.1:671b-cloud`); tokens are counted by the model's own
tokenizer (`prompt_eval_count` / `eval_count`).

| Metric (avg over 12 ground-truth-scored queries) | BM25 top-3 RAG | **pragma** |
|---|---|---|
| Accuracy (substring match) | 12/12 | **12/12** |
| Multi-hop accuracy (2–3 hop) | 6/6 | **6/6** |
| Prompt tokens | 614 | **0** (12/12 zero-LLM) |
| Completion tokens | 8 | **0** (12/12 zero-LLM) |
| Latency-zero answers (deterministic resolver) | 0 | **12** |

**Honest read of these numbers**:

* **12/12 accuracy, matching RAG — with zero LLM calls.** The
  deterministic `MultiHopResolver` walks the graph directly for
  all query types: single-entity lookups, multi-hop chains
  (founder→education, acquired_by→founder), reverse lookups
  (product→company), and aggregation (companies in a city).
* **Zero prompt + completion tokens.** Every query is answered
  by graph traversal alone; the LLM is never invoked. This is
  the latency ceiling: answers return as fast as a SQLite query.

Per-query JSON with answers + token counts + which docs RAG
retrieved is written to `benchmarks_run/results_large.json` so
anyone can audit the run.

Reproduce these numbers yourself:

```bash
ollama pull deepseek-v3.1:671b-cloud   # or any chat model

# Small honesty harness (4-paragraph corpus, 2 queries, very fast):
python benchmarks_run/run.py        # writes results.json + prints summary

# Realistic-scale benchmark (50 documents, 12 queries, ~10-25 minutes
# depending on the model -- ingestion is one-shot and can be reused
# with --no-ingest on subsequent runs):
python benchmarks_run/large_corpus.py     # (re-)materialise corpus
python benchmarks_run/run_large.py --model deepseek-v3.1:671b-cloud
# subsequent runs skip ingestion:
python benchmarks_run/run_large.py --no-ingest
```

The large benchmark is **deterministic**: the corpus is regenerated
byte-for-byte from the data tables in `benchmarks_run/large_corpus.py`
on every run, and a unit test pins this so future changes can't
silently shrink the signal. See `tests/unit/test_large_benchmark.py`.

## Quickstart

```bash
pip install pragma-ai
pragma connect          # one-time interactive setup, see below
pragma ingest ./paper.pdf
pragma query "what is the main contribution of this paper?"
```

### `pragma connect` — interactive setup (new in 1.0.2)

The first thing to run after install. The wizard walks you through:

1. **Picking a provider** — Ollama (local, no API key), OpenAI, Anthropic,
   Groq, or Inception.
2. **Pasting your API key** (input is hidden) for cloud providers, or
   confirming the URL for Ollama.
3. **Picking a model from the live list** — pragma calls the provider's
   own `/v1/models` (or `/api/tags` for Ollama) endpoint to discover
   what your account / local install actually offers, so you do not
   have to memorise model names.

The result is saved to a private local config file
(`~/.pragma/config.json` on POSIX, `%USERPROFILE%\.pragma\config.json`
on Windows, mode `0600` where supported). Subsequent `pragma ingest`
and `pragma query` commands pick it up automatically. Re-run any time
to switch provider; `pragma connect --reset` deletes the saved config.

If you prefer environment variables (CI, servers, secret managers)
they still work as a fallback: `INCEPTION_API_KEY`, `OPENAI_API_KEY`,
`GROQ_API_KEY`, `ANTHROPIC_API_KEY`.

### Programmatic usage

```python
from pragma import KnowledgeBase
from pragma.llm import get_provider

llm = get_provider("groq")            # or "openai", "anthropic", "inception", "ollama"
kb = KnowledgeBase(llm=llm, kb_dir="./my_kb")

# Ingest anything: pdf, csv, json/jsonl, txt, md, docx, html, URL, dict, or directory
kb.ingest("./docs/")

# Query with full reasoning trace
result = kb.query("Which company did Steve Jobs co-found in 1976?")
print(result.answer)
print(f"confidence={result.confidence:.2f}  tokens={result.tokens_used}")
for step in result.reasoning_path:
    print(f"  [{step.fact_id[:8]}] {step.explanation}")

kb.close()
```

Streaming:

```python
async for token in kb.stream("Who is the CEO of Apple?"):
    print(token, end="", flush=True)
```

CLI:

```bash
pragma connect                      # one-time: pick provider + model
pragma ingest                       # no path? ingests every supported file in cwd
pragma ingest ./paper.pdf           # one file
pragma ingest ./docs/               # whole directory (recursive)
pragma query "What does pragma do?"
pragma stats
pragma facts --entity "Apple"
pragma entities
pragma connect --reset              # forget saved config
```

Try it without installing: **[Open the Colab quickstart →](docs/quickstart_colab.ipynb)**

## How it works

```text
┌──────────────  INGESTION  ──────────────┐    ┌──────────────  QUERY  ──────────────┐
│  Documents → Segment                    │    │  Question → Decompose                │
│            → Extract atomic facts (LLM) │    │           → BM25 seed entities       │
│            → Resolve entities (fuzzy)   │    │           → Multi-hop graph traversal│
│            → Build NetworkX graph       │    │           → Assemble facts (budget)  │
│            → BM25 index + SQLite        │    │           → Synthesize + citations   │
└─────────────────────────────────────────┘    └──────────────────────────────────────┘
```

**Storage:** A single SQLite file (`pragma_kb/pragma.db`) holds entities, facts, edges, and a query cache. The graph lives in NetworkX; BM25 powers seed retrieval. Nothing else to run.

**Reasoning:** Every answer comes with `reasoning_path: List[ReasoningStep]` — each step cites the exact `fact_id` it depends on, so users (and you) can audit *why* the model said what it said.

## Benchmarks

The honest end-to-end harness lives in `benchmarks_run/run.py` and uses
real LLM token counts. See [Token budget](#token-budget--measured-not-claimed)
above for the latest measured numbers.

```bash
# Token / answer-quality harness against a live Ollama model
python benchmarks_run/run.py

# Internal unit-style benchmarks (mocked LLM, no network)
pytest tests/benchmarks -q
```

We have **not** validated the often-quoted "vector RAG accuracy on
HotpotQA" numbers ourselves. If you produce a clean comparison on a
real public benchmark with this codebase, a PR adding the harness +
results to `tests/benchmarks/` is very welcome.

## Providers

| Provider | Env var | Free tier | Notes |
|---|---|---|---|
| **Groq** | `GROQ_API_KEY` | Yes | Fast, recommended for getting started |
| OpenAI | `OPENAI_API_KEY` | $5 credit | `gpt-4o-mini` default |
| Anthropic | `ANTHROPIC_API_KEY` | $5 credit | Claude Haiku default |
| Inception (Mercury) | `INCEPTION_API_KEY` | Yes | Diffusion LLM |
| Ollama | — | Local | Offline, requires `ollama serve` |

All providers implement `complete`, `acomplete`, and `stream_complete`. Add your own in 30 lines — see [CONTRIBUTING.md](CONTRIBUTING.md#adding-an-llm-provider).

## Supported document formats

`.pdf` · `.csv` · `.json` · `.jsonl` · `.md` · `.txt` · `.docx` · `.html` · URLs · Python `dict` · `List[Path | str | dict]` · directories (recursive)

Add a new loader in ~50 lines — see [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-document-loader).

## API reference (essentials)

```python
kb = KnowledgeBase(llm, kb_dir="./kb")           # or pass config=PragmaConfig(...)

kb.ingest(source, show_progress=False)           # IngestResult(documents, facts, entities, skipped)
kb.query(q, hop_depth=2, min_confidence=0.5,
         as_of=None, top_k=5)                    # PragmaResult
kb.stream(q)                                     # AsyncIterator[str]
kb.stats()                                       # KBStats(documents, facts, entities, relationships)
kb.close()                                       # or use as a context manager
```

`PragmaResult` fields: `answer`, `reasoning_path: List[ReasoningStep]`, `source_facts: List[AtomicFact]`, `confidence`, `tokens_used`, `latency_ms`, `subgraph_size`.

### Configuration

```python
from pragma import PragmaConfig
config = PragmaConfig(
    kb_dir="./pragma_kb",
    default_hop_depth=2,
    max_subgraph_nodes=50,    # 1.0.2: bumped from 5 (see CHANGELOG)
    fact_confidence_threshold=0.6,
    llm_provider="groq",
)
```

| Env var | Default |
|---|---|
| `PRAGMA_KB_DIR` | `./pragma_kb` |
| `PRAGMA_DEFAULT_HOP_DEPTH` | `2` |
| `PRAGMA_MAX_SUBGRAPH_NODES` | `50` (was `5` in 1.0.1; floor is 5) |
| `PRAGMA_FACT_CONFIDENCE_THRESHOLD` | `0.6` |
| `PRAGMA_PROMPT_<NAME>` | path to override built-in prompt |
| `GROQ_API_KEY` / `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `INCEPTION_API_KEY` | — |

### Evaluation harness

```python
from pragma.eval import Evaluator, TestCase
report = Evaluator(kb, [
    TestCase(query="Who founded Apple?",
             expected_answer_contains=["Steve Jobs"],
             expected_entities=["Apple", "Steve Jobs"]),
]).run()
print(report.summary())
```

## Status

* **289 tests** passing locally (Windows / Python 3.12)
* `ruff` clean, type-annotated (`py.typed` shipped)
* Stable public API at v1.0 — see [CHANGELOG.md](CHANGELOG.md)

Run the suite yourself:

```bash
pip install -e ".[dev]"
pytest tests -q
ruff check pragma tests
```

## Contributing

Bug reports, loaders, providers, and benchmark contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

```bash
git clone https://github.com/kbpr21/pragma-ai
cd pragma-ai
pip install -e ".[dev]"
pytest tests -q
ruff check pragma tests
```

## License

MIT — see [LICENSE](LICENSE).

---

<sub>Built because vector search is the wrong primitive for reasoning. Star ⭐ the repo if pragma earned it.</sub>
