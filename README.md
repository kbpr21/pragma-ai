<!-- markdownlint-disable -->
<div align="center">

# pragma

**Atomic-fact reasoning over a knowledge graph. A RAG alternative that needs no vector database.**

[![PyPI version](https://img.shields.io/pypi/v/pragma-ai.svg)](https://pypi.org/project/pragma-ai/)
[![Python versions](https://img.shields.io/pypi/pyversions/pragma-ai.svg)](https://pypi.org/project/pragma-ai/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-289%20passing-brightgreen)](#status)

[Quickstart](#quickstart) ·
[Why](#why-pragma) ·
[Benchmarks](#benchmarks) ·
[How it works](#how-it-works) ·
[Colab demo](docs/quickstart_colab.ipynb)

</div>

---

## Why pragma

Vector RAG fails silently in predictable ways: keyword mismatch, irrelevant context, no multi-hop reasoning, no citations, no temporal awareness, ~3,000 tokens per query.

**pragma** stores documents as a graph of atomic `(subject, predicate, object)` facts. Queries traverse the graph, return cited reasoning paths, and use ~6× fewer tokens than vector RAG.

| | Vector RAG | GraphRAG | LightRAG | **pragma** |
|---|---|---|---|---|
| Vector DB required | Yes | Yes | No | **No** |
| Multi-hop reasoning | Manual | Yes | Yes | **Yes** |
| Tokens per query | ~3,000 | ~5,000 | ~400 | **~280** |
| Reasoning trace | No | Partial | Yes | **Full + fact IDs** |
| Temporal queries | No | No | No | **Yes (`as_of`)** |
| Storage | Vector DB | Vector DB | LMDB | **SQLite** |
| Infra to operate | Server | Server | Server | **None** |

## Quickstart

```bash
pip install pragma-ai
```

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
pragma ingest ./docs/
pragma query "What does pragma do?"
pragma stats
pragma facts --entity "Apple"
pragma entities
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

```bash
pytest tests/benchmarks -q
```

Representative runs (Groq Llama-3.3-70B, 100 multi-hop questions over 50-document corpus):

| Metric | Vector RAG | GraphRAG | **pragma** |
|---|---|---|---|
| Tokens / query (avg) | 3,142 | 4,890 | **278** |
| 2-hop accuracy | 41 % | 76 % | **82 %** |
| Cite-able reasoning | ✗ | partial | **✓** |
| Cold-start infra | Pinecone/Qdrant | Neo4j+Pinecone | **0 services** |

> Numbers vary with corpus and model; see `tests/benchmarks/` for reproducible harnesses.

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
    max_subgraph_nodes=5,
    fact_confidence_threshold=0.6,
    llm_provider="groq",
)
```

| Env var | Default |
|---|---|
| `PRAGMA_KB_DIR` | `./pragma_kb` |
| `PRAGMA_DEFAULT_HOP_DEPTH` | `2` |
| `PRAGMA_MAX_SUBGRAPH_NODES` | `5` |
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
