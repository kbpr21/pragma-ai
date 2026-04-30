# Welcome to pragma

> **"Not retrieval. Reasoning over atomic facts."**

A zero-infrastructure Python library that replaces vector-based RAG with structured reasoning over atomic facts stored in a knowledge graph.

## Why pragma?

Vector-based RAG fails in 8 key ways:

| Problem | pragma Solution |
|---------|----------------|
| Keyword matching misses similar queries | BM25 + fuzzy entity matching |
| Cosine similarity selects wrong context | Graph traversal with reasoning |
| No multi-hop reasoning | Multi-hop graph traversal |
| Black-box answers | Full reasoning trace with fact IDs |
| Token bloat (~3000/query) | ~280 tokens/query (6x smaller) |
| No temporal awareness | `valid_from`/`valid_until` filtering |
| No entity resolution | Fuzzy matching with clustering |
| No confidence scoring | Per-fact confidence + aggregation |

## Quick Installation

```bash
pip install pragma
```

## Quick Example

```python
from pragma import KnowledgeBase
from pragma.llm import get_provider

# Create knowledge base (uses free Groq tier by default)
llm = get_provider("inception")  # or "groq"
kb = KnowledgeBase(llm=llm, kb_dir="./my_kb")

# Ingest documents
kb.ingest("./docs/")  # pdf, csv, json, txt, md, docx, html

# Query with full reasoning trace
result = kb.query("What is Apple's revenue?")
print(result.answer)

# Each answer includes reasoning path
for step in result.reasoning_path:
    print(f"  {step.fact_id}: {step.explanation}")

kb.close()
```

## Key Features

- **Zero infrastructure** — SQLite only, no vector DB
- **Atomic facts** — Every sentence becomes an interconnected fact
- **Graph reasoning** — Multi-hop traversal instead of similarity search
- **Full explainability** — Every answer includes reasoning path
- **~6x token efficient** — vs traditional RAG

## Next Steps

- [5-Minute Quickstart](quickstart.md) — Get up and running
- [Concepts](concepts/index.md) — Understand the architecture
- [API Reference](api-reference.md) — Full API documentation
- [Provider Setup](providers.md) — Configure LLM providers

## License

MIT — see [LICENSE](LICENSE)
