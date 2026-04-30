# Quickstart

Get pragma working in 5 minutes.

## 1. Install

```bash
pip install pragma
```

## 2. Set API Key

Choose a provider:

```bash
# Option A: Inception (recommended, free tier available)
export INCEPTION_API_KEY="sk_your_key"

# Option B: Groq (free)
export GROQ_API_KEY="sk_your_key"

# Option C: OpenAI
export OPENAI_API_KEY="sk_your_key"

# Option D: Ollama (offline)
# No API key needed
```

## 3. Create a Knowledge Base

```python
from pragma import KnowledgeBase
from pragma.llm import get_provider

# Initialize with your LLM provider
llm = get_provider("inception")  # or "groq", "openai", "anthropic", "ollama"

kb = KnowledgeBase(llm=llm, kb_dir="./my_knowledge")
```

## 4. Ingest Documents

Create a sample file `apple.txt`:

```
Apple Inc. is a technology company founded in 1976 by Steve Jobs.
Apple is headquartered in Cupertino, California.
Apple's main products include iPhone, Mac, and iPad.
```

Then ingest:

```python
kb.ingest("./apple.txt")
```

Supported formats: `.pdf`, `.csv`, `.json`, `.txt`, `.md`, `.docx`, `.html`

## 5. Query

```python
result = kb.query("What is Apple?")
print(result.answer)

# Full reasoning trace
for step in result.reasoning_path:
    print(f"  Fact {step.fact_id[:8]}: {step.explanation}")

print(f"Confidence: {result.confidence}")
print(f"Latency: {result.latency_ms:.0f}ms")
```

## CLI Alternative

```bash
# Ingest documents
pragma ingest ./docs/

# Query
pragma query "What is Apple?"

# Check stats
pragma stats
```

## Next Steps

- [Concepts](concepts/index.md) — Deep dive into how it works
- [Provider Setup](providers.md) — Configure different LLM providers
- [Formats](formats.md) — Supported document formats
