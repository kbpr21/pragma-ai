# Provider Setup

Configure different LLM providers for pragma.

## Overview

| Provider | Env Variable | Free Tier | Notes |
|----------|------------|----------|-------|
| **Inception** (recommended) | `INCEPTION_API_KEY` | 1000/min | mercury-2 model |
| Groq | `GROQ_API_KEY` | Yes | llama-3.3-70b |
| OpenAI | `OPENAI_API_KEY` | $5 credit | gpt-4o-mini |
| Anthropic | `ANTHROPIC_API_KEY` | $5 credit | claude-haiku |
| Ollama | None | Unlimited | Offline |

## Inception (Recommended)

```python
from pragma.llm import get_provider

llm = get_provider("inception")  # Uses mercury-2
```

```bash
export INCEPTION_API_KEY="sk_your_key"
```

Get key at: https://console.inceptionlabs.ai

## Groq (Free)

```python
llm = get_provider("groq")  # Uses llama-3.3-70b-versatile
```

```bash
export GROQ_API_KEY="sk_your_key"
```

Get key at: https://console.groq.com

## OpenAI

```python
llm = get_provider("openai")  # Uses gpt-4o-mini
```

```bash
export OPENAI_API_KEY="sk_your_key"
```

## Anthropic

```python
llm = get_provider("anthropic")  # Uses claude-haiku-4-5-20251001
```

```bash
export ANTHROPIC_API_KEY="sk_your_key"
```

## Ollama (Offline)

```python
llm = get_provider("ollama", base_url="http://localhost:11434")

# Or pull a model
# ollama pull llama3
```

```bash
# No API key needed
# Install: brew install ollama (macOS) or pip install ollama
# Run: ollama serve
```

## Full Example with All Providers

```python
from pragma import KnowledgeBase
from pragma.llm import get_provider

# Inception (fastest)
llm = get_provider("inception")
kb = KnowledgeBase(llm=llm, kb_dir="./kb")
kb.ingest("./docs")
result = kb.query("What is X?")
```
```python
# Groq (free)
llm = get_provider("groq")
kb = KnowledgeBase(llm=llm, kb_dir="./kb")
# ... same as above
```
```python
# Ollama (offline)
llm = get_provider("ollama", base_url="http://localhost:11434")
kb = KnowledgeBase(llm=llm, kb_dir="./kb")
# ... same as above
```
