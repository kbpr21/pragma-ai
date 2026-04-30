# Concepts

Understanding the pragma architecture.

## Overview

pragma replaces vector-based retrieval with structured reasoning:

1. **Atomic Facts** — Each fact is extracted as subject|predicate|object
2. **Knowledge Graph** — Facts are nodes in a NetworkX graph
3. **Graph Traversal** — Multi-hop reasoning via traversal
4. **LLM Synthesis** — Generate answer with fact citations

## Architecture Diagram

```
Document → Segment → Extract → Resolve → Graph
                          ↓
Query → Decompose → BM25 → Traverse → Assemble → Answer
                                    ↓
                              reasoning_path
```

## Key Concepts

- [Atomic Facts](concepts/atomic-facts.md) — What makes a fact "atomic"
- [Knowledge Graph](concepts/knowledge-graph.md) — How facts form a graph
- [Reasoning](concepts/reasoning.md) — How multi-hop traversal works
