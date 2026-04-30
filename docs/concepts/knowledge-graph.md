# Knowledge Graph

How atomic facts form a queryable knowledge structure.

## Structure

pragma uses NetworkX to build a multi-di-graph:

```
┌─────────────────────────────────────────┐
│           Knowledge Graph              │
├─────────────────────────────────────────┤
│  Nodes: Entities (Apple, iPhone, ...)  │
│  Edges: Facts (founded, sells, ...)    │
│  Edge Data: confidence, source, ...    │
└─────────────────────────────────────────┘
```

## Graph Properties

| Property | Value |
|----------|-------|
| Type | MultiDiGraph (multiple edges between nodes) |
| Storage | SQLite + NetworkX in-memory |
| Index | BM25 for entity search |
| Persistence | On-disk SQLite |

## Example Graph

For facts:
- "Apple | founded in | 1976"
- "Steve Jobs | co-founded | Apple"
- "Apple | sells | iPhone"
- "iPhone | is a | product"

```
    ┌──────────┐     ┌──────────┐
    │  Apple  │────►│  1976   │
    └──────────┘     └──────────┘
    ▲        │
    │        ▼
┌───┴───┐   ▼
│Steve  │ ┌──────┐
│Jobs   │►│iPhone │
└──────┘ └──────┘
```

## Entity Resolution

Multiple mentions map to one entity:

```
"Apple Inc." → "Apple"
"the tech giant" → "Apple"
"Apple Incorporated" → "Apple"
```

Uses fuzzy matching (rapidfuzz) with configurable threshold.

## BM25 Index

For fast entity lookup:

```python
from pragma.graph.builder import GraphBuilder

builder = GraphBuilder(storage, llm)
entities = builder.search_entities_bm25("tech company")
# Returns: ["Apple", "Microsoft", "Google", ...]
```

## Query Flow

1. **Find** seed entities via BM25
2. **Traverse** hops from seeds
3. **Assemble** facts from traversed edges
4. **Synthesize** answer via LLM
