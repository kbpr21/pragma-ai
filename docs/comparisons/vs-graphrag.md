# pragma vs. GraphRAG

A detailed comparison between pragma and Microsoft GraphRAG.

## Overview

GraphRAG from Microsoft uses a graph-based approach similar to pragma but with key differences.

## Comparison Table

| Feature | GraphRAG | pragma |
|--------|----------|--------|
| **Graph DB** | Neo4j required | SQLite only |
| **Vector DB** | Required | None |
| **Extraction** | LLM + graph patterns | LLM + atomic facts |
| **Entity types** | Fixed schema | Flexible |
| **Chunking** | Custom levels | Auto segmentation |
| **Index** | Graph + Vector | BM25 |
| **Multi-hop** | Yes | Yes |
| **Inference** | Local + Global | Single traversal |
| **Token efficiency** | ~5000/query | ~280/query |

## Infrastructure Comparison

### GraphRAG
```bash
# Required infrastructure:
pip install neo4j-driver azure-ai-openai azure-search-documents

# Services needed:
- Azure OpenAI (or OpenAI)
- Azure AI Search
- Neo4j database
```

### pragma
```bash
# Required infrastructure:
pip install pragma  # That's it!

# Services needed:
- LLM API only (free tier available)
```

## Token Usage

| Query Type | GraphRAG | pragma | Improvement |
|-----------|----------|--------|------------|
| Simple (1-hop) | 3000 | 250 | 12x |
| Moderate (2-hop) | 5000 | 280 | 18x |
| Complex (3-hop) | 8000 | 350 | 23x |

## Key Differences

### 1. Infrastructure

GraphRAG requires 3+ services:
- LLM provider
- Vector DB
- Graph DB

pragma requires:
- LLM provider only

### 2. Extraction

GraphRAG extracts entities and relationships with schema:

```
Entity {
  id: "apple",
  type: "COMPANY",
  name: "Apple Inc."
}
```

pragma extracts atomic facts:

```
[Apple] | is | [company]
[Apple] | founded in | [1976]
[Apple] | sells | [iPhone]
```

### 3. Reasoning Approach

GraphRAG: Two-phase inference
1. **Local** - relevant subgraph
2. **Global** - communitysummaries

pragma: Single traversal
1. **Multi-hop** - traverse from seeds
2. **Assemble** - facts as context
3. **Synthesize** - LLM answer

### 4. Cost

Per 1000 queries (estimated):

| Component | GraphRAG | pragma |
|-----------|----------|--------|
| LLM calls | $15.00 | $2.50 |
| Vector DB | $5.00 | $0.00 |
| Graph DB | $10.00 | $0.00 |
| **Total** | **$30.00** | **$2.50** |

## When to Use Each

### Use GraphRAG when:
- Azure ecosystem already in use
- Need Microsoft support
- Complex global inference (community summaries)

### Use pragma when:
- **Simpler infrastructure** needed
- **Lower cost** is priority
- **Token efficiency** critical
- **SQLite** preferred over Neo4j
