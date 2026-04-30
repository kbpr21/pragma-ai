# Reasoning Traversal

How multi-hop reasoning works over the knowledge graph.

## Why Multi-Hop?

Single-hop retrieves connected facts:
- "What is Apple?" → "Apple is a company"

Multi-hop chains relationships:
- "Who founded Apple?" → "Steve Jobs co-founded Apple" (fact 1)
                         → "Apple founded in 1976" (fact 2)

## Traversal Algorithm

```
1. Find seed entities (BM25)
2. Expand to 1-hop neighbors
3. For each hop:
   - Add discovered entities to queue
   - Extract facts from new edges
4. Return assembled subgraph
```

## Hop Depth

Control traversal depth:

```python
result = kb.query(
    "What is Apple's supply chain?",
    hop_depth=3  # Default: 2
)
```

| hop_depth | Entities | Facts | Latency |
|-----------|----------|-------|---------|
| 1 | 5-10 | 10-20 | ~100ms |
| 2 | 20-50 | 50-100 | ~500ms |
| 3 | 100+ | 200+ | ~1s |

## Reasoning Path

Each answer includes the reasoning chain:

```python
result = kb.query("Who founded Apple?")

for step in result.reasoning_path:
    print(f"Fact: {step.fact_id[:8]}")
    print(f"  {step.explanation}")
    print(f"  Hop: {step.hop_number}")
```

Output:
```
Fact: a1b2c3d4
  Steve Jobs co-founded Apple (1997-2011)
  Hop: 1
Fact: e5f6g7h8
  Apple founded in 1976 by Steve Jobs
  Hop: 2
```

## Confidence Aggregation

Multiple facts → single confidence:

```
total_confidence = avg(fact_confidences) + reasoning_bonus
                 = 0.85 + 0.05 = 0.90
```

Bonus increases with more reasoning steps.

## Temporal Reasoning

Filter by time:

```python
result = kb.query(
    "What was Apple's status?",
    as_of="2010-01-01"  # Query historical state
)
```

SQL filter:
```sql
WHERE (valid_from IS NULL OR valid_from <= '2010-01-01')
  AND (valid_until IS NULL OR valid_until > '2010-01-01')
```
