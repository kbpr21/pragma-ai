# Atomic Facts

An **atomic fact** is the smallest standalone claim that can be verified.

## Definition

An atomic fact has three components:

```
[Subject] | [Predicate] | [Object]

Example:
Apple | founded in | 1976
Apple | headquartered in | Cupertino
Apple | sells | iPhone
```

## Properties

| Property | Description |
|----------|-------------|
| Subject | The entity being described |
| Predicate | The relationship (is, founded, sells, etc.) |
| Object | The value or related entity |
| Confidence | 0.0 to 1.0 (LLM-assigned) |
| valid_from | When this fact became true |
| valid_until | When this fact stopped being true |

## Examples

### Atomic (good)
- "Apple | is | a company"
- "Steve Jobs | co-founded | Apple"
- "iPhone | released in | 2007"

### Non-Atomic (needs splitting)
- "Apple is a company that sells phones and computers"
  → Split into: "Apple | sells | phones", "Apple | sells | computers"

## Extraction

pragma uses an LLM to extract atomic facts from documents:

```python
from pragma.ingestion.extractor import FactExtractor

extractor = FactExtractor(llm)
facts = extractor.extract("Apple is a company that sells iPhones.")
# Returns: [AtomicFact(...), AtomicFact(...)]
```

## Confidence Scoring

Each fact gets a confidence score from the LLM:

- 0.9-1.0: Strong factual claim
- 0.7-0.9: Likely true
- 0.5-0.7: Possible but uncertain
- <0.5: Low confidence, filtered out

## Temporal Facts

Facts can have temporal bounds:

```python
fact = AtomicFact(
    subject="Apple",
    predicate="was",
    object="public",
    valid_from=datetime(1980, 12, 12),
    valid_until=datetime(2020, 8, 31),
)
```

This enables "What was Apple's status in 2010?" queries.
