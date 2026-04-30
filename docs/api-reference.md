# API Reference

## KnowledgeBase

```python
from pragma import KnowledgeBase

kb = KnowledgeBase(llm, kb_dir="./kb")
```

### Methods

#### `kb.ingest(source)`

Ingest documents into the knowledge base.

```python
result = kb.ingest("./docs/")

# Returns IngestResult
#   - result.documents: int
#   - result.facts: int
#   - result.entities: int
```

#### `kb.query(query, **kwargs)`

Query the knowledge base.

```python
result = kb.query("What is Apple?")

# Returns PragmaResult
#   - result.answer: str
#   - result.reasoning_path: List[ReasoningStep]
#   - result.confidence: float
#   - result.latency_ms: float
#   - result.tokens_used: int
#   - result.subgraph_size: int
```

Parameters:
- `query`: Natural language question
- `hop_depth`: Maximum traversal depth (default: 2)
- `min_confidence`: Minimum fact confidence (default: 0.5)
- `as_of`: Filter facts valid at this date
- `top_k`: Maximum sub-questions (default: 5)

#### `kb.stream(query, **kwargs)`

Streaming query (async).

```python
async for token in kb.stream("What is Apple?"):
    print(token, end="", flush=True)
```

#### `kb.stats()`

Get knowledge base statistics.

```python
stats = kb.stats()

# Returns KBStats
#   - stats.documents: int
#   - stats.facts: int
#   - stats.entities: int
#   - stats.relationships: int
```

#### `kb.close()`

Close the knowledge base.

```python
kb.close()
# Or use context manager:
with KnowledgeBase(llm=llm) as kb:
    ...
```

## PragmaConfig

```python
from pragma import PragmaConfig

config = PragmaConfig(
    kb_dir="./pragma_kb",
    default_hop_depth=2,
    max_subgraph_nodes=5,
    llm_provider="inception",
)
```

## Models

### PragmaResult

```python
@dataclass
class PragmaResult:
    answer: str
    reasoning_path: List[ReasoningStep]
    confidence: float
    latency_ms: float
    tokens_used: int
    subgraph_size: int
```

### ReasoningStep

```python
@dataclass
class ReasoningStep:
    fact_id: str
    explanation: str
    hop_number: int
```

### AtomicFact

```python
@dataclass
class AtomicFact:
    id: str
    subject_id: str
    predicate: str
    object_id: Optional[str]
    object_value: Optional[str]
    context: str
    source_doc: str
    source_page: Optional[int]
    confidence: float
    ingested_at: datetime
    valid_from: Optional[datetime]
    valid_until: Optional[datetime]
```

### IngestResult

```python
@dataclass
class IngestResult:
    documents: int
    facts: int
    entities: int
    skipped: int
```

## CLI Commands

```bash
# Ingest documents
pragma ingest ./docs/

# Query
pragma query "What is Apple?"

# Stats
pragma stats

# Facts for entity
pragma facts --entity "Apple"

# Entities list
pragma entities --limit 20

# Configuration
pragma config

# Clear KB (with confirmation)
pragma clear
```

## Exceptions

```python
from pragma import (
    PragmaError,
    LLMError,
    IngestionError,
    StorageError,
    QueryError,
    GraphError,
    ConfigurationError,
)

try:
    kb.query("What is X?")
except LLMError as e:
    print(e.message)
    print(e.remediation)  # How to fix
```
