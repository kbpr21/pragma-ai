from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, List, Optional


@dataclass
class AtomicFact:
    """Represents an atomic fact extracted from a document."""

    id: str
    subject_id: str
    predicate: str
    object_id: Optional[str] = None
    object_value: Optional[str] = None
    context: str = ""
    source_doc: str = ""
    source_page: Optional[int] = None
    confidence: float = 1.0
    ingested_at: Optional[datetime] = None
    valid_from: Optional[datetime] = None
    valid_until: Optional[datetime] = None
    is_active: bool = True

    def __post_init__(self) -> None:
        if self.ingested_at is None:
            self.ingested_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "predicate": self.predicate,
            "object_id": self.object_id,
            "object_value": self.object_value,
            "context": self.context,
            "source_doc": self.source_doc,
            "source_page": self.source_page,
            "confidence": self.confidence,
            "ingested_at": self.ingested_at.isoformat() if self.ingested_at else None,
            "valid_from": self.valid_from.isoformat() if self.valid_from else None,
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "is_active": self.is_active,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AtomicFact":
        ingested_at = None
        if data.get("ingested_at"):
            ingested_at = datetime.fromisoformat(data["ingested_at"])
        valid_from = None
        if data.get("valid_from"):
            valid_from = datetime.fromisoformat(data["valid_from"])
        valid_until = None
        if data.get("valid_until"):
            valid_until = datetime.fromisoformat(data["valid_until"])

        return cls(
            id=data["id"],
            subject_id=data["subject_id"],
            predicate=data["predicate"],
            object_id=data.get("object_id"),
            object_value=data.get("object_value"),
            context=data.get("context", ""),
            source_doc=data.get("source_doc", ""),
            source_page=data.get("source_page"),
            confidence=data.get("confidence", 1.0),
            ingested_at=ingested_at,
            valid_from=valid_from,
            valid_until=valid_until,
            is_active=data.get("is_active", True),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AtomicFact):
            return NotImplemented
        return self.id == other.id


@dataclass
class Entity:
    """Represents an entity in the knowledge graph."""

    id: str
    name: str
    entity_type: Optional[str] = None
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None
    created_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "entity_type": self.entity_type,
            "aliases": self.aliases,
            "description": self.description,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Entity":
        created_at = None
        if data.get("created_at"):
            created_at = datetime.fromisoformat(data["created_at"])

        return cls(
            id=data["id"],
            name=data["name"],
            entity_type=data.get("entity_type"),
            aliases=data.get("aliases", []),
            description=data.get("description"),
            created_at=created_at,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.id == other.id


@dataclass
class ReasoningStep:
    """Represents a step in the reasoning process."""

    fact_id: str
    explanation: str
    hop_number: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "explanation": self.explanation,
            "hop_number": self.hop_number,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReasoningStep":
        return cls(
            fact_id=data["fact_id"],
            explanation=data["explanation"],
            hop_number=data["hop_number"],
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ReasoningStep):
            return NotImplemented
        return self.fact_id == other.fact_id and self.hop_number == other.hop_number


@dataclass
class PragmaResult:
    """Result of a query to the knowledge base."""

    answer: str
    reasoning_path: List[ReasoningStep]
    source_facts: List[AtomicFact]
    confidence: float
    tokens_used: int
    latency_ms: float
    subgraph_size: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "reasoning_path": [step.to_dict() for step in self.reasoning_path],
            "source_facts": [fact.to_dict() for fact in self.source_facts],
            "confidence": self.confidence,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "subgraph_size": self.subgraph_size,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PragmaResult":
        return cls(
            answer=data["answer"],
            reasoning_path=[
                ReasoningStep.from_dict(s) for s in data.get("reasoning_path", [])
            ],
            source_facts=[
                AtomicFact.from_dict(f) for f in data.get("source_facts", [])
            ],
            confidence=data.get("confidence", 0.0),
            tokens_used=data.get("tokens_used", 0),
            latency_ms=data.get("latency_ms", 0.0),
            subgraph_size=data.get("subgraph_size", 0),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, PragmaResult):
            return NotImplemented
        return self.answer == other.answer and self.confidence == other.confidence


@dataclass
class KBStats:
    """Statistics about the knowledge base."""

    documents: int
    facts: int
    entities: int
    relationships: int
    kb_dir: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "facts": self.facts,
            "entities": self.entities,
            "relationships": self.relationships,
            "kb_dir": self.kb_dir,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KBStats":
        return cls(
            documents=data.get("documents", 0),
            facts=data.get("facts", 0),
            entities=data.get("entities", 0),
            relationships=data.get("relationships", 0),
            kb_dir=data.get("kb_dir", ""),
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, KBStats):
            return NotImplemented
        return (
            self.documents == other.documents
            and self.facts == other.facts
            and self.entities == other.entities
        )
