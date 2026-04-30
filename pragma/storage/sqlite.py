import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional

from pragma.models import AtomicFact, Entity, KBStats, PragmaResult, ReasoningStep


def _adapt_datetime(dt: datetime) -> str:
    return dt.isoformat()


def _convert_datetime(value: bytes) -> datetime:
    return datetime.fromisoformat(value.decode())


sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_converter("datetime", _convert_datetime)


class SQLiteStore:
    """SQLite-based storage for the knowledge base."""

    def __init__(self, kb_dir: str = "./pragma_kb") -> None:
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.kb_dir / "knowledge.db"
        self._conn: Optional[sqlite3.Connection] = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        schema_path = Path(__file__).parent / "migrations" / "001_initial.sql"
        with open(schema_path, encoding="utf-8") as f:
            schema = f.read()
        conn.executescript(schema)
        conn.commit()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def save_document(
        self,
        doc_id: str,
        path: str,
        doc_type: str,
        char_count: int = 0,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, path, doc_type, ingested_at, char_count, metadata) VALUES (?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                path,
                doc_type,
                now,
                char_count,
                json.dumps(metadata) if metadata else None,
            ),
        )
        conn.commit()
        return doc_id

    def document_exists(self, doc_id: str) -> bool:
        conn = self._get_connection()
        cursor = conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,))
        return cursor.fetchone() is not None

    def save_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: Optional[str] = None,
        aliases: Optional[List[str]] = None,
        description: Optional[str] = None,
    ) -> str:
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()

        existing = conn.execute(
            "SELECT id FROM entities WHERE name = ?", (name,)
        ).fetchone()

        if existing:
            entity_id = existing["id"]
            conn.execute(
                "UPDATE entities SET entity_type = ?, aliases = ?, description = ? WHERE id = ?",
                (
                    entity_type,
                    json.dumps(aliases) if aliases else None,
                    description,
                    entity_id,
                ),
            )
        else:
            conn.execute(
                "INSERT INTO entities (id, name, entity_type, aliases, description, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    entity_id,
                    name,
                    entity_type,
                    json.dumps(aliases) if aliases else None,
                    description,
                    now,
                ),
            )
        conn.commit()
        return entity_id

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        conn = self._get_connection()
        row = conn.execute("SELECT * FROM entities WHERE name = ?", (name,)).fetchone()
        if not row:
            return None
        return self._row_to_entity(row)

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM entities WHERE id = ?", (entity_id,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_entity(row)

    def get_all_entities(self) -> List[Entity]:
        conn = self._get_connection()
        rows = conn.execute("SELECT * FROM entities").fetchall()
        return [self._row_to_entity(row) for row in rows]

    def save_fact(self, fact: AtomicFact) -> str:
        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            """INSERT OR REPLACE INTO facts
            (id, subject_id, predicate, object_id, object_value, context, source_doc, source_page, confidence, ingested_at, valid_from, valid_until, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                fact.id,
                fact.subject_id,
                fact.predicate,
                fact.object_id,
                fact.object_value,
                fact.context,
                fact.source_doc,
                fact.source_page,
                fact.confidence,
                fact.ingested_at.isoformat() if fact.ingested_at else now,
                fact.valid_from.isoformat() if fact.valid_from else None,
                fact.valid_until.isoformat() if fact.valid_until else None,
                1 if fact.is_active else 0,
            ),
        )
        conn.commit()
        return fact.id

    def get_facts_by_subject(self, subject_id: str) -> List[AtomicFact]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM facts WHERE subject_id = ?", (subject_id,)
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_facts_as_of(self, entity_ids: List[str], date) -> List[AtomicFact]:
        """Get facts valid at a specific point in time."""
        conn = self._get_connection()
        placeholders = ",".join("?" * len(entity_ids))
        query = f"""
            SELECT * FROM facts
            WHERE subject_id IN ({placeholders})
            AND (valid_from IS NULL OR valid_from <= ?)
            AND (valid_until IS NULL OR valid_until > ?)
        """
        params = entity_ids + [date, date]
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_facts_by_object(self, object_id: str) -> List[AtomicFact]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM facts WHERE object_id = ?", (object_id,)
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_facts_by_entities(
        self, subject_id: str, object_id: str
    ) -> List[AtomicFact]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM facts WHERE subject_id = ? AND object_id = ?",
            (subject_id, object_id),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def get_active_facts(self, min_confidence: float = 0.0) -> List[AtomicFact]:
        conn = self._get_connection()
        rows = conn.execute(
            "SELECT * FROM facts WHERE is_active = 1 AND confidence >= ?",
            (min_confidence,),
        ).fetchall()
        return [self._row_to_fact(row) for row in rows]

    def invalidate_fact(self, fact_id: str) -> None:
        conn = self._get_connection()
        conn.execute("UPDATE facts SET is_active = 0 WHERE id = ?", (fact_id,))
        conn.commit()

    def get_kb_stats(self) -> KBStats:
        conn = self._get_connection()

        doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        fact_count = conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        entity_count = conn.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        rel_count = conn.execute("SELECT COUNT(*) FROM relationships").fetchone()[0]

        return KBStats(
            documents=doc_count,
            facts=fact_count,
            entities=entity_count,
            relationships=rel_count,
            kb_dir=str(self.kb_dir),
        )

    def save_query_cache(
        self, query_hash: str, query_text: str, result: PragmaResult
    ) -> None:
        import uuid

        conn = self._get_connection()
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "reasoning_path": [r.to_dict() for r in result.reasoning_path],
            "source_facts": [f.to_dict() for f in result.source_facts],
            "confidence": result.confidence,
            "tokens_used": result.tokens_used,
            "subgraph_size": result.subgraph_size,
        }
        conn.execute(
            """INSERT OR REPLACE INTO query_cache
            (id, query_hash, query_text, answer, reasoning, created_at, ttl_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                str(uuid.uuid4()),
                query_hash,
                query_text,
                result.answer,
                json.dumps(payload),
                now,
                3600,
            ),
        )
        conn.commit()

    def get_query_cache(self, query_hash: str) -> Optional[PragmaResult]:
        conn = self._get_connection()
        row = conn.execute(
            "SELECT * FROM query_cache WHERE query_hash = ?", (query_hash,)
        ).fetchone()
        if not row:
            return None
        return self._row_to_result(row)

    def _row_to_entity(self, row: sqlite3.Row) -> Entity:
        return Entity(
            id=row["id"],
            name=row["name"],
            entity_type=row["entity_type"],
            aliases=json.loads(row["aliases"]) if row["aliases"] else [],
            description=row["description"],
            created_at=datetime.fromisoformat(row["created_at"])
            if row["created_at"]
            else None,
        )

    def _row_to_fact(self, row: sqlite3.Row) -> AtomicFact:
        return AtomicFact(
            id=row["id"],
            subject_id=row["subject_id"],
            predicate=row["predicate"],
            object_id=row["object_id"],
            object_value=row["object_value"],
            context=row["context"] or "",
            source_doc=row["source_doc"] or "",
            source_page=row["source_page"],
            confidence=row["confidence"],
            ingested_at=datetime.fromisoformat(row["ingested_at"])
            if row["ingested_at"]
            else None,
            valid_from=datetime.fromisoformat(row["valid_from"])
            if row["valid_from"]
            else None,
            valid_until=datetime.fromisoformat(row["valid_until"])
            if row["valid_until"]
            else None,
            is_active=bool(row["is_active"]),
        )

    def _row_to_result(self, row: sqlite3.Row) -> PragmaResult:
        from pragma.models import AtomicFact

        raw = json.loads(row["reasoning"]) if row["reasoning"] else {}

        # Backwards-compatible: legacy rows stored a bare list of reasoning steps.
        if isinstance(raw, list):
            reasoning_path = [ReasoningStep.from_dict(r) for r in raw]
            return PragmaResult(
                answer=row["answer"],
                reasoning_path=reasoning_path,
                source_facts=[],
                confidence=1.0,
                tokens_used=0,
                latency_ms=0.0,
            )

        reasoning_path = [
            ReasoningStep.from_dict(r) for r in raw.get("reasoning_path", [])
        ]
        source_facts = [AtomicFact.from_dict(f) for f in raw.get("source_facts", [])]
        return PragmaResult(
            answer=row["answer"],
            reasoning_path=reasoning_path,
            source_facts=source_facts,
            confidence=float(raw.get("confidence", 1.0)),
            tokens_used=int(raw.get("tokens_used", 0)),
            latency_ms=0.0,
            subgraph_size=int(raw.get("subgraph_size", 0)),
        )

    def __enter__(self) -> "SQLiteStore":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
