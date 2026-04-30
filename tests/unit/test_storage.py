import tempfile
from pathlib import Path

import pytest

from pragma.models import AtomicFact, PragmaResult, ReasoningStep
from pragma.storage.sqlite import SQLiteStore


@pytest.fixture
def temp_kb_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def store(temp_kb_dir):
    store = SQLiteStore(kb_dir=temp_kb_dir)
    yield store
    store.close()


class TestSQLiteStoreInit:
    def test_creates_db_in_directory(self, temp_kb_dir):
        store = SQLiteStore(kb_dir=temp_kb_dir)
        store.close()
        db_path = Path(temp_kb_dir) / "knowledge.db"
        assert db_path.exists()

    def test_wal_mode_enabled(self, store):
        conn = store._get_connection()
        cursor = conn.execute("PRAGMA journal_mode")
        assert cursor.fetchone()[0].upper() == "WAL"


class TestDocumentOperations:
    def test_save_document(self, store):
        doc_id = store.save_document(
            doc_id="doc1",
            path="/test/doc.pdf",
            doc_type="pdf",
            char_count=1000,
        )
        assert doc_id == "doc1"

    def test_document_exists(self, store):
        store.save_document(doc_id="doc1", path="/test.pdf", doc_type="pdf")
        assert store.document_exists("doc1") is True
        assert store.document_exists("nonexistent") is False


class TestEntityOperations:
    def test_save_entity(self, store):
        entity_id = store.save_entity(
            entity_id="e1",
            name="Apple Inc.",
            entity_type="ORG",
        )
        assert entity_id == "e1"

    def test_save_entity_upsert_on_name(self, store):
        store.save_entity(entity_id="e1", name="Apple", entity_type="ORG")
        entity_id = store.save_entity(
            entity_id="e2", name="Apple", entity_type="COMPANY"
        )

        assert entity_id == "e1"

    def test_get_entity_by_name(self, store):
        store.save_entity(entity_id="e1", name="Apple Inc.", entity_type="ORG")
        entity = store.get_entity_by_name("Apple Inc.")

        assert entity is not None
        assert entity.name == "Apple Inc."
        assert entity.entity_type == "ORG"

    def test_get_entity_by_name_not_found(self, store):
        entity = store.get_entity_by_name("Nonexistent")
        assert entity is None

    def test_get_all_entities(self, store):
        store.save_entity(entity_id="e1", name="Apple")
        store.save_entity(entity_id="e2", name="Microsoft")

        entities = store.get_all_entities()
        assert len(entities) == 2


class TestFactOperations:
    def test_save_fact(self, store):
        store.save_entity(entity_id="e1", name="CYP3A4")
        store.save_entity(entity_id="e2", name="DrugX")

        fact = AtomicFact(
            id="f1",
            subject_id="e1",
            predicate="metabolizes",
            object_id="e2",
            confidence=0.95,
        )
        fact_id = store.save_fact(fact)
        assert fact_id == "f1"

    def test_get_facts_by_subject(self, store):
        store.save_entity(entity_id="e1", name="CYP3A4")
        store.save_entity(entity_id="e2", name="DrugX")

        fact1 = AtomicFact(
            id="f1", subject_id="e1", predicate="metabolizes", object_id="e2"
        )
        fact2 = AtomicFact(
            id="f2", subject_id="e1", predicate="inhibits", object_id="e2"
        )
        store.save_fact(fact1)
        store.save_fact(fact2)

        facts = store.get_facts_by_subject("e1")
        assert len(facts) == 2

    def test_get_facts_by_object(self, store):
        store.save_entity(entity_id="e1", name="CYP3A4")
        store.save_entity(entity_id="e2", name="DrugX")

        fact = AtomicFact(
            id="f1", subject_id="e1", predicate="metabolizes", object_id="e2"
        )
        store.save_fact(fact)

        facts = store.get_facts_by_object("e2")
        assert len(facts) == 1

    def test_get_facts_by_entities(self, store):
        store.save_entity(entity_id="e1", name="CYP3A4")
        store.save_entity(entity_id="e2", name="DrugX")

        fact = AtomicFact(
            id="f1", subject_id="e1", predicate="metabolizes", object_id="e2"
        )
        store.save_fact(fact)

        facts = store.get_facts_by_entities("e1", "e2")
        assert len(facts) == 1

    def test_get_active_facts(self, store):
        store.save_entity(entity_id="e1", name="Entity1")
        store.save_entity(entity_id="e2", name="Entity2")

        fact1 = AtomicFact(
            id="f1",
            subject_id="e1",
            predicate="relates",
            object_id="e2",
            confidence=0.9,
        )
        fact2 = AtomicFact(
            id="f2",
            subject_id="e1",
            predicate="relates2",
            object_id="e2",
            confidence=0.5,
        )
        store.save_fact(fact1)
        store.save_fact(fact2)

        facts = store.get_active_facts(min_confidence=0.7)
        assert len(facts) == 1

    def test_invalidate_fact(self, store):
        store.save_entity(entity_id="e1", name="Entity1")
        store.save_entity(entity_id="e2", name="Entity2")

        fact = AtomicFact(id="f1", subject_id="e1", predicate="relates", object_id="e2")
        store.save_fact(fact)

        store.invalidate_fact("f1")
        facts = store.get_active_facts()
        assert len(facts) == 0


class TestKBStats:
    def test_get_kb_stats(self, store):
        store.save_document(doc_id="d1", path="/test.pdf", doc_type="pdf")
        store.save_entity(entity_id="e1", name="Apple")
        store.save_entity(entity_id="e2", name="Microsoft")
        store.save_entity(entity_id="e3", name="Google")

        store.save_entity(entity_id="e1_rel", name="Apple2")
        store.save_entity(entity_id="e2_rel", name="Microsoft2")

        stats = store.get_kb_stats()

        assert stats.documents == 1
        assert stats.entities == 5


class TestQueryCache:
    def test_save_and_get_query_cache(self, store):
        result = PragmaResult(
            answer="Test answer",
            reasoning_path=[
                ReasoningStep(fact_id="f1", explanation="step1", hop_number=1)
            ],
            source_facts=[],
            confidence=0.9,
            tokens_used=100,
            latency_ms=500.0,
        )
        store.save_query_cache("hash123", "What is X?", result)

        cached = store.get_query_cache("hash123")
        assert cached is not None
        assert cached.answer == "Test answer"

    def test_get_query_cache_not_found(self, store):
        cached = store.get_query_cache("nonexistent")
        assert cached is None


class TestPortability:
    def test_db_portable_across_os(self, temp_kb_dir):
        store1 = SQLiteStore(kb_dir=temp_kb_dir)
        store1.save_document(doc_id="d1", path="/test.pdf", doc_type="pdf")
        store1.close()

        db_path = Path(temp_kb_dir) / "knowledge.db"
        content = db_path.read_bytes()
        assert len(content) > 0
