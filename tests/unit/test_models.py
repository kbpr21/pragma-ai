from pragma.models import AtomicFact, Entity, KBStats, PragmaResult, ReasoningStep


class TestAtomicFact:
    def test_create_basic(self):
        fact = AtomicFact(id="f1", subject_id="e1", predicate="test_predicate")

        assert fact.id == "f1"
        assert fact.subject_id == "e1"
        assert fact.predicate == "test_predicate"
        assert fact.confidence == 1.0
        assert fact.is_active is True

    def test_to_dict(self):
        fact = AtomicFact(
            id="f1",
            subject_id="e1",
            predicate="metabolizes",
            object_id="e2",
            source_doc="doc1",
            confidence=0.9,
        )

        data = fact.to_dict()

        assert data["id"] == "f1"
        assert data["subject_id"] == "e1"
        assert data["predicate"] == "metabolizes"
        assert data["object_id"] == "e2"
        assert data["confidence"] == 0.9

    def test_from_dict(self):
        data = {
            "id": "f1",
            "subject_id": "e1",
            "predicate": "metabolizes",
            "object_id": "e2",
            "confidence": 0.9,
        }

        fact = AtomicFact.from_dict(data)

        assert fact.id == "f1"
        assert fact.subject_id == "e1"
        assert fact.object_id == "e2"

    def test_equality(self):
        fact1 = AtomicFact(id="f1", subject_id="e1", predicate="test")
        fact2 = AtomicFact(id="f1", subject_id="e1", predicate="test2")
        fact3 = AtomicFact(id="f2", subject_id="e1", predicate="test")

        assert fact1 == fact2
        assert fact1 != fact3


class TestEntity:
    def test_create_basic(self):
        entity = Entity(id="e1", name="Apple Inc.")

        assert entity.id == "e1"
        assert entity.name == "Apple Inc."
        assert entity.aliases == []

    def test_with_aliases(self):
        entity = Entity(id="e1", name="Apple Inc.", aliases=["Apple", "AAPL"])

        assert entity.aliases == ["Apple", "AAPL"]

    def test_to_dict(self):
        entity = Entity(id="e1", name="Apple Inc.", entity_type="ORG")

        data = entity.to_dict()

        assert data["id"] == "e1"
        assert data["name"] == "Apple Inc."
        assert data["entity_type"] == "ORG"

    def test_from_dict(self):
        data = {
            "id": "e1",
            "name": "Apple Inc.",
            "entity_type": "ORG",
            "aliases": ["Apple", "AAPL"],
        }

        entity = Entity.from_dict(data)

        assert entity.id == "e1"
        assert entity.name == "Apple Inc."
        assert entity.aliases == ["Apple", "AAPL"]

    def test_equality(self):
        entity1 = Entity(id="e1", name="Apple")
        entity2 = Entity(id="e1", name="Apple Inc.")
        entity3 = Entity(id="e2", name="Apple")

        assert entity1 == entity2
        assert entity1 != entity3


class TestReasoningStep:
    def test_create(self):
        step = ReasoningStep(fact_id="f1", explanation="uses fact", hop_number=1)

        assert step.fact_id == "f1"
        assert step.explanation == "uses fact"
        assert step.hop_number == 1

    def test_to_dict(self):
        step = ReasoningStep(fact_id="f1", explanation="test", hop_number=1)

        data = step.to_dict()

        assert data["fact_id"] == "f1"
        assert data["hop_number"] == 1

    def test_from_dict(self):
        data = {"fact_id": "f1", "explanation": "test", "hop_number": 1}

        step = ReasoningStep.from_dict(data)

        assert step.fact_id == "f1"
        assert step.hop_number == 1


class TestPragmaResult:
    def test_create(self):
        result = PragmaResult(
            answer="test answer",
            reasoning_path=[],
            source_facts=[],
            confidence=0.9,
            tokens_used=100,
            latency_ms=500.0,
        )

        assert result.answer == "test answer"
        assert result.confidence == 0.9
        assert result.tokens_used == 100

    def test_to_dict(self):
        result = PragmaResult(
            answer="test",
            reasoning_path=[
                ReasoningStep(fact_id="f1", explanation="step1", hop_number=1)
            ],
            source_facts=[],
            confidence=0.9,
            tokens_used=100,
            latency_ms=500.0,
        )

        data = result.to_dict()

        assert data["answer"] == "test"
        assert len(data["reasoning_path"]) == 1
        assert data["tokens_used"] == 100

    def test_from_dict(self):
        data = {
            "answer": "test",
            "reasoning_path": [
                {"fact_id": "f1", "explanation": "step1", "hop_number": 1}
            ],
            "source_facts": [],
            "confidence": 0.9,
            "tokens_used": 100,
            "latency_ms": 500.0,
        }

        result = PragmaResult.from_dict(data)

        assert result.answer == "test"
        assert result.confidence == 0.9


class TestKBStats:
    def test_create(self):
        stats = KBStats(documents=10, facts=100, entities=50, relationships=75)

        assert stats.documents == 10
        assert stats.facts == 100
        assert stats.entities == 50
        assert stats.relationships == 75

    def test_to_dict(self):
        stats = KBStats(
            documents=10, facts=100, entities=50, relationships=75, kb_dir="./kb"
        )

        data = stats.to_dict()

        assert data["documents"] == 10
        assert data["facts"] == 100

    def test_from_dict(self):
        data = {
            "documents": 10,
            "facts": 100,
            "entities": 50,
            "kb_dir": "./kb",
        }

        stats = KBStats.from_dict(data)

        assert stats.documents == 10
        assert stats.facts == 100

    def test_equality(self):
        stats1 = KBStats(documents=10, facts=100, entities=50, relationships=75)
        stats2 = KBStats(documents=10, facts=100, entities=50, relationships=75)
        stats3 = KBStats(documents=20, facts=100, entities=50, relationships=75)

        assert stats1 == stats2
        assert stats1 != stats3
