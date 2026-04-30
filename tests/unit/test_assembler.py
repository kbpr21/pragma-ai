import networkx as nx
from pragma.query.assembler import FactAssembler
from pragma.models import Entity


def make_fact_dict(
    id,
    subject_id,
    predicate,
    object_id=None,
    object_value=None,
    confidence=1.0,
    is_active=True,
):
    return {
        "id": id,
        "subject_id": subject_id,
        "predicate": predicate,
        "object_id": object_id,
        "object_value": object_value,
        "confidence": confidence,
        "is_active": is_active,
    }


class MockGraphBuilder:
    def __init__(self):
        self.entities = {}
        self.facts = []

    @property
    def storage(self):
        return self

    def get_entity_by_id(self, entity_id: str):
        return self.entities.get(entity_id)

    def get_facts_by_subject(self, subject_id: str):
        return [f for f in self.facts if f.get("subject_id") == subject_id]

    def get_facts_by_object(self, object_id: str):
        return [f for f in self.facts if f.get("object_id") == object_id]


class TestFactAssembler:
    """FactAssembler tests."""

    def test_assemble_facts_empty_subgraph(self):
        builder = MockGraphBuilder()
        assembler = FactAssembler(builder)
        g = nx.MultiDiGraph()
        result = assembler.assemble_facts(g)
        assert result == []

    def test_filter_by_confidence(self):
        builder = MockGraphBuilder()
        facts = [
            make_fact_dict("f1", "e1", "rel", confidence=0.9),
            make_fact_dict("f2", "e1", "rel", confidence=0.3),
        ]

        assembler = FactAssembler(builder, min_confidence=0.5)
        filtered = assembler._filter_facts(facts)

        assert len(filtered) == 1
        assert filtered[0]["id"] == "f1"

    def test_deduplicate_facts(self):
        builder = MockGraphBuilder()
        facts = [
            make_fact_dict("f1", "e1", "rel", "e2", confidence=0.8),
            make_fact_dict("f2", "e1", "rel", "e2", confidence=0.9),
        ]

        assembler = FactAssembler(builder)
        deduped = assembler._deduplicate_facts(facts)

        assert len(deduped) == 1

    def test_sort_by_confidence(self):
        builder = MockGraphBuilder()
        facts = [
            make_fact_dict("f1", "e1", "rel", confidence=0.5),
            make_fact_dict("f2", "e1", "rel", confidence=0.9),
            make_fact_dict("f3", "e1", "rel", confidence=0.7),
        ]

        assembler = FactAssembler(builder)
        sorted_facts = assembler._sort_facts(facts)

        assert sorted_facts[0]["confidence"] == 0.9

    def test_trim_by_token_budget(self):
        builder = MockGraphBuilder()
        builder.entities["e1"] = Entity("e1", "Entity1", "ORG", [], None)

        facts = [
            make_fact_dict(f"f{i}", "e1", "rel", confidence=1.0) for i in range(100)
        ]

        assembler = FactAssembler(builder, max_tokens=100)
        trimmed = assembler._trim_by_token_budget(facts)

        assert len(trimmed) < 100

    def test_format_fact_dict_uses_compact_format(self):
        """v1.0.1: render mirrors AnswerSynthesizer._format_fact, no
        confidence in output, no zero-padded F001 prefix."""
        builder = MockGraphBuilder()
        builder.entities["e1"] = Entity("e1", "Apple", "ORG", [], None)
        builder.entities["e2"] = Entity("e2", "Tim Cook", "PERSON", [], None)

        fact = make_fact_dict("f1", "e1", "CEO is", "e2", confidence=0.95)

        assembler = FactAssembler(builder)
        formatted = assembler.format_fact_dict(fact, index=0)

        # New compact format
        assert formatted.startswith("F1: ")
        assert "Apple" in formatted
        assert "CEO is" in formatted
        assert "Tim Cook" in formatted
        # Confidence intentionally NOT in the prompt (saves tokens)
        assert "confidence" not in formatted.lower()
        # No zero-padding (was "F001" before, now "F1:")
        assert "F001" not in formatted
