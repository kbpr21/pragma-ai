from datetime import datetime

from pragma.query.assembler import FactAssembler
from pragma.storage.sqlite import SQLiteStore


class MockGraphBuilder:
    def __init__(self):
        self.storage = MockStorage()


class MockStorage:
    def get_facts_as_of(self, entity_ids, date):
        return []

    def get_facts_by_subject(self, entity_id):
        return []


class MockFact:
    def __init__(
        self,
        id="F1",
        subject_id="Apple",
        predicate="is",
        object_id="C1",
        object_value="company",
        valid_from=None,
        valid_until=None,
    ):
        self.id = id
        self.subject_id = subject_id
        self.predicate = predicate
        self.object_id = object_id
        self.object_value = object_value
        self.context = "test"
        self.source_doc = "doc1"
        self.source_page = 1
        self.confidence = 0.95
        self.ingested_at = datetime.now()
        self.is_active = True
        self.valid_from = valid_from
        self.valid_until = valid_until


class TestTemporalQueries:
    """Temporal query tests."""

    def test_assemble_facts_accepts_as_of_param(self):
        """assemble_facts accepts as_of parameter."""
        import networkx as nx

        gb = MockGraphBuilder()
        assembler = FactAssembler(gb)
        graph = nx.MultiDiGraph()
        graph.add_node("Apple")

        result = assembler.assemble_facts(graph, as_of=datetime(2024, 1, 1))

        assert isinstance(result, list)

    def test_get_facts_as_of_returns_empty_for_no_entities(self):
        """Returns empty list when no entities."""
        gb = MockGraphBuilder()
        assembler = FactAssembler(gb)

        result = assembler._get_facts_as_of([], datetime(2024, 1, 1))

        assert result == []

    def test_get_facts_as_of_filters_by_valid_from(self, tmp_path):
        """Filters facts by valid_from date."""
        store = SQLiteStore(str(tmp_path))

        fact = MockFact(
            id="F1",
            subject_id="Apple",
            predicate="is",
            object_value="company",
            valid_from=datetime(2024, 1, 1),
            valid_until=datetime(2025, 1, 1),
        )
        store.save_fact(fact)

        result = store.get_facts_as_of(["Apple"], datetime(2024, 6, 1))

        assert len(result) == 1
        store.close()

    def test_get_facts_as_of_excludes_expired_facts(self, tmp_path):
        """Excludes facts that expired before query date."""
        store = SQLiteStore(str(tmp_path))

        fact = MockFact(
            id="F1",
            subject_id="Apple",
            predicate="is",
            object_value="private",
            valid_from=datetime(2020, 1, 1),
            valid_until=datetime(2023, 1, 1),
        )
        store.save_fact(fact)

        result = store.get_facts_as_of(["Apple"], datetime(2024, 6, 1))

        assert len(result) == 0
        store.close()

    def test_get_facts_as_of_includes_future_facts(self, tmp_path):
        """Includes facts valid in the future."""
        store = SQLiteStore(str(tmp_path))

        fact = MockFact(
            id="F1",
            subject_id="Apple",
            predicate="is",
            object_value="tech leader",
            valid_from=datetime(2025, 1, 1),
            valid_until=None,
        )
        store.save_fact(fact)

        result = store.get_facts_as_of(["Apple"], datetime(2025, 6, 1))

        assert len(result) == 1
        store.close()

    def test_get_facts_as_of_returns_current_if_no_temporal(self, tmp_path):
        """Returns facts without temporal bounds as current."""
        store = SQLiteStore(str(tmp_path))

        fact = MockFact(
            id="F1",
            subject_id="Apple",
            predicate="is",
            object_value="company",
            valid_from=None,
            valid_until=None,
        )
        store.save_fact(fact)

        result = store.get_facts_as_of(["Apple"], datetime(2024, 6, 1))

        assert len(result) == 1
        store.close()
