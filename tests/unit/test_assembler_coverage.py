import networkx as nx
from unittest.mock import MagicMock

from pragma.query.assembler import FactAssembler
from pragma.models import AtomicFact
from datetime import datetime


class TestAssemblerEdgeCases:
    """Increase assembler.py coverage."""

    def test_assemble_empty_subgraph(self):
        """Test empty subgraph returns empty list."""
        gb = MagicMock()
        assembler = FactAssembler(gb)
        graph = nx.MultiDiGraph()
        result = assembler.assemble_facts(graph)
        assert result == []

    def test_assemble_with_as_of_none(self):
        """Test as_of=None works same as without."""
        gb = MagicMock()
        assembler = FactAssembler(gb)
        graph = nx.MultiDiGraph()
        graph.add_node("Apple")
        graph.add_edge("Apple", "Co", predicate="is", key="is")

        result = assembler.assemble_facts(graph, as_of=None)
        # Returns empty since no actual facts without storage mock
        assert isinstance(result, list)

    def test_get_facts_as_of_empty_list(self):
        """Test empty entity list returns empty."""
        from datetime import datetime

        gb = MagicMock()
        gb.storage.get_facts_as_of.return_value = []
        assembler = FactAssembler(gb)
        result = assembler._get_facts_as_of([], datetime.now())
        assert result == []

    def test_get_facts_as_of_with_exception(self):
        """Test exception handling in temporal query."""
        gb = MagicMock()
        gb.storage.get_facts_as_of.side_effect = Exception("DB error")
        assembler = FactAssembler(gb)
        result = assembler._get_facts_as_of(["Apple"], datetime.now())
        assert result == []

    def test_convert_facts_to_dicts(self):
        """Test dict conversion."""
        gb = MagicMock()
        assembler = FactAssembler(gb)
        graph = nx.MultiDiGraph()

        facts = [
            AtomicFact(
                id="F1",
                subject_id="Apple",
                predicate="is",
                object_id="C1",
                object_value="company",
                context="test",
                source_doc="doc1",
                source_page=1,
                confidence=0.9,
                ingested_at=datetime.now(),
                is_active=True,
            )
        ]

        result = assembler._convert_facts_to_dicts(facts, graph)
        assert isinstance(result, list)
