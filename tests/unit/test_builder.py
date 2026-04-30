from typing import List, Optional

import pytest
from pragma.graph.builder import GraphBuilder
from pragma.models import AtomicFact, Entity


class MockStorage:
    """Mock storage for testing GraphBuilder."""

    def __init__(self):
        self.entities: dict[str, Entity] = {}

    def save_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: Optional[str] = None,
        aliases: Optional[List[str]] = None,
    ) -> str:
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            aliases=aliases or [],
            created_at=None,
        )
        self.entities[entity_id] = entity
        return entity_id

    def get_entity_by_id(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_all_entities(self) -> List[Entity]:
        return list(self.entities.values())


class TestGraphBuilderInit:
    """Tests for GraphBuilder initialization."""

    def test_create_empty_graph(self, tmp_path):
        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        assert builder.graph is not None
        assert builder.graph.number_of_nodes() == 0

    def test_graph_persists_across_instances(self, tmp_path):
        storage = MockStorage()
        storage.save_entity("e1", "Test Entity", "ORG", [])

        builder1 = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder1.add_entity(Entity("e1", "Test Entity", "ORG", [], None))
        builder1.save()

        builder2 = GraphBuilder(storage, kb_dir=str(tmp_path))
        assert builder2.graph.number_of_nodes() == 1


class TestGraphBuilderAddEntity:
    """Tests for adding entities to graph."""

    def test_add_single_entity(self, tmp_path):
        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        entity = Entity("e1", "Apple", "ORG", ["Apple Inc"], None)
        builder.add_entity(entity)

        assert builder.graph.number_of_nodes() == 1
        assert builder.graph.has_node("e1")
        assert builder.graph.nodes["e1"]["name"] == "Apple"

    def test_add_multiple_entities(self, tmp_path):
        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        entities = [
            Entity("e1", "Apple", "ORG", [], None),
            Entity("e2", "Google", "ORG", [], None),
            Entity("e3", "Microsoft", "ORG", [], None),
        ]
        for e in entities:
            builder.add_entity(e)

        assert builder.graph.number_of_nodes() == 3

    def test_add_duplicate_entity(self, tmp_path):
        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        entity = Entity("e1", "Apple", "ORG", [], None)
        builder.add_entity(entity)
        builder.add_entity(entity)

        assert builder.graph.number_of_nodes() == 1


class TestGraphBuilderAddFact:
    """Tests for adding facts to graph."""

    def test_add_fact_with_entities(self, tmp_path):
        storage = MockStorage()
        storage.save_entity("e1", "Apple", "ORG", [])
        storage.save_entity("e2", "Tim Cook", "PERSON", [])

        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        fact = AtomicFact(
            id="f1",
            subject_id="e1",
            predicate="CEO is",
            object_id="e2",
            confidence=1.0,
            source_doc="test.txt",
        )
        builder.add_fact(fact)

        assert builder.graph.number_of_nodes() == 2
        assert builder.graph.has_edge("e1", "e2", key="f1")

    def test_add_fact_metadata(self, tmp_path):
        storage = MockStorage()
        storage.save_entity("e1", "Apple", "ORG", [])
        storage.save_entity("e2", "Tim Cook", "PERSON", [])

        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        fact = AtomicFact(
            id="f1",
            subject_id="e1",
            predicate="CEO is",
            object_id="e2",
            confidence=0.95,
            source_doc="ceo.txt",
            context="According to the article...",
        )
        builder.add_fact(fact)

        edge_data = builder.graph.edges["e1", "e2", "f1"]
        assert edge_data["confidence"] == 0.95
        assert edge_data["source_doc"] == "ceo.txt"

    def test_add_fact_updates_bm25(self, tmp_path):
        storage = MockStorage()
        storage.save_entity("e1", "Apple", "ORG", [])

        builder = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder.add_entity(Entity("e1", "Apple", "ORG", [], None))

        fact = AtomicFact(
            id="f1",
            subject_id="e1",
            predicate="is",
            object_id=None,
            object_value="a company",
            confidence=1.0,
        )
        builder.add_fact(fact)

        assert builder._bm25_index is None


class TestGraphBuilderSaveLoad:
    """Tests for graph persistence."""

    def test_save_graph_with_content(self, tmp_path):
        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder.add_entity(Entity("e1", "Apple", "ORG", [], None))
        builder.save()

        graph_path = tmp_path / "graph.json"
        assert graph_path.exists()

    def test_save_and_load_graph(self, tmp_path):
        storage = MockStorage()
        storage.save_entity("e1", "Apple", "ORG", [])
        storage.save_entity("e2", "Google", "ORG", [])

        builder1 = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder1.add_entity(Entity("e1", "Apple", "ORG", [], None))
        builder1.add_entity(Entity("e2", "Google", "ORG", [], None))
        builder1.add_fact(
            AtomicFact(
                id="f1",
                subject_id="e1",
                predicate="competes with",
                object_id="e2",
                confidence=1.0,
            )
        )
        builder1.save()

        builder2 = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder2.load()

        assert builder2.graph.number_of_nodes() == 2
        assert builder2.graph.number_of_edges() == 1

    def test_load_nonexistent_graph(self, tmp_path):
        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder.load()

        assert builder.graph.number_of_nodes() == 0


class TestGraphBuilderBM25:
    """Tests for BM25 index."""

    def test_rebuild_bm25_index(self, tmp_path):
        pytest.importorskip("rank_bm25")

        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        builder.add_entity(Entity("e1", "Apple Inc", "ORG", ["Apple"], None))
        builder.add_entity(Entity("e2", "Google", "ORG", [], None))
        builder.rebuild_bm25_index()

        assert builder._bm25_index is not None

    def test_search_entities_bm25(self, tmp_path):
        pytest.importorskip("rank_bm25")

        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        builder.add_entity(Entity("e1", "Apple", "ORG", [], None))
        builder.add_entity(Entity("e2", "Google", "ORG", [], None))
        builder.add_entity(Entity("e3", "Microsoft", "ORG", [], None))

        builder.rebuild_bm25_index()
        results = builder.search_entities_bm25("Apple", top_k=1)

        assert len(results) >= 1

    def test_save_load_bm25_index(self, tmp_path):
        pytest.importorskip("rank_bm25")

        storage = MockStorage()
        builder1 = GraphBuilder(storage, kb_dir=str(tmp_path))

        builder1.add_entity(Entity("e1", "Apple", "ORG", [], None))
        builder1.rebuild_bm25_index()
        builder1.save_bm25_index()

        builder2 = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder2.load_bm25_index()

        assert builder2._bm25_index is not None


class TestGraphBuilderSubgraph:
    """Tests for subgraph extraction."""

    def test_get_subgraph_single_seed(self, tmp_path):
        storage = MockStorage()
        storage.save_entity("e1", "Apple", "ORG", [])
        storage.save_entity("e2", "Tim Cook", "PERSON", [])

        builder = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder.add_entity(Entity("e1", "Apple", "ORG", [], None))
        builder.add_entity(Entity("e2", "Tim Cook", "PERSON", [], None))
        builder.add_fact(
            AtomicFact(
                id="f1",
                subject_id="e1",
                predicate="CEO is",
                object_id="e2",
                confidence=1.0,
            )
        )

        subgraph = builder.get_subgraph(["e1"], hop_depth=1)

        assert subgraph.number_of_nodes() >= 1
        assert subgraph.has_node("e1")

    def test_get_subgraph_max_nodes(self, tmp_path):
        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        for i in range(100):
            storage.save_entity(f"e{i}", f"Entity{i}", "ORG", [])
            builder.add_entity(Entity(f"e{i}", f"Entity{i}", "ORG", [], None))

        subgraph = builder.get_subgraph(["e0"], hop_depth=2, max_nodes=10)

        assert subgraph.number_of_nodes() <= 10


class TestGraphBuilderStats:
    """Tests for graph statistics."""

    def test_stats_empty_graph(self, tmp_path):
        storage = MockStorage()
        builder = GraphBuilder(storage, kb_dir=str(tmp_path))

        stats = builder.stats()
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_stats_with_data(self, tmp_path):
        storage = MockStorage()
        storage.save_entity("e1", "Apple", "ORG", [])
        storage.save_entity("e2", "Google", "ORG", [])

        builder = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder.add_entity(Entity("e1", "Apple", "ORG", [], None))
        builder.add_entity(Entity("e2", "Google", "ORG", [], None))
        builder.add_fact(
            AtomicFact(
                id="f1",
                subject_id="e1",
                predicate="competes with",
                object_id="e2",
                confidence=1.0,
            )
        )

        stats = builder.stats()
        assert stats["nodes"] == 2
        assert stats["edges"] == 1


class TestGraphBuilderClear:
    """Tests for clearing graph."""

    def test_clear_graph(self, tmp_path):
        storage = MockStorage()
        storage.save_entity("e1", "Apple", "ORG", [])

        builder = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder.add_entity(Entity("e1", "Apple", "ORG", [], None))
        assert builder.graph.number_of_nodes() == 1

        builder.clear()
        assert builder.graph.number_of_nodes() == 0
        assert builder._bm25_index is None


class TestGraphBuilderRemoveFact:
    """Tests for removing facts."""

    def test_remove_fact(self, tmp_path):
        storage = MockStorage()
        storage.save_entity("e1", "Apple", "ORG", [])
        storage.save_entity("e2", "Google", "ORG", [])

        builder = GraphBuilder(storage, kb_dir=str(tmp_path))
        builder.add_entity(Entity("e1", "Apple", "ORG", [], None))
        builder.add_entity(Entity("e2", "Google", "ORG", [], None))
        builder.add_fact(
            AtomicFact(
                id="f1",
                subject_id="e1",
                predicate="competes with",
                object_id="e2",
                confidence=1.0,
            )
        )

        assert builder.graph.number_of_edges() == 1

        builder.remove_fact("f1")
        assert builder.graph.number_of_edges() == 0
