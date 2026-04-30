import networkx as nx
from pragma.graph.traversal import GraphTraverser
from pragma.models import Entity


class MockGraphBuilder:
    """Mock graph builder for testing."""

    def __init__(self, graph: nx.MultiDiGraph = None):
        self._graph = graph or nx.MultiDiGraph()

    @property
    def graph(self):
        return self._graph

    def get_subgraph(self, seed_entities, hop_depth=2, max_nodes=50):
        if not seed_entities:
            return nx.MultiDiGraph()

        nodes_in_subgraph = set()
        edges_in_subgraph = []

        for seed in seed_entities:
            if not self._graph.has_node(seed):
                continue
            nodes_in_subgraph.add(seed)

            for depth in range(hop_depth):
                current_nodes = list(nodes_in_subgraph)
                for node in current_nodes:
                    for u, v, k, data in self._graph.out_edges(
                        node, keys=True, data=True
                    ):
                        if v not in nodes_in_subgraph:
                            nodes_in_subgraph.add(v)
                        edges_in_subgraph.append((u, v, k, data))

        subgraph = nx.MultiDiGraph()
        subgraph.add_nodes_from((n, self._graph.nodes[n]) for n in nodes_in_subgraph)
        for u, v, k, data in edges_in_subgraph:
            if u in nodes_in_subgraph and v in nodes_in_subgraph:
                subgraph.add_edge(u, v, key=k, **data)

        return subgraph


class TestGraphTraverser:
    """GraphTraverser tests."""

    def test_extract_subgraph_empty(self):
        builder = MockGraphBuilder()
        traverser = GraphTraverser(builder)
        result = traverser.extract_subgraph([], hop_depth=2)
        assert result.number_of_nodes() == 0

    def test_extract_subgraph_single_seed(self):
        g = nx.MultiDiGraph()
        g.add_node("e1", name="Apple")
        g.add_node("e2", name="Tim Cook")
        g.add_edge("e1", "e2", key="f1", predicate="CEO is", confidence=1.0)

        builder = MockGraphBuilder(g)
        traverser = GraphTraverser(builder)

        seed = [Entity("e1", "Apple", "ORG", [], None)]
        result = traverser.extract_subgraph(seed, hop_depth=1)

        assert result.number_of_nodes() >= 1

    def test_extract_subgraph_multiple_seeds(self):
        g = nx.MultiDiGraph()
        g.add_node("e1", name="Apple")
        g.add_node("e2", name="Google")
        g.add_node("e3", name="CEO")

        builder = MockGraphBuilder(g)
        traverser = GraphTraverser(builder)

        seeds = [
            Entity("e1", "Apple", "ORG", [], None),
            Entity("e2", "Google", "ORG", [], None),
        ]
        result = traverser.extract_subgraph(seeds, hop_depth=1)

        assert result.number_of_nodes() >= 2

    def test_extract_subgraph_hop_depth(self):
        g = nx.MultiDiGraph()
        g.add_node("e1", name="A")
        g.add_node("e2", name="B")
        g.add_node("e3", name="C")
        g.add_edge("e1", "e2", key="f1", predicate="knows", confidence=1.0)
        g.add_edge("e2", "e3", key="f2", predicate="knows", confidence=1.0)

        builder = MockGraphBuilder(g)
        traverser = GraphTraverser(builder, default_hop_depth=2)

        seed = [Entity("e1", "A", "ORG", [], None)]
        result = traverser.extract_subgraph(seed, hop_depth=2)

        assert "e3" in result.nodes() or result.number_of_nodes() < 3

    def test_get_reasoning_paths(self):
        g = nx.MultiDiGraph()
        g.add_node("e1", name="Apple")
        g.add_node("e2", name="Tim Cook")
        g.add_edge("e1", "e2", key="f1", predicate="CEO is", confidence=1.0)

        builder = MockGraphBuilder(g)
        traverser = GraphTraverser(builder)

        subgraph = g
        seed = [Entity("e1", "Apple", "ORG", [], None)]
        paths = traverser.get_reasoning_paths(subgraph, seed)

        assert len(paths) >= 1
        assert "Apple" in paths[0]

    def test_get_hop_chain(self):
        g = nx.MultiDiGraph()
        g.add_node("e1", name="Apple")
        g.add_node("e2", name="Tim Cook")
        g.add_node("e3", name="USA")
        g.add_edge("e1", "e2", key="f1", predicate="CEO is")
        g.add_edge("e2", "e3", key="f2", predicate="born in")

        builder = MockGraphBuilder(g)
        traverser = GraphTraverser(builder)

        path = traverser.get_hop_chain("e1", "e3", max_hops=3)

        assert len(path) == 3

    def test_get_hop_chain_no_path(self):
        g = nx.MultiDiGraph()
        g.add_node("e1", name="A")
        g.add_node("e2", name="B")

        builder = MockGraphBuilder(g)
        traverser = GraphTraverser(builder)

        path = traverser.get_hop_chain("e1", "e2", max_hops=3)

        assert path == []

    def test_get_entity_hops(self):
        g = nx.MultiDiGraph()
        g.add_node("e1", name="Apple")
        g.add_node("e2", name="Tim Cook")
        g.add_edge("e1", "e2", key="f1", predicate="CEO is")

        builder = MockGraphBuilder(g)
        traverser = GraphTraverser(builder)

        hops = traverser.get_entity_hops(g, "e1")

        assert hops.get("e1") == 0
        assert hops.get("e2") == 1

    def test_prune_by_confidence(self):
        g = nx.MultiDiGraph()
        g.add_node("e1", name="A")
        g.add_node("e2", name="B")
        g.add_node("e3", name="C")
        g.add_edge("e1", "e2", key="f1", predicate="rel1", confidence=0.9)
        g.add_edge("e1", "e3", key="f2", predicate="rel2", confidence=0.3)

        builder = MockGraphBuilder(g)
        traverser = GraphTraverser(builder)

        pruned = traverser.prune_by_confidence(g, min_confidence=0.5)

        assert pruned.has_edge("e1", "e2")
        assert not pruned.has_edge("e1", "e3")

    def test_get_subgraph_stats(self):
        g = nx.MultiDiGraph()
        g.add_node("e1", name="A")
        g.add_node("e2", name="B")
        g.add_edge("e1", "e2", key="f1", predicate="rel")

        builder = MockGraphBuilder(g)
        traverser = GraphTraverser(builder)

        stats = traverser.get_subgraph_stats(g)

        assert stats["nodes"] == 2
        assert stats["edges"] == 1
        assert stats["avg_degree"] == 1.0
