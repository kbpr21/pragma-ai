import logging
from typing import Any, Dict, List

import networkx as nx

from pragma.graph.builder import GraphBuilder
from pragma.models import Entity

logger = logging.getLogger(__name__)


class GraphTraverser:
    """Graph traversal for multi-hop reasoning."""

    def __init__(
        self,
        graph_builder: GraphBuilder,
        max_subgraph_nodes: int = 50,
        default_hop_depth: int = 2,
    ) -> None:
        self.graph_builder = graph_builder
        self.max_subgraph_nodes = max_subgraph_nodes
        self.default_hop_depth = default_hop_depth

    def extract_subgraph(
        self,
        seed_entities: List[Entity],
        hop_depth: int = None,
    ) -> nx.MultiDiGraph:
        """Extract subgraph around seed entities.

        Args:
            seed_entities: Starting entities
            hop_depth: Maximum traversal depth (default: self.default_hop_depth)

        Returns:
            Subgraph containing seed entities and neighbors
        """
        if hop_depth is None:
            hop_depth = self.default_hop_depth

        if not seed_entities:
            return nx.MultiDiGraph()

        seed_ids = [e.id for e in seed_entities if e.id]
        if not seed_ids:
            return nx.MultiDiGraph()

        return self.graph_builder.get_subgraph(
            seed_entities=seed_ids,
            hop_depth=hop_depth,
            max_nodes=self.max_subgraph_nodes,
        )

    def get_reasoning_paths(
        self,
        subgraph: nx.MultiDiGraph,
        seed_entities: List[Entity],
    ) -> List[str]:
        """Convert subgraph edges to human-readable paths.

        Args:
            subgraph: The extracted subgraph
            seed_entities: Starting seed entities

        Returns:
            List of human-readable path strings
        """
        if not subgraph.edges():
            return []

        paths = []

        for u, v, data in subgraph.edges(data=True):
            u_name = subgraph.nodes[u].get("name", u) if u in subgraph.nodes else u
            v_name = subgraph.nodes[v].get("name", v) if v in subgraph.nodes else v
            predicate = data.get("predicate", "related to")

            path = f"{u_name} [{predicate}] {v_name}"
            paths.append(path)

        return paths

    def get_hop_chain(
        self,
        entity_a_id: str,
        entity_b_id: str,
        max_hops: int = 3,
    ) -> List[str]:
        """Find shortest path between two entities.

        Args:
            entity_a_id: Source entity ID
            entity_b_id: Target entity ID
            max_hops: Maximum hops to search

        Returns:
            List of entity names forming the path
        """
        g = self.graph_builder.graph

        if not g.has_node(entity_a_id) or not g.has_node(entity_b_id):
            return []

        try:
            path = nx.shortest_path(g, entity_a_id, entity_b_id)
            if len(path) - 1 > max_hops:
                return []
        except nx.NetworkXNoPath:
            return []

        return [g.nodes[n].get("name", n) for n in path]

    def get_entity_hops(
        self,
        subgraph: nx.MultiDiGraph,
        seed_id: str,
    ) -> Dict[str, int]:
        """Get hop distance for each node from a seed.

        Args:
            subgraph: The subgraph to analyze
            seed_id: Starting entity ID

        Returns:
            Dict mapping entity_id to hop distance
        """
        if not subgraph.has_node(seed_id):
            return {}

        hops = {}
        try:
            lengths = nx.single_source_shortest_path_length(subgraph, seed_id)
            hops = {n: d for n, d in lengths.items()}
        except nx.NetworkXError:
            pass

        return hops

    def prune_by_confidence(
        self,
        subgraph: nx.MultiDiGraph,
        min_confidence: float = 0.5,
    ) -> nx.MultiDiGraph:
        """Prune subgraph by confidence threshold.

        Args:
            subgraph: Input subgraph
            min_confidence: Minimum confidence to keep

        Returns:
            Pruned subgraph
        """
        pruned = nx.MultiDiGraph()
        pruned.add_nodes_from(subgraph.nodes(data=True))

        for u, v, key, data in subgraph.edges(keys=True, data=True):
            confidence = data.get("confidence", 1.0)
            if confidence >= min_confidence:
                pruned.add_edge(u, v, key=key, **data)

        return pruned

    def get_subgraph_stats(self, subgraph: nx.MultiDiGraph) -> Dict[str, Any]:
        """Get statistics about a subgraph.

        Args:
            subgraph: The subgraph to analyze

        Returns:
            Dict with stats
        """
        return {
            "nodes": subgraph.number_of_nodes(),
            "edges": subgraph.number_of_edges(),
            "avg_degree": sum(dict(subgraph.degree()).values())
            / max(subgraph.number_of_nodes(), 1),
        }
