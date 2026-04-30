import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from pragma.models import AtomicFact, Entity

logger = logging.getLogger(__name__)


class GraphBuilder:
    """Build and maintain the knowledge graph using NetworkX."""

    def __init__(
        self,
        storage: Any,
        kb_dir: str = "./pragma_kb",
        entity_index: Optional[Dict[str, Entity]] = None,
    ) -> None:
        self.storage = storage
        self.kb_dir = Path(kb_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)

        self._graph: Optional[nx.MultiDiGraph] = None
        self._bm25_index: Optional[Any] = None
        self._entity_index = entity_index or {}

        self._graph_path = self.kb_dir / "graph.json"
        self._bm25_path = self.kb_dir / "bm25_index.pkl"

    @property
    def graph(self) -> nx.MultiDiGraph:
        """Get or create the graph."""
        if self._graph is None:
            self._graph = self._load_graph()
        return self._graph

    def _create_empty_graph(self) -> nx.MultiDiGraph:
        """Create an empty MultiDiGraph."""
        return nx.MultiDiGraph()

    def _load_graph(self) -> nx.MultiDiGraph:
        """Load graph from JSON or create new."""
        if self._graph_path.exists():
            try:
                with open(self._graph_path, encoding="utf-8") as f:
                    data = json.load(f)
                g = nx.node_link_graph(data, directed=True, multigraph=True)
                logger.info(f"Loaded graph with {g.number_of_nodes()} nodes")
                return g
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"Failed to load graph: {e}. Creating new.")
        return self._create_empty_graph()

    def save(self) -> None:
        """Serialize graph to JSON in kb_dir."""
        if self._graph is None:
            return

        data = nx.node_link_data(self._graph)
        with open(self._graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        logger.debug(f"Saved graph with {self._graph.number_of_nodes()} nodes")

    def load(self) -> nx.MultiDiGraph:
        """Deserialize graph from JSON."""
        self._graph = self._load_graph()
        return self._graph

    def add_entity(self, entity: Entity) -> None:
        """Add entity node to graph."""
        g = self.graph
        if not g.has_node(entity.id):
            g.add_node(
                entity.id,
                name=entity.name,
                entity_type=entity.entity_type,
                aliases=json.dumps(entity.aliases) if entity.aliases else "[]",
            )
            self._entity_index[entity.id] = entity
            self._invalidate_bm25()

    def add_fact(self, fact: AtomicFact) -> None:
        """Add fact as edge in NetworkX graph.

        Adds subject and object entities as nodes if not present,
        then adds edge with fact metadata.
        """
        g = self.graph

        if fact.subject_id and not g.has_node(fact.subject_id):
            subject_entity = self.storage.get_entity_by_id(fact.subject_id)
            if subject_entity:
                g.add_node(
                    fact.subject_id,
                    name=subject_entity.name,
                    entity_type=subject_entity.entity_type,
                    aliases=json.dumps(subject_entity.aliases)
                    if subject_entity.aliases
                    else "[]",
                )

        if fact.object_id and not g.has_node(fact.object_id):
            object_entity = self.storage.get_entity_by_id(fact.object_id)
            if object_entity:
                g.add_node(
                    fact.object_id,
                    name=object_entity.name,
                    entity_type=object_entity.entity_type,
                    aliases=json.dumps(object_entity.aliases)
                    if object_entity.aliases
                    else "[]",
                )

        edge_key = fact.id
        edge_attrs = {
            "predicate": fact.predicate,
            "fact_id": fact.id,
            "source_doc": fact.source_doc or "",
            "confidence": fact.confidence,
            "context": fact.context or "",
        }

        if fact.subject_id and fact.object_id:
            if g.has_edge(fact.subject_id, fact.object_id, key=edge_key):
                g.remove_edge(fact.subject_id, fact.object_id, key=edge_key)
            g.add_edge(fact.subject_id, fact.object_id, key=edge_key, **edge_attrs)

        self._invalidate_bm25()

    def remove_fact(self, fact_id: str) -> None:
        """Remove a fact edge from the graph."""
        g = self.graph
        edges_to_remove = [(u, v, k) for u, v, k in g.edges(keys=True) if k == fact_id]
        for u, v, k in edges_to_remove:
            g.remove_edge(u, v, key=k)
        self._invalidate_bm25()

    def get_neighbors(
        self,
        entity_id: str,
        max_depth: int = 1,
    ) -> List[Tuple[str, Dict[str, Any]]]:
        """Get neighboring entities up to max_depth."""
        g = self.graph
        if not g.has_node(entity_id):
            return []

        neighbors = []
        for depth in range(1, max_depth + 1):
            if depth == 1:
                nodes = [entity_id]
            else:
                nodes = [
                    n
                    for n in g.nodes()
                    if nx.shortest_path_length(g, entity_id, n) == depth - 1
                ]

            for node in nodes:
                for _, neighbor, data in g.out_edges(node, data=True):
                    if (neighbor, data) not in neighbors:
                        neighbors.append((neighbor, data))

        return neighbors

    def get_subgraph(
        self,
        seed_entities: List[str],
        hop_depth: int = 2,
        max_nodes: int = 50,
    ) -> nx.MultiDiGraph:
        """Extract subgraph around seed entities.

        Args:
            seed_entities: Starting entity IDs
            hop_depth: Maximum traversal depth
            max_nodes: Maximum nodes to include

        Returns:
            Subgraph containing seed entities and neighbors
        """
        g = self.graph
        if not seed_entities:
            return self._create_empty_graph()

        nodes_in_subgraph = set()
        edges_in_subgraph = []

        for seed in seed_entities:
            if not g.has_node(seed):
                continue
            nodes_in_subgraph.add(seed)

            for depth in range(hop_depth):
                current_nodes = list(nodes_in_subgraph)
                for node in current_nodes:
                    for u, v, k, data in g.out_edges(node, keys=True, data=True):
                        if v not in nodes_in_subgraph:
                            nodes_in_subgraph.add(v)
                        edges_in_subgraph.append((u, v, k, data))

                    for u, v, k, data in g.in_edges(node, keys=True, data=True):
                        if u not in nodes_in_subgraph:
                            nodes_in_subgraph.add(u)
                        edges_in_subgraph.append((u, v, k, data))

                if len(nodes_in_subgraph) >= max_nodes:
                    break

        subgraph = nx.MultiDiGraph()
        subgraph.add_nodes_from((n, g.nodes[n]) for n in nodes_in_subgraph)
        for u, v, k, data in edges_in_subgraph:
            if u in nodes_in_subgraph and v in nodes_in_subgraph:
                subgraph.add_edge(u, v, key=k, **data)

        return subgraph

    def rebuild_bm25_index(self) -> None:
        """Rebuild BM25 index over all entity names."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank_bm25 not installed")
            return

        g = self.graph
        entities = []

        for node_id in g.nodes():
            node_data = g.nodes[node_id]
            name = node_data.get("name", "")
            aliases_raw = node_data.get("aliases", "[]")
            try:
                aliases = (
                    json.loads(aliases_raw)
                    if isinstance(aliases_raw, str)
                    else aliases_raw
                )
            except json.JSONDecodeError:
                aliases = []
            entities.append({"id": node_id, "name": name, "aliases": aliases})

        if not entities:
            self._bm25_index = None
            return

        tokenized_corpus = []
        for entity in entities:
            tokens = (
                (entity["name"] + " " + " ".join(entity.get("aliases", [])))
                .lower()
                .split()
            )
            tokenized_corpus.append(tokens)

        self._bm25_index = BM25Okapi(tokenized_corpus)
        self._bm25_entity_ids = [e["id"] for e in entities]

    def save_bm25_index(self) -> None:
        """Save BM25 index to pickle file."""
        if self._bm25_index is None:
            return

        data = {
            "index": self._bm25_index,
            "entity_ids": getattr(self, "_bm25_entity_ids", []),
        }

        with open(self._bm25_path, "wb") as f:
            pickle.dump(data, f)
        logger.debug("Saved BM25 index")

    def load_bm25_index(self) -> None:
        """Load BM25 index from pickle file."""
        if not self._bm25_path.exists():
            return

        try:
            with open(self._bm25_path, "rb") as f:
                data = pickle.load(f)
            self._bm25_index = data["index"]
            self._bm25_entity_ids = data.get("entity_ids", [])
            logger.debug("Loaded BM25 index")
        except Exception as e:
            logger.warning(f"Failed to load BM25 index: {e}")

    def search_entities_bm25(self, query: str, top_k: int = 5) -> List[str]:
        """Search entities using BM25 index.

        Args:
            query: Search query
            top_k: Number of results to return

        Returns:
            List of entity IDs
        """
        if self._bm25_index is None:
            self.rebuild_bm25_index()

        if self._bm25_index is None:
            return []

        tokenized_query = query.lower().split()
        scores = self._bm25_index.get_scores(tokenized_query)

        scored = list(enumerate(scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in scored[:top_k]:
            if hasattr(self, "_bm25_entity_ids") and idx < len(self._bm25_entity_ids):
                results.append(self._bm25_entity_ids[idx])

        return results

    def _invalidate_bm25(self) -> None:
        """Invalidate BM25 index after graph changes."""
        self._bm25_index = None

    def stats(self) -> Dict[str, int]:
        """Get graph statistics."""
        g = self.graph
        return {
            "nodes": g.number_of_nodes(),
            "edges": g.number_of_edges(),
        }

    def clear(self) -> None:
        """Clear the graph."""
        self._graph = self._create_empty_graph()
        self._bm25_index = None
        self._entity_index = {}
