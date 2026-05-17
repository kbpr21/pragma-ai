import logging
from typing import Dict, List, Optional

from pragma.graph.builder import GraphBuilder
from pragma.models import Entity

logger = logging.getLogger(__name__)


class BM25Retriever:
    """BM25-based entity retrieval."""

    def __init__(
        self,
        graph_builder: GraphBuilder,
        top_k_per_question: int = 3,
        max_total_seeds: int = 10,
    ) -> None:
        self.graph_builder = graph_builder
        self.top_k_per_question = top_k_per_question
        self.max_total_seeds = max_total_seeds

    def find_seed_entities(
        self,
        sub_questions: List[str],
    ) -> List[Entity]:
        """Find seed entities for sub-questions.

        Args:
            sub_questions: List of sub-questions from decomposer

        Returns:
            List of entities (deduplicated)
        """
        if not sub_questions:
            return []

        all_entity_scores: Dict[str, float] = {}

        # For multi-question queries, allow more seeds so each
        # sub-question has a chance of finding its entities.
        effective_max_seeds = self.max_total_seeds
        if len(sub_questions) >= 3:
            effective_max_seeds = max(self.max_total_seeds, len(sub_questions) * 2)

        for question in sub_questions:
            if not question or not question.strip():
                continue

            entity_ids = self._search_question(question)

            for i, entity_id in enumerate(entity_ids):
                score = (self.top_k_per_question - i) / self.top_k_per_question
                if entity_id in all_entity_scores:
                    all_entity_scores[entity_id] += score
                else:
                    all_entity_scores[entity_id] = score

        sorted_entities = sorted(
            all_entity_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )

        top_entities = []
        for entity_id, score in sorted_entities[:effective_max_seeds]:
            entity = self._get_entity(entity_id)
            if entity:
                top_entities.append(entity)

        if not top_entities:
            logger.warning("BM25 found no entities for any sub-question")

        return top_entities

    def _search_question(self, question: str) -> List[str]:
        """Search for entities matching a single question."""
        try:
            entity_ids = self.graph_builder.search_entities_bm25(
                question,
                top_k=self.top_k_per_question,
            )
            return entity_ids
        except Exception as e:
            logger.warning(f"BM25 search failed for '{question}': {e}")
            return []

    def _get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID from storage."""
        try:
            return self.graph_builder.storage.get_entity_by_id(entity_id)
        except Exception:
            return None

    def find_seed_entities_simple(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Entity]:
        """Simple single-query entity search.

        Args:
            query: Search query
            top_k: Number of entities to return

        Returns:
            List of entities
        """
        entity_ids = self.graph_builder.search_entities_bm25(query, top_k=top_k)

        entities = []
        for entity_id in entity_ids:
            entity = self._get_entity(entity_id)
            if entity:
                entities.append(entity)

        return entities


class HybridRetriever:
    """Fuses BM25 keyword retrieval with semantic vector retrieval
    using Reciprocal Rank Fusion (RRF).

    When the semantic retriever is unavailable (``sentence-transformers``
    not installed or embeddings disabled), silently degrades to
    BM25-only retrieval — identical to ``BM25Retriever``.

    RRF score for a document d across rankings R1..Rn:
        RRF(d) = Σ 1 / (k + rank_i(d))
    where k is a smoothing constant (default 60).
    """

    def __init__(
        self,
        graph_builder: GraphBuilder,
        semantic_retriever: Optional["SemanticRetriever"] = None,
        top_k_per_question: int = 3,
        max_total_seeds: int = 10,
        rrf_k: int = 60,
        semantic_top_k: int = 10,
    ) -> None:
        self.bm25 = BM25Retriever(
            graph_builder,
            top_k_per_question=top_k_per_question,
            max_total_seeds=max_total_seeds,
        )
        self.semantic = semantic_retriever
        self.rrf_k = rrf_k
        self.semantic_top_k = semantic_top_k
        self.graph_builder = graph_builder

    def find_seed_entities(
        self,
        sub_questions: List[str],
    ) -> List[Entity]:
        """Find seed entities by fusing BM25 + semantic retrieval."""
        # BM25 path — always runs.
        bm25_entities = self.bm25.find_seed_entities(sub_questions)

        # Semantic path — only runs if available.
        if self.semantic is None or not self.semantic.available:
            return bm25_entities

        # Get semantic fact hits for each sub-question.
        semantic_fact_ids: Dict[str, float] = {}
        for question in sub_questions:
            if not question or not question.strip():
                continue
            hits = self.semantic.query(question, top_k=self.semantic_top_k)
            for rank, (fact_id, sim) in enumerate(hits):
                if fact_id not in semantic_fact_ids:
                    semantic_fact_ids[fact_id] = 0.0
                # RRF score contribution from this ranking.
                semantic_fact_ids[fact_id] += 1.0 / (self.rrf_k + rank + 1)

        if not semantic_fact_ids:
            return bm25_entities

        # Map semantic fact_ids → entity_ids via storage.
        semantic_entity_scores: Dict[str, float] = {}
        try:
            conn = self.graph_builder.storage._get_connection()
            for fact_id, rrf_score in semantic_fact_ids.items():
                row = conn.execute(
                    "SELECT subject_id, object_id FROM facts WHERE id = ?",
                    (fact_id,),
                ).fetchone()
                if row:
                    for eid in (row["subject_id"], row["object_id"]):
                        if eid:
                            semantic_entity_scores[eid] = (
                                semantic_entity_scores.get(eid, 0.0) + rrf_score
                            )
        except Exception as e:  # noqa: BLE001
            logger.warning("Hybrid retrieval: fact→entity mapping failed: %s", e)
            return bm25_entities

        # RRF fusion: combine BM25 entity ranks with semantic entity scores.
        fused_scores: Dict[str, float] = {}

        # BM25 entities: assign RRF scores by rank position.
        for rank, entity in enumerate(bm25_entities):
            fused_scores[entity.id] = 1.0 / (self.rrf_k + rank + 1)

        # Merge semantic scores.
        for eid, score in semantic_entity_scores.items():
            fused_scores[eid] = fused_scores.get(eid, 0.0) + score

        # Sort by fused score, retrieve Entity objects.
        sorted_ids = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # Deduplicate: keep order, cap at max_total_seeds.
        seen = set()
        result: List[Entity] = []
        max_seeds = self.bm25.max_total_seeds
        if len(sub_questions) >= 3:
            max_seeds = max(max_seeds, len(sub_questions) * 2)

        for eid, _ in sorted_ids:
            if eid in seen:
                continue
            seen.add(eid)
            # First check if we already have the Entity from BM25.
            entity = next((e for e in bm25_entities if e.id == eid), None)
            if entity is None:
                entity = self.bm25._get_entity(eid)
            if entity:
                result.append(entity)
            if len(result) >= max_seeds:
                break

        return result


# Type import for annotations only.
try:
    from pragma.query.semantic import SemanticRetriever  # noqa: F401
except ImportError:
    pass
