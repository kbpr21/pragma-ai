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
