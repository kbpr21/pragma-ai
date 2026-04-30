import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx

from pragma.graph.builder import GraphBuilder
from pragma.models import AtomicFact

logger = logging.getLogger(__name__)


class FactAssembler:
    """Assemble facts from subgraph for LLM synthesis."""

    def __init__(
        self,
        graph_builder: GraphBuilder,
        min_confidence: float = 0.5,
        max_tokens: int = 600,
    ) -> None:
        self.graph_builder = graph_builder
        self.min_confidence = min_confidence
        self.max_tokens = max_tokens

    def assemble_facts(
        self,
        subgraph: nx.MultiDiGraph,
        as_of: datetime = None,
    ) -> List[Dict[str, Any]]:
        """Assemble facts from subgraph edges.

        Args:
            subgraph: Extracted subgraph from traversal
            as_of: Filter facts valid at this point in time

        Returns:
            List of fact dicts ready for LLM prompt
        """
        if not subgraph.edges():
            return []

        entity_ids = list(subgraph.nodes())

        if as_of:
            all_facts = self._get_facts_as_of(entity_ids, as_of)
            return self._convert_facts_to_dicts(all_facts, subgraph)

        all_facts = []

        for u, v, key, edge_data in subgraph.edges(keys=True, data=True):
            subject_id = u
            object_id = v
            predicate = edge_data.get("predicate", "related to")

            subject_facts = self._get_facts_for_entity(subject_id, "subject")
            object_facts = self._get_facts_for_entity(object_id, "object")

            combined = subject_facts + object_facts

            for fact in combined:
                fact_dict = {
                    "id": fact.id,
                    "subject_id": fact.subject_id,
                    "predicate": fact.predicate,
                    "object_id": fact.object_id,
                    "object_value": fact.object_value,
                    "context": fact.context,
                    "source_doc": fact.source_doc,
                    "source_page": fact.source_page,
                    "confidence": fact.confidence,
                    "ingested_at": fact.ingested_at,
                    "is_active": fact.is_active,
                    "_edge_predicate": predicate,
                }
                all_facts.append(fact_dict)

        filtered = self._filter_facts(all_facts)
        deduplicated = self._deduplicate_facts(filtered)
        sorted_facts = self._sort_facts(deduplicated)
        trimmed = self._trim_by_token_budget(sorted_facts)

        return trimmed

    def _get_facts_for_entity(
        self,
        entity_id: str,
        role: str = "subject",
    ) -> List[AtomicFact]:
        """Get facts for an entity from storage."""
        try:
            if role == "subject":
                return self.graph_builder.storage.get_facts_by_subject(entity_id)
            elif role == "object":
                return self.graph_builder.storage.get_facts_by_object(entity_id)
            else:
                return []
        except Exception as e:
            logger.warning(f"Failed to get facts for {entity_id}: {e}")
            return []

    def _get_facts_as_of(
        self,
        entity_ids: List[str],
        date: datetime,
    ) -> List[AtomicFact]:
        """Get facts valid at a specific point in time."""
        try:
            return self.graph_builder.storage.get_facts_as_of(entity_ids, date)
        except Exception as e:
            logger.warning(f"Failed temporal query: {e}")
            return []

    def _convert_facts_to_dicts(
        self,
        facts: List[AtomicFact],
        subgraph: nx.MultiDiGraph,
    ) -> List[Dict[str, Any]]:
        """Convert AtomicFact objects to dicts with edge predicates."""
        result = []
        for u, v, key, edge_data in subgraph.edges(keys=True, data=True):
            subject_id = u
            predicate = edge_data.get("predicate", "related to")

            for fact in facts:
                if fact.subject_id != subject_id:
                    continue
                fact_dict = {
                    "id": fact.id,
                    "subject_id": fact.subject_id,
                    "predicate": fact.predicate,
                    "object_id": fact.object_id,
                    "object_value": fact.object_value,
                    "context": fact.context,
                    "source_doc": fact.source_doc,
                    "source_page": fact.source_page,
                    "confidence": fact.confidence,
                    "ingested_at": fact.ingested_at,
                    "is_active": fact.is_active,
                    "_edge_predicate": predicate,
                }
                result.append(fact_dict)
        return result

    def _filter_facts(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter facts by active status and confidence."""
        filtered = []

        for fact in facts:
            if not fact.get("is_active", True):
                continue

            confidence = fact.get("confidence", 1.0)
            if confidence < self.min_confidence:
                continue

            filtered.append(fact)

        return filtered

    def _deduplicate_facts(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deduplicate facts, keep highest confidence."""
        seen: Dict[Tuple[str, str, str], Dict[str, Any]] = {}

        for fact in facts:
            key = (
                fact.get("subject_id", ""),
                fact.get("predicate", ""),
                str(fact.get("object_id", "")) + str(fact.get("object_value", "")),
            )

            if key not in seen:
                seen[key] = fact
            else:
                existing_confidence = seen[key].get("confidence", 0)
                new_confidence = fact.get("confidence", 0)
                if new_confidence > existing_confidence:
                    seen[key] = fact

        return list(seen.values())

    def _sort_facts(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Sort by confidence DESC, then ingested_at DESC."""
        return sorted(
            facts,
            key=lambda f: (
                f.get("confidence", 0),
                str(f.get("ingested_at", "")),
            ),
            reverse=True,
        )

    def _trim_by_token_budget(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Trim facts to fit token budget."""
        if not facts:
            return facts

        total_tokens = 0
        kept_facts = []

        for fact in facts:
            fact_str = self.format_fact_dict(fact, index=len(kept_facts))
            estimated_tokens = len(fact_str.split()) * 1.3

            if total_tokens + estimated_tokens > self.max_tokens:
                break

            kept_facts.append(fact)
            total_tokens += estimated_tokens

        return kept_facts

    def format_fact_dict(
        self,
        fact: Dict[str, Any],
        index: int,
    ) -> str:
        """Render a fact compactly for prompt-size estimation.

        Mirrors :meth:`pragma.query.synthesizer.AnswerSynthesizer._format_fact`
        so the assembler's token-budget estimate matches what's actually sent
        to the LLM. The format is intentionally simple
        (``F<idx>: <subject> -- <predicate> --> <object>``) and excludes the
        confidence value -- that's metadata pragma uses internally and the
        LLM does not need to see it.
        """
        fid = f"F{index + 1}"
        subject = self._get_entity_name(fact.get("subject_id"))
        predicate = fact.get("predicate", "related to")
        object_id = fact.get("object_id")
        object_value = fact.get("object_value")
        object_val = self._get_entity_name(object_id) if object_id else object_value
        return f"{fid}: {subject} -- {predicate} --> {object_val}"

    def _get_entity_name(self, entity_id: Optional[str]) -> str:
        """Get entity name by ID; returns the id (or ``unknown``) on failure."""
        if not entity_id:
            return "unknown"
        try:
            entity = self.graph_builder.storage.get_entity_by_id(entity_id)
            return entity.name if entity else entity_id
        except Exception:  # noqa: BLE001
            return entity_id
