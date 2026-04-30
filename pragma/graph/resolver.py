import logging
import uuid
from typing import Any, List, Optional

from pragma.models import Entity

logger = logging.getLogger(__name__)


class EntityResolver:
    """Resolve entities with fuzzy matching and aliasing."""

    def __init__(
        self,
        storage: Any,
        fuzzy_threshold: int = 85,
    ) -> None:
        self.storage = storage
        self.fuzzy_threshold = fuzzy_threshold
        self._init_fuzzy()

    def _init_fuzzy(self) -> None:
        try:
            from rapidfuzz import fuzz

            self._fuzz = fuzz
        except ImportError:
            logger.warning("rapidfuzz not installed. Using slow fuzzy matching.")
            self._fuzz = None

    def resolve(
        self,
        name: str,
        entity_type: Optional[str] = None,
    ) -> Entity:
        """Resolve an entity by name, with fuzzy matching.

        Strategies (in order):
        1. Exact match (case-insensitive)
        2. Alias lookup
        3. Fuzzy match (rapidfuzz token_sort_ratio >= threshold)
        4. Create new entity

        Args:
            name: Entity name to resolve
            entity_type: Optional entity type (PERSON, ORG, CONCEPT, etc.)

        Returns:
            Entity object (new or existing)
        """
        if not name or not name.strip():
            return self._create_entity("unknown", entity_type)

        name = name.strip()

        entity = self._exact_match(name)
        if entity:
            logger.debug(f"Exact match: {name} -> {entity.id}")
            if entity_type and entity.entity_type != entity_type:
                entity.entity_type = entity_type
                self.storage.save_entity(
                    entity.id, entity.name, entity.entity_type, entity.aliases
                )
            return entity

        entity = self._alias_lookup(name)
        if entity:
            logger.debug(f"Alias match: {name} -> {entity.id}")
            return entity

        entity = self._fuzzy_match(name)
        if entity:
            logger.debug(f"Fuzzy match: {name} -> {entity.id}")
            new_aliases = entity.aliases + [name]
            self.storage.save_entity(
                entity.id, entity.name, entity.entity_type, new_aliases
            )
            return entity

        return self._create_entity(name, entity_type)

    def _exact_match(self, name: str) -> Optional[Entity]:
        """Strategy 1: Exact match (case-insensitive)."""
        entity = self.storage.get_entity_by_name(name)
        if entity:
            return entity
        entity = self.storage.get_entity_by_name(name.lower())
        return entity

    def _alias_lookup(self, name: str) -> Optional[Entity]:
        """Strategy 2: Alias lookup (scan aliases JSON array)."""
        all_entities = self.storage.get_all_entities()
        name_lower = name.lower()

        for entity in all_entities:
            if not entity.aliases:
                continue
            aliases = [a.lower() for a in entity.aliases]
            if name_lower in aliases:
                return entity
            for alias in entity.aliases:
                if alias.lower() == name_lower:
                    return entity

        return None

    def _fuzzy_match(self, name: str) -> Optional[Entity]:
        """Strategy 3: Fuzzy match with rapidfuzz."""
        if self._fuzz is None:
            return self._slow_fuzzy_match(name)

        all_entities = self.storage.get_all_entities()
        if not all_entities:
            return None

        name_lower = name.lower()
        best_match = None
        best_score = 0

        for entity in all_entities:
            scores = [
                self._fuzz.token_sort_ratio(name_lower, entity.name.lower()),
            ]
            scores.extend(
                self._fuzz.token_sort_ratio(name_lower, alias.lower())
                for alias in entity.aliases
            )

            max_score = max(scores) if scores else 0
            if max_score >= self.fuzzy_threshold and max_score > best_score:
                best_score = max_score
                best_match = entity

        return best_match

    def _slow_fuzzy_match(self, name: str) -> Optional[Entity]:
        """Fallback fuzzy matching without rapidfuzz."""
        all_entities = self.storage.get_all_entities()
        if not all_entities:
            return None

        name_lower = name.lower().split()
        best_match = None
        best_score = 0

        for entity in all_entities:
            entity_words = entity.name.lower().split()
            common = set(name_lower) & set(entity_words)
            score = len(common) / max(len(name_lower), len(entity_words), 1) * 100

            if score >= self.fuzzy_threshold and score > best_score:
                best_score = score
                best_match = entity

        return best_match

    def _create_entity(self, name: str, entity_type: Optional[str]) -> Entity:
        """Strategy 4: Create new entity."""
        entity_id = str(uuid.uuid4())
        self.storage.save_entity(entity_id, name, entity_type, [])
        return Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            aliases=[],
        )

    def merge_entities(
        self,
        entity_a: Entity,
        entity_b: Entity,
    ) -> Entity:
        """Merge two entities, consolidating facts and aliases.

        The merged entity keeps entity_a's ID and adds entity_b as an alias.

        Args:
            entity_a: Primary entity (keeps ID)
            entity_b: Secondary entity (becomes alias)

        Returns:
            Merged entity (entity_a with updated aliases)
        """
        merged_aliases = list(
            set(entity_a.aliases + [entity_b.name] + entity_b.aliases)
        )

        self.storage.save_entity(
            entity_a.id,
            entity_a.name,
            entity_a.entity_type or entity_b.entity_type,
            merged_aliases,
        )

        return Entity(
            id=entity_a.id,
            name=entity_a.name,
            entity_type=entity_a.entity_type or entity_b.entity_type,
            aliases=merged_aliases,
        )

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities by name/alias substring match."""
        all_entities = self.storage.get_all_entities()
        query_lower = query.lower()

        matches = [
            e
            for e in all_entities
            if query_lower in e.name.lower()
            or any(query_lower in a.lower() for a in e.aliases)
        ]

        return matches[:limit]
