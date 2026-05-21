import logging
import re
import uuid
from typing import Any, List, Optional

from pragma.models import Entity
from pragma.graph.synonyms import SynonymDictionary

logger = logging.getLogger(__name__)

# Corporate suffixes to strip during normalization.
_CORP_SUFFIXES = re.compile(
    r"\b(inc\.?|corp\.?|ltd\.?|llc|co\.?|plc|gmbh|s\.?a\.?|ag)\s*$",
    re.IGNORECASE,
)


class EntityResolver:
    """Resolve entities with normalization, fuzzy matching, and aliasing.

    Resolution pipeline (v2.0):
        0. Normalize name (synonym expansion, case, suffix stripping)
        1. Exact match (case-insensitive)
        2. Alias lookup
        3. Fuzzy match (rapidfuzz token_sort_ratio >= threshold)
        4. Create new entity
    """

    def __init__(
        self,
        storage: Any,
        fuzzy_threshold: int = 85,
        synonym_dict_path: Optional[str] = None,
    ) -> None:
        self.storage = storage
        self.fuzzy_threshold = fuzzy_threshold
        self._synonyms = SynonymDictionary(user_dict_path=synonym_dict_path)
        self._init_fuzzy()
        self._entities_cache: Optional[List[Entity]] = None

    def _get_cached_entities(self) -> List[Entity]:
        if self._entities_cache is None:
            self._entities_cache = self.storage.get_all_entities()
        return self._entities_cache

    def _init_fuzzy(self) -> None:
        try:
            from rapidfuzz import fuzz

            self._fuzz = fuzz
        except ImportError:
            logger.warning("rapidfuzz not installed. Using slow fuzzy matching.")
            self._fuzz = None

    # ------------------------------------------------------------------
    # Normalization pipeline (v2.0)
    # ------------------------------------------------------------------

    def normalize(self, name: str) -> str:
        """Apply the normalization pipeline to an entity name.

        Steps:
        1. Strip whitespace and collapse multiple spaces
        2. Expand known abbreviations/synonyms ("ML" → "Machine Learning")
        3. Strip corporate suffixes ("Inc.", "Corp.", "Ltd.")
        4. Title-case for consistent storage
        """
        if not name:
            return name

        # Step 1: whitespace normalization.
        cleaned = " ".join(name.split()).strip()

        # Step 2: synonym expansion (check the full name first, then
        # individual tokens for abbreviation expansion).
        expansion = self._synonyms.expand(cleaned)
        if expansion is not None:
            if expansion:  # non-empty expansion
                cleaned = expansion
            else:
                # Empty expansion means it's a stripped suffix — keep as is
                pass

        # Step 3: strip corporate suffixes.
        cleaned = _CORP_SUFFIXES.sub("", cleaned).strip()

        # Step 4: consistent casing.
        # - Preserve all-uppercase tokens that look like acronyms (<=5 chars)
        # - Preserve words with internal capitals (camelCase, PascalCase)
        # - Title-case only fully lowercase words
        words = cleaned.split()
        normalized_words = []
        for word in words:
            if word.isupper() and len(word) <= 5:
                normalized_words.append(word)  # preserve acronyms
            elif any(c.isupper() for c in word[1:]):
                normalized_words.append(word)  # preserve internal caps
            elif word.islower():
                normalized_words.append(word.capitalize())
            else:
                normalized_words.append(word)  # preserve as-is
        cleaned = " ".join(normalized_words)

        return cleaned.strip() or name.strip()

    def resolve(
        self,
        name: str,
        entity_type: Optional[str] = None,
    ) -> Entity:
        """Resolve an entity by name, with normalization and fuzzy matching.

        Strategies (in order):
        0. Normalize name (synonym expansion, case, suffix stripping)
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

        # Step 0: normalize before any matching.
        original_name = name.strip()
        name = self.normalize(original_name)

        entity = self._exact_match(name)
        if entity:
            logger.debug(f"Exact match: {name} -> {entity.id}")
            if entity_type and entity.entity_type != entity_type:
                entity.entity_type = entity_type
                self.storage.save_entity(
                    entity.id, entity.name, entity.entity_type, entity.aliases
                )
                self._entities_cache = None  # Invalidate cache
            return entity

        # Also try matching the original (un-normalized) name.
        if name != original_name:
            entity = self._exact_match(original_name)
            if entity:
                logger.debug(f"Exact match (original): {original_name} -> {entity.id}")
                return entity

        entity = self._alias_lookup(name)
        if entity:
            logger.debug(f"Alias match: {name} -> {entity.id}")
            return entity

        entity = self._fuzzy_match(name)
        if entity:
            logger.debug(f"Fuzzy match: {name} -> {entity.id}")
            # Add both normalized and original as aliases.
            new_aliases = list(set(entity.aliases + [name, original_name]))
            self.storage.save_entity(
                entity.id, entity.name, entity.entity_type, new_aliases
            )
            self._entities_cache = None  # Invalidate cache
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
        """Strategy 2: Alias lookup (point query on indexed aliases)."""
        if hasattr(self.storage, "get_entity_id_by_alias") and hasattr(
            self.storage, "get_entity_by_id"
        ):
            entity_id = self.storage.get_entity_id_by_alias(name)
            if entity_id:
                return self.storage.get_entity_by_id(entity_id)
            return None

        # Fallback for storage backends without point query support (e.g. MockStorage)
        all_entities = self._get_cached_entities()
        name_lower = name.lower()
        for entity in all_entities:
            if not entity.aliases:
                continue
            if any(a.lower() == name_lower for a in entity.aliases):
                return entity
        return None

    def _fuzzy_match(self, name: str) -> Optional[Entity]:
        """Strategy 3: Fuzzy match with rapidfuzz."""
        if self._fuzz is None:
            return self._slow_fuzzy_match(name)

        all_entities = self._get_cached_entities()
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
        all_entities = self._get_cached_entities()
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
        self._entities_cache = None  # Invalidate cache
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
        self._entities_cache = None  # Invalidate cache

        return Entity(
            id=entity_a.id,
            name=entity_a.name,
            entity_type=entity_a.entity_type or entity_b.entity_type,
            aliases=merged_aliases,
        )

    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        """Search entities by name/alias substring match."""
        all_entities = self._get_cached_entities()
        query_lower = query.lower()

        matches = [
            e
            for e in all_entities
            if query_lower in e.name.lower()
            or any(query_lower in a.lower() for a in e.aliases)
        ]

        return matches[:limit]
