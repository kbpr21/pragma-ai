import pytest
from typing import List, Optional

from pragma.graph.resolver import EntityResolver
from pragma.models import Entity


class MockStorage:
    """Mock storage for testing EntityResolver."""

    def __init__(self):
        self.entities: dict[str, Entity] = {}
        self.save_called = []

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
        self.save_called.append(("save", entity_id, name, entity_type, aliases))
        return entity_id

    def get_entity_by_name(self, name: str) -> Optional[Entity]:
        for entity in self.entities.values():
            if entity.name == name or entity.name.lower() == name.lower():
                return entity
        return None

    def get_all_entities(self) -> List[Entity]:
        return list(self.entities.values())


class TestEntityResolverExactMatch:
    """Tests for exact match resolution."""

    def test_exact_match_returns_existing(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple", "ORG", ["Apple Inc"])

        resolver = EntityResolver(storage)
        result = resolver.resolve("Apple")

        assert result.name == "Apple"

    def test_exact_match_case_insensitive(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple", "ORG", [])

        resolver = EntityResolver(storage)
        result = resolver.resolve("apple")

        assert result.name == "Apple"

    def test_exact_match_empty_name(self):
        storage = MockStorage()

        resolver = EntityResolver(storage)
        result = resolver.resolve("")

        assert result.name == "unknown"


class TestEntityResolverAlias:
    """Tests for alias lookup resolution."""

    def test_alias_lookup_finds_entity(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple", "ORG", ["Apple Inc", "AAPL"])

        resolver = EntityResolver(storage)
        result = resolver.resolve("Apple Inc")

        assert result.name == "Apple"

    def test_alias_lookup_case_insensitive(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple", "ORG", ["Apple Inc"])

        resolver = EntityResolver(storage)
        result = resolver.resolve("APPLE INC")

        assert result.name == "Apple"

    def test_alias_not_found_continues(self):
        storage = MockStorage()
        storage.save_entity("id1", "Google", "ORG", ["Alphabet"])

        resolver = EntityResolver(storage)
        result = resolver.resolve("Microsoft")

        assert result.name == "Microsoft"
        assert result.id != "id1"


class TestEntityResolverFuzzy:
    """Tests for fuzzy match resolution."""

    def test_fuzzy_match_threshold_requires_rapidfuzz(self):
        pytest.importorskip("rapidfuzz")

        storage = MockStorage()
        storage.save_entity("id1", "Apple Inc", "ORG", [])

        resolver = EntityResolver(storage, fuzzy_threshold=65)
        result = resolver.resolve("Apple Inc.")

        assert result.name == "Apple Inc"

    def test_fuzzy_below_threshold_creates_new(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple", "ORG", [])

        resolver = EntityResolver(storage, fuzzy_threshold=90)
        result = resolver.resolve("Googles")

        assert result.name == "Googles"
        assert result.id != "id1"

    def test_fuzzy_no_match_without_rapidfuzz(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple Inc", "ORG", [])

        resolver = EntityResolver(storage, fuzzy_threshold=60)
        result = resolver.resolve("Microsoft Corp")

        assert result.name == "Microsoft Corp"


class TestEntityResolverCreate:
    """Tests for new entity creation."""

    def test_new_entity_created(self):
        storage = MockStorage()

        resolver = EntityResolver(storage)
        result = resolver.resolve("NewCompany", "ORG")

        assert result.name == "NewCompany"
        assert result.entity_type == "ORG"

    def test_new_entity_with_type(self):
        storage = MockStorage()

        resolver = EntityResolver(storage)
        result = resolver.resolve("Tim Cook", "PERSON")

        assert result.entity_type == "PERSON"


class TestEntityResolverMerge:
    """Tests for entity merging."""

    def test_merge_entities(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple", "ORG", ["Apple Inc"])
        storage.save_entity("id2", "Apple Inc", "ORG", [])

        resolver = EntityResolver(storage)

        e1 = Entity("id1", "Apple", "ORG", ["Apple Inc"], None, None)
        e2 = Entity("id2", "Apple Inc", "ORG", [], None, None)

        result = resolver.merge_entities(e1, e2)

        assert result.name == "Apple"
        assert "Apple Inc" in result.aliases

    def test_merge_preserves_type(self):
        storage = MockStorage()

        resolver = EntityResolver(storage)

        e1 = Entity("id1", "Apple", "ORG", [], None, None)
        e2 = Entity("id2", "Apple Inc", None, [], None, None)

        result = resolver.merge_entities(e1, e2)

        assert result.entity_type == "ORG"


class TestEntityResolverSearch:
    """Tests for entity search."""

    def test_search_by_name(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple", "ORG", [])
        storage.save_entity("id2", "Google", "ORG", [])

        resolver = EntityResolver(storage)
        results = resolver.search_entities("Apple")

        assert len(results) == 1
        assert results[0].name == "Apple"

    def test_search_by_alias(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple", "ORG", ["Apple Inc"])

        resolver = EntityResolver(storage)
        results = resolver.search_entities("Apple Inc")

        assert len(results) == 1

    def test_search_limit(self):
        storage = MockStorage()
        for i in range(20):
            storage.save_entity(f"id{i}", f"Company{i}", "ORG", [])

        resolver = EntityResolver(storage)
        results = resolver.search_entities("Company", limit=5)

        assert len(results) == 5


class TestEntityResolverEdgeCases:
    """Edge case tests."""

    def test_whitespace_only(self):
        storage = MockStorage()

        resolver = EntityResolver(storage)
        result = resolver.resolve("   ")

        assert result.name == "unknown"

    def test_none_type_preserved(self):
        storage = MockStorage()
        storage.save_entity("id1", "Test", None, [])

        resolver = EntityResolver(storage)
        result = resolver.resolve("Test")

        assert result.name == "Test"

    def test_update_existing_type(self):
        storage = MockStorage()
        storage.save_entity("id1", "Apple", "ORG", [])

        resolver = EntityResolver(storage)
        result = resolver.resolve("Apple", "ORG")

        assert result.entity_type == "ORG"
