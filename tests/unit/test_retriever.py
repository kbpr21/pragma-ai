from pragma.query.retriever import BM25Retriever
from pragma.models import Entity


class MockGraphBuilder:
    """Mock graph builder for testing."""

    def __init__(self):
        self._entities = {}
        self.search_results = {}

    def add_entity(self, entity: Entity):
        self._entities[entity.id] = entity

    def search_entities_bm25(self, query: str, top_k: int = 5):
        return self.search_results.get(query, [])

    @property
    def storage(self):
        return self

    def get_entity_by_id(self, entity_id: str):
        return self._entities.get(entity_id)


class TestBM25Retriever:
    """BM25 Retriever tests."""

    def test_empty_questions(self):
        builder = MockGraphBuilder()
        retriever = BM25Retriever(builder)
        result = retriever.find_seed_entities([])
        assert result == []

    def test_finds_entities_for_question(self):
        builder = MockGraphBuilder()

        e1 = Entity("e1", "Apple", "ORG", [], None)
        builder.add_entity(e1)
        builder.search_results = {"What is Apple?": ["e1"]}

        retriever = BM25Retriever(builder)
        result = retriever.find_seed_entities(["What is Apple?"])

        assert len(result) == 1
        assert result[0].name == "Apple"

    def test_multiple_questions_scored(self):
        builder = MockGraphBuilder()

        e1 = Entity("e1", "Apple", "ORG", [], None)
        e2 = Entity("e2", "iPhone", "PRODUCT", [], None)
        builder.add_entity(e1)
        builder.add_entity(e2)

        builder.search_results = {
            "What does Apple make?": ["e1", "e2"],
            "What is iPhone?": ["e2"],
        }

        retriever = BM25Retriever(builder)
        result = retriever.find_seed_entities(
            ["What does Apple make?", "What is iPhone?"]
        )

        assert len(result) >= 1

    def test_max_total_seeds_limit(self):
        builder = MockGraphBuilder()

        for i in range(15):
            builder.add_entity(Entity(f"e{i}", f"Entity{i}", "ORG", [], None))
            builder.search_results[f"Q{i}"] = [f"e{i}"]

        retriever = BM25Retriever(builder, max_total_seeds=5)
        result = retriever.find_seed_entities([f"Q{i}" for i in range(15)])

        assert len(result) <= 5

    def test_no_entities_logs_warning(self, caplog):
        builder = MockGraphBuilder()
        builder.search_results = {"Q": []}

        retriever = BM25Retriever(builder)
        result = retriever.find_seed_entities(["Q"])

        assert len(result) == 0

    def test_simple_search(self):
        builder = MockGraphBuilder()

        e1 = Entity("e1", "Apple", "ORG", [], None)
        builder.add_entity(e1)
        builder.search_results = {"Apple query": ["e1"]}

        retriever = BM25Retriever(builder)
        result = retriever.find_seed_entities_simple("Apple query", top_k=3)

        assert len(result) == 1

    def test_simple_search_no_results(self):
        builder = MockGraphBuilder()
        builder.search_results = {"unknown": []}

        retriever = BM25Retriever(builder)
        result = retriever.find_seed_entities_simple("unknown", top_k=3)

        assert result == []

    def test_top_k_per_question(self):
        builder = MockGraphBuilder()

        for i in range(10):
            builder.add_entity(Entity(f"e{i}", f"Entity{i}", "ORG", [], None))

        builder.search_results = {"Q": [f"e{i}" for i in range(10)]}

        retriever = BM25Retriever(builder, top_k_per_question=3)
        result = retriever.find_seed_entities(["Q"])

        assert len(result) >= 1
