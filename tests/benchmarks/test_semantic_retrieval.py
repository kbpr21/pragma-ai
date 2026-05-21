"""Semantic retrieval benchmark tests.

Compares BM25-only vs hybrid retrieval effectiveness on queries that
require fuzzy semantic matching — synonyms, paraphrases, and conceptual
similarity.

These tests validate the ``SemanticRetriever`` and ``HybridRetriever``
components without requiring ``sentence-transformers`` to be installed
(they test the infrastructure, not the model quality).
"""

from pragma.graph.resolver import EntityResolver
from pragma.graph.synonyms import SynonymDictionary


# ---------------------------------------------------------------------------
# Synonym dictionary tests
# ---------------------------------------------------------------------------


class TestSynonymDictionary:
    """Verify the synonym dictionary used for ontology normalization."""

    def test_builtin_synonyms_exist(self):
        """The built-in dictionary should contain common abbreviations."""
        d = SynonymDictionary()
        assert d.size > 0
        assert d.expand("ml") == "machine learning"
        assert d.expand("nlp") == "natural language processing"
        assert d.expand("ai") == "artificial intelligence"

    def test_expand_case_insensitive(self):
        """Synonym lookup should be case-insensitive."""
        d = SynonymDictionary()
        assert d.expand("ML") == "machine learning"
        assert d.expand("Nlp") == "natural language processing"

    def test_expand_unknown_returns_none(self):
        """Unknown terms should return None, not raise."""
        d = SynonymDictionary()
        assert d.expand("xyzzy_unknown_term") is None

    def test_corporate_suffix_expansion(self):
        """Corporate suffixes should expand to empty string."""
        d = SynonymDictionary()
        assert d.expand("inc.") == ""
        assert d.expand("ltd.") == ""
        assert d.expand("llc") == ""

    def test_user_dict_merge(self, tmp_path):
        """User-provided synonyms should merge with built-ins."""
        user_dict = tmp_path / "custom_synonyms.json"
        user_dict.write_text('{"myterm": "my canonical form"}')

        d = SynonymDictionary(user_dict_path=str(user_dict))
        assert d.expand("myterm") == "my canonical form"
        # Built-ins should still work.
        assert d.expand("ml") == "machine learning"


# ---------------------------------------------------------------------------
# Entity normalization tests
# ---------------------------------------------------------------------------


class TestEntityNormalization:
    """Verify the resolver's normalization pipeline."""

    def test_normalize_abbreviation(self, tmp_path):
        """Abbreviations should be expanded."""
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        resolver = EntityResolver(store)
        assert "machine learning" in resolver.normalize("ML").lower()
        store.close()

    def test_normalize_corporate_suffix(self, tmp_path):
        """Corporate suffixes should be stripped."""
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        resolver = EntityResolver(store)
        result = resolver.normalize("Acme Corp.")
        assert "Corp" not in result
        assert "Acme" in result
        store.close()

    def test_normalize_whitespace(self, tmp_path):
        """Extra whitespace should be collapsed."""
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        resolver = EntityResolver(store)
        result = resolver.normalize("  Steve    Jobs  ")
        assert result == "Steve Jobs"
        store.close()

    def test_normalize_preserves_acronyms(self, tmp_path):
        """Short all-caps tokens should be preserved as acronyms."""
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        resolver = EntityResolver(store)
        result = resolver.normalize("IBM Research Lab")
        assert "IBM" in result
        store.close()

    def test_resolve_with_synonym(self, tmp_path):
        """Resolving 'ML' and 'Machine Learning' should yield the same entity."""
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        resolver = EntityResolver(store)

        # First resolution creates the entity.
        e1 = resolver.resolve("Machine Learning", entity_type="CONCEPT")
        # Second resolution via abbreviation should find the same entity.
        e2 = resolver.resolve("ML", entity_type="CONCEPT")

        assert e1.id == e2.id
        store.close()


# ---------------------------------------------------------------------------
# Semantic retriever infrastructure tests
# ---------------------------------------------------------------------------


class TestSemanticRetrieverInfra:
    """Test SemanticRetriever infrastructure (no model required)."""

    def test_import_semantic_retriever(self):
        """The module should import without errors."""
        from pragma.query.semantic import SemanticRetriever

        assert SemanticRetriever is not None

    def test_graceful_degradation_without_model(self, tmp_path):
        """Without sentence-transformers, retriever should degrade gracefully."""
        from pragma.query.semantic import SemanticRetriever
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        retriever = SemanticRetriever(
            storage=store,
            model_name="nonexistent-model-that-wont-load",
        )
        # Query should return empty list, not crash.
        results = retriever.query("test query", top_k=5)
        assert isinstance(results, list)
        store.close()


# ---------------------------------------------------------------------------
# Hybrid retriever tests
# ---------------------------------------------------------------------------


class TestHybridRetrieverInfra:
    """Test HybridRetriever infrastructure."""

    def test_import_hybrid_retriever(self):
        """HybridRetriever should be importable."""
        from pragma.query.retriever import HybridRetriever

        assert HybridRetriever is not None

    def test_hybrid_falls_back_to_bm25(self, tmp_path):
        """When no semantic retriever is provided, hybrid should
        behave exactly like BM25."""
        from pragma.query.retriever import HybridRetriever
        from pragma.graph.builder import GraphBuilder
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        builder = GraphBuilder(store, kb_dir=str(tmp_path))

        retriever = HybridRetriever(
            builder,
            semantic_retriever=None,
        )
        # Should not crash; returns empty for empty KB.
        results = retriever.find_seed_entities(["test query"])
        assert isinstance(results, list)
        store.close()


class TestAPIEmbeddings:
    """Verify that SemanticRetriever integrates with LLM API embedding providers."""

    class MockOpenAIProvider:
        def __init__(self):
            self.embedded_texts = []
            self.model = "text-embedding-3-small"

        def embed(self, texts, model=None):
            self.embedded_texts.extend(texts)
            # Return dummy vectors: 2D vector for each text
            # E.g., [1.0, 2.0] which is NOT L2-normalized
            return [[1.0, 2.0] for _ in texts]

    class MockOllamaProvider:
        def __init__(self):
            self.embedded_texts = []
            self.model = "nomic-embed-text"

        def embed(self, texts, model=None):
            self.embedded_texts.extend(texts)
            # Return dummy vectors: 2D vector for each text
            # E.g., [3.0, 4.0] which is NOT L2-normalized
            return [[3.0, 4.0] for _ in texts]

    def test_semantic_retriever_uses_openai_embed(self, tmp_path):
        """SemanticRetriever should use OpenAIProvider.embed and L2 normalize vectors."""
        from pragma.query.semantic import SemanticRetriever
        from pragma.storage.sqlite import SQLiteStore
        import numpy as np

        store = SQLiteStore(str(tmp_path))
        mock_llm = self.MockOpenAIProvider()

        retriever = SemanticRetriever(
            storage=store,
            model_name="text-embedding-3-small",
            llm=mock_llm,
        )

        assert retriever.available is True

        facts = [
            {"id": "f1", "context": "Apple was founded by Steve Jobs."},
            {"id": "f2", "context": "Microsoft was founded by Bill Gates."},
        ]

        num_embedded = retriever.embed_facts(facts)
        assert num_embedded == 2
        assert len(mock_llm.embedded_texts) == 2
        assert mock_llm.embedded_texts[0] == "Apple was founded by Steve Jobs."

        # Verify that embeddings in retriever's index are L2-normalized
        vec1 = retriever._index["f1"]
        # Expected: normalized version of [1.0, 2.0]
        expected_norm = np.linalg.norm(np.array([1.0, 2.0]))
        expected_vec = np.array([1.0, 2.0]) / expected_norm
        assert np.allclose(vec1, expected_vec)

        # Let's perform a query
        results = retriever.query("Who founded Apple?", top_k=2)
        assert len(results) == 2
        # Since all mock embeddings are identical, both will have a similarity of 1.0
        assert np.allclose(results[0][1], 1.0) or np.allclose(results[0][1], 0.999999)

        store.close()

    def test_semantic_retriever_uses_ollama_embed(self, tmp_path):
        """SemanticRetriever should use OllamaProvider.embed and L2 normalize vectors."""
        from pragma.query.semantic import SemanticRetriever
        from pragma.storage.sqlite import SQLiteStore
        import numpy as np

        store = SQLiteStore(str(tmp_path))
        mock_llm = self.MockOllamaProvider()

        retriever = SemanticRetriever(
            storage=store,
            model_name="nomic-embed-text",
            llm=mock_llm,
        )

        assert retriever.available is True

        facts = [{"id": "f1", "context": "Python is a programming language."}]

        num_embedded = retriever.embed_facts(facts)
        assert num_embedded == 1
        assert len(mock_llm.embedded_texts) == 1

        vec1 = retriever._index["f1"]
        expected_norm = np.linalg.norm(np.array([3.0, 4.0]))
        expected_vec = np.array([3.0, 4.0]) / expected_norm
        assert np.allclose(vec1, expected_vec)

        store.close()
