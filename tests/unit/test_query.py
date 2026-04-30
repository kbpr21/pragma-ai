from unittest.mock import patch

from pragma import KnowledgeBase


class MockLLM:
    def complete(self, messages, **kwargs):
        return '{"answer": "Test answer", "reasoning_steps": []}'

    async def acomplete(self, messages, **kwargs):
        return self.complete(messages, **kwargs)

    @property
    def model_name(self):
        return "mock"


class TestKnowledgeBaseQuery:
    """KnowledgeBase.query() tests."""

    def test_query_returns_pragaresult(self, tmp_path):
        kb = KnowledgeBase(llm=MockLLM(), kb_dir=str(tmp_path))

        with patch.object(kb._graph_builder, "search_entities_bm25", return_value=[]):
            result = kb.query("What is Apple?")

        assert hasattr(result, "answer")
        assert hasattr(result, "confidence")
        assert hasattr(result, "latency_ms")
        kb.close()

    def test_query_with_hop_depth(self, tmp_path):
        kb = KnowledgeBase(llm=MockLLM(), kb_dir=str(tmp_path))

        with patch.object(kb._graph_builder, "search_entities_bm25", return_value=[]):
            result = kb.query("What is Apple?", hop_depth=3)

        assert result is not None
        kb.close()

    def test_query_with_min_confidence(self, tmp_path):
        kb = KnowledgeBase(llm=MockLLM(), kb_dir=str(tmp_path))

        with patch.object(kb._graph_builder, "search_entities_bm25", return_value=[]):
            result = kb.query("What is Apple?", min_confidence=0.8)

        assert result is not None
        kb.close()

    def test_query_with_top_k(self, tmp_path):
        kb = KnowledgeBase(llm=MockLLM(), kb_dir=str(tmp_path))

        with patch.object(kb._graph_builder, "search_entities_bm25", return_value=[]):
            result = kb.query("What is Apple?", top_k=10)

        assert result is not None
        kb.close()

    def test_query_no_entities_returns_insufficient(self, tmp_path):
        kb = KnowledgeBase(llm=MockLLM(), kb_dir=str(tmp_path))

        with patch.object(kb._graph_builder, "search_entities_bm25", return_value=[]):
            result = kb.query("What is nonexistent thing?")

        assert "Insufficient knowledge" in result.answer
        kb.close()

    def test_query_cache_key_deterministic(self, tmp_path):
        kb = KnowledgeBase(llm=MockLLM(), kb_dir=str(tmp_path))

        key1 = kb._compute_query_cache_key("test query", 2, 0.5, None)
        key2 = kb._compute_query_cache_key("test query", 2, 0.5, None)

        assert key1 == key2
        kb.close()

    def test_query_cache_key_varies_by_params(self, tmp_path):
        kb = KnowledgeBase(llm=MockLLM(), kb_dir=str(tmp_path))

        key1 = kb._compute_query_cache_key("test query", 2, 0.5, None)
        key2 = kb._compute_query_cache_key("test query", 3, 0.5, None)

        assert key1 != key2
        kb.close()

    def test_query_cache_key_varies_by_as_of(self, tmp_path):
        """Cache key includes as_of parameter."""
        kb = KnowledgeBase(llm=MockLLM(), kb_dir=str(tmp_path))

        key1 = kb._compute_query_cache_key("test query", 2, 0.5, "2024-01-01")
        key2 = kb._compute_query_cache_key("test query", 2, 0.5, None)

        assert key1 != key2
        kb.close()
