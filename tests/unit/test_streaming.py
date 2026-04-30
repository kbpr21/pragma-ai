import pytest
from unittest.mock import MagicMock, patch


class MockLLMWithStream:
    """Mock LLM with streaming support."""

    def __init__(self):
        self.model = "mock-stream"

    def complete(self, messages, **kwargs):
        return '{"answer": "test", "reasoning_steps": []}'

    async def acomplete(self, messages, **kwargs):
        return self.complete(messages, **kwargs)

    async def stream_complete(self, messages, **kwargs):
        """Yield tokens one at a time."""
        text = '{"answer": "streaming works", "reasoning_steps": []}'
        for char in text:
            yield char
        yield "\n"

    @property
    def model_name(self):
        return "mock-stream"


class TestStreaming:
    """Streaming tests."""

    def test_stream_method_exists(self):
        """KnowledgeBase has stream method."""
        from pragma import KnowledgeBase

        assert hasattr(KnowledgeBase, "stream")

    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self):
        """stream() yields tokens."""
        from pragma import KnowledgeBase
        import networkx as nx

        llm = MockLLMWithStream()

        with patch("pragma.KnowledgeBase.__init__", return_value=None):
            kb = KnowledgeBase.__new__(KnowledgeBase)
            kb._llm = llm
            kb.config = MagicMock()
            kb.config.default_hop_depth = 1
            kb.config.max_subgraph_nodes = 3
            kb._graph_builder = MagicMock()
            kb._storage = MagicMock()

            gb = kb._graph_builder
            gb.search_entities_bm25 = MagicMock(return_value=["Apple"])

            graph = nx.MultiDiGraph()
            graph.add_node("Apple")
            graph.add_edge("Apple", "Apple", predicate="is", key="is")

            traverser = MagicMock()
            traverser.extract_subgraph = MagicMock(return_value=graph)
            traverser.get_reasoning_paths = MagicMock(return_value=[])

            with patch("pragma.graph.traversal.GraphTraverser", return_value=traverser):
                with patch("pragma.query.decomposer.QueryDecomposer") as dc:
                    d = MagicMock()
                    d.decompose = MagicMock(return_value=["What is Apple?"])
                    dc.return_value = d

                    with patch("pragma.query.retriever.BM25Retriever") as rt:
                        r = MagicMock()
                        r.find_seed_entities = MagicMock(return_value=["Apple"])
                        rt.return_value = r

                        with patch("pragma.query.assembler.FactAssembler") as am:
                            a = MagicMock()
                            a.assemble_facts = MagicMock(
                                return_value=[
                                    {
                                        "id": "F1",
                                        "subject_id": "Apple",
                                        "predicate": "is",
                                        "object_value": "company",
                                        "confidence": 0.9,
                                    }
                                ]
                            )
                            am.return_value = a

                            tokens = []
                            async for token in kb.stream("What is Apple?"):
                                tokens.append(token)

                            assert len(tokens) > 0

    @pytest.mark.asyncio
    async def test_stream_returns_async_generator(self):
        """stream() returns AsyncGenerator type."""
        from pragma import KnowledgeBase
        import inspect

        kb = KnowledgeBase.__new__(KnowledgeBase)

        result = kb.stream("test")
        assert inspect.isasyncgen(result)

    def test_llm_provider_has_stream_complete(self):
        """LLMProvider protocol has stream_complete."""
        from pragma.llm.base import LLMProvider

        assert hasattr(LLMProvider, "stream_complete")

    @pytest.mark.asyncio
    async def test_stream_integration_with_mock(self):
        """Full stream with mock works."""
        from pragma import KnowledgeBase
        import networkx as nx

        llm = MockLLMWithStream()

        gb = MagicMock()
        gb.search_entities_bm25 = MagicMock(return_value=["Apple"])
        gb.storage.get_entities_by_name = MagicMock(return_value=[])

        with patch("pragma.KnowledgeBase.__init__", return_value=None):
            kb = KnowledgeBase.__new__(KnowledgeBase)
            kb._llm = llm
            kb.config = MagicMock()
            kb.config.default_hop_depth = 1
            kb.config.max_subgraph_nodes = 3
            kb._graph_builder = gb
            kb._storage = MagicMock()

        graph = nx.MultiDiGraph()
        graph.add_node("Apple")
        gb.get_subgraph = MagicMock(return_value=graph)

        tokens = []
        async for token in kb.stream("What is Apple?"):
            tokens.append(token)

        assert len(tokens) > 0
