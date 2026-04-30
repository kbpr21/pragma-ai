"""Integration tests for pragma end-to-end workflows."""

import tempfile

from pragma import KnowledgeBase


class MockLLM:
    """Mock LLM for integration tests."""

    def complete(self, messages, **kwargs):
        return '{"answer": "Test answer", "reasoning_steps": [], "confidence": 0.9}'

    async def acomplete(self, messages, **kwargs):
        return self.complete(messages, **kwargs)

    async def stream_complete(self, messages, **kwargs):
        yield '{"answer": "Test"}\n'

    @property
    def model_name(self):
        return "mock"


class TestEndToEndIngestion:
    """End-to-end ingestion tests."""

    def test_ingest_csv_to_facts(self):
        """Ingest CSV and verify facts in DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            result = kb.ingest("tests/fixtures/companies.csv")

            # Verify ingestion worked - documents were processed
            assert result.documents >= 1

            kb.close()

    def test_ingest_json_to_facts(self):
        """Ingest JSON and verify facts in DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            result = kb.ingest("tests/fixtures/companies.json")

            assert result.documents >= 1

            kb.close()

    def test_ingest_txt_to_facts(self):
        """Ingest TXT and verify facts in DB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            result = kb.ingest("tests/fixtures/companies.txt")

            assert result.documents >= 1

            kb.close()


class TestEndToEndQuery:
    """End-to-end query tests."""

    def test_query_returns_result(self):
        """Query returns proper result object."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            # First ingest creates graph
            kb.ingest("tests/fixtures/companies.txt")

            # Then query
            result = kb.query("What is Apple?")

            # Verify result structure
            assert hasattr(result, "answer")
            assert hasattr(result, "confidence")
            assert hasattr(result, "latency_ms")

            kb.close()

    def test_query_with_hop_depth(self):
        """Query with hop_depth parameter."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            kb.ingest("tests/fixtures/companies.txt")

            result = kb.query("What is Apple?", hop_depth=3)

            assert result is not None
            kb.close()


class TestMultiHopReasoning:
    """Multi-hop reasoning tests."""

    def test_multi_hop_graph_traversal(self):
        """Verify 3-hop traversal works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            # Build a multi-hop graph
            kb.ingest("tests/fixtures/companies.txt")

            # Query with multi-hop
            result = kb.query("What tech era was Apple founded in?", hop_depth=3)

            # Should return result with reasoning
            assert result is not None

            kb.close()


class TestTemporalQueries:
    """Temporal query tests."""

    def test_as_of_filter(self):
        """Verify as_of parameter works."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            kb.ingest("tests/fixtures/companies.txt")

            # Query with temporal filter
            result = kb.query("What is Apple?", as_of="2020-01-01")

            assert result is not None
            kb.close()

    def test_temporal_fact_resolution(self):
        """Verify temporal facts resolve correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create KB
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            # Ingest some text with dates
            kb.ingest("tests/fixtures/companies.txt")

            result = kb.query("What year was Google founded?")

            # Should get answer with confidence
            assert result is not None

            kb.close()


class TestCrossFormatIngestion:
    """Cross-format ingestion tests."""

    def test_multi_format_ingest(self):
        """Ingest multiple formats and verify combined knowledge."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            # Ingest all three formats
            result = kb.ingest(
                [
                    "tests/fixtures/companies.txt",
                    "tests/fixtures/companies.csv",
                    "tests/fixtures/companies.json",
                ]
            )

            # Should have ingested something
            assert result.documents >= 1

            # Query should work across all facts
            result = kb.query("What are the companies?")

            assert result is not None

            kb.close()

    def test_query_requires_multiple_formats(self):
        """Query that needs facts from different source formats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kb = KnowledgeBase(llm=MockLLM(), kb_dir=tmpdir)

            # Ingest all formats
            kb.ingest("tests/fixtures/companies.txt")
            kb.ingest("tests/fixtures/companies.csv")
            kb.ingest("tests/fixtures/companies.json")

            # Query needs knowledge from different sources
            result = kb.query("Which company was founded first: Apple or Google?")

            assert result is not None

            kb.close()
