from unittest.mock import patch
from typing import List, Dict, Any

from pragma.ingestion.extractor import FactExtractor
from pragma.ingestion.preprocessor import ProcessedSegment


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = "", should_fail: bool = False):
        self.response = response
        self.should_fail = should_fail
        self.call_count = 0

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        self.call_count += 1
        if self.should_fail:
            raise Exception("Mock LLM error")
        return self.response

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        return self.complete(messages, **kwargs)

    @property
    def model_name(self) -> str:
        return "mock-model"


def make_segment(content: str, source: str = "test.txt") -> ProcessedSegment:
    """Helper to create a ProcessedSegment."""
    return ProcessedSegment(
        content=content,
        source=source,
        doc_type="txt",
        chunk_index=0,
        content_hash="abc123",
        metadata={
            "source_doc": source,
            "page": 1,
            "filename": source,
            "char_count": len(content),
        },
    )


class TestFactExtractorBasic:
    """Basic extraction tests."""

    def test_extract_empty_segments(self):
        llm = MockLLMProvider()
        extractor = FactExtractor(llm)
        result = extractor.extract([])
        assert result == []

    def test_extract_single_segment(self):
        llm = MockLLMProvider(
            response="""[
                {"subject": "Apple", "predicate": "is", "object": "a company", "confidence": 1.0}
            ]"""
        )
        extractor = FactExtractor(llm)
        segments = [make_segment("Apple is a company.")]
        result = extractor.extract(segments)

        assert len(result) == 1
        assert result[0]["subject"] == "Apple"
        assert result[0]["predicate"] == "is"

    def test_llm_failure_returns_empty(self):
        llm = MockLLMProvider(should_fail=True)
        extractor = FactExtractor(llm)
        segments = [make_segment("Some content")]
        result = extractor.extract(segments)
        assert result == []


class TestFactExtractorJsonParsing:
    """JSON parsing tests with various formats."""

    def test_parse_plain_json(self):
        llm = MockLLMProvider()
        extractor = FactExtractor(llm)
        segments = [make_segment("Test.")]

        json_response = """[
            {"subject": "Test", "predicate": "is", "object": "valid", "confidence": 1.0}
        ]"""
        with patch.object(llm, "complete", return_value=json_response):
            result = extractor.extract(segments)

        assert len(result) == 1

    def test_parse_markdown_fences(self):
        llm = MockLLMProvider(
            response="""```json
[
    {"subject": "Test", "predicate": "is", "object": "wrapped", "confidence": 1.0}
]
```"""
        )
        extractor = FactExtractor(llm)
        segments = [make_segment("Test.")]

        with patch.object(llm, "complete", return_value=llm.response):
            result = extractor.extract(segments)

        assert len(result) == 1
        assert result[0]["object"] == "wrapped"

    def test_parse_triple_backticks(self):
        llm = MockLLMProvider(
            response="""```
[
    {"subject": "Test", "predicate": "works", "object": "yes", "confidence": 1.0}
]
```"""
        )
        extractor = FactExtractor(llm)
        segments = [make_segment("Test.")]

        with patch.object(llm, "complete", return_value=llm.response):
            result = extractor.extract(segments)

        assert len(result) == 1

    def test_parse_partial_json(self):
        llm = MockLLMProvider(
            response="""[
    {"subject": "First", "predicate": "valid", "object": "yes", "confidence": 1.0},
    {"subject": "Second", "predicate": "broken","""
        )
        extractor = FactExtractor(llm)
        segments = [make_segment("Test.")]

        with patch.object(llm, "complete", return_value=llm.response):
            result = extractor.extract(segments)

        assert len(result) >= 1


class TestFactExtractorValidation:
    """Validation and filtering tests."""

    def test_filter_low_confidence(self):
        llm = MockLLMProvider(
            response="""[
                {"subject": "A", "predicate": "is", "object": "fact", "confidence": 1.0},
                {"subject": "B", "predicate": "is", "object": "uncertain", "confidence": 0.3}
            ]"""
        )
        extractor = FactExtractor(llm, min_confidence=0.6)
        segments = [make_segment("A is fact. B is uncertain.")]

        with patch.object(llm, "complete", return_value=llm.response):
            result = extractor.extract(segments)

        assert len(result) == 1
        assert result[0]["subject"] == "A"

    def test_missing_subject_filtered(self):
        llm = MockLLMProvider(
            response="""[
                {"predicate": "is", "object": "invalid"},
                {"subject": "Valid", "predicate": "is", "object": "ok", "confidence": 1.0}
            ]"""
        )
        extractor = FactExtractor(llm)
        segments = [make_segment("Test.")]

        with patch.object(llm, "complete", return_value=llm.response):
            result = extractor.extract(segments)

        assert len(result) == 1
        assert result[0]["subject"] == "Valid"

    def test_missing_predicate_filtered(self):
        llm = MockLLMProvider(
            response="""[
                {"subject": "Valid", "predicate": "is", "object": "ok", "confidence": 1.0},
                {"subject": "NoPred", "object": "invalid", "confidence": 1.0}
            ]"""
        )
        extractor = FactExtractor(llm)
        segments = [make_segment("Test.")]

        with patch.object(llm, "complete", return_value=llm.response):
            result = extractor.extract(segments)

        assert result[0]["subject"] == "Valid"


class TestFactExtractorBatch:
    """Batch extraction tests."""

    def test_batch_extract(self):
        response_text = """[
            {"subject": "Fact1", "predicate": "from", "object": "batch", "confidence": 1.0}
        ]"""

        class BatchLLM(MockLLMProvider):
            def __init__(self):
                super().__init__(response=response_text)
                self.call_count = 0

        llm = BatchLLM()
        extractor = FactExtractor(llm)
        segments = [make_segment("Segment 1."), make_segment("Segment 2.")]
        result = extractor.extract_batch(segments, max_tokens=4000)

        assert len(result) >= 1
        assert llm.call_count == 1


class TestFactExtractorMetadata:
    """Tests for metadata assignment."""

    def test_metadata_assignment(self):
        llm = MockLLMProvider(
            response="""[
                {"subject": "Test", "predicate": "works", "object": "well", "confidence": 1.0}
            ]"""
        )
        extractor = FactExtractor(llm)
        segments = [make_segment("Test works well.", "doc.pdf")]

        with patch.object(llm, "complete", return_value=llm.response):
            result = extractor.extract(segments)

        assert len(result) == 1
        assert result[0]["_source_doc"] == "doc.pdf"
        assert result[0]["_source_page"] == 1
        assert result[0]["_context"] == "Test works well."


class TestFactExtractorNegation:
    """Tests for negation preservation."""

    def test_preserve_negation(self):
        llm = MockLLMProvider(
            response="""[
                {"subject": "Drug X", "predicate": "does NOT cause", "object": "liver damage", "confidence": 1.0}
            ]"""
        )
        extractor = FactExtractor(llm)
        segments = [make_segment("Drug X does not cause liver damage.")]

        with patch.object(llm, "complete", return_value=llm.response):
            result = extractor.extract(segments)

        assert len(result) == 1
        assert "NOT" in result[0]["predicate"]


class TestFactExtractorMaxFacts:
    """Test max facts limit."""

    def test_max_facts_per_segment(self):
        llm = MockLLMProvider(
            response="""[
                {"subject": "A", "predicate": "relates", "object": "B", "confidence": 1.0},
                {"subject": "C", "predicate": "relates", "object": "D", "confidence": 1.0},
                {"subject": "E", "predicate": "relates", "object": "F", "confidence": 1.0},
                {"subject": "G", "predicate": "relates", "object": "H", "confidence": 1.0}
            ]"""
        )
        extractor = FactExtractor(llm, max_facts_per_segment=2)
        segments = [make_segment("Multiple facts here.")]

        with patch.object(llm, "complete", return_value=llm.response):
            result = extractor.extract(segments)

        assert len(result) == 2
