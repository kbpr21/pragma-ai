from pragma.query.decomposer import QueryDecomposer


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response: str = "[]"):
        self.response = response

    def complete(self, messages, **kwargs):
        return self.response

    async def acomplete(self, messages, **kwargs):
        return self.response

    @property
    def model_name(self):
        return "mock"


class TestQueryDecomposerBasic:
    """Basic decomposition tests."""

    def test_decompose_empty_query(self):
        decomposer = QueryDecomposer(MockLLM())
        result = decomposer.decompose("")
        assert result == [""]

    def test_decompose_simple_json(self):
        llm = MockLLM('["What is Apple?", "Who is Tim Cook?"]')
        decomposer = QueryDecomposer(llm)
        # Multi-hop hint ("and") forces the LLM-call path.
        result = decomposer.decompose("Tell me about Apple and Tim Cook")
        assert len(result) == 2
        assert "Apple" in result[0]

    def test_decompose_max_limit(self):
        llm = MockLLM('["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7"]')
        decomposer = QueryDecomposer(MockLLM(llm.response), max_subquestions=5)
        # Long query with hint forces decomposition.
        result = decomposer.decompose(
            "Compare X and Y and Z and W and V across many dimensions please now"
        )
        assert len(result) == 5

    def test_decompose_fallback_single(self):
        llm = MockLLM("This is a single question")
        decomposer = QueryDecomposer(llm)
        result = decomposer.decompose("What is AI?")
        assert result == ["What is AI?"]

    def test_decompose_llm_failure(self):
        class FailingLLM:
            def complete(self, *args, **kwargs):
                raise Exception("API failed")

        decomposer = QueryDecomposer(FailingLLM())
        result = decomposer.decompose("What is AI?")
        assert result == ["What is AI?"]


class TestQueryDecomposerJsonParsing:
    """JSON parsing tests."""

    def test_parse_markdown_fences(self):
        llm = MockLLM('```json\n["Q1", "Q2"]\n```')
        decomposer = QueryDecomposer(llm)
        # "and" hint forces the LLM-call path.
        result = decomposer.decompose("What is X and Y in this domain?")
        assert len(result) == 2

    def test_parse_plain_json(self):
        llm = MockLLM('["Question one", "Question two"]')
        decomposer = QueryDecomposer(llm)
        # "and" hint forces the LLM-call path.
        result = decomposer.decompose("What is X and Y here?")
        assert result == ["Question one", "Question two"]

    def test_parse_invalid_json(self):
        llm = MockLLM("Not JSON at all")
        decomposer = QueryDecomposer(llm)
        result = decomposer.decompose("Some query")
        assert isinstance(result, list)


class TestQueryDecomposerFallback:
    """Fallback parsing tests."""

    def test_fallback_numbered_list(self):
        llm = MockLLM("1. First question\n2. Second question\n3. Third")
        decomposer = QueryDecomposer(llm)
        result = decomposer._fallback_parse(llm.response)
        assert len(result) >= 1

    def test_fallback_comma_separated(self):
        llm = MockLLM("Q1, Q2, Q3")
        decomposer = QueryDecomposer(llm)
        result = decomposer._fallback_parse(llm.response)
        assert len(result) >= 1

    def test_fallback_and_split(self):
        llm = MockLLM("Question one and question two")
        decomposer = QueryDecomposer(llm)
        result = decomposer._fallback_parse(llm.response)
        assert len(result) >= 1
