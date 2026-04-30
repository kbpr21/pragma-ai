from pragma.query.synthesizer import AnswerSynthesizer


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self, response: str = "{}"):
        self.response = response

    def complete(self, messages, **kwargs):
        return self.response

    async def acomplete(self, messages, **kwargs):
        return self.response

    @property
    def model_name(self):
        return "mock"


class TestAnswerSynthesizer:
    """AnswerSynthesizer tests."""

    def test_synthesize_empty_facts(self):
        llm = MockLLM()
        synthesizer = AnswerSynthesizer(llm)

        result = synthesizer.synthesize("What is X?", [], [])

        assert result.answer == "Insufficient information to answer"
        assert result.confidence == 0.0

    def test_synthesize_json_response(self):
        llm = MockLLM(
            '{"answer": "Apple is a company", "reasoning_steps": [{"fact_id": "F001", "explanation": "States Apple is a company"}]}'
        )
        synthesizer = AnswerSynthesizer(llm)

        facts = ["[F001] Apple | is | company (confidence: 0.95)"]
        result = synthesizer.synthesize("What is Apple?", facts, [])

        assert "Apple" in result.answer
        assert result.confidence > 0

    def test_synthesize_markdown_fences(self):
        llm = MockLLM('```json\n{"answer": "Test", "reasoning_steps": []}\n```')
        synthesizer = AnswerSynthesizer(llm)

        result = synthesizer.synthesize("Q?", ["fact"], [])
        assert "Test" in result.answer

    def test_parse_invalid_json_fallback(self):
        llm = MockLLM("This is plain text answer without JSON")
        synthesizer = AnswerSynthesizer(llm)

        result = synthesizer.synthesize("Q?", ["fact"], [])

        assert isinstance(result.answer, str)

    def test_llm_failure(self):
        class FailingLLM:
            def complete(self, *args, **kwargs):
                raise Exception("API failed")

        synthesizer = AnswerSynthesizer(FailingLLM())
        result = synthesizer.synthesize("Q?", ["fact"], [])

        assert result.answer == "Error during synthesis"

    def test_calculate_confidence(self):
        llm = MockLLM()
        synthesizer = AnswerSynthesizer(llm)

        facts = [
            {"confidence": 0.9},
            {"confidence": 0.8},
            {"confidence": 0.7},
        ]
        confidence = synthesizer._calculate_confidence(facts)

        assert 0.7 < confidence < 1.0

    def test_calculate_confidence_empty(self):
        llm = MockLLM()
        synthesizer = AnswerSynthesizer(llm)

        confidence = synthesizer._calculate_confidence([])

        assert confidence == 0.0

    def test_confidence_bounded(self):
        llm = MockLLM()
        synthesizer = AnswerSynthesizer(llm)

        facts = [{"confidence": 1.0}] * 100
        confidence = synthesizer._calculate_confidence(facts)

        assert confidence <= 1.0
