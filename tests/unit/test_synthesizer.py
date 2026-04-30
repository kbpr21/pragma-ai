"""Unit tests for :class:`pragma.query.synthesizer.AnswerSynthesizer`.

Pin the v1.0.1 token-efficiency contract:

* Empty facts -> structured "insufficient" output, no LLM call.
* Compact JSON schema (``{"a": ..., "f": [...]}``) is normalised to the
  legacy ``{"answer": ..., "reasoning_steps": [...]}`` shape downstream
  consumers expect.
* Legacy verbose schema is still parsed for backward compatibility.
* Markdown code fences are stripped.
* Plain-text fallback when JSON parsing fails.
* LLM exceptions surface as a structured error answer.
* Direct-answer fast-path returns the object value with ZERO LLM calls
  when exactly one fact's subject + predicate match the query.
* Query-keyword pre-filter drops facts with no overlap before the prompt.
* ``_compute_confidence`` is empty/full/bounded.
"""

from __future__ import annotations

from typing import Any, Dict, List

from pragma.query.synthesizer import AnswerSynthesizer


class MockLLM:
    """Mock LLM that returns a fixed response and counts calls."""

    def __init__(self, response: str = "{}") -> None:
        self.response = response
        self.calls: List[List[Dict[str, str]]] = []

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        self.calls.append(messages)
        return self.response

    async def acomplete(self, messages, **kwargs):
        return self.complete(messages, **kwargs)

    @property
    def model_name(self) -> str:
        return "mock"


# ---------------------------------------------------------------------------
# Empty / error paths
# ---------------------------------------------------------------------------


def test_synthesize_empty_facts() -> None:
    llm = MockLLM()
    syn = AnswerSynthesizer(llm)
    result = syn.synthesize("What is X?", [])
    assert result.answer == "Insufficient information to answer"
    assert result.confidence == 0.0
    assert llm.calls == []


def test_synthesize_llm_failure() -> None:
    class FailingLLM:
        def complete(self, *a: Any, **kw: Any) -> str:
            raise RuntimeError("API failed")

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(FailingLLM())
    result = syn.synthesize("Q?", [{"subject_id": "x", "predicate": "p"}])
    assert result.answer == "Error during synthesis"


def test_synthesize_empty_response() -> None:
    syn = AnswerSynthesizer(MockLLM(""))
    result = syn.synthesize(
        "Q?",
        [{"subject_id": "x", "predicate": "p", "object_value": "v"}],
    )
    assert "Empty response" in result.answer


# ---------------------------------------------------------------------------
# Schema parsing
# ---------------------------------------------------------------------------


def test_synthesize_compact_schema() -> None:
    """The new v1.0.1 ultra-compact schema must round-trip cleanly."""
    syn = AnswerSynthesizer(MockLLM('{"a": "Apple is a company", "f": ["F1"]}'))
    result = syn.synthesize(
        "What is Apple?",
        [
            {
                "subject_id": "e1",
                "predicate": "is",
                "object_value": "company",
                "confidence": 0.95,
            }
        ],
        entity_names={"e1": "Apple"},
    )
    assert "Apple" in result.answer
    assert result.reasoning_steps == [{"fact_id": "F1", "explanation": ""}]
    assert result.confidence > 0


def test_synthesize_legacy_schema() -> None:
    """The verbose legacy schema is still understood for back-compat."""
    syn = AnswerSynthesizer(
        MockLLM(
            '{"answer": "Apple is a company", '
            '"reasoning_steps": [{"fact_id": "F1", "explanation": "states it"}]}'
        )
    )
    result = syn.synthesize(
        "What is Apple?",
        [
            {
                "subject_id": "e1",
                "predicate": "is",
                "object_value": "company",
                "confidence": 0.95,
            }
        ],
        entity_names={"e1": "Apple"},
    )
    assert "Apple" in result.answer
    assert result.reasoning_steps[0]["fact_id"] == "F1"
    assert result.reasoning_steps[0]["explanation"] == "states it"


def test_synthesize_markdown_fences() -> None:
    syn = AnswerSynthesizer(MockLLM('```json\n{"a":"Test","f":[]}\n```'))
    result = syn.synthesize(
        "Q?",
        [{"subject_id": "x", "predicate": "p", "object_value": "v"}],
    )
    assert "Test" in result.answer


def test_synthesize_plain_text_fallback() -> None:
    syn = AnswerSynthesizer(MockLLM("This is plain text"))
    result = syn.synthesize(
        "Q?",
        [{"subject_id": "x", "predicate": "p", "object_value": "v"}],
    )
    assert "plain text" in result.answer


# ---------------------------------------------------------------------------
# Direct-answer fast-path: ZERO LLM calls
# ---------------------------------------------------------------------------


def test_direct_answer_fast_path_skips_llm() -> None:
    """When one fact's subject+predicate match the query strongly, return
    its object directly with no LLM call."""
    llm = MockLLM('{"a":"should NOT see this","f":[]}')
    syn = AnswerSynthesizer(llm)
    facts = [
        {
            "subject_id": "apple",
            "predicate": "was founded by",
            "object_value": "Steve Jobs, Steve Wozniak, and Ronald Wayne",
            "confidence": 1.0,
        }
    ]
    result = syn.synthesize(
        "Who founded Apple?",
        facts,
        entity_names={"apple": "Apple Inc."},
    )
    assert "Steve Jobs" in result.answer
    assert llm.calls == [], "fast-path must not invoke the LLM"
    assert result.reasoning_steps[0]["fact_id"] == "F1"


def test_direct_answer_skipped_when_multiple_candidates() -> None:
    llm = MockLLM('{"a":"goes through LLM","f":["F1"]}')
    syn = AnswerSynthesizer(llm)
    facts = [
        {
            "subject_id": "apple",
            "predicate": "was founded by",
            "object_value": "Steve Jobs",
            "confidence": 1.0,
        },
        {
            "subject_id": "apple",
            "predicate": "was founded by",
            "object_value": "Steve Wozniak",
            "confidence": 1.0,
        },
    ]
    result = syn.synthesize(
        "Who founded Apple?",
        facts,
        entity_names={"apple": "Apple Inc."},
    )
    assert "LLM" in result.answer
    assert len(llm.calls) == 1


def test_direct_answer_skipped_when_low_confidence() -> None:
    llm = MockLLM('{"a":"via LLM","f":["F1"]}')
    syn = AnswerSynthesizer(llm)
    facts = [
        {
            "subject_id": "apple",
            "predicate": "was founded by",
            "object_value": "Steve Jobs",
            "confidence": 0.5,  # below 0.85 threshold
        }
    ]
    result = syn.synthesize(
        "Who founded Apple?",
        facts,
        entity_names={"apple": "Apple Inc."},
    )
    assert "LLM" in result.answer
    assert len(llm.calls) == 1


# ---------------------------------------------------------------------------
# Query-keyword pre-filter
# ---------------------------------------------------------------------------


def test_keyword_filter_drops_irrelevant_facts() -> None:
    captured: Dict[str, str] = {}

    class CapturingLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            captured["user"] = messages[-1]["content"]
            return '{"a":"Cupertino","f":["F1"]}'

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(CapturingLLM())
    facts = [
        # Relevant. Confidence kept below the direct-answer threshold (0.85)
        # so we exercise the LLM-call path and can inspect the prompt.
        {
            "subject_id": "a",
            "predicate": "headquartered in",
            "object_value": "Cupertino",
            "confidence": 0.7,
        },
        # Irrelevant noise (no overlap with "headquartered" / "Apple")
        {
            "subject_id": "b",
            "predicate": "won the World Cup in",
            "object_value": "2018",
            "confidence": 0.7,
        },
    ]
    syn.synthesize(
        "Where is Apple headquartered?",
        facts,
        entity_names={"a": "Apple", "b": "France"},
    )
    prompt = captured["user"]
    assert "Cupertino" in prompt
    assert "World Cup" not in prompt


def test_keyword_filter_does_not_starve_llm_when_no_overlap() -> None:
    """If keyword filtering would drop ALL facts, fall back to sending them
    anyway -- better to spend tokens than answer 'unknown'."""
    captured: Dict[str, str] = {}

    class CapturingLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            captured["user"] = messages[-1]["content"]
            return '{"a":"unknown","f":[]}'

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(CapturingLLM())
    facts = [
        {
            "subject_id": "a",
            "predicate": "is",
            "object_value": "round",
            "confidence": 0.9,
        }
    ]
    syn.synthesize(
        "What did the President say yesterday?",
        facts,
        entity_names={"a": "Earth"},
    )
    assert "round" in captured["user"]


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_compute_confidence_basic() -> None:
    facts = [{"confidence": 0.9}, {"confidence": 0.8}, {"confidence": 0.7}]
    c = AnswerSynthesizer._compute_confidence(facts)
    assert 0.7 < c < 1.0


def test_compute_confidence_empty() -> None:
    assert AnswerSynthesizer._compute_confidence([]) == 0.0


def test_compute_confidence_bounded() -> None:
    facts = [{"confidence": 1.0}] * 100
    assert AnswerSynthesizer._compute_confidence(facts) <= 1.0


# ---------------------------------------------------------------------------
# Entity-name rendering, structural
# ---------------------------------------------------------------------------


def test_format_fact_context_safety_net_when_object_empty() -> None:
    """When the extractor truncated the value into the predicate (object
    slot empty), the renderer must fall back to the original sentence so
    the LLM can answer instead of returning 'unknown'."""
    syn = AnswerSynthesizer(MockLLM())
    fact = {
        "subject_id": "apple",
        "predicate": "reached market cap of",
        "object_value": None,
        "object_id": None,
        "context": "Apple reached a market capitalization of $3 trillion in January 2022.",
        "confidence": 1.0,
    }
    rendered = syn._format_fact(fact, 1, {"apple": "Apple Inc."})
    assert "context:" in rendered
    assert "$3 trillion" in rendered
    assert "January 2022" in rendered


def test_format_fact_context_truncated_at_limit() -> None:
    syn = AnswerSynthesizer(MockLLM())
    long_ctx = "X " * 200  # 400 chars, well above the 160-char limit
    fact = {
        "subject_id": "s",
        "predicate": "p",
        "context": long_ctx,
        "confidence": 1.0,
    }
    rendered = syn._format_fact(fact, 1, {"s": "Subj"})
    # Truncated and ellipsis added
    assert "..." in rendered
    # And capped at the configured limit (+ structural overhead)
    assert len(rendered) < 220


def test_format_fact_uses_entity_names_not_uuids() -> None:
    syn = AnswerSynthesizer(MockLLM())
    fact = {
        "subject_id": "uuid-apple",
        "predicate": "was founded by",
        "object_id": "uuid-jobs",
        "confidence": 1.0,
    }
    rendered = syn._format_fact(
        fact, 1, {"uuid-apple": "Apple Inc.", "uuid-jobs": "Steve Jobs"}
    )
    assert rendered == "F1: Apple Inc. -- was founded by --> Steve Jobs"


def test_safety_rail_caps_runaway_facts() -> None:
    seen_lines: Dict[str, int] = {"n": 0}

    class CountingLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            import re as _re

            seen_lines["n"] = sum(
                1
                for ln in messages[-1]["content"].splitlines()
                if _re.match(r"^F\d+:", ln)
            )
            return '{"a":"ok","f":[]}'

        @property
        def model_name(self) -> str:
            return "mock"

    # Default path: 20 in, 20 out (no artificial cap).
    syn = AnswerSynthesizer(CountingLLM())
    facts = [
        {
            "subject_id": f"s{i}",
            "predicate": "test predicate",
            "object_value": f"value{i}",
            "confidence": 0.5,  # under direct-answer threshold
        }
        for i in range(20)
    ]
    syn.synthesize("test predicate query", facts)
    assert seen_lines["n"] == 20

    # Explicit safety-rail engages when configured.
    syn2 = AnswerSynthesizer(CountingLLM(), max_facts=3)
    syn2.synthesize("test predicate query", facts)
    assert seen_lines["n"] == 3
