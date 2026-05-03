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
    llm = MockLLM('{"a":"Steve Jobs founded the company","f":["F1"]}')
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
    assert "Steve Jobs" in result.answer
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


# ---------------------------------------------------------------------------
# Fragment detection + F-id stripping
# ---------------------------------------------------------------------------


def test_fragment_answer_triggers_refinement() -> None:
    """When the LLM returns a fragment (starts with 'with'), the
    post-processor should detect it and retry with a refinement prompt."""

    class TwoCallLLM:
        def __init__(self) -> None:
            self.calls: int = 0

        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            self.calls += 1
            if self.calls == 1:
                return '{"a":"with learned softmax attention over depth","f":["F1"]}'
            return "AttnRes uses learned softmax attention over the depth dimension"

        @property
        def model_name(self) -> str:
            return "mock"

    llm = TwoCallLLM()
    syn = AnswerSynthesizer(llm)
    facts = [
        {
            "subject_id": "attnres",
            "predicate": "is a drop-in replacement for",
            "object_value": "with learned softmax attention over depth",
            "confidence": 0.9,
        }
    ]
    result = syn.synthesize(
        "What is the core idea behind AttnRes?",
        facts,
        entity_names={"attnres": "AttnRes"},
    )
    assert llm.calls == 2
    assert "AttnRes" in result.answer or "attention" in result.answer.lower()


def test_fid_artifacts_stripped_from_answer() -> None:
    """F-id labels like (F1), [F2] should be removed from the answer."""

    class FidLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            return '{"a":"AttnRes (F1) is a method [F2] that uses depth-wise attention","f":["F1","F2"]}'

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(FidLLM())
    facts = [
        {
            "subject_id": "attnres",
            "predicate": "uses",
            "object_value": "depth-wise attention",
            "confidence": 0.8,
        }
    ]
    result = syn.synthesize(
        "What does AttnRes use?",
        facts,
        entity_names={"attnres": "AttnRes"},
    )
    assert "(F1)" not in result.answer
    assert "[F2]" not in result.answer


def test_confidence_penalized_for_fragment_answer() -> None:
    """Fragment answers should get a confidence penalty."""

    class FragLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            return '{"a":"by maintaining parallel streams","f":["F1"]}'

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(FragLLM())
    facts = [
        {
            "subject_id": "x",
            "predicate": "does",
            "object_value": "something",
            "confidence": 0.9,
        }
    ]
    result = syn.synthesize("What does X do?", facts, entity_names={"x": "X"})
    # The fragment "by maintaining parallel streams" starts with "by",
    # so confidence should be penalized.
    assert result.confidence < 0.9


def test_confidence_zero_for_unknown() -> None:
    """'unknown' answers should get zero confidence."""

    class UnkLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            return '{"a":"unknown","f":[]}'

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(UnkLLM())
    facts = [
        {
            "subject_id": "x",
            "predicate": "does",
            "object_value": "something",
            "confidence": 0.9,
        }
    ]
    result = syn.synthesize("What does X do?", facts, entity_names={"x": "X"})
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Task-type classification
# ---------------------------------------------------------------------------


def test_classify_factoid() -> None:
    from pragma.query.synthesizer import _classify_task, TaskType

    assert _classify_task("Who founded Apple?") == TaskType.FACTOID
    assert _classify_task("What is the capital of France?") == TaskType.FACTOID


def test_classify_summary() -> None:
    from pragma.query.synthesizer import _classify_task, TaskType

    assert _classify_task("Summarize the paper in 3 sentences") == TaskType.SUMMARY
    assert _classify_task("Give an overview of the method") == TaskType.SUMMARY


def test_classify_plan() -> None:
    from pragma.query.synthesizer import _classify_task, TaskType

    assert (
        _classify_task("Turn the core idea into a 10-step implementation plan")
        == TaskType.PLAN
    )
    assert _classify_task("Step-by-step roadmap for deployment") == TaskType.PLAN


def test_classify_analogy() -> None:
    from pragma.query.synthesizer import _classify_task, TaskType

    assert (
        _classify_task("Relate AttnRes to database query optimization")
        == TaskType.ANALOGY
    )
    assert (
        _classify_task("Draw an analogy between transformers and RNNs")
        == TaskType.ANALOGY
    )


def test_classify_multi_question() -> None:
    from pragma.query.synthesizer import _classify_task, TaskType

    q = "What parts discuss efficiency? Which sections talk about scaling? Where is BlockAttnRes justified?"
    assert _classify_task(q) == TaskType.MULTI_QUESTION


# ---------------------------------------------------------------------------
# Multi-question splitting
# ---------------------------------------------------------------------------


def test_split_questions() -> None:
    from pragma.query.synthesizer import _split_questions

    q = "What parts discuss efficiency? Which sections talk about scaling? Where is BlockAttnRes justified?"
    parts = _split_questions(q)
    assert len(parts) == 3
    assert all(p.endswith("?") for p in parts)


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------


def test_truncation_detected_incomplete_json() -> None:
    from pragma.query.synthesizer import AnswerSynthesizer

    assert AnswerSynthesizer._looks_truncated('{"a":"The paper introduces mHC')


def test_truncation_detected_mid_word() -> None:
    from pragma.query.synthesizer import AnswerSynthesizer

    assert AnswerSynthesizer._looks_truncated(
        '{"a":"The paper introduces mHC-lite, a streamlined variant of the '
        "Manifold-Constrained Hyper-Connection framework that replaces the "
        'standard 20-iteration Sinkhorn-Knop","f":["F1"]}'
    )


def test_truncation_not_detected_for_complete_answer() -> None:
    from pragma.query.synthesizer import AnswerSynthesizer

    assert not AnswerSynthesizer._looks_truncated(
        '{"a":"Steve Jobs founded the company","f":["F1"]}'
    )
    assert not AnswerSynthesizer._looks_truncated(
        '{"a":"The answer is 42.","f":["F1"]}'
    )


# ---------------------------------------------------------------------------
# Hallucination detection (via confidence scoring)
# ---------------------------------------------------------------------------


def test_hallucination_penalized_in_confidence() -> None:
    """Answers with content words not grounded in facts should get
    a confidence penalty."""

    class HallucLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            return '{"a":"The method uses database indexing and query optimization to speed up retrieval","f":["F1"]}'

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(HallucLLM())
    # Use low-confidence facts so the direct-answer fast-path is skipped.
    facts = [
        {
            "subject_id": "attnres",
            "predicate": "uses",
            "object_value": "depth-wise softmax attention",
            "confidence": 0.5,
        }
    ]
    result = syn.synthesize(
        "What does AttnRes use?",
        facts,
        entity_names={"attnres": "AttnRes"},
    )
    assert result.confidence < 0.9


def test_grounded_answer_not_penalized() -> None:
    """Answers grounded in facts should NOT get a hallucination penalty."""

    class GroundedLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            return '{"a":"AttnRes uses depth-wise softmax attention over the layer dimension","f":["F1"]}'

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(GroundedLLM())
    facts = [
        {
            "subject_id": "attnres",
            "predicate": "uses",
            "object_value": "depth-wise softmax attention",
            "confidence": 0.9,
        }
    ]
    result = syn.synthesize(
        "What does AttnRes use?",
        facts,
        entity_names={"attnres": "AttnRes"},
    )
    assert result.confidence >= 0.7


def test_multi_question_partial_coverage_penalized() -> None:
    """Multi-question answers with 'Not covered' sub-questions
    should get a proportional confidence penalty."""

    class PartialLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            return (
                '{"a":"1. AttnRes uses depth-wise softmax attention. '
                "2. Not covered in available facts. "
                "3. Not covered in available facts."
                '","f":["F1"]}'
            )

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(PartialLLM())
    facts = [
        {
            "subject_id": "attnres",
            "predicate": "uses",
            "object_value": "depth-wise softmax attention",
            "confidence": 0.9,
        }
    ]
    query = "What does AttnRes use? Why is it better? How does it scale?"
    result = syn.synthesize(query, facts, entity_names={"attnres": "AttnRes"})
    # 2/3 sub-questions uncovered → penalty of ~0.33
    assert result.confidence < 0.8


def test_multi_question_full_coverage_not_penalized() -> None:
    """Multi-question answers with no 'Not covered' should not get
    the partial coverage penalty."""

    class FullLLM:
        def complete(self, messages: List[Dict[str, str]], **kw: Any) -> str:
            return (
                '{"a":"1. AttnRes uses depth-wise softmax attention. '
                "2. It is better because it weights contributions. "
                "3. It scales via Block AttnRes."
                '","f":["F1","F2","F3"]}'
            )

        @property
        def model_name(self) -> str:
            return "mock"

    syn = AnswerSynthesizer(FullLLM())
    facts = [
        {
            "subject_id": "attnres",
            "predicate": "uses",
            "object_value": "depth-wise softmax attention",
            "confidence": 0.9,
        },
        {
            "subject_id": "attnres",
            "predicate": "is better because",
            "object_value": "it weights contributions",
            "confidence": 0.9,
        },
        {
            "subject_id": "blockattnres",
            "predicate": "scales via",
            "object_value": "Block AttnRes",
            "confidence": 0.9,
        },
    ]
    query = "What does AttnRes use? Why is it better? How does it scale?"
    result = syn.synthesize(
        query,
        facts,
        entity_names={"attnres": "AttnRes", "blockattnres": "BlockAttnRes"},
    )
    # No "Not covered" → no partial coverage penalty
    assert result.confidence >= 0.7
