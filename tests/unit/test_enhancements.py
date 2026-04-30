"""Regression tests for enhancements (Apr 2026 audit).

Each test pins behaviour that was either broken or inconsistent in the
original LLM-generated codebase.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List
from unittest.mock import patch

import pytest

from pragma import KnowledgeBase
from pragma.eval import Evaluator, TestCase
from pragma.ingestion.extractor import FactExtractor
from pragma.llm.base import LLMError
from pragma.models import AtomicFact, PragmaResult, ReasoningStep
from pragma.prompts import load_prompt
from pragma.storage.sqlite import SQLiteStore


class _MockLLM:
    """Minimal mock LLM with all Protocol methods."""

    model = "mock"

    def __init__(self, response: str = "[]") -> None:
        self.response = response

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        return self.response

    async def acomplete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        return self.response

    async def stream_complete(
        self, messages: List[Dict[str, str]], **kwargs: Any
    ) -> AsyncGenerator[str, None]:
        for ch in self.response:
            yield ch

    @property
    def model_name(self) -> str:
        return "mock"


# ---------------------------------------------------------------------------
# 1. LLMError must be importable from pragma.llm.base (regression)
# ---------------------------------------------------------------------------


def test_llm_error_importable_from_base() -> None:
    """`from pragma.llm.base import LLMError` is what every provider does."""
    assert issubclass(LLMError, Exception)


# ---------------------------------------------------------------------------
# 2. All providers honour the LLMProvider protocol (stream_complete present)
# ---------------------------------------------------------------------------


def test_all_providers_have_stream_complete() -> None:
    from pragma.llm.anthropic import AnthropicProvider
    from pragma.llm.groq import GroqProvider
    from pragma.llm.inception import InceptionProvider
    from pragma.llm.ollama import OllamaProvider
    from pragma.llm.openai import OpenAIProvider

    for cls in (
        GroqProvider,
        OpenAIProvider,
        AnthropicProvider,
        OllamaProvider,
        InceptionProvider,
    ):
        assert hasattr(
            cls, "stream_complete"
        ), f"{cls.__name__} missing stream_complete"
        assert hasattr(cls, "complete")
        assert hasattr(cls, "acomplete")


# ---------------------------------------------------------------------------
# 3. acomplete must work with kwargs (Inception bug)
# ---------------------------------------------------------------------------


def test_inception_acomplete_accepts_kwargs(monkeypatch) -> None:
    monkeypatch.setenv("INCEPTION_API_KEY", "test-key")
    from pragma.llm.inception import InceptionProvider

    p = InceptionProvider()

    async def fake_complete_async() -> str:
        # complete() will be called via run_in_executor with kwargs bound by partial
        with patch.object(p, "complete", return_value="ok") as mock:
            out = await p.acomplete(
                [{"role": "user", "content": "hi"}], temperature=0.7
            )
            # Verify kwargs were forwarded
            mock.assert_called_once()
            _, kwargs = mock.call_args
            assert kwargs.get("temperature") == 0.7
            return out

    assert asyncio.run(fake_complete_async()) == "ok"


# ---------------------------------------------------------------------------
# 4. doc_id hashes file content, not just path
# ---------------------------------------------------------------------------


def test_doc_id_dedupes_identical_content_at_different_paths(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("Hello world", encoding="utf-8")
    b.write_text("Hello world", encoding="utf-8")

    kb = KnowledgeBase(llm=_MockLLM(), kb_dir=str(tmp_path / "kb"))
    try:
        assert kb._compute_doc_id(str(a)) == kb._compute_doc_id(str(b))
        # Different content -> different id
        c = tmp_path / "c.txt"
        c.write_text("Different content", encoding="utf-8")
        assert kb._compute_doc_id(str(a)) != kb._compute_doc_id(str(c))
    finally:
        kb.close()


# ---------------------------------------------------------------------------
# 5. Confidence is clamped to [0, 1]
# ---------------------------------------------------------------------------


def test_extractor_clamps_confidence() -> None:
    extractor = FactExtractor(_MockLLM())
    facts = extractor._validate_facts(
        [
            {"subject": "A", "predicate": "is", "object": "B", "confidence": 1.7},
            {"subject": "C", "predicate": "is", "object": "D", "confidence": -0.4},
            {"subject": "E", "predicate": "is", "object": "F", "confidence": "junk"},
        ]
    )
    assert facts[0]["confidence"] == 1.0
    assert facts[1]["confidence"] == 0.0
    assert facts[2]["confidence"] == 1.0  # default on parse failure


# ---------------------------------------------------------------------------
# 6. Prompts are loadable from .txt files with env override
# ---------------------------------------------------------------------------


def test_load_prompt_returns_file_contents() -> None:
    text = load_prompt("fact_extraction", default="FALLBACK")
    assert "atomic fact extractor" in text.lower()
    assert text != "FALLBACK"


def test_load_prompt_env_override(tmp_path: Path, monkeypatch) -> None:
    custom = tmp_path / "custom.txt"
    custom.write_text("CUSTOM PROMPT", encoding="utf-8")
    monkeypatch.setenv("PRAGMA_PROMPT_FACT_EXTRACTION", str(custom))
    load_prompt.cache_clear()
    assert load_prompt("fact_extraction", default="x") == "CUSTOM PROMPT"
    load_prompt.cache_clear()


# ---------------------------------------------------------------------------
# 7. Query cache round-trips confidence and source_facts
# ---------------------------------------------------------------------------


def test_query_cache_preserves_confidence_and_source_facts(tmp_path: Path) -> None:
    store = SQLiteStore(str(tmp_path))
    fact = AtomicFact(
        id="f1",
        subject_id="A",
        predicate="is",
        object_value="company",
        confidence=0.92,
    )
    result = PragmaResult(
        answer="A is a company",
        reasoning_path=[ReasoningStep(fact_id="f1", explanation="cited", hop_number=0)],
        source_facts=[fact],
        confidence=0.87,
        tokens_used=42,
        latency_ms=12.3,
        subgraph_size=3,
    )
    store.save_query_cache("h1", "what is A?", result)

    loaded = store.get_query_cache("h1")
    assert loaded is not None
    assert loaded.confidence == pytest.approx(0.87)
    assert loaded.tokens_used == 42
    assert loaded.subgraph_size == 3
    assert len(loaded.source_facts) == 1
    assert loaded.source_facts[0].id == "f1"
    assert loaded.source_facts[0].confidence == pytest.approx(0.92)
    store.close()


# ---------------------------------------------------------------------------
# 8. Evaluator scaffold works end-to-end with a stub KB
# ---------------------------------------------------------------------------


class _StubKB:
    def query(self, q: str, **_: Any) -> PragmaResult:
        return PragmaResult(
            answer="Steve Jobs co-founded Apple",
            reasoning_path=[
                ReasoningStep(
                    fact_id="f1",
                    explanation="Apple founded by Steve Jobs",
                    hop_number=0,
                )
            ],
            source_facts=[
                AtomicFact(
                    id="f1",
                    subject_id="Apple",
                    predicate="founded by",
                    object_id="Steve Jobs",
                    confidence=0.95,
                )
            ],
            confidence=0.9,
            tokens_used=120,
            latency_ms=42.0,
        )


def test_evaluator_runs_and_aggregates() -> None:
    cases = [
        TestCase(
            query="Who founded Apple?",
            expected_answer_contains=["Steve Jobs"],
            expected_entities=["Apple", "Steve Jobs"],
        ),
        TestCase(
            query="Trick question",
            expected_answer_contains=["Bill Gates"],
        ),
    ]
    report = Evaluator(_StubKB(), cases).run()
    assert report.n == 2
    assert 0.0 < report.pass_rate <= 1.0
    assert report.results[0].answer_match == pytest.approx(1.0)
    assert report.results[0].entity_recall == pytest.approx(1.0)
    assert report.results[1].answer_match == pytest.approx(0.0)
    summary = report.summary()
    assert "pragma eval" in summary
    assert "pass=" in summary


# ---------------------------------------------------------------------------
# 9. CLI does not crash on entities/clear/facts (smoke test)
# ---------------------------------------------------------------------------


def test_cli_clear_help_does_not_raise() -> None:
    from typer.testing import CliRunner

    from pragma.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["clear", "--help"])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# 10. KnowledgeBase.query returns populated source_facts
# ---------------------------------------------------------------------------


def test_decomposer_skips_llm_for_simple_queries() -> None:
    """Short single-clause queries must NOT trigger the decompose LLM call.

    Pins the v1.0.2 token-efficiency optimisation: simple queries take the
    fast-path so we save one LLM round-trip per query.
    """
    from pragma.query.decomposer import QueryDecomposer

    class CountingLLM:
        def __init__(self) -> None:
            self.calls = 0

        def complete(self, *a: Any, **kw: Any) -> str:
            self.calls += 1
            return '["should not see this"]'

        @property
        def model_name(self) -> str:
            return "mock"

    llm = CountingLLM()
    dec = QueryDecomposer(llm)
    for q in [
        "Who founded Apple?",
        "What year was Tim Cook born?",
        "Where is Apple Park located?",
    ]:
        out = dec.decompose(q)
        assert out == [q], f"expected fast-path for {q!r}, got {out}"
    assert llm.calls == 0, "fast-path must not call the LLM"


# Synthesizer-specific behaviour (entity-name rendering, safety-rail,
# direct-answer fast-path, query-keyword filter, schema parsing) is fully
# covered by tests/unit/test_synthesizer.py. They were duplicated here in
# the v1.0.2 work-in-progress branch and have been consolidated.


def test_ingest_long_text_string_does_not_crash(tmp_path: Path) -> None:
    """Regression: ``kb.ingest(text)`` with a string longer than the
    POSIX NAME_MAX (255 chars) used to crash on Linux because
    ``Path(source).is_dir()`` raised OSError. Pin: it must be treated
    as raw text and reach the ingestion pipeline.

    Reproduces the CI failure in
    ``tests/benchmarks/test_ingestion.py::test_pdf_pages_per_minute``.
    """
    kb = KnowledgeBase(llm=_MockLLM(), kb_dir=str(tmp_path))
    long_text = "Apple Inc. was founded in 1976. " * 30  # ~960 chars
    # Should not raise OSError("File name too long").
    result = kb.ingest(long_text)
    assert result is not None
    kb.close()


def test_query_populates_source_facts(tmp_path: Path) -> None:
    """Mocking the full query pipeline to confirm source_facts is populated."""
    import networkx as nx

    kb = KnowledgeBase(llm=_MockLLM(), kb_dir=str(tmp_path))

    # Fake decomposer / retriever / traverser / assembler / synthesizer.
    fake_subgraph = nx.MultiDiGraph()
    fake_subgraph.add_node("A")
    fake_subgraph.add_node("B")
    fake_subgraph.add_edge("A", "B", key="e1", predicate="is")

    facts = [
        {
            "id": "f1",
            "subject_id": "A",
            "predicate": "is",
            "object_id": "B",
            "object_value": None,
            "confidence": 0.9,
            "context": "ctx",
            "is_active": True,
        }
    ]

    with (
        patch("pragma.query.decomposer.QueryDecomposer") as Dec,
        patch("pragma.query.retriever.BM25Retriever") as Ret,
        patch("pragma.graph.traversal.GraphTraverser") as Trav,
        patch("pragma.query.assembler.FactAssembler") as Ass,
        patch("pragma.query.synthesizer.AnswerSynthesizer") as Syn,
    ):
        Dec.return_value.decompose.return_value = ["q"]

        seed = type("E", (), {"id": "A", "name": "A"})()
        Ret.return_value.find_seed_entities.return_value = [seed]

        traverser_inst = Trav.return_value
        traverser_inst.extract_subgraph.return_value = fake_subgraph
        traverser_inst.get_reasoning_paths.return_value = ["A [is] B"]

        Ass.return_value.assemble_facts.return_value = facts

        synth_out = type(
            "S",
            (),
            {
                "answer": "A is B",
                "reasoning_steps": [{"fact_id": "f1", "explanation": "cited"}],
                "confidence": 0.9,
            },
        )()
        Syn.return_value.synthesize.return_value = synth_out

        result = kb.query("what is A?")

    assert result.answer == "A is B"
    assert len(result.source_facts) == 1
    assert result.source_facts[0].id == "f1"
    assert result.source_facts[0].confidence == pytest.approx(0.9)
    kb.close()
