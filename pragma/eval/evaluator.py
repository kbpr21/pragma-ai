"""Evaluator and metric primitives for pragma.

The evaluator is intentionally LLM-agnostic: it inspects ``PragmaResult``
objects produced by the KnowledgeBase and computes deterministic metrics
(no second-pass LLM scoring). This keeps eval reproducible.

Metrics
-------
* **answer_match**: 1.0 if every ``expected_answer_contains`` substring is
  present in the answer (case-insensitive); else fraction matched.
* **entity_recall**: fraction of ``expected_entities`` that appear in the
  reasoning path or source facts.
* **fact_count**: number of source facts cited.
* **avg_confidence**: mean confidence of source facts.
* **tokens_used**: token estimate produced by the KB.
* **latency_ms**: wall-clock latency for ``kb.query``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class TestCase:
    """A single evaluation case."""

    query: str
    expected_answer_contains: List[str] = field(default_factory=list)
    expected_entities: List[str] = field(default_factory=list)
    expected_hop_depth: Optional[int] = None
    tags: List[str] = field(default_factory=list)

    # Pytest collects classes named ``Test*`` by default. Mark this dataclass
    # as not-a-test so it is silently ignored during collection.
    __test__ = False


@dataclass
class CaseResult:
    """Per-case evaluation output."""

    case: TestCase
    answer: str
    answer_match: float
    entity_recall: float
    fact_count: int
    avg_confidence: float
    tokens_used: int
    latency_ms: float

    def passed(self, threshold: float = 0.5) -> bool:
        return self.answer_match >= threshold


@dataclass
class EvalReport:
    """Aggregate report across all test cases."""

    results: List[CaseResult] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.results)

    @property
    def pass_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed()) / len(self.results)

    @property
    def avg_answer_match(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.answer_match for r in self.results) / len(self.results)

    @property
    def avg_entity_recall(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.entity_recall for r in self.results) / len(self.results)

    @property
    def avg_tokens(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.tokens_used for r in self.results) / len(self.results)

    @property
    def avg_latency_ms(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.latency_ms for r in self.results) / len(self.results)

    def summary(self) -> str:
        return (
            f"pragma eval: {self.n} cases | pass={self.pass_rate:.1%} "
            f"answer_match={self.avg_answer_match:.2f} "
            f"entity_recall={self.avg_entity_recall:.2f} "
            f"avg_tokens={self.avg_tokens:.0f} "
            f"avg_latency={self.avg_latency_ms:.0f}ms"
        )

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "pass_rate": self.pass_rate,
            "avg_answer_match": self.avg_answer_match,
            "avg_entity_recall": self.avg_entity_recall,
            "avg_tokens": self.avg_tokens,
            "avg_latency_ms": self.avg_latency_ms,
            "results": [
                {
                    "query": r.case.query,
                    "answer": r.answer,
                    "answer_match": r.answer_match,
                    "entity_recall": r.entity_recall,
                    "fact_count": r.fact_count,
                    "avg_confidence": r.avg_confidence,
                    "tokens_used": r.tokens_used,
                    "latency_ms": r.latency_ms,
                }
                for r in self.results
            ],
        }


def _answer_match(answer: str, expected: List[str]) -> float:
    """Fraction of expected substrings present in the answer."""
    if not expected:
        return 1.0
    answer_l = (answer or "").lower()
    hits = sum(1 for e in expected if e.lower() in answer_l)
    return hits / len(expected)


def _entity_recall(result: Any, expected: List[str]) -> float:
    """Fraction of expected entities appearing in reasoning/source facts."""
    if not expected:
        return 1.0

    haystack_parts: List[str] = []
    for step in getattr(result, "reasoning_path", []) or []:
        haystack_parts.append(getattr(step, "explanation", "") or "")
    for fact in getattr(result, "source_facts", []) or []:
        haystack_parts.append(getattr(fact, "subject_id", "") or "")
        haystack_parts.append(getattr(fact, "object_id", "") or "")
        haystack_parts.append(getattr(fact, "object_value", "") or "")
        haystack_parts.append(getattr(fact, "context", "") or "")
    haystack = " ".join(haystack_parts).lower()

    hits = sum(1 for e in expected if e.lower() in haystack)
    return hits / len(expected)


class Evaluator:
    """Run a battery of test cases against a KnowledgeBase."""

    def __init__(self, kb: Any, test_cases: List[TestCase]) -> None:
        self.kb = kb
        self.test_cases = test_cases

    def run(self, **query_kwargs: Any) -> EvalReport:
        """Execute every test case and return an aggregate report."""
        results: List[CaseResult] = []
        for case in self.test_cases:
            result = self.kb.query(case.query, **query_kwargs)

            confidences = [
                getattr(f, "confidence", 1.0)
                for f in (getattr(result, "source_facts", []) or [])
            ]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

            results.append(
                CaseResult(
                    case=case,
                    answer=getattr(result, "answer", "") or "",
                    answer_match=_answer_match(
                        getattr(result, "answer", "") or "",
                        case.expected_answer_contains,
                    ),
                    entity_recall=_entity_recall(result, case.expected_entities),
                    fact_count=len(getattr(result, "source_facts", []) or []),
                    avg_confidence=avg_conf,
                    tokens_used=int(getattr(result, "tokens_used", 0) or 0),
                    latency_ms=float(getattr(result, "latency_ms", 0.0) or 0.0),
                )
            )

        return EvalReport(results=results)
