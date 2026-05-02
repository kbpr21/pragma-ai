"""Tests for the deterministic large-corpus benchmark fixtures.

We don't (and cannot) run the full benchmark in CI -- that needs a live
Ollama -- but we DO test the corpus generator and the baseline's
retrieval pieces, because they are the "ground truth" the benchmark's
honesty depends on. If the corpus generator silently changes between
runs, every reported number is suspect.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def benchmarks_path():
    """Add benchmarks_run/ to sys.path for the duration of the test so
    the modules under it are importable."""
    p = Path(__file__).resolve().parents[2] / "benchmarks_run"
    sys.path.insert(0, str(p))
    try:
        yield p
    finally:
        sys.path.remove(str(p))


def test_corpus_is_deterministic(tmp_path: Path, benchmarks_path: Path) -> None:
    """Materialising the corpus twice must produce byte-identical files
    -- otherwise the benchmark cannot claim reproducibility."""
    lc = importlib.import_module("large_corpus")
    out1 = tmp_path / "run1"
    out2 = tmp_path / "run2"
    q1 = tmp_path / "q1.json"
    q2 = tmp_path / "q2.json"
    s1 = lc.materialise(corpus_dir=out1, queries_path=q1)
    s2 = lc.materialise(corpus_dir=out2, queries_path=q2)
    # Compare only the content-bearing fields; the path fields differ
    # by construction since each run wrote to a different tmpdir.
    keep = {"documents", "words", "approx_tokens", "queries"}
    assert {k: v for k, v in s1.items() if k in keep} == {
        k: v for k, v in s2.items() if k in keep
    }
    files1 = sorted(out1.glob("*.txt"))
    files2 = sorted(out2.glob("*.txt"))
    assert [f.name for f in files1] == [f.name for f in files2]
    for a, b in zip(files1, files2):
        assert (
            a.read_bytes() == b.read_bytes()
        ), f"non-deterministic content in {a.name}"
    assert q1.read_bytes() == q2.read_bytes()


def test_corpus_has_at_least_50_docs_and_5k_tokens(
    tmp_path: Path, benchmarks_path: Path
) -> None:
    """The user explicitly asked for >=50 documents and >=5k tokens of
    knowledge -- pin that as a regression test so any future trim of
    the data tables fails loudly here rather than silently shrinking
    the benchmark's signal."""
    lc = importlib.import_module("large_corpus")
    stats = lc.materialise(
        corpus_dir=tmp_path / "corpus", queries_path=tmp_path / "q.json"
    )
    assert stats["documents"] >= 50, stats
    assert stats["approx_tokens"] >= 5000, stats
    assert stats["queries"] >= 10, stats


def test_every_company_founder_resolves_to_a_person(benchmarks_path: Path) -> None:
    """Cross-reference invariant: every Company.founder must match a
    Person.name exactly. If this drifts, multi-hop queries will silently
    degrade because the graph cannot link the two halves of the path."""
    lc = importlib.import_module("large_corpus")
    person_names = {p.name for p in lc.PEOPLE}
    for c in lc.COMPANIES:
        assert (
            c.founder in person_names
        ), f"company {c.name!r} founder {c.founder!r} not in PEOPLE"


def test_every_query_expectation_appears_in_corpus(
    tmp_path: Path, benchmarks_path: Path
) -> None:
    """Sanity: at least one of each query's expected substrings must
    actually appear in the corpus. If not, the benchmark would mark
    pragma 'wrong' even when the pipeline is functioning correctly --
    a worst-of-both-worlds failure mode that this test prevents."""
    lc = importlib.import_module("large_corpus")
    lc.materialise(corpus_dir=tmp_path / "c", queries_path=tmp_path / "q.json")
    all_text = "\n".join(
        p.read_text(encoding="utf-8").lower() for p in (tmp_path / "c").glob("*.txt")
    )
    for q in lc.QUERIES:
        assert any(
            e.lower() in all_text for e in q["expects"]
        ), f"query {q['q']!r} expects {q['expects']} but none appear in the corpus"


def test_bm25_retrieves_obviously_relevant_doc(benchmarks_path: Path) -> None:
    """Smoke test on the BM25 index: the doc explicitly named in the
    query should be in the top-3. Nothing fancy -- just guarding against
    a future regression that breaks the baseline's retrieval."""
    rl = importlib.import_module("run_large")
    docs = {
        "alpha.txt": "Helix Robotics is a robotics company headquartered in Austin.",
        "beta.txt": "BlueCell Storage builds grid-scale batteries from Stockholm.",
        "gamma.txt": "QubitForge designs quantum chips out of Cambridge, UK.",
    }
    bm25 = rl.BM25Index(docs)
    top = [d for d, _ in bm25.search("Where is Helix Robotics?", k=2)]
    assert "alpha.txt" in top


def test_is_correct_or_semantics(benchmarks_path: Path) -> None:
    """``is_correct`` must short-circuit on the first matching expected
    substring (case-insensitive) and never raise for empty inputs."""
    rl = importlib.import_module("run_large")
    assert rl.is_correct("Maya Chen founded it.", ["Maya Chen"])
    assert rl.is_correct("MAYA chen", ["maya chen"])
    assert rl.is_correct(
        "PalmPay was the prior company.", ["PalmPay Africa", "PalmPay"]
    )
    assert not rl.is_correct("I do not know.", ["Maya Chen"])
    assert not rl.is_correct("", ["anything"])
