"""Realistic-scale benchmark for pragma vs a top-k vector-RAG baseline.

Why this script exists
----------------------

The original ``benchmarks_run/run.py`` runs against a 4-paragraph
corpus -- great for proving the token claims hold token-for-token, but
not enough to convince anyone that pragma's graph approach scales.
This runner uses the deterministic 50-document corpus from
``large_corpus.py`` (~7k words / ~9k tokens) and a BM25-top-k baseline
that mimics what a realistic vector-RAG implementation would do:
retrieve a handful of relevant chunks rather than stuffing the whole
corpus.

It produces three honest numbers per query:

1. **prompt_tokens / completion_tokens / total_tokens** -- counted by
   the model itself (``prompt_eval_count`` / ``eval_count`` from
   Ollama), not by pragma's word-count approximation.
2. **correctness** -- a substring match against the ground-truth
   ``expects`` list from ``large_corpus.py``. Cheap, but unambiguous
   because we own the corpus.
3. **latency_ms**.

Both pragma and the baseline call the same Ollama model so the only
moving part is the retrieval / reasoning strategy.

Usage
-----

    # First-time / when corpus changes:
    python benchmarks_run/large_corpus.py

    # Fast pragma-only smoke run (skip vector-RAG baseline):
    python benchmarks_run/run_large.py --skip-baseline

    # Full benchmark, default Ollama model:
    python benchmarks_run/run_large.py

    # Different model:
    python benchmarks_run/run_large.py --model llama3.2:3b

    # Reuse an existing pragma KB instead of re-ingesting (huge speedup):
    python benchmarks_run/run_large.py --no-ingest
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import httpx

# Force UTF-8 stdout so Windows PowerShell does not crash on em-dashes.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pragma import KnowledgeBase
from pragma.llm.base import LLMError

DEFAULT_MODEL = "minimax-m2.7:cloud"
OLLAMA_URL = "http://localhost:11434"
ROOT = Path(__file__).parent
CORPUS_DIR = ROOT / "large_corpus"
QUERIES_PATH = ROOT / "queries_large.json"
KB_DIR = ROOT / "kb_large"
RESULTS_PATH = ROOT / "results_large.json"


# ---------------------------------------------------------------------------
# Token-counting Ollama wrapper (same shape as run.py's CountingOllama)
# ---------------------------------------------------------------------------


class CountingOllama:
    """LLMProvider that records true prompt + completion tokens per call.

    Both pragma and the baseline use this so the comparison is
    apples-to-apples on a single model. ``calls`` is a list of dicts
    -- one entry per LLM call -- with ``prompt_tokens``,
    ``completion_tokens``, and a couple of char-level diagnostics.
    """

    def __init__(self, model: str, base_url: str = OLLAMA_URL) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.calls: List[Dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self.model

    def _post(self, payload: dict) -> dict:
        with httpx.Client(timeout=600.0) as client:
            r = client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if "temperature" in kwargs:
            payload["options"] = {"temperature": kwargs["temperature"]}
        # Per-call heartbeat to stderr so a multi-minute ingestion does
        # not look hung. Cheap (a single line per call) and only goes
        # to stderr so the captured stdout summary stays clean.
        n = len(self.calls) + 1
        t0 = time.time()
        sys.stderr.write(f"  [llm #{n:>3}] -> {self.model}  msgs={len(messages)} ... ")
        sys.stderr.flush()
        try:
            data = self._post(payload)
        except httpx.HTTPError as e:
            sys.stderr.write(f"FAILED: {e}\n")
            raise LLMError(f"Ollama call failed: {e}") from e
        content = (data.get("message") or {}).get("content", "")
        prompt_tok = int(data.get("prompt_eval_count") or 0)
        compl_tok = int(data.get("eval_count") or 0)
        sys.stderr.write(
            f"OK in {time.time() - t0:.1f}s "
            f"(prompt={prompt_tok} compl={compl_tok})\n"
        )
        self.calls.append(
            {
                "prompt_tokens": prompt_tok,
                "completion_tokens": compl_tok,
                "char_prompt": sum(len(m.get("content", "")) for m in messages),
                "char_completion": len(content),
            }
        )
        return content

    async def acomplete(self, messages, **kwargs):  # pragma: no cover
        return self.complete(messages, **kwargs)

    async def stream_complete(self, messages, **kwargs):  # pragma: no cover
        yield self.complete(messages, **kwargs)


# ---------------------------------------------------------------------------
# Tiny BM25 implementation -- intentionally simple so the baseline is
# transparent and reproducible; no scikit-learn / no rank-bm25 dep.
# ---------------------------------------------------------------------------


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenise(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class BM25Index:
    """Okapi BM25 over a tiny in-memory corpus.

    Parameters follow the textbook defaults (k1=1.5, b=0.75). The
    implementation is ~30 lines because clarity matters more here than
    the last few percent of retrieval quality -- the point is to give
    the baseline a fair shake, not to beat the state of the art.
    """

    def __init__(self, docs: Dict[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = list(docs.keys())
        self.tokens: Dict[str, List[str]] = {d: _tokenise(t) for d, t in docs.items()}
        self.length: Dict[str, int] = {d: len(toks) for d, toks in self.tokens.items()}
        self.avgdl = (
            (sum(self.length.values()) / len(self.length)) if self.length else 0
        )
        # Document frequency
        df: Counter = Counter()
        for toks in self.tokens.values():
            df.update(set(toks))
        n = len(docs)
        # Robertson-Sparck-Jones IDF clipped at 0 to avoid negatives
        # destroying the score for very common terms.
        self.idf: Dict[str, float] = {
            term: max(0.0, math.log((n - f + 0.5) / (f + 0.5) + 1.0))
            for term, f in df.items()
        }
        # Per-doc term frequency, computed once.
        self.tf: Dict[str, Counter] = {
            d: Counter(toks) for d, toks in self.tokens.items()
        }

    def search(self, query: str, k: int = 3) -> List[Tuple[str, float]]:
        q_terms = _tokenise(query)
        scores: Dict[str, float] = {}
        for d in self.doc_ids:
            tf = self.tf[d]
            dl = self.length[d]
            score = 0.0
            for t in q_terms:
                if t not in tf:
                    continue
                idf = self.idf.get(t, 0.0)
                f = tf[t]
                denom = f + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                score += idf * (f * (self.k1 + 1)) / denom
            if score > 0:
                scores[d] = score
        return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:k]


# ---------------------------------------------------------------------------
# Vector-RAG-style baseline (BM25 top-k chunks)
# ---------------------------------------------------------------------------


def baseline_topk(
    llm: CountingOllama,
    bm25: BM25Index,
    docs: Dict[str, str],
    question: str,
    k: int = 3,
) -> Dict[str, Any]:
    """Retrieve top-``k`` docs by BM25, send them as context, ask the
    question. This is what a basic vector-RAG implementation would do
    after embedding+ANN retrieval -- using BM25 here keeps the
    benchmark dependency-free without changing the qualitative shape
    of the comparison (vector RAG would retrieve roughly the same docs
    on a corpus this small)."""
    hits = bm25.search(question, k=k)
    context = "\n\n---\n\n".join(docs[d] for d, _ in hits)
    system = (
        "You are a helpful assistant. Use ONLY the context below to answer "
        "the user's question. If the context does not contain the answer, "
        "say you don't know. Answer concisely."
    )
    user = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    before = len(llm.calls)
    answer = llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.0,
    )
    new = llm.calls[before:]
    return {
        "answer": answer.strip(),
        "prompt_tokens": sum(c["prompt_tokens"] for c in new),
        "completion_tokens": sum(c["completion_tokens"] for c in new),
        "retrieved_docs": [d for d, _ in hits],
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def is_correct(answer: str, expects: List[str]) -> bool:
    """Substring match (case-insensitive). The ``expects`` list is OR'd:
    any single hit means correct. We own the corpus, so the strings are
    chosen to be unambiguous (we never put 'MIT' as an expected answer
    when 'Massachusetts Institute of Technology' could also appear)."""
    a = answer.lower()
    return any(e.lower() in a for e in expects)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model id")
    ap.add_argument("--ollama-url", default=OLLAMA_URL)
    ap.add_argument("--top-k", type=int, default=3, help="BM25 top-k for baseline")
    ap.add_argument(
        "--no-ingest",
        action="store_true",
        help="Reuse existing kb_large/ instead of re-ingesting",
    )
    ap.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Run pragma only; skip the vector-RAG baseline",
    )
    ap.add_argument(
        "--queries",
        type=int,
        default=None,
        help="Run only the first N queries (for fast smoke tests)",
    )
    args = ap.parse_args()

    if not CORPUS_DIR.exists() or not any(CORPUS_DIR.glob("*.txt")):
        print(
            f"Corpus not materialised. Run:  python {Path(__file__).parent.name}/large_corpus.py",
            file=sys.stderr,
        )
        sys.exit(2)

    # ---- Load corpus + queries -----------------------------------------
    docs: Dict[str, str] = {
        p.name: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.txt"))
    }
    queries: List[Dict[str, Any]] = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    if args.queries:
        queries = queries[: args.queries]

    total_words = sum(len(_tokenise(t)) for t in docs.values())
    print("=" * 80)
    print(f"  pragma large-scale benchmark   (model: {args.model})")
    print("=" * 80)
    print(f"  Corpus  : {len(docs)} documents, ~{total_words} words")
    print(f"  Queries : {len(queries)}")
    print(f"  KB dir  : {KB_DIR}")
    print()

    # ---- Pragma ingestion ---------------------------------------------
    pragma_llm = CountingOllama(model=args.model, base_url=args.ollama_url)
    if not args.no_ingest:
        if KB_DIR.exists():
            shutil.rmtree(KB_DIR)
        kb = KnowledgeBase(llm=pragma_llm, kb_dir=str(KB_DIR))
        print(f"[1] Ingesting {len(docs)} documents into pragma KB...")
        t0 = time.time()
        res = kb.ingest(str(CORPUS_DIR), show_progress=False)
        ing_calls = list(pragma_llm.calls)
        pragma_llm.calls.clear()
        print(
            f"    docs={res.documents} facts={res.facts} entities={res.entities} "
            f"({time.time() - t0:.1f}s, {len(ing_calls)} LLM calls, "
            f"{sum(c['prompt_tokens'] + c['completion_tokens'] for c in ing_calls):,} "
            f"total ingest tokens)"
        )
    else:
        kb = KnowledgeBase(llm=pragma_llm, kb_dir=str(KB_DIR))
        s = kb.stats()
        print(
            f"[1] Reusing existing KB: {s.documents} docs / {s.facts} facts / "
            f"{s.entities} entities"
        )

    # ---- Baseline setup ------------------------------------------------
    bm25 = BM25Index(docs)
    rag_llm = CountingOllama(model=args.model, base_url=args.ollama_url)

    # ---- Per-query evaluation ------------------------------------------
    print("\n[2] Per-query: pragma vs vector-RAG (BM25 top-k=%d)" % args.top_k)
    print("-" * 80)
    rows: List[Dict[str, Any]] = []
    for i, q in enumerate(queries, 1):
        question = q["q"]
        expects = q["expects"]

        # --- pragma
        pragma_llm.calls.clear()
        t0 = time.time()
        pr = kb.query(question)
        pr_lat = (time.time() - t0) * 1000
        pr_prompt = sum(c["prompt_tokens"] for c in pragma_llm.calls)
        pr_compl = sum(c["completion_tokens"] for c in pragma_llm.calls)
        pr_correct = is_correct(pr.answer, expects)

        # --- baseline
        if args.skip_baseline:
            rg = {
                "answer": "(skipped)",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "retrieved_docs": [],
            }
            rg_lat = 0.0
            rg_correct = False
        else:
            rag_llm.calls.clear()
            t0 = time.time()
            rg = baseline_topk(rag_llm, bm25, docs, question, k=args.top_k)
            rg_lat = (time.time() - t0) * 1000
            rg_correct = is_correct(rg["answer"], expects)

        rows.append(
            {
                "q": question,
                "expects": expects,
                "hops": q.get("hops"),
                "kind": q.get("kind"),
                "pragma_answer": pr.answer,
                "pragma_correct": pr_correct,
                "pragma_prompt_tokens": pr_prompt,
                "pragma_completion_tokens": pr_compl,
                "pragma_total_tokens": pr_prompt + pr_compl,
                "pragma_calls": len(pragma_llm.calls),
                "pragma_facts_cited": len(pr.source_facts),
                "pragma_lat_ms": pr_lat,
                "rag_answer": rg["answer"],
                "rag_correct": rg_correct,
                "rag_retrieved": rg["retrieved_docs"],
                "rag_prompt_tokens": rg["prompt_tokens"],
                "rag_completion_tokens": rg["completion_tokens"],
                "rag_total_tokens": rg["prompt_tokens"] + rg["completion_tokens"],
                "rag_lat_ms": rg_lat,
            }
        )

        ok_p = "OK " if pr_correct else "FAIL"
        ok_r = "OK " if rg_correct else "FAIL"
        print(
            f"  [{i:>2}/{len(queries)}] hop={q.get('hops')} {q.get('kind'):<11} "
            f"| pragma {ok_p} {pr_prompt + pr_compl:>5}t  "
            f"| rag {ok_r} {rg['prompt_tokens'] + rg['completion_tokens']:>5}t"
        )

    # ---- Summary -------------------------------------------------------
    n = len(rows)

    def avg(key: str) -> float:
        return sum(r[key] for r in rows) / n if n else 0.0

    print("\n" + "=" * 80)
    print("  Summary")
    print("=" * 80)
    print(f"  Queries: {n}")
    print()
    print("  pragma:")
    print(f"    accuracy             : {sum(r['pragma_correct'] for r in rows)}/{n}")
    print(f"    avg prompt tokens    : {avg('pragma_prompt_tokens'):.0f}")
    print(f"    avg completion tokens: {avg('pragma_completion_tokens'):.0f}")
    print(f"    avg total tokens     : {avg('pragma_total_tokens'):.0f}")
    print(f"    avg LLM calls        : {avg('pragma_calls'):.1f}")
    print(f"    avg latency          : {avg('pragma_lat_ms'):.0f} ms")
    if not args.skip_baseline:
        print()
        print(f"  vector-RAG (BM25 top-{args.top_k}):")
        print(f"    accuracy             : {sum(r['rag_correct'] for r in rows)}/{n}")
        print(f"    avg prompt tokens    : {avg('rag_prompt_tokens'):.0f}")
        print(f"    avg completion tokens: {avg('rag_completion_tokens'):.0f}")
        print(f"    avg total tokens     : {avg('rag_total_tokens'):.0f}")
        print(f"    avg latency          : {avg('rag_lat_ms'):.0f} ms")
        if avg("rag_total_tokens"):
            reduction = (1 - avg("pragma_total_tokens") / avg("rag_total_tokens")) * 100
            print(f"\n  Token reduction (pragma vs RAG): {reduction:+.1f}%")

    RESULTS_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nFull per-query JSON: {RESULTS_PATH}")
    kb.close()


if __name__ == "__main__":
    main()
