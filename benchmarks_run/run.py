"""End-to-end honesty test for pragma's tokens-per-query and atomic-fact claims.

Runs the full ingest + query pipeline against a real Ollama model
(`minimax-m2.7:cloud`) and captures TRUE prompt/completion token counts as
reported by the model, not pragma's word-count approximation.

Usage (from repo root):
    python benchmarks_run/run.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List

import httpx

# Force UTF-8 stdout/stderr so PowerShell's cp1252 default doesn't crash on
# em-dashes / non-breaking hyphens that the model emits.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pragma import KnowledgeBase
from pragma.llm.base import LLMError

OLLAMA_URL = "http://localhost:11434"
MODEL = "minimax-m2.7:cloud"

DOC = Path("benchmarks_run/apple.txt")
KB_DIR = Path("benchmarks_run/kb")

QUESTIONS = [
    # 1. Single-fact lookup -- exercises the founding triple.
    "Who founded Apple?",
    # 2. Multi-hop: needs to find Tim Cook (mentioned in para 2) AND his
    # birthplace fact. Both pieces are extracted as separate atomic
    # facts in pragma's KB; the synthesizer must link them.
    "Where was Tim Cook born?",
]


# ----------------------------------------------------------------------------
# Token-counting Ollama wrapper
# ----------------------------------------------------------------------------


class CountingOllama:
    """Drop-in LLMProvider that records true prompt + completion token counts.

    Uses the Ollama /api/chat endpoint with stream=False so the response
    JSON includes `prompt_eval_count` and `eval_count`, which are produced
    by the model's own tokenizer.
    """

    def __init__(self, model: str = MODEL, base_url: str = OLLAMA_URL) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.calls: List[Dict[str, Any]] = []  # one entry per LLM call

    @property
    def model_name(self) -> str:
        return self.model

    def _post(self, payload: dict) -> dict:
        with httpx.Client(timeout=300.0) as client:
            r = client.post(f"{self.base_url}/api/chat", json=payload)
            r.raise_for_status()
            return r.json()

    def complete(self, messages: List[Dict[str, str]], **kwargs: Any) -> str:
        payload = {"model": self.model, "messages": messages, "stream": False}
        if "temperature" in kwargs:
            payload["options"] = {"temperature": kwargs["temperature"]}
        import sys

        t0 = time.time()
        sys.stderr.write(f"  -> ollama call (msg={len(messages)} ...) ")
        sys.stderr.flush()
        try:
            data = self._post(payload)
        except httpx.HTTPError as e:
            sys.stderr.write(f" FAILED: {e}\n")
            raise LLMError(f"Ollama call failed: {e}") from e
        content = (data.get("message") or {}).get("content", "")
        prompt_tok = int(data.get("prompt_eval_count") or 0)
        compl_tok = int(data.get("eval_count") or 0)
        sys.stderr.write(
            f"OK in {time.time() - t0:.1f}s (prompt={prompt_tok} compl={compl_tok})\n"
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

    async def stream_complete(  # pragma: no cover
        self, messages, **kwargs
    ) -> AsyncGenerator[str, None]:
        yield self.complete(messages, **kwargs)


# ----------------------------------------------------------------------------
# Vector-RAG-style baseline (top-k chunks, no graph)
# ----------------------------------------------------------------------------


def vector_rag_baseline(llm: CountingOllama, doc_text: str, question: str) -> dict:
    """Naive vector-RAG simulation: send top-k chunks (or whole doc if small)
    plus question to the LLM, measure tokens. The doc here is small enough
    that "top-k" reduces to "the whole doc"; that's representative because
    real vector RAG sends ~3,000 tokens by stuffing every retrieved chunk."""
    system = (
        "You are a helpful assistant. Use the context below to answer the "
        "user's question. If the answer is not in the context, say you don't "
        "know."
    )
    user = f"Context:\n{doc_text}\n\nQuestion: {question}\n\nAnswer:"
    before = len(llm.calls)
    answer = llm.complete(
        [{"role": "system", "content": system}, {"role": "user", "content": user}]
    )
    new = llm.calls[before:]
    return {
        "answer": answer.strip(),
        "prompt_tokens": sum(c["prompt_tokens"] for c in new),
        "completion_tokens": sum(c["completion_tokens"] for c in new),
    }


# ----------------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------------


def main() -> None:
    if KB_DIR.exists():
        import shutil

        shutil.rmtree(KB_DIR)

    print("=" * 76)
    print(f"  pragma honesty benchmark    (Ollama model: {MODEL})")
    print("=" * 76)

    # --- Ingest -------------------------------------------------------------
    pragma_llm = CountingOllama()
    kb = KnowledgeBase(llm=pragma_llm, kb_dir=str(KB_DIR))

    print("\n[1] Ingesting", DOC, "...")
    t0 = time.time()
    res = kb.ingest(str(DOC))
    print(
        f"    documents={res.documents}  facts={res.facts}  "
        f"entities={res.entities}  ({time.time() - t0:.1f}s)"
    )
    ingest_calls = list(pragma_llm.calls)
    pragma_llm.calls.clear()

    stats = kb.stats()
    print(
        f"    KB stats: {stats.documents} docs / {stats.facts} facts / "
        f"{stats.entities} entities / {stats.relationships} edges"
    )

    # --- Show every extracted atomic fact -----------------------------------
    print("\n[2] Atomic facts extracted (first 30):")
    print("-" * 76)
    conn = kb._storage._get_connection()
    rows = conn.execute(
        "SELECT id, subject_id, predicate, object_id, object_value, confidence "
        "FROM facts WHERE is_active = 1 ORDER BY confidence DESC"
    ).fetchall()
    ent_name = {}
    for r in conn.execute("SELECT id, name FROM entities").fetchall():
        ent_name[r["id"]] = r["name"]
    for r in rows[:30]:
        subj = ent_name.get(r["subject_id"], r["subject_id"] or "?")
        obj = (
            ent_name.get(r["object_id"], r["object_id"])
            if r["object_id"]
            else (r["object_value"] or "?")
        )
        print(
            f"  [{r['id'][:8]}] {subj} -- {r['predicate']} --> {obj} "
            f"(conf={r['confidence']:.2f})"
        )
    if len(rows) > 30:
        print(f"  ... and {len(rows) - 30} more")

    print(f"\n    Ingestion LLM calls: {len(ingest_calls)}")
    print(
        f"    Total ingestion prompt tokens:   "
        f"{sum(c['prompt_tokens'] for c in ingest_calls):,}"
    )
    print(
        f"    Total ingestion completion tokens: "
        f"{sum(c['completion_tokens'] for c in ingest_calls):,}"
    )

    # --- Per-query benchmark ------------------------------------------------
    print("\n[3] Per-query: pragma  vs  vector-RAG-style baseline")
    print("-" * 76)

    rag_llm = CountingOllama()  # separate counter for clean comparison
    doc_text = DOC.read_text(encoding="utf-8")

    rows = []
    for q in QUESTIONS:
        # pragma
        pragma_llm.calls.clear()
        t0 = time.time()
        r = kb.query(q)
        prag_lat = (time.time() - t0) * 1000
        prag_prompt = sum(c["prompt_tokens"] for c in pragma_llm.calls)
        prag_compl = sum(c["completion_tokens"] for c in pragma_llm.calls)
        prag_calls = len(pragma_llm.calls)

        # vector-RAG baseline
        rag_llm.calls.clear()
        t0 = time.time()
        rag = vector_rag_baseline(rag_llm, doc_text, q)
        rag_lat = (time.time() - t0) * 1000

        rows.append(
            {
                "q": q,
                "pragma_answer": r.answer,
                "pragma_prompt": prag_prompt,
                "pragma_compl": prag_compl,
                "pragma_total": prag_prompt + prag_compl,
                "pragma_approx": r.tokens_used,  # pragma's own approx
                "pragma_calls": prag_calls,
                "pragma_facts_cited": len(r.source_facts),
                "pragma_lat_ms": prag_lat,
                "rag_answer": rag["answer"],
                "rag_prompt": rag["prompt_tokens"],
                "rag_compl": rag["completion_tokens"],
                "rag_total": rag["prompt_tokens"] + rag["completion_tokens"],
                "rag_lat_ms": rag_lat,
            }
        )

        print(f"\nQ: {q}")
        print(f"  pragma     -> {r.answer[:120]}")
        print(
            f"               prompt={prag_prompt:>4}  compl={prag_compl:>4}  "
            f"total={prag_prompt + prag_compl:>4}  calls={prag_calls}  "
            f"facts={len(r.source_facts)}  approx={r.tokens_used}  "
            f"lat={prag_lat:.0f}ms"
        )
        print(f"  vector-RAG -> {rag['answer'][:120]}")
        print(
            f"               prompt={rag['prompt_tokens']:>4}  "
            f"compl={rag['completion_tokens']:>4}  "
            f"total={rag['prompt_tokens'] + rag['completion_tokens']:>4}  "
            f"calls=1  lat={rag_lat:.0f}ms"
        )

    # --- Summary ------------------------------------------------------------
    print("\n" + "=" * 76)
    print("  Summary")
    print("=" * 76)
    n = len(rows)
    avg = lambda k: sum(r[k] for r in rows) / n  # noqa: E731
    print(f"  Avg pragma     prompt tokens : {avg('pragma_prompt'):.0f}")
    print(f"  Avg pragma     compl  tokens : {avg('pragma_compl'):.0f}")
    print(f"  Avg pragma     total  tokens : {avg('pragma_total'):.0f}")
    print(f"  Avg pragma     approx (README claim metric) : {avg('pragma_approx'):.0f}")
    print(f"  Avg vector-RAG prompt tokens : {avg('rag_prompt'):.0f}")
    print(f"  Avg vector-RAG total  tokens : {avg('rag_total'):.0f}")
    print(
        f"  Token reduction              : "
        f"{(1 - avg('pragma_total') / avg('rag_total')) * 100:.1f}%  "
        f"(prompt-only: "
        f"{(1 - avg('pragma_prompt') / avg('rag_prompt')) * 100:.1f}%)"
    )

    out = Path("benchmarks_run/results.json")
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nFull per-query JSON written to {out}")

    kb.close()


if __name__ == "__main__":
    main()
