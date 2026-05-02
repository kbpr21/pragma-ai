"""Answer synthesis: turn (query, facts) into a cited answer with minimum
token cost.

Token-efficiency techniques used here, all empirically justified by the
benchmark in ``benchmarks_run/run.py``:

1. **Compact fact rendering** -- ``F1: <subj> -- <pred> --> <obj>`` using
   entity NAMES, not UUIDs (~10 tokens/fact vs ~30 before).
2. **Query-keyword pre-filter** -- only facts whose subject/predicate/object
   text overlaps with the query make it into the prompt. This commonly drops
   the fact list from ~25 to ~3-5 without hurting answer quality.
3. **Direct-answer fast-path** -- when exactly one filtered fact's subject
   AND predicate both match the query strongly, the answer is its object
   value, returned WITHOUT an LLM call.
4. **Compact JSON output** -- ``{"a": "...", "f": ["F1"]}`` instead of a
   nested ``reasoning_steps`` array. Saves ~50 completion tokens per query.
5. **No graph_path in the prompt** -- the facts already encode structure.
6. **No system-prompt boilerplate** -- single-line directive, no rules list.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from pragma.llm.base import LLMProvider
from pragma.prompts import load_prompt

logger = logging.getLogger(__name__)


# Prompt that teaches the LLM to chain facts for multi-hop questions.
# The original single-line prompt was too terse: the LLM returned
# "unknown" when it needed to compose two facts (e.g. founder + alma
# mater). The revised prompt includes an explicit chaining example and
# a rule that says "combine facts if no single fact answers".
_BUILTIN_SYNTHESIS_PROMPT = (
    "Answer briefly using ONLY these facts. "
    "If no single fact answers, COMBINE multiple facts: e.g. if F1 says "
    "'QubitForge --was founded by--> Sofia Petrova' and F2 says "
    "'Sofia Petrova --studied at--> Cambridge', then for 'Where did the "
    "founder of QubitForge study?' answer Cambridge using [F1,F2]. "
    'Output one JSON object: {"a":"answer","f":["F1","F2"]} -- "f" is the '
    "list of fact IDs you used. If none answer even after combining, set "
    '"a":"unknown" and "f":[].'
)


DEFAULT_SYNTHESIS_PROMPT = load_prompt("synthesis", default=_BUILTIN_SYNTHESIS_PROMPT)


# Tokens we don't count as meaningful for the keyword overlap filter. Keep
# small; over-aggressive filtering hurts recall.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "with",
        "by",
        "from",
        "as",
        "and",
        "or",
        "but",
        "not",
        "no",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "we",
        "you",
        "they",
        "he",
        "she",
        "him",
        "her",
        "his",
        "their",
        "what",
        "who",
        "whom",
        "whose",
        "where",
        "when",
        "why",
        "how",
        "which",
        "tell",
        "me",
        "about",
        "any",
        "some",
        "all",
    }
)

_WORD_RE = re.compile(r"[a-zA-Z0-9]+")


def _query_keywords(query: str) -> List[str]:
    """Return content-bearing lowercase tokens from the query."""
    return [
        w.lower()
        for w in _WORD_RE.findall(query)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    ]


class SynthesisOutput:
    """Output from answer synthesis. Stable shape for downstream consumers."""

    def __init__(
        self,
        answer: str,
        reasoning_steps: List[Dict[str, str]],
        confidence: float,
    ) -> None:
        self.answer = answer
        self.reasoning_steps = reasoning_steps
        self.confidence = confidence


class AnswerSynthesizer:
    """Synthesize answer from facts with minimum token cost.

    The synthesizer is the LAST line of defence on prompt size: the assembler
    has already done coarse filtering, but our pre-filter here uses the
    actual query string to drop irrelevant facts before they enter the prompt.

    ``max_facts`` is a pure safety-rail, not a budget. Real token-budgeting
    is in :class:`pragma.query.assembler.FactAssembler`.
    """

    def __init__(
        self,
        llm: LLMProvider,
        max_facts: Optional[int] = None,
    ) -> None:
        self.llm = llm
        self.max_facts = max_facts if max_facts is not None else 200

    # ------------------------------------------------------------------
    # Fact rendering
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve(
        eid: Any,
        names: Dict[str, str],
        fallback: Any = None,
    ) -> str:
        if not eid:
            return str(fallback) if fallback is not None else "?"
        return names.get(str(eid), str(eid))

    # When the structured triple is malformed (the extractor truncated a
    # value into the predicate, object slot is empty), include up to this
    # many characters of the fact's original sentence so the LLM can still
    # answer. Trade-off: ~20-40 extra prompt tokens per degenerate fact in
    # exchange for not silently answering "unknown".
    _CONTEXT_SAFETY_NET_CHARS: int = 160

    def _format_fact(
        self,
        fact: Any,
        idx: int,
        entity_names: Optional[Dict[str, str]] = None,
    ) -> str:
        """Render one fact compactly, using entity names not UUIDs.

        Falls back to including a truncated ``context`` sentence when the
        object slot is empty -- this guards against extractor failures
        where the value was absorbed into the predicate.
        """
        if isinstance(fact, str):
            return f"F{idx}: {fact}"

        names = entity_names or {}
        subj = self._resolve(fact.get("subject_id"), names, fact.get("subject_name"))
        obj_value = fact.get("object_value")
        obj = obj_value if obj_value else self._resolve(fact.get("object_id"), names)
        pred = fact.get("predicate", "?")

        # Safety net: if the object slot is still empty / "?", emit the
        # source sentence so the LLM has a fallback to read the actual
        # value from. This catches the "predicate absorbed the value"
        # extractor failure mode in v1.0.1 fact data.
        if obj in (None, "", "?", "unknown") or not str(obj).strip():
            context = str(fact.get("context") or "").strip()
            if context:
                if len(context) > self._CONTEXT_SAFETY_NET_CHARS:
                    context = (
                        context[: self._CONTEXT_SAFETY_NET_CHARS - 1].rstrip() + "..."
                    )
                return f"F{idx}: {subj} -- {pred} (context: {context})"

        return f"F{idx}: {subj} -- {pred} --> {obj}"

    # ------------------------------------------------------------------
    # Query-keyword pre-filter
    # ------------------------------------------------------------------

    def _filter_facts_by_query(
        self,
        facts: List[Dict[str, Any]],
        query: str,
        entity_names: Dict[str, str],
    ) -> List[Dict[str, Any]]:
        """Drop facts that share no content keyword with the query.

        Falls back to returning the original list if filtering would leave
        nothing -- it's better to spend a few extra prompt tokens than to
        leave the LLM with no facts at all.
        """
        keywords = _query_keywords(query)
        if not keywords:
            return facts

        scored: List[Tuple[int, Dict[str, Any]]] = []
        for f in facts:
            subj = self._resolve(f.get("subject_id"), entity_names).lower()
            obj_value = (f.get("object_value") or "").lower()
            obj = obj_value or self._resolve(f.get("object_id"), entity_names).lower()
            pred = str(f.get("predicate") or "").lower()
            blob = f"{subj} {pred} {obj}"
            score = sum(1 for kw in keywords if kw in blob)
            if score > 0:
                scored.append((score, f))

        if not scored:
            return facts  # don't starve the LLM if nothing matched

        # Higher overlap first; preserve original order on ties.
        scored.sort(key=lambda t: t[0], reverse=True)
        return [f for _, f in scored]

    # ------------------------------------------------------------------
    # Direct-answer fast-path (zero LLM calls)
    # ------------------------------------------------------------------

    def _try_direct_answer(
        self,
        query: str,
        facts: List[Dict[str, Any]],
        entity_names: Dict[str, str],
    ) -> Optional[SynthesisOutput]:
        """If exactly one fact's subject+predicate match the query, return its
        object as the answer without calling the LLM.

        Heuristics, kept conservative on purpose:
        - need >=1 query keyword in the subject AND >=1 in the predicate
        - require fact confidence >= 0.85
        - only one candidate fact (otherwise we can't know which to pick)
        """
        keywords = _query_keywords(query)
        if not keywords:
            return None

        candidates: List[Dict[str, Any]] = []
        for f in facts:
            if float(f.get("confidence", 0)) < 0.85:
                continue
            subj = self._resolve(f.get("subject_id"), entity_names).lower()
            pred = str(f.get("predicate") or "").lower()
            subj_hit = any(kw in subj for kw in keywords)
            pred_hit = any(kw in pred for kw in keywords)
            if subj_hit and pred_hit:
                candidates.append(f)

        if len(candidates) != 1:
            return None

        f = candidates[0]
        obj_value = f.get("object_value")
        obj = (
            obj_value if obj_value else self._resolve(f.get("object_id"), entity_names)
        )
        if not obj or obj == "?":
            return None

        fact_id_short = "F1"
        return SynthesisOutput(
            answer=str(obj),
            reasoning_steps=[
                {
                    "fact_id": fact_id_short,
                    "explanation": "direct-match: subject+predicate of this fact match the query",
                }
            ],
            confidence=float(f.get("confidence", 1.0)),
        )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def synthesize(
        self,
        query: str,
        facts: List[Dict[str, Any]],
        graph_path: Optional[List[str]] = None,
        entity_names: Optional[Dict[str, str]] = None,
    ) -> SynthesisOutput:
        """Synthesize an answer from facts.

        ``graph_path`` is accepted for backward compatibility but is no
        longer included in the LLM prompt -- the facts already encode the
        structural information and the path text was pure prompt bloat.
        """
        del graph_path  # intentionally ignored (kept for API compatibility)

        if not facts:
            return SynthesisOutput(
                answer="Insufficient information to answer",
                reasoning_steps=[],
                confidence=0.0,
            )

        names = entity_names or {}

        # 1. Pre-filter by query overlap. Often shrinks 25 facts -> 3-5.
        filtered = self._filter_facts_by_query(facts, query, names)

        # 2. Try the direct-answer fast-path (zero LLM calls).
        direct = self._try_direct_answer(query, filtered, names)
        if direct is not None:
            logger.debug("synthesize: direct-answer fast-path used")
            return direct

        # 3. Apply safety-rail cap, render compactly, call the LLM.
        capped_facts = filtered[: self.max_facts]
        facts_text = "\n".join(
            self._format_fact(f, i + 1, names) for i, f in enumerate(capped_facts)
        )

        # No "Q:" / "Facts:" labels -- the model can read the structure.
        user_prompt = f"{query}\n{facts_text}"

        try:
            response = self.llm.complete(
                [
                    {"role": "system", "content": DEFAULT_SYNTHESIS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Synthesis LLM call failed: {e}")
            return SynthesisOutput(
                answer="Error during synthesis",
                reasoning_steps=[],
                confidence=0.0,
            )

        if not response or not response.strip():
            return SynthesisOutput(
                answer="Empty response from LLM",
                reasoning_steps=[],
                confidence=0.0,
            )

        result = self._parse_response(response)
        if result is None:
            return SynthesisOutput(
                answer="Could not parse synthesis response",
                reasoning_steps=[],
                confidence=0.0,
            )

        return SynthesisOutput(
            answer=result.get("answer", ""),
            reasoning_steps=result.get("reasoning_steps", []),
            confidence=self._compute_confidence(capped_facts),
        )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = re.sub(r"^```(?:json)?\s*", "", text.strip())
        text = re.sub(r"```\s*$", "", text)
        return text.strip()

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse the LLM response into a normalised
        ``{answer, reasoning_steps}`` dict.

        Accepts both the new compact schema (``{"a": ..., "f": [...]}``) and
        the legacy verbose schema (``{"answer": ..., "reasoning_steps":
        [...]}``) for forward/backward compatibility. Falls back to
        plain-text parsing if JSON fails.
        """
        text = self._strip_code_fences(response)

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            logger.debug("synthesize: JSON parse failed; using fallback")
            return self._fallback_parse(text)

        if not isinstance(parsed, dict):
            return self._fallback_parse(text)

        # Compact schema  ->  normalise.
        if "a" in parsed:
            answer = str(parsed.get("a", "")).strip()
            ids = parsed.get("f") or []
            if not isinstance(ids, list):
                ids = []
            steps = [{"fact_id": str(fid), "explanation": ""} for fid in ids if fid]
            return {"answer": answer, "reasoning_steps": steps}

        # Legacy schema  ->  pass through with light cleanup.
        answer = str(parsed.get("answer", "")).strip()
        raw_steps = parsed.get("reasoning_steps") or []
        steps = []
        if isinstance(raw_steps, list):
            for s in raw_steps:
                if isinstance(s, dict):
                    steps.append(
                        {
                            "fact_id": str(s.get("fact_id", "")),
                            "explanation": str(s.get("explanation", "")),
                        }
                    )
        return {"answer": answer, "reasoning_steps": steps}

    def _fallback_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Last-resort plain-text parsing when JSON fails."""
        if not text:
            return None
        # Heuristic: take the first non-empty line as the answer.
        for line in text.splitlines():
            line = line.strip()
            if line:
                return {"answer": line, "reasoning_steps": []}
        return None

    # ------------------------------------------------------------------
    # Confidence
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_confidence(facts: List[Dict[str, Any]]) -> float:
        """Confidence = mean(fact.confidence) + small recall bonus.

        Replaces the previous two confidence helpers (``_calculate_confidence``
        and ``_calculate_confidence_from_facts``); they did the same thing
        with different inputs and were both partially broken when the fact
        format changed.
        """
        if not facts:
            return 0.0
        values = [float(f.get("confidence", 1.0)) for f in facts]
        avg = sum(values) / len(values)
        recall_bonus = min(len(values) * 0.02, 0.2)
        return min(avg + recall_bonus, 1.0)
