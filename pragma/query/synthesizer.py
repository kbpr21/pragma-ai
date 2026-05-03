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
7. **Task-type detection** -- queries are classified (factoid, summary,
   plan, analogy, multi-question) and receive specialised prompts.
8. **Hallucination detection** -- answers are checked for grounding in
   the provided facts; ungrounded answers get confidence penalties.
9. **Truncation detection** -- mid-sentence cutoffs trigger a retry.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

from pragma.llm.base import LLMProvider
from pragma.prompts import load_prompt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Task-type classification
# ---------------------------------------------------------------------------


class TaskType(Enum):
    """What kind of reasoning task does the query require?"""

    FACTOID = auto()  # Single fact or simple composition
    SUMMARY = auto()  # "Summarize / overview / describe"
    PLAN = auto()  # "Implementation plan / steps / roadmap"
    ANALOGY = auto()  # "Relate X to Y / compare / analogy"
    MULTI_QUESTION = auto()  # Multiple distinct questions in one query


def _classify_task(query: str) -> TaskType:
    """Classify the query's reasoning task type.

    Uses keyword heuristics — fast, no LLM call needed.
    """
    q = query.strip().lower()

    # Multi-question: multiple question marks or line breaks with questions.
    if q.count("?") >= 2:
        return TaskType.MULTI_QUESTION

    # Summary / overview / describe.
    summary_keywords = {
        "summarize",
        "summary",
        "overview",
        "describe",
        "explain in detail",
        "give an overview",
        "main points",
        "key takeaways",
        "in 3 sentences",
        "in 5 sentences",
        "brief overview",
        "high-level overview",
    }
    if any(kw in q for kw in summary_keywords):
        return TaskType.SUMMARY

    # Plan / steps / roadmap / implementation.
    plan_keywords = {
        "implementation plan",
        "step-by-step",
        "steps to",
        "roadmap",
        "how to implement",
        "how would you build",
        "turn into a plan",
        "implementation steps",
    }
    if any(kw in q for kw in plan_keywords):
        return TaskType.PLAN

    # Analogy / relate / compare.
    analogy_keywords = {
        "relate",
        "analogy",
        "analogous",
        "compare to",
        "similar to",
        "like in",
        "parallel between",
        "map to",
        "bridge between",
    }
    if any(kw in q for kw in analogy_keywords):
        return TaskType.ANALOGY

    return TaskType.FACTOID


def _split_questions(query: str) -> List[str]:
    """Split a multi-question query into individual questions.

    Handles:
    - Multiple sentences ending in '?'
    - Line-break separated questions
    - Semicolon-separated questions
    """
    # Split on question marks followed by whitespace or line breaks.
    parts = re.split(r"\?\s*[\n\r]+|\?\s+", query.strip())
    # Re-attach the '?' to each part.
    result = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not p.endswith("?"):
            p += "?"
        result.append(p)
    if not result:
        return [query]
    return result


# Prompt that teaches the LLM to chain facts for multi-hop questions.
# The original single-line prompt was too terse: the LLM returned
# "unknown" when it needed to compose two facts (e.g. founder + alma
# mater). The revised prompt includes an explicit chaining example and
# a rule that says "combine facts if no single fact answers".
_BUILTIN_SYNTHESIS_PROMPT = (
    "You are a precise reasoning engine. Answer the question using ONLY the "
    "provided facts. Follow these rules strictly:\n"
    "\n"
    "1. SYNTHESIZE, do not paraphrase. Combine multiple facts when no single "
    "fact answers the question. Chain across facts: if F1 links A to B and F2 "
    "links B to C, you may conclude A→C.\n"
    "\n"
    "2. For questions asking about problems, methods, or results, structure "
    "your answer as: problem → method → result. For 'what are the N issues' "
    "questions, list ALL issues found in the facts, not just one.\n"
    "\n"
    "3. ANSWER COMPLETENESS: If the question asks about multiple things (e.g. "
    "'what are the three issues'), your answer must address ALL of them. If "
    "the facts only cover some, state what is covered and note gaps.\n"
    "\n"
    "4. NEVER return a bare fragment like 'with learned softmax attention' as "
    "an answer. Every answer must be a complete, self-contained sentence or "
    "phrase that makes sense without seeing the facts.\n"
    "\n"
    "5. NEVER include fact IDs (F1, F2, etc.) or labels in the answer text. "
    "They go in the 'f' array only.\n"
    "\n"
    "6. If the facts are insufficient to answer even after combining, output "
    '{"a":"unknown","f":[],"reason":"brief explanation of what is missing"}.\n'
    "\n"
    "7. Do NOT hallucinate information not present in the facts.\n"
    "\n"
    'Output one JSON object: {"a":"your answer","f":["F1","F2"]}. '
    'The "f" array lists the fact IDs you used. The answer "a" must be '
    "a complete, readable response — never a raw predicate fragment."
)

# Task-specific prompts. Each extends the base prompt with rules
# tailored to the reasoning task type.

_SUMMARY_PROMPT = (
    "You are summarizing a research document from its atomic facts. "
    "Follow these rules strictly:\n"
    "\n"
    "1. SYNTHESIZE an abstract-level understanding from the facts. "
    "Do NOT just list facts — build a coherent narrative.\n"
    "\n"
    "2. Structure: problem → approach → key result. Cover each in "
    "1-2 sentences. Preserve technical terms and quantities.\n"
    "\n"
    "3. If the facts are insufficient for a full summary, summarize "
    "what IS covered and note what is missing. Never fabricate.\n"
    "\n"
    "4. NEVER include fact IDs (F1, F2) in the answer text.\n"
    "\n"
    "5. Complete all sentences. Do NOT truncate mid-sentence.\n"
    "\n"
    'Output one JSON object: {"a":"your summary","f":["F1","F2"]}.'
)

_PLAN_PROMPT = (
    "You are turning research findings into an implementation plan. "
    "Follow these rules strictly:\n"
    "\n"
    "1. Derive each step from the facts. If a step has no factual "
    "basis, mark it as [inferred] and keep it minimal.\n"
    "\n"
    "2. Number each step. Be specific — reference actual methods, "
    "quantities, and design choices from the facts.\n"
    "\n"
    "3. If the facts are insufficient for a complete plan, produce "
    "the steps that ARE grounded and note what is missing.\n"
    "\n"
    "4. NEVER include fact IDs (F1, F2) in the answer text.\n"
    "\n"
    'Output one JSON object: {"a":"numbered plan steps","f":["F1","F2"]}.'
)

_ANALOGY_PROMPT = (
    "You are drawing an analogy between a research concept and an "
    "external domain, using the provided facts. Rules:\n"
    "\n"
    "1. Ground every claim in a specific fact. If no fact directly "
    "supports the analogy, try to construct one from the factual "
    "attributes: what the concept DOES, how it WORKS, what it "
    "REPLACES. Then map those attributes to the target domain.\n"
    "\n"
    "2. Mark inferred analogies as [speculative]. Keep them brief.\n"
    "\n"
    "3. Only output "
    '{"a":"unknown","f":[],"reason":"..."} if the facts contain '
    "ABSOLUTELY NO information about the source concept. If there "
    "are ANY facts about the concept, use them to build the analogy.\n"
    "\n"
    "4. NEVER include fact IDs (F1, F2) in the answer text.\n"
    "\n"
    'Output one JSON object: {"a":"your analogy","f":["F1","F2"]}.'
)

_MULTI_QUESTION_PROMPT = (
    "You are answering a multi-part question. Follow these rules:\n"
    "\n"
    "1. Answer EVERY sub-question separately. Label each answer "
    "with the sub-question it addresses.\n"
    "\n"
    "2. Before saying 'Not covered in available facts' for a "
    "sub-question, try to INFER the answer by combining multiple "
    "facts. Chain across facts: if F1 says X does A and F2 says A "
    "causes B, you may conclude X leads to B. Mark such inferences "
    "as [inferred].\n"
    "\n"
    "3. Only say 'Not covered' if NO combination of facts provides "
    "even a partial answer to that sub-question.\n"
    "\n"
    "4. NEVER return a single fragment that only addresses one part.\n"
    "\n"
    "5. NEVER include fact IDs (F1, F2) in the answer text.\n"
    "\n"
    'Output one JSON object: {"a":"your multi-part answer","f":["F1","F2"]}.'
)

# Map task types to their specialised prompts.
_TASK_PROMPTS: Dict[TaskType, str] = {
    TaskType.FACTOID: _BUILTIN_SYNTHESIS_PROMPT,
    TaskType.SUMMARY: _SUMMARY_PROMPT,
    TaskType.PLAN: _PLAN_PROMPT,
    TaskType.ANALOGY: _ANALOGY_PROMPT,
    TaskType.MULTI_QUESTION: _MULTI_QUESTION_PROMPT,
}


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


def _resolve_eid(
    eid: Any,
    names: Dict[str, str],
    fallback: Any = None,
) -> str:
    """Resolve an entity ID to a name, for use outside the class."""
    if not eid:
        return str(fallback) if fallback is not None else "?"
    return names.get(str(eid), str(eid))


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

        Also matches against the fact's ``context`` field so that facts
        whose subject/predicate/object are terse but whose source
        sentence is rich in query-relevant terms still survive the
        filter. Falls back to returning the original list if filtering
        would leave nothing.
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
            context = str(f.get("context") or "").lower()
            blob = f"{subj} {pred} {obj} {context}"
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

        # 1. Classify the task type and select the appropriate prompt.
        task = _classify_task(query)
        prompt = _TASK_PROMPTS.get(task, _BUILTIN_SYNTHESIS_PROMPT)
        logger.debug("synthesize: task type=%s, prompt selected", task.name)

        # For multi-question queries, increase max_tokens since the
        # answer needs to cover multiple parts.
        max_tokens = 600
        if task == TaskType.MULTI_QUESTION:
            max_tokens = 1000
        elif task == TaskType.SUMMARY:
            max_tokens = 900
        elif task == TaskType.PLAN:
            max_tokens = 900

        # 2. Pre-filter by query overlap. Often shrinks 25 facts -> 3-5.
        # For multi-question queries, be less aggressive — keep more
        # facts so each sub-question has a chance of finding its facts.
        filtered = self._filter_facts_by_query(facts, query, names)
        if task == TaskType.MULTI_QUESTION and len(filtered) < 5:
            # Too few facts after filtering — use the full set.
            filtered = facts

        # 3. Try the direct-answer fast-path (zero LLM calls).
        # Only for FACTOID tasks — summary/plan/analogy/multi-question
        # always need LLM synthesis.
        if task == TaskType.FACTOID:
            direct = self._try_direct_answer(query, filtered, names)
            if direct is not None:
                logger.debug("synthesize: direct-answer fast-path used")
                return direct

        # 4. Apply safety-rail cap, render compactly, call the LLM.
        capped_facts = filtered[: self.max_facts]
        facts_text = "\n".join(
            self._format_fact(f, i + 1, names) for i, f in enumerate(capped_facts)
        )

        # No "Q:" / "Facts:" labels -- the model can read the structure.
        user_prompt = f"{query}\n{facts_text}"

        try:
            response = self.llm.complete(
                [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=max_tokens,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Synthesis LLM call failed: {e}")
            return SynthesisOutput(
                answer="Error during synthesis",
                reasoning_steps=[],
                confidence=0.0,
            )

        # Retry with 2x budget if the model returned empty (common with
        # diffusion models like Mercury that consume reasoning tokens).
        if not response or not response.strip():
            logger.debug("Synthesis: empty response, retrying with 2x max_tokens")
            try:
                response = self.llm.complete(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens * 2,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(f"Synthesis LLM retry failed: {e}")

        if not response or not response.strip():
            return SynthesisOutput(
                answer="Empty response from LLM",
                reasoning_steps=[],
                confidence=0.0,
            )

        # 5. Truncation detection: if the response looks cut off
        # (ends mid-sentence without closing the JSON), retry with
        # more tokens and a continuation hint.
        if self._looks_truncated(response):
            logger.debug("synthesize: detected truncated response, retrying")
            try:
                cont_response = self.llm.complete(
                    [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens * 2,
                )
                if cont_response and cont_response.strip():
                    # Use the longer response if it parses.
                    cont_result = self._parse_response(cont_response)
                    if cont_result is not None:
                        response = cont_response
            except Exception:  # noqa: BLE001
                pass  # Keep the original truncated response.

        result = self._parse_response(response)
        if result is None:
            return SynthesisOutput(
                answer="Could not parse synthesis response",
                reasoning_steps=[],
                confidence=0.0,
            )

        answer = result.get("answer", "")
        reasoning_steps = result.get("reasoning_steps", [])

        # 6. Post-processing: clean and validate the answer.
        answer = self._postprocess_answer(answer, query, capped_facts, names)

        # 7. Compute confidence with hallucination-aware scoring.
        confidence = self._compute_confidence(capped_facts, answer, query, names)

        return SynthesisOutput(
            answer=answer,
            reasoning_steps=reasoning_steps,
            confidence=confidence,
        )

    # ------------------------------------------------------------------
    # Truncation detection
    # ------------------------------------------------------------------

    @staticmethod
    def _looks_truncated(response: str) -> bool:
        """Heuristic: does the response look like it was cut off?

        Signs of truncation:
        - JSON is incomplete (no closing ``}``)
        - Answer text ends mid-word (last token is a partial word)
        - Answer text ends with a comma or colon mid-sentence

        NOT truncation (these are normal):
        - Answer just missing a trailing period
        - Answer ending with a closing quote or paren
        """
        text = response.strip()
        if not text:
            return False

        # Check for incomplete JSON — definitive truncation.
        if text.startswith("{") and not text.rstrip().endswith("}"):
            return True

        # Check if the answer text inside JSON is cut off mid-word.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "a" in parsed:
                answer_text = str(parsed["a"]).strip()
                if answer_text:
                    last = answer_text[-1]
                    # Mid-word truncation: ends with a partial word
                    # (letter followed by nothing, common when
                    # max_tokens cuts off mid-token).
                    # E.g. "Sinkhorn-Knop" → truncated.
                    # But "The answer is 42" → not truncated.
                    if last.isalpha() and len(answer_text) > 50:
                        # Long answer ending with a letter (no
                        # punctuation) is suspicious.
                        # Check if the last few chars look like a
                        # cut-off word (no space before the last
                        # 5 chars).
                        tail = answer_text[-10:]
                        if " " not in tail[-5:]:
                            return True
                    # Ends with comma/colon mid-enumeration.
                    if last in {",", ":"} and len(answer_text) > 30:
                        return True
        except json.JSONDecodeError:
            pass

        return False

    # ------------------------------------------------------------------
    # Answer post-processing
    # ------------------------------------------------------------------

    # Regex that matches F-id artifacts like (F1), [F1], F1, F12, etc.
    _FID_RE = re.compile(r"\s*[\(\[]?F\d+[\)\]]?")

    # Refinement prompt used when the initial answer is a fragment or
    # otherwise underspecified.
    _REFINEMENT_PROMPT = (
        "The previous answer was a fragment or incomplete. "
        "Given the question and facts below, write a COMPLETE, "
        "self-contained answer that directly addresses the question. "
        "Do not repeat the fragment — restructure it into a proper "
        "sentence. Output plain text only, no JSON."
    )

    def _postprocess_answer(
        self,
        answer: str,
        query: str,
        facts: List[Dict[str, Any]],
        entity_names: Dict[str, str],
    ) -> str:
        """Clean and validate the synthesized answer.

        Steps:
        1. Strip F-id artifacts (F1, (F2), [F3]) from the answer text.
        2. Detect fragment answers (starts with lowercase preposition,
           very short, or looks like a predicate tail).
        3. If the answer is a fragment, retry with a refinement prompt.
        4. Strip trailing punctuation artifacts.
        """
        if not answer or not answer.strip():
            return answer

        # Step 1: Strip F-id artifacts.
        cleaned = self._FID_RE.sub("", answer).strip()

        # Step 2: Detect fragment answers.
        if self._is_fragment(cleaned, query):
            logger.debug(
                "synthesize: detected fragment answer, retrying with refinement"
            )
            refined = self._refine_answer(cleaned, query, facts, entity_names)
            if refined and refined.strip():
                return refined

        # Step 3: Clean trailing artifacts (orphan commas, dashes).
        cleaned = re.sub(r"[\s,;:]+$", "", cleaned)
        return cleaned

    @staticmethod
    def _is_fragment(answer: str, query: str) -> bool:
        """Heuristic to detect underspecified / fragment answers.

        A fragment is an answer that:
        - Starts with a lowercase preposition/conjunction ("with", "by",
          "from", "of", "and", "or", "than", "over", "through", "via")
        - Is very short (1-2 words) and does not contain any noun from
          the query AND doesn't start with a capital letter (proper noun)
        - Looks like a predicate tail rather than a noun phrase
        """
        if not answer:
            return True
        words = answer.split()

        # Check for preposition/conjunction starters.
        first = words[0].lower()
        fragment_starters = {
            "with",
            "by",
            "from",
            "of",
            "and",
            "or",
            "than",
            "over",
            "through",
            "via",
            "using",
            "than",
            "which",
            "that",
            "where",
            "when",
            "while",
        }
        if first in fragment_starters:
            return True

        # Very short answers (1-2 words) that don't start with a
        # capital letter and don't contain query nouns are fragments.
        if len(words) <= 2:
            if words[0][0].isupper() or words[0][0].isdigit():
                return False  # Proper noun or number — not a fragment
            query_lower = query.lower()
            answer_lower = answer.lower()
            content_words = [w for w in answer_lower.split() if len(w) > 3]
            if not any(w in query_lower for w in content_words):
                return True

        return False

    def _refine_answer(
        self,
        fragment: str,
        query: str,
        facts: List[Dict[str, Any]],
        entity_names: Dict[str, str],
    ) -> Optional[str]:
        """Retry with a refinement prompt when the answer is a fragment."""
        facts_text = "\n".join(
            self._format_fact(f, i + 1, entity_names)
            for i, f in enumerate(facts[: self.max_facts])
        )
        user_prompt = (
            f"Question: {query}\n"
            f"Fragment answer: {fragment}\n"
            f"Facts:\n{facts_text}\n\n"
            f"{self._REFINEMENT_PROMPT}"
        )
        try:
            response = self.llm.complete(
                [
                    {"role": "system", "content": self._REFINEMENT_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=400,
            )
        except Exception:  # noqa: BLE001
            return None

        if not response or not response.strip():
            return None

        # Strip any JSON/code fences the model might still produce.
        text = self._strip_code_fences(response).strip()
        # If the model returned JSON, try to extract the "a" field.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and "a" in parsed:
                return str(parsed["a"]).strip()
        except json.JSONDecodeError:
            pass
        # Otherwise return the plain text, stripping F-id artifacts.
        return self._FID_RE.sub("", text).strip()

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
    def _compute_confidence(
        facts: List[Dict[str, Any]],
        answer: str = "",
        query: str = "",
        entity_names: Optional[Dict[str, str]] = None,
    ) -> float:
        """Confidence = mean(fact.confidence) + recall bonus - penalties.

        Penalties applied:
        - Fragment answer: answer starts with a preposition or is ≤2
          words without query nouns → −0.3
        - Very short answer (≤3 words): −0.15
        - "unknown" answer: 0.0
        - Low keyword overlap between answer and query: −0.1
        - Hallucination: answer contains significant content not
          grounded in the provided facts → −0.3

        The base confidence is the mean of the underlying fact
        confidences plus a small recall bonus (capped at 0.2).
        """
        if not facts:
            return 0.0

        # "unknown" answers get zero confidence.
        if answer and answer.strip().lower() == "unknown":
            return 0.0

        values = [float(f.get("confidence", 1.0)) for f in facts]
        avg = sum(values) / len(values)
        recall_bonus = min(len(values) * 0.02, 0.2)
        base = min(avg + recall_bonus, 1.0)

        # No answer text to validate — return base.
        if not answer or not query:
            return base

        penalty = 0.0

        # Fragment penalty: answer starts with a preposition.
        answer_words = answer.strip().split()
        if answer_words:
            first = answer_words[0].lower()
            fragment_starters = {
                "with",
                "by",
                "from",
                "of",
                "and",
                "or",
                "than",
                "over",
                "through",
                "via",
                "using",
            }
            if first in fragment_starters:
                penalty += 0.3

        # Very short answer penalty (but not for proper nouns).
        if len(answer_words) <= 3:
            if not (answer_words[0][0].isupper() or answer_words[0][0].isdigit()):
                penalty += 0.15

        # Low keyword overlap between answer and query.
        answer_lower = answer.lower()
        query_kw = [
            w.lower()
            for w in _WORD_RE.findall(query)
            if len(w) > 2 and w.lower() not in _STOPWORDS
        ]
        if query_kw:
            overlap = sum(1 for kw in query_kw if kw in answer_lower)
            overlap_ratio = overlap / len(query_kw)
            if overlap_ratio < 0.2:
                penalty += 0.1

        # Hallucination detection: check if the answer's content
        # words are grounded in the provided facts. Extract
        # significant content words from the answer, then check
        # what fraction appear in the fact texts.
        names = entity_names or {}
        fact_text_blob = " ".join(
            f"{_resolve_eid(f.get('subject_id'), names)} "
            f"{f.get('predicate', '')} "
            f"{f.get('object_value', '')} "
            f"{_resolve_eid(f.get('object_id'), names)} "
            f"{f.get('context', '')}"
            for f in facts
        ).lower()

        # Extract content words from the answer (longer than 4 chars,
        # not stopwords, not common verbs).
        answer_content_words = [
            w.lower()
            for w in _WORD_RE.findall(answer)
            if len(w) > 4
            and w.lower() not in _STOPWORDS
            and w.lower()
            not in {
                "which",
                "where",
                "while",
                "therefore",
                "however",
                "because",
                "although",
                "further",
                "between",
                "through",
                "without",
                "another",
                "whether",
                "either",
                "neither",
                "instead",
                "despite",
                "during",
                "before",
                "after",
                "since",
                "until",
                "unless",
                "whereas",
            }
        ]

        if answer_content_words:
            grounded = sum(1 for w in answer_content_words if w in fact_text_blob)
            grounding_ratio = grounded / len(answer_content_words)
            # If less than 30% of answer content words appear in
            # the facts, the answer likely contains hallucinations.
            if grounding_ratio < 0.3:
                penalty += 0.3
                logger.debug(
                    "confidence: low grounding %.0f%% (%d/%d words in facts)",
                    grounding_ratio * 100,
                    grounded,
                    len(answer_content_words),
                )

        # Multi-question partial coverage penalty: if the answer
        # contains "Not covered" for some sub-questions, reduce
        # confidence proportionally. E.g. 3/8 "Not covered" → 0.375
        # penalty, so confidence drops from ~1.0 to ~0.6.
        if query.count("?") >= 2:
            # Count sub-questions from the query.
            sub_q_count = query.count("?")
            # Count "Not covered" occurrences in the answer.
            not_covered_count = answer_lower.count("not covered")
            if sub_q_count > 0 and not_covered_count > 0:
                coverage_ratio = 1.0 - (not_covered_count / sub_q_count)
                # Penalty = (1 - coverage_ratio) * 0.5
                # So 50% uncovered → 0.25 penalty, 75% uncovered → 0.375
                partial_penalty = (1.0 - coverage_ratio) * 0.5
                penalty += partial_penalty
                logger.debug(
                    "confidence: multi-question coverage %.0f%% "
                    "(%d/%d covered), penalty +%.2f",
                    coverage_ratio * 100,
                    sub_q_count - not_covered_count,
                    sub_q_count,
                    partial_penalty,
                )

        return max(base - penalty, 0.0)
