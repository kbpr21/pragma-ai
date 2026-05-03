import json
import logging
import re
from typing import List

from pragma.llm.base import LLMProvider
from pragma.prompts import load_prompt

logger = logging.getLogger(__name__)

_BUILTIN_DECOMPOSE_PROMPT = (
    "Split the question into atomic sub-questions. "
    "If it's already atomic, return it unchanged. "
    "Output ONLY a JSON array of strings, no preamble. "
    'Example: ["What company makes iPhone?", "Who is its CEO?"]'
)


# Heuristic: queries that are short and have a single question word are
# almost never multi-hop. Decomposing them via an LLM call is pure waste.
_SIMPLE_QUERY_MAX_WORDS = 14
_MULTI_HOP_HINTS = (
    " and ",
    " then ",
    " who also ",
    " whose ",
    " which is ",
    " that is ",
)


def _looks_simple(query: str) -> bool:
    q = query.strip().lower()
    # Multi-question queries are NEVER simple — they need decomposition
    # so each sub-question gets its own BM25 retrieval pass.
    if q.count("?") >= 2:
        return False
    if len(q.split()) <= _SIMPLE_QUERY_MAX_WORDS and q.count("?") <= 1:
        if not any(hint in q for hint in _MULTI_HOP_HINTS):
            return True
    return False


DEFAULT_DECOMPOSE_PROMPT = load_prompt(
    "query_decompose", default=_BUILTIN_DECOMPOSE_PROMPT
)


class QueryDecomposer:
    """Decompose complex queries into sub-questions."""

    def __init__(
        self,
        llm: LLMProvider,
        max_subquestions: int = 5,
    ) -> None:
        self.llm = llm
        self.max_subquestions = max_subquestions

    def decompose(self, query: str) -> List[str]:
        """Decompose a query into sub-questions.

        Returns the original query as a single-element list when the query
        looks simple enough to skip the LLM call (most common case).
        """
        if not query or not query.strip():
            return [query]

        # Fast-path: simple queries don't need decomposition. This eliminates
        # ~80% of decompose LLM calls in real workloads.
        if _looks_simple(query):
            return [query]

        try:
            response = self.llm.complete(
                [
                    {"role": "system", "content": DEFAULT_DECOMPOSE_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
                max_tokens=200,
            )
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}")
            return [query]

        sub_questions = self._parse_response(response)

        if not sub_questions or len(sub_questions) <= 1:
            return [query]

        return sub_questions[: self.max_subquestions]

    def _parse_response(self, response: str) -> List[str]:
        """Parse LLM response into list of sub-questions."""
        text = response.strip()

        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [str(q).strip() for q in parsed if str(q).strip()]
        except json.JSONDecodeError:
            pass

        return self._fallback_parse(text)

    def _fallback_parse(self, text: str) -> List[str]:
        """Fallback parsing when JSON fails."""
        lines = re.split(r"[\n\r]+", text)
        questions = []

        for line in lines:
            line = line.strip()
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r'^"\s*', "", line)
            line = re.sub(r'\s*"$', "", line)

            if line and ("?" in line or len(line) > 10):
                questions.append(line)

        if not questions and text:
            if "," in text:
                questions = [q.strip() for q in text.split(",")]
            elif " and " in text.lower():
                parts = re.split(r"\s+and\s+", text, maxsplit=4)
                questions = [p.strip() for p in parts if p.strip()]

        return [q for q in questions if q][: self.max_subquestions]
