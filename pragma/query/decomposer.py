import json
import logging
import re
from typing import List

from pragma.llm.base import LLMProvider
from pragma.prompts import load_prompt

logger = logging.getLogger(__name__)

_BUILTIN_DECOMPOSE_PROMPT = """You are a query decomposition assistant. Your task is to break down a complex question into simpler atomic sub-questions.

Rules:
1. Each sub-question should be self-contained and answerable independently
2. Break down multi-hop questions into stepwise questions
3. Maximum 5 sub-questions
4. Output only valid JSON array of strings

Output format:
["sub-question 1", "sub-question 2", "sub-question 3"]

Example:
Input: "What country is the CEO of the company that makes iPhone from?"
Output: ["What company makes iPhone?", "Who is the CEO of Apple?", "Where is the CEO born?"]"""


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

        Args:
            query: The original query string

        Returns:
            List of sub-question strings
        """
        if not query or not query.strip():
            return [query]

        user_prompt = f"Input: {query}\nOutput:"

        try:
            response = self.llm.complete(
                [
                    {"role": "system", "content": DEFAULT_DECOMPOSE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=500,
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
