import json
import logging
import re
from typing import Any, Dict, List, Optional

from pragma.llm.base import LLMProvider
from pragma.prompts import load_prompt

logger = logging.getLogger(__name__)

_BUILTIN_SYNTHESIS_PROMPT = """You are a precise knowledge synthesis assistant. Your task is to answer questions based solely on the provided atomic facts and reasoning path.

Instructions:
1. Answer ONLY from the provided facts - do not infer information not present
2. Provide step-by-step reasoning showing how facts connect to your answer
3. For each reasoning step, reference the fact ID (e.g., [F001])
4. If facts are insufficient, state "Insufficient information to answer"
5. Assign confidence based on quality and coverage of supporting facts

Output format (JSON):
{
  "answer": "Your answer here",
  "reasoning_steps": [
    {"fact_id": "F001", "explanation": "How this fact supports the answer"},
    {"fact_id": "F002", "explanation": "How this fact supports the answer"}
  ]
}"""


DEFAULT_SYNTHESIS_PROMPT = load_prompt("synthesis", default=_BUILTIN_SYNTHESIS_PROMPT)


class SynthesisOutput:
    """Output from answer synthesis."""

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
    """Synthesize answer from facts and graph path."""

    def __init__(
        self,
        llm: LLMProvider,
    ) -> None:
        self.llm = llm

    def _format_fact(self, fact) -> str:
        """Format fact (dict or string) as string."""
        if isinstance(fact, str):
            return fact
        return f"[{fact.get('id', '?')}] {fact.get('subject_id', '?')} | {fact.get('predicate', '?')} | {fact.get('object_value', fact.get('object_id', '?'))} (confidence: {fact.get('confidence', 1.0):.2f})"

    def synthesize(
        self,
        query: str,
        facts: List[Dict[str, Any]],
        graph_path: List[str],
    ) -> SynthesisOutput:
        """Synthesize answer from facts and graph path.

        Args:
            query: Original user query
            facts: List of fact dicts from assembler
            graph_path: List of reasoning path strings

        Returns:
            SynthesisOutput with answer, reasoning, and confidence
        """
        if not facts:
            return SynthesisOutput(
                answer="Insufficient information to answer",
                reasoning_steps=[],
                confidence=0.0,
            )

        facts_strings = [self._format_fact(f) for f in facts]
        facts_text = "\n".join(facts_strings)
        path_text = "\n".join(graph_path) if graph_path else "No direct path found"

        user_prompt = f"""Question: {query}

Atomic Facts:
{facts_text}

Reasoning Path:
{path_text}

Provide your answer in the specified JSON format."""

        try:
            response = self.llm.complete(
                [
                    {"role": "system", "content": DEFAULT_SYNTHESIS_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                max_tokens=1000,
            )
            if not response or not response.strip():
                logger.warning("Empty LLM response")
                return SynthesisOutput(
                    answer="Empty response from LLM",
                    reasoning_steps=[],
                    confidence=0.0,
                )
            logger.info(f"LLM response: {response[:200]}")
        except Exception as e:
            logger.warning(f"Synthesis LLM call failed: {e}")
            return SynthesisOutput(
                answer="Error during synthesis",
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

        confidence = self._calculate_confidence_from_facts(facts)

        return SynthesisOutput(
            answer=result.get("answer", ""),
            reasoning_steps=result.get("reasoning_steps", []),
            confidence=confidence,
        )

    def _parse_response(self, response: str) -> Optional[Dict[str, Any]]:
        """Parse LLM JSON response."""
        text = response.strip()

        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse error: {e}, trying fallback")

        return self._fallback_parse(text)

    def _fallback_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """Fallback parsing for plain text."""
        lines = text.split("\n")
        answer_lines = []
        reasoning_steps = []

        in_reasoning = False
        for line in lines:
            line = line.strip()
            if not line:
                continue

            if "reasoning" in line.lower():
                in_reasoning = True
                continue

            if in_reasoning and (line.startswith("[") or "fact" in line.lower()):
                reasoning_steps.append({"fact_id": "F001", "explanation": line})
            else:
                answer_lines.append(line)

        if not answer_lines:
            return None

        return {
            "answer": " ".join(answer_lines),
            "reasoning_steps": reasoning_steps[:5],
        }

    def _calculate_confidence(self, facts: List[Dict[str, Any]]) -> float:
        """Calculate confidence from fact dicts."""
        if not facts:
            return 0.0

        confidences = [f.get("confidence", 1.0) for f in facts]
        avg_confidence = sum(confidences) / len(confidences)

        reasoning_bonus = min(len(facts) * 0.02, 0.2)

        return min(avg_confidence + reasoning_bonus, 1.0)

    def _calculate_confidence_from_facts(self, facts: List[str]) -> float:
        """Calculate confidence from formatted fact strings."""
        if not facts:
            return 0.0

        confidences = []
        for fact in facts:
            match = re.search(r"confidence:\s*([\d.]+)", str(fact))
            if match:
                confidences.append(float(match.group(1)))
            else:
                confidences.append(1.0)

        avg = sum(confidences) / len(confidences) if confidences else 0.0
        return min(avg + 0.1, 1.0)
