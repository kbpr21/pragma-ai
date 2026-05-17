import json
import logging
import re
from typing import Any, Dict, List

from pragma.ingestion.preprocessor import ProcessedSegment
from pragma.llm.base import LLMProvider
from pragma.prompts import load_prompt

logger = logging.getLogger(__name__)

_BUILTIN_FACT_PROMPT = """You are an industrial-grade atomic fact extractor. Your job is to decompose text into the smallest possible self-contained propositions while PRESERVING SEMANTIC NUANCE.

CORE RULES:
1. Each fact must be independently verifiable without reading other facts.
2. Resolve all pronouns to their referents.
3. Extract numbers and dates exactly as they appear.
4. Preserve negation: "X does NOT cause Y" is a valid fact.
5. Do NOT infer facts not in the text.

FIELD DISCIPLINE (makes facts queryable):
6. The "predicate" is a SHORT verb phrase (1-5 words), e.g. "founded",
   "is CEO of", "reached market cap of", "was born on". It MUST NOT
   contain values, numbers, dates, money amounts, or named entities.
7. Concrete values (numbers, dates, money like "$1 trillion", places,
   percentages, durations) belong in "object_value" — NEVER in the
   predicate. If the value is itself an entity, put its name in "object".
8. WRONG: predicate="reached a market capitalization of $1 trillion".
   RIGHT: predicate="reached market cap of", object_value="$1 trillion in August 2018".
9. Always populate "context" with the full original sentence.

SEMANTIC NUANCE PRESERVATION (critical for reasoning quality):
10. "modality" classifies the epistemic status of the claim:
    - "assertion": definitive statement of fact ("X is Y", "X does Y")
    - "hypothesis": speculative or uncertain ("X may Y", "X appears to Y", "X suggests Y")
    - "negation": explicit denial ("X does NOT Y", "X is not Y")
    - "conditional": depends on conditions ("If X then Y", "X under Z conditions")
    - "comparative": relative claim ("X outperforms Y", "X is better than Y")
11. "is_speculative": set to true when the text uses hedging language
    (e.g. "appears to", "may", "might", "suggests", "could", "seems",
    "it is possible that", "under certain conditions", "arguably").
12. "hedge_phrase": when is_speculative is true, copy the EXACT hedging
    phrase from the source text (e.g. "appears to", "under scaling-law
    conditions"). Leave null for assertions.
13. Confidence tiers:
    - 1.0 = explicitly and definitively stated
    - 0.85 = strongly implied with clear evidence
    - 0.7 = inferred from context
    - 0.55 = speculative, hedged, or uncertain language

EXAMPLES:
Input: "AttnRes appears to mitigate PreNorm dilution under scaling-law conditions."
Output: {
  "subject": "AttnRes",
  "predicate": "mitigates",
  "object": "PreNorm dilution",
  "object_value": null,
  "context": "AttnRes appears to mitigate PreNorm dilution under scaling-law conditions.",
  "confidence": 0.55,
  "modality": "hypothesis",
  "is_speculative": true,
  "hedge_phrase": "appears to ... under scaling-law conditions"
}

Input: "Apple was founded on April 1, 1976."
Output: {
  "subject": "Apple",
  "predicate": "was founded on",
  "object": null,
  "object_value": "April 1, 1976",
  "context": "Apple was founded on April 1, 1976.",
  "confidence": 1.0,
  "modality": "assertion",
  "is_speculative": false,
  "hedge_phrase": null
}

Output ONLY valid JSON, no preamble, no markdown fences:
[
  {
    "subject": "string (canonical noun phrase)",
    "predicate": "string (short verb phrase, NO values)",
    "object": "string (entity name) OR null",
    "object_value": "string (literal value/number/date/money) OR null",
    "context": "string (original sentence this came from)",
    "confidence": float,
    "modality": "assertion|hypothesis|negation|conditional|comparative",
    "is_speculative": bool,
    "hedge_phrase": "string OR null"
  }
]"""


DEFAULT_SYSTEM_PROMPT = load_prompt("fact_extraction", default=_BUILTIN_FACT_PROMPT)


class FactExtractor:
    """Extract atomic facts from text segments using LLM."""

    def __init__(
        self,
        llm: LLMProvider,
        max_facts_per_segment: int = 50,
        min_confidence: float = 0.6,
        max_completion_tokens: int = 6000,
    ) -> None:
        self.llm = llm
        self.max_facts_per_segment = max_facts_per_segment
        self.min_confidence = min_confidence
        self.max_completion_tokens = max_completion_tokens

    def extract(
        self,
        segments: List[ProcessedSegment],
    ) -> List[Dict[str, Any]]:
        """Extract facts from preprocessed segments.

        Args:
            segments: List of ProcessedSegment from preprocessor

        Returns:
            List of raw fact dicts (before entity resolution)
        """
        if not segments:
            return []

        all_facts = []

        for segment in segments:
            facts = self._extract_from_segment(segment)
            all_facts.extend(facts)

        return all_facts

    def _extract_from_segment(
        self,
        segment: ProcessedSegment,
    ) -> List[Dict[str, Any]]:
        """Extract facts from a single segment."""
        user_prompt = f"""[DOCUMENT SEGMENT]
{segment.content}
[/DOCUMENT SEGMENT]"""

        response = self._call_llm(user_prompt)
        if not response:
            logger.warning(f"Empty LLM response for segment from {segment.source}")
            return []

        facts = self._parse_json_response(response)
        facts = self._filter_by_confidence(facts)

        if not facts:
            logger.info(
                f"0 facts parsed from {segment.source} "
                f"(response {len(response)} chars)"
            )

        for fact in facts:
            fact["_source_doc"] = segment.metadata.get("source_doc", "")
            fact["_source_page"] = segment.metadata.get("page")
            fact["_context"] = segment.content
            fact["_content_hash"] = segment.content_hash

        return facts

    def _call_llm(self, user_prompt: str) -> str:
        """Call the LLM with retry on empty response.

        Diffusion-based models (e.g. Mercury) consume reasoning tokens
        from the ``max_tokens`` budget. If the budget is too low the
        model produces an empty completion. We retry once with a
        larger budget when that happens.
        """
        messages = [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        try:
            response = self.llm.complete(
                messages,
                temperature=0.0,
                max_tokens=self.max_completion_tokens,
            )
        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return ""

        # Some models return empty when the token budget is exhausted
        # by internal reasoning. Retry once with 2x budget.
        if not response or not response.strip():
            logger.debug("Empty response, retrying with 2x max_tokens")
            try:
                response = self.llm.complete(
                    messages,
                    temperature=0.0,
                    max_tokens=self.max_completion_tokens * 2,
                )
            except Exception as e:
                logger.warning(f"LLM retry failed: {e}")
                return ""

        return response or ""

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse JSON from LLM response robustly.

        Handles:
        - Markdown code fences
        - Partial/incomplete JSON
        - Extra text before/after JSON
        - Mercury-style output (direct objects without array wrapper)
        """
        text = response.strip()

        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"^```\s*", "", text)
        text = re.sub(r"```$", "", text)
        text = text.strip()

        # Try direct JSON array
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return self._validate_facts(parsed)
            elif isinstance(parsed, dict):
                return self._validate_facts([parsed])
        except json.JSONDecodeError:
            pass

        # Try wrap in array brackets
        text = "[" + text + "]"
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                valid = self._validate_facts(parsed)
                if valid:
                    return valid
        except json.JSONDecodeError:
            pass

        # Recovery for partial JSON (silent)
        return self._recover_partial_json(text, silent=True)

    def _recover_partial_json(
        self, text: str, silent: bool = False
    ) -> List[Dict[str, Any]]:
        """Attempt to recover facts from malformed JSON."""
        facts = []

        brace_depth = 0
        in_string = False
        current_start = -1

        for i, char in enumerate(text):
            if char == '"' and (i == 0 or text[i - 1] != "\\"):
                in_string = not in_string
            if not in_string:
                if char == "{":
                    if brace_depth == 0:
                        current_start = i
                    brace_depth += 1
                elif char == "}":
                    brace_depth -= 1
                    if brace_depth == 0 and current_start >= 0:
                        try:
                            obj_str = text[current_start : i + 1]
                            obj = json.loads(obj_str)
                            if isinstance(obj, dict):
                                facts.append(obj)
                        except json.JSONDecodeError:
                            pass
                        current_start = -1

        return self._validate_facts(facts)

    def _validate_facts(self, facts: List[Any]) -> List[Dict[str, Any]]:
        """Validate and clean parsed facts."""
        valid_facts = []
        _valid_modalities = {
            "assertion",
            "hypothesis",
            "negation",
            "conditional",
            "comparative",
        }

        for item in facts:
            if not isinstance(item, dict):
                continue

            subject = item.get("subject", "")
            predicate = item.get("predicate", "")

            if not subject or not predicate:
                continue

            try:
                raw_conf = float(item.get("confidence", 1.0))
            except (TypeError, ValueError):
                raw_conf = 1.0
            confidence = max(0.0, min(1.0, raw_conf))

            # -- Semantic metadata (v2.0) --
            raw_modality = str(item.get("modality", "assertion")).lower().strip()
            modality = (
                raw_modality if raw_modality in _valid_modalities else "assertion"
            )
            is_speculative = bool(item.get("is_speculative", False))
            hedge_phrase = item.get("hedge_phrase")
            if hedge_phrase is not None:
                hedge_phrase = str(hedge_phrase).strip() or None

            fact = {
                "subject": str(subject).strip(),
                "predicate": str(predicate).strip(),
                "object": item.get("object"),
                "object_value": item.get("object_value"),
                "context": item.get("context", ""),
                "confidence": confidence,
                "modality": modality,
                "is_speculative": is_speculative,
                "hedge_phrase": hedge_phrase,
            }

            if fact["object"] is not None:
                fact["object"] = str(fact["object"]).strip()
            if fact["object_value"] is not None:
                fact["object_value"] = str(fact["object_value"]).strip()

            valid_facts.append(fact)

        return valid_facts

    def _filter_by_confidence(
        self,
        facts: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Filter facts by minimum confidence threshold."""
        return [
            f for f in facts if self.min_confidence <= f.get("confidence", 1.0) <= 1.0
        ][: self.max_facts_per_segment]

    def extract_batch(
        self,
        segments: List[ProcessedSegment],
        max_tokens: int = 8000,
    ) -> List[Dict[str, Any]]:
        """Extract facts from multiple segments in one LLM call.

        Useful when segments are short and can fit together.

        Args:
            segments: List of segments to batch
            max_tokens: Approximate token budget for combined prompt

        Returns:
            List of raw fact dicts
        """
        if not segments:
            return []

        combined_text = ""
        segment_metadata = []

        for segment in segments:
            segment_text = (
                f"\n--- SEGMENT {segment.chunk_index} ---\n{segment.content}\n"
            )
            if len(combined_text) + len(segment_text) > max_tokens * 4:
                break
            combined_text += segment_text
            segment_metadata.append(segment.metadata)

        user_prompt = f"""[DOCUMENT SEGMENTS]
{combined_text}
[/DOCUMENT SEGMENTS]"""

        response = self._call_llm(user_prompt)
        if not response:
            logger.warning("Empty LLM response for batch extraction")
            return []

        facts = self._parse_json_response(response)
        facts = self._filter_by_confidence(facts)

        for fact in facts:
            fact["_source_doc"] = segment_metadata[0].get("source_doc", "")
            fact["_source_page"] = segment_metadata[0].get("page")
            fact["_context"] = combined_text[:500]

        return facts
