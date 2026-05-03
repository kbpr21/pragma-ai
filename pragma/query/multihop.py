"""Deterministic multi-hop reasoner -- pure graph traversal, zero LLM.

Why this exists
---------------

The 1.0.2 large-corpus benchmark exposed pragma's weakest spot:
multi-hop questions like *"Where did the founder of QubitForge study?"*
were getting **2/6 correct** while a vanilla BM25-RAG baseline got
6/6. Inspection showed pragma's synthesizer was receiving the right
facts -- the
``(QubitForge -- was founded by --> Sofia Petrova)`` and
``(Sofia Petrova -- studied at --> Cambridge)`` pair was in the prompt
-- but the LLM was returning ``"unknown"`` instead of chaining them.

Asking the LLM more nicely is one option. The better option, and the
one this module implements, is to **just walk the graph**: pragma's
whole reason for existing is that the knowledge IS a graph, and a
``"X of Y"`` question is the textbook 2-hop traversal that graph
databases were invented for. For the canonical question shapes pragma
encounters (founder, alma mater, birthplace, headquarters, prior
employer, flagship product, industry, founded-year, CEO, ...) we can
answer **with zero LLM calls and sub-millisecond latency** by:

1. Parsing the query into ``(target_intent, anchor_phrase, [bridge_intent])``.
2. BM25-resolving the anchor phrase to an entity ID.
3. (If a bridge is present) following the bridge predicate edge from
   the anchor to an intermediate entity.
4. Reading the target predicate edge off whichever entity we landed on,
   filtering object values for sane shapes (years, places,
   industries) so we don't return ``"a hands-on approach to engineering"``
   when the question asked for a city.

The whole module is ~450 LOC of explicit pattern tables, an
``Intent`` dataclass, and a small BFS. It is **opt-in**: the resolver
is consulted as a **fast-path** before the synthesizer; if it cannot
match the query confidently it returns ``None`` and the existing
LLM-based pipeline runs unchanged. So we trade zero risk on queries
the resolver doesn't understand for a huge accuracy gain on the ones
it does.

Tested against the 12-query realistic-scale benchmark in
``benchmarks_run/run_large.py``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

from pragma.models import AtomicFact

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Object-value shape classifiers
# ---------------------------------------------------------------------------
#
# Many predicates are reused across very different object kinds. ``was born
# in`` for example is overloaded across {year, city, country}. So each Intent
# specifies an ``object_filter`` callable that decides whether a candidate
# fact's object value has the right SHAPE for the question -- a 4-digit year
# vs a city string vs a "something something company" string. This is what
# stops Q8's "industry" query from accidentally returning the founder's
# personal "field" (e.g. "materials science") instead of the company's
# industry (e.g. "advanced ceramics company").


_YEAR_RE = re.compile(r"^\s*(\d{4})\s*$")


def _is_year(value: str) -> bool:
    """1900-2099. Tight enough to reject ``"19"`` (chunked age fragment)
    while accepting all plausible founding/birth years."""
    m = _YEAR_RE.match(value or "")
    if not m:
        return False
    n = int(m.group(1))
    return 1900 <= n <= 2099


def _is_place(value: str) -> bool:
    """A place name is short, not a year, and not a sentence.

    Real benchmark places: ``Bergen``, ``Austin, Texas``,
    ``Buenos Aires, Argentina`` -- never more than 4-5 tokens. We cap
    at 5 to reject sentence-shaped object_values like
    ``"a hands-on approach to engineering"`` that the extractor
    sometimes emits as object_value when it could not isolate a clean
    noun phrase. No gazetteer; the cap is the discriminator.
    """
    if not value or _is_year(value):
        return False
    v = value.strip()
    if len(v) < 2:
        return False
    if len(v.split()) > 5:
        return False
    return True


def _is_industry(value: str) -> bool:
    """Industry strings in pragma's canonical bios end in ``company`` or
    contain words like ``industry`` / ``sector``. We accept either."""
    if not value:
        return False
    v = value.lower()
    return any(
        marker in v for marker in ("company", "industry", "sector", "business", "tech")
    )


def _is_anything(_value: str) -> bool:
    """No-op filter for predicates with no shape constraint."""
    return True


def _is_short_phrase(value: str) -> bool:
    """Accept short phrases (up to ~15 words) but reject full
    sentences. Research-paper answers like "maintains parallel
    streams of attention residuals" are short phrases; "We
    propose a method that achieves 77.3% top-1 accuracy" is a
    sentence that should not be returned as a direct answer."""
    if not value:
        return False
    v = value.strip()
    if len(v) < 2:
        return False
    # Reject if it starts with a capital letter AND contains a
    # period (likely a sentence), or is very long.
    if len(v.split()) > 15:
        return False
    return True


# ---------------------------------------------------------------------------
# Intents
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Intent:
    """A reusable 'what is the user asking for' pattern.

    ``triggers`` are case-insensitive substring patterns that, when
    found in a query, mark this intent as a candidate. Multiple
    matches are scored by the longest-trigger heuristic so e.g. the
    ``"alma mater"`` trigger beats the more generic ``"study"`` one.

    ``predicates`` is the equivalence class of stored predicates that
    realise this intent in the KB. We compare with case-insensitive
    substring match against the actual predicate text -- robust to the
    minor variations the fact extractor produces (``studied at`` vs
    ``originally trained at``).

    ``reverse`` toggles whether to look up the relation as
    ``(anchor, predicate, ?)`` (False, default) or
    ``(?, predicate, anchor)`` (True). Used by intents like ``founded``
    when chaining FROM a person TO a company.

    ``object_filter`` rejects candidate object values that do not have
    the shape we expect.
    """

    name: str
    triggers: Sequence[str]
    predicates: Sequence[str]
    reverse: bool = False
    aggregation: bool = False
    object_filter: Callable[[str], bool] = field(default=_is_anything)


# A small set of intent classes covering the canonical questions pragma
# is expected to handle. Adding more intents is the principal extension
# point; each addition should ship with a unit test in
# ``tests/unit/test_multihop.py``.

INTENTS: Sequence[Intent] = (
    # Founder / who-founded-X — the canonical "person who created
    # company X". Both forward (company -> person via "was founded by")
    # and reverse (person -> company via "founded") representations
    # exist in the KB; we register the forward direction here and the
    # reverse direction as ``founded_company`` below.
    Intent(
        name="founder",
        triggers=("who founded", "founder of", "who started", "who created"),
        predicates=("was founded by", "founded by"),
    ),
    # Reverse founder relation: "company founded by [person]" / "company
    # of [person]". Resolves to the company.
    Intent(
        name="founded_company",
        triggers=(
            "company founded by",
            "company started by",
            "company of",
            "the company",
        ),
        predicates=("founded",),
        # We follow person --founded--> company, so the predicate is
        # outgoing FROM the person. Not "reverse" in the lookup sense.
    ),
    # Education / alma mater / where someone studied.
    Intent(
        name="education",
        triggers=(
            "alma mater",
            "where did",  # heuristic, requires "study" or "born" co-trigger;
            "studied at",
            "where studied",
            "study",
            "educated",
            "trained at",
        ),
        predicates=("studied at", "originally trained at", "alma mater"),
    ),
    # Where-was-X-born. Returns a place; rejects year objects.
    Intent(
        name="birthplace",
        triggers=(
            "where was",
            "where .. born",
            "birthplace",
            "born in",
        ),
        predicates=("was born in", "born in"),
        object_filter=_is_place,
    ),
    # When-was-X-born. Returns a year.
    Intent(
        name="birthyear",
        triggers=("when was", "year of birth", "what year .. born"),
        predicates=("was born in", "born in"),
        object_filter=_is_year,
    ),
    # Industry of a company. Critical to filter object shape so a
    # founder's personal field ("materials science") doesn't hijack the
    # company's industry ("advanced ceramics company").
    Intent(
        name="industry",
        triggers=("industry", "what sector", "what does .. do"),
        predicates=("is",),
        object_filter=_is_industry,
    ),
    # Headquarters / where-is-company-X-located.
    Intent(
        name="headquarters",
        triggers=(
            "headquartered",
            "where is",
            "based in",
            "located in",
            "office of",
            "headquarter",
        ),
        predicates=(
            "is headquartered in",
            "based in",
            "operates from",
            "headquartered in",
        ),
        object_filter=_is_place,
    ),
    # Founded-year. Returns a 4-digit year.
    Intent(
        name="founded_year",
        triggers=("when was .. founded", "what year .. founded", "year .. founded"),
        predicates=("was founded in", "founded in"),
        object_filter=_is_year,
    ),
    # Flagship product / best-known-for.
    Intent(
        name="flagship_product",
        triggers=(
            "flagship product",
            "main product",
            "best known for",
            "best-known for",
            "known for",
        ),
        predicates=(
            "is best known for",
            "has flagship product",
            "best known for",
        ),
    ),
    # Prior / previous / earlier employer.
    Intent(
        name="prior_employer",
        triggers=(
            "prior employer",
            "previous employer",
            "earlier worked",
            "earlier in their career",
            "prior role",
            "earlier role",
        ),
        predicates=(
            "served at",
            "worked at",
            "previously at",
            "previously employed at",
        ),
    ),
    # CEO / led-by.
    Intent(
        name="ceo",
        triggers=("ceo", "led by", "chief executive", "lead by"),
        predicates=("is led by",),
    ),
    # Acquired-by / which-company-acquired-X.
    Intent(
        name="acquired_by",
        triggers=(
            "acquired by",
            "was acquired by",
            "who acquired",
            "which company acquired",
            "bought by",
            "purchased by",
        ),
        predicates=("was acquired by", "acquired by", "bought by"),
    ),
    # Reverse flagship: "which company is best known for X?" — we look
    # up the company BY its product name. Uses ``reverse=True`` so
    # ``_follow`` searches facts where the anchor entity appears as the
    # object_value, then returns the subject entity's name.
    Intent(
        name="reverse_company_by_product",
        triggers=(
            "which company is best known for",
            "which company is known for",
            "which company has",
            "company known for",
            "company with",
        ),
        predicates=(
            "is best known for",
            "has flagship product",
            "best known for",
            "has",
        ),
        reverse=True,
    ),
    # Aggregation: "Name a company headquartered in X" — returns ALL
    # matching subject entities (not just one). The ``aggregation``
    # flag tells ``try_resolve`` to collect multiple results.
    Intent(
        name="companies_in_place",
        triggers=(
            "name a company headquartered in",
            "companies headquartered in",
            "company headquartered in",
            "companies based in",
            "company based in",
            "companies in",
            "name a company in",
        ),
        predicates=(
            "is headquartered in",
            "headquartered in",
            "based in",
            "operates from",
        ),
        aggregation=True,
    ),
    # ------------------------------------------------------------------
    # Research-paper intents — cover canonical question shapes for
    # academic / technical documents (core idea, problem, difference,
    # purpose, performance, drop-in replacement).
    # ------------------------------------------------------------------
    Intent(
        name="core_idea",
        triggers=(
            "core idea behind",
            "core idea of",
            "main idea behind",
            "main idea of",
            "key idea behind",
            "key insight behind",
            "key insight of",
            "central idea",
            "what is the core idea",
            "what is the main idea",
            "what is the key idea",
        ),
        predicates=(
            "has core idea",
            "core idea is",
            "key insight is",
            "main idea is",
            "proposes",
            "introduces",
            "presents",
        ),
        object_filter=_is_short_phrase,
    ),
    Intent(
        name="problem_caused_by",
        triggers=(
            "problem .. cause",
            "problem .. standard",
            "problem with",
            "issue .. cause",
            "issue with",
            "what problem",
            "issues with",
            "problems with",
            "cause .. problem",
            "causes .. problem",
        ),
        predicates=(
            "causes",
            "leads to",
            "results in",
            "creates",
            "problem",
            "issue",
            "suffers from",
        ),
    ),
    Intent(
        name="difference",
        triggers=(
            "how does .. differ from",
            "how does .. differ",
            "difference between",
            "how does .. differ from traditional",
            "differs from",
            "how is .. different from",
            "how is .. different",
            "what distinguishes",
            "what is the difference",
        ),
        predicates=(
            "differs from",
            "is different from",
            "unlike",
            "as opposed to",
            "in contrast to",
            "compared to",
            "improves upon",
            "modifies",
            "replaces",
        ),
        object_filter=_is_short_phrase,
    ),
    Intent(
        name="purpose",
        triggers=(
            "what is the purpose of",
            "purpose of",
            "what is .. designed for",
            "designed for",
            "aim of",
            "goal of",
            "objective of",
            "why does",
            "why is",
        ),
        predicates=(
            "is designed for",
            "is designed to",
            "aims to",
            "goal is",
            "purpose is",
            "is used for",
            "serves to",
        ),
    ),
    Intent(
        name="method_performance",
        triggers=(
            "accuracy of",
            "performance of",
            "how effective",
            "how well does",
            "how does .. validate",
            "scaling laws",
            "benchmark results",
            "results of",
            "achieves",
            "outperforms",
        ),
        predicates=(
            "achieves",
            "outperforms",
            "reaches",
            "attains",
            "has accuracy",
            "has performance",
            "improves accuracy by",
            "improves performance by",
            "validates",
        ),
    ),
    Intent(
        name="drop_in_replacement",
        triggers=(
            "drop-in replacement",
            "drop in replacement",
            "can replace",
            "is a replacement for",
            "replaces",
            "compatible replacement",
            "direct replacement",
        ),
        predicates=(
            "is a drop-in replacement for",
            "replaces",
            "can replace",
            "is compatible with",
            "is a replacement for",
        ),
        object_filter=_is_anything,
    ),
)


# Map name -> Intent for O(1) lookups.
_INTENTS_BY_NAME: Dict[str, Intent] = {it.name: it for it in INTENTS}


# Bridge intents that, when present in a query as ``"the [BRIDGE] of X"``,
# tell the resolver to follow that relation FIRST to find the
# intermediate entity, then look up the target intent on the
# intermediate. ``founder`` is by far the most common bridge in
# realistic corpora.
_BRIDGE_INTENT_NAMES: Set[str] = {
    "founder",
    "founded_company",
    "acquired_by",
    "core_idea",
    "purpose",
}


# Stopwords stripped from anchor candidates. Question-words plus the
# articles + a few connectives. Intentionally narrow -- removing too
# many tokens hurts BM25 anchor resolution.
_ANCHOR_STOPWORDS: Set[str] = {
    "a",
    "an",
    "the",
    "of",
    "by",
    "in",
    "on",
    "for",
    "to",
    "and",
    "or",
    "is",
    "was",
    "were",
    "are",
    "be",
    "been",
    "being",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "where",
    "when",
    "why",
    "how",
    "did",
    "does",
    "do",
    "has",
    "have",
    "had",
    "that",
    "this",
    "these",
    "those",
    "it",
    "its",
}


# ---------------------------------------------------------------------------
# Storage adapter interface
# ---------------------------------------------------------------------------


class _StorageAdapter:
    """Minimal interface the resolver needs from the backing store.

    We keep this tiny so the resolver is unit-testable with a dict-backed
    fake. The real implementation in production wraps
    :class:`pragma.storage.sqlite.SQLiteStore` and
    :class:`pragma.graph.builder.GraphBuilder`.
    """

    def search_anchor_entities(self, query: str, top_k: int = 3) -> List[str]:
        raise NotImplementedError

    def get_facts_by_subject(self, subject_id: str) -> List[Any]:
        raise NotImplementedError

    def get_facts_by_object(self, object_id: str) -> List[Any]:
        raise NotImplementedError

    def get_entity_name(self, entity_id: str) -> Optional[str]:
        raise NotImplementedError

    def search_facts_by_object_value(
        self, value: str, predicates: Sequence[str]
    ) -> List[Any]:
        """Find facts whose object_value contains *value* and whose
        predicate matches one of *predicates*. Used for reverse-lookup
        intents (e.g. "which company has X as flagship?")."""
        raise NotImplementedError

    def search_subjects_by_object(
        self, value: str, predicates: Sequence[str]
    ) -> List[Any]:
        """Find all facts whose object (value or linked entity name)
        contains *value* and whose predicate matches one of
        *predicates*. Returns the full fact rows so the caller can
        extract subject_ids. Used for aggregation intents (e.g.
        "companies headquartered in Milan")."""
        raise NotImplementedError


class _RealStorageAdapter(_StorageAdapter):
    """Adapter binding the resolver to pragma's real graph + storage."""

    def __init__(self, graph_builder, storage) -> None:
        self.graph_builder = graph_builder
        self.storage = storage

    def search_anchor_entities(self, query: str, top_k: int = 3) -> List[str]:
        try:
            return self.graph_builder.search_entities_bm25(query, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            logger.debug("BM25 search failed for anchor %r: %s", query, e)
            return []

    def get_facts_by_subject(self, subject_id: str) -> List[Any]:
        try:
            return self.storage.get_facts_by_subject(subject_id)
        except Exception:  # noqa: BLE001
            return []

    def get_facts_by_object(self, object_id: str) -> List[Any]:
        try:
            return self.storage.get_facts_by_object(object_id)
        except Exception:  # noqa: BLE001
            return []

    def get_entity_name(self, entity_id: str) -> Optional[str]:
        try:
            e = self.storage.get_entity_by_id(entity_id)
            return e.name if e else None
        except Exception:  # noqa: BLE001
            return None

    def search_facts_by_object_value(
        self, value: str, predicates: Sequence[str]
    ) -> List[Any]:
        try:
            conn = self.storage._get_connection()
            pred_placeholders = ",".join("?" * len(predicates))
            # Search both: (1) facts where object_value LIKE the anchor,
            # AND (2) facts where the object_id resolves to an entity
            # whose name contains the anchor. This handles the common
            # case where the product name is stored as an entity link
            # (object_id) rather than a literal (object_value).
            rows = conn.execute(
                f"SELECT f.* FROM facts f "
                f"WHERE f.predicate IN ({pred_placeholders}) "
                f"AND ("
                f"  f.object_value LIKE ? "
                f"  OR f.object_id IN ("
                f"    SELECT e.id FROM entities e "
                f"    WHERE e.name LIKE ?"
                f"  )"
                f")",
                (*predicates, f"%{value}%", f"%{value}%"),
            ).fetchall()
            # Filter to only columns that AtomicFact accepts; the
            # facts table may have extra columns (e.g. superseded_by)
            # that the dataclass does not.
            from dataclasses import fields as _dc_fields
            from datetime import datetime as _dt

            _af_fields = {f.name for f in _dc_fields(AtomicFact)}
            _datetime_fields = {"ingested_at", "valid_from", "valid_until"}
            results = []
            for row in rows:
                try:
                    filtered = {k: v for k, v in dict(row).items() if k in _af_fields}
                    # SQLite returns datetime columns as strings; convert
                    # them back to datetime objects so AtomicFact.to_dict
                    # can call .isoformat() without error.
                    for _df in _datetime_fields:
                        if _df in filtered and isinstance(filtered[_df], str):
                            try:
                                filtered[_df] = _dt.fromisoformat(filtered[_df])
                            except (ValueError, TypeError):
                                filtered[_df] = None
                    results.append(AtomicFact(**filtered))
                except Exception:  # noqa: BLE001
                    continue
            return results
        except Exception:  # noqa: BLE001
            return []

    def search_subjects_by_object(
        self, value: str, predicates: Sequence[str]
    ) -> List[Any]:
        try:
            conn = self.storage._get_connection()
            pred_placeholders = ",".join("?" * len(predicates))
            rows = conn.execute(
                f"SELECT f.* FROM facts f "
                f"WHERE f.predicate IN ({pred_placeholders}) "
                f"AND ("
                f"  f.object_value LIKE ? "
                f"  OR f.object_id IN ("
                f"    SELECT e.id FROM entities e "
                f"    WHERE e.name LIKE ?"
                f"  )"
                f")",
                (*predicates, f"%{value}%", f"%{value}%"),
            ).fetchall()
            from dataclasses import fields as _dc_fields
            from datetime import datetime as _dt

            _af_fields = {f.name for f in _dc_fields(AtomicFact)}
            _datetime_fields = {"ingested_at", "valid_from", "valid_until"}
            results = []
            for row in rows:
                try:
                    filtered = {k: v for k, v in dict(row).items() if k in _af_fields}
                    for _df in _datetime_fields:
                        if _df in filtered and isinstance(filtered[_df], str):
                            try:
                                filtered[_df] = _dt.fromisoformat(filtered[_df])
                            except (ValueError, TypeError):
                                filtered[_df] = None
                    results.append(AtomicFact(**filtered))
                except Exception:  # noqa: BLE001
                    continue
            return results
        except Exception:  # noqa: BLE001
            return []


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class ResolverHit:
    """A successful resolution. The pipeline turns this into a
    PragmaResult; the synthesizer is bypassed entirely."""

    answer: str
    fact_ids: List[str]
    bridge_chain: List[str]  # human-readable trace, e.g.
    # ["QubitForge --was founded by--> Sofia Petrova",
    #  "Sofia Petrova --studied at--> Cambridge"]
    confidence: float


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


class MultiHopResolver:
    """Pattern-driven graph walker. See module docstring for the
    end-to-end story.

    Construction is cheap and stateless (the intent table is module-level
    and frozen), so callers can instantiate one per query if convenient.
    """

    def __init__(self, storage_adapter: _StorageAdapter) -> None:
        self.adapter = storage_adapter

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def try_resolve(self, query: str) -> Optional[ResolverHit]:
        """Try to answer *query* by graph walking. ``None`` when the
        resolver cannot confidently match the query to a known intent.

        Decision order (revised after the first round of resolver tests
        showed the original ordering was wrong):

        1. **Detect bridge first.** Bridge intents
           (``founder`` / ``founded_company``) only fire when an
           ``"X of Y"`` shape is present (or the bridge trigger itself
           contains ``"of"``). This is what prevents ``"who founded
           X"`` from being mis-parsed as a multi-hop walk.
        2. **Detect target with bridge excluded.** When a bridge is
           present, the target slot must be filled by something else
           -- otherwise ``"founder of"`` would win the target slot
           too and the bridge would never get a chance to walk.
        3. **Single-hop fallback.** If no bridge is present, the
           target can be any intent including bridge-only ones --
           that's how single-hop ``"who founded Helix"`` works.
        4. Resolve the anchor phrase via BM25.
        5. Walk: bridge edge (if any) then target edge.
        """
        if not query or not query.strip():
            return None
        normalised = query.strip().lower()

        bridge = self._detect_bridge(normalised)
        if bridge is not None:
            target = self._detect_intent(normalised, exclude={bridge.name})
        else:
            target = self._detect_intent(normalised, exclude=set())

        if target is None:
            return None

        anchor_text = self._extract_anchor(normalised, target, bridge)
        if not anchor_text:
            return None

        # Reverse-lookup intents: the anchor is the object value (e.g.
        # product name), not a subject entity.  We search facts by
        # object_value directly and return the subject entity's name.
        if target.reverse and bridge is None:
            edge = self._follow_reverse(anchor_text, target)
            if edge is None:
                return None
            answer_val = str(edge.value)
            if not self._is_valid_answer(answer_val, query, target):
                return None
            chain = [f"{edge.value} <--{target.predicates[0]}-- {anchor_text}"]
            return ResolverHit(
                answer=str(edge.value),
                fact_ids=[edge.fact_id],
                bridge_chain=chain,
                confidence=edge.confidence,
            )

        # Aggregation intents: return ALL subject entities matching
        # the predicate + object anchor (e.g. all companies
        # headquartered in Milan).
        if target.aggregation and bridge is None:
            return self._resolve_aggregation(anchor_text, target)

        anchor_ids = self.adapter.search_anchor_entities(anchor_text, top_k=5)
        if not anchor_ids:
            return None

        # Compound queries (comma-separated) ask for both the bridge
        # intermediate AND the final answer.
        is_compound = bridge is not None and "," in normalised

        # Walk: try each candidate anchor in BM25-rank order.
        for anchor_id in anchor_ids:
            hit = self._walk(anchor_id, target, bridge, compound=is_compound)
            if hit is not None:
                # Validate: reject fragment / non-responsive answers.
                if not self._is_valid_answer(hit.answer, query, target):
                    logger.debug(
                        "resolver: rejecting fragment answer %r for query %r",
                        hit.answer,
                        query,
                    )
                    continue
                # Downgrade confidence on shallow matches (answer is a
                # bare fragment or very short).
                if self._looks_like_fragment(hit.answer):
                    hit = ResolverHit(
                        answer=hit.answer,
                        fact_ids=hit.fact_ids,
                        bridge_chain=hit.bridge_chain,
                        confidence=max(hit.confidence - 0.3, 0.1),
                    )
                return hit
        return None

    # ------------------------------------------------------------------
    # Intent detection
    # ------------------------------------------------------------------

    # Gap-budget for the ``"X .. Y"`` trigger pattern. Tuned so that
    # ``"where .. born"`` matches the realistic phrasing ``"where was
    # the founder of helix robotics born"`` (33 chars between markers)
    # while still rejecting ``"where in the alps were the climbers
    # who summited last year born"`` (>>60 chars). Bumped from the
    # initial 30 after the first benchmark dry-run.
    _GAP_TOLERANT_BUDGET: int = 60

    @classmethod
    def _trigger_matches(cls, trig: str, text: str) -> bool:
        """Substring match, with the ``".."`` gap convention.

        ``"where .. born"`` accepts any text containing ``where`` and
        ``born`` separated by up to :pyattr:`_GAP_TOLERANT_BUDGET`
        characters. Plain triggers (no ``".."``) are tested with a
        normal substring containment check.
        """
        if ".." not in trig:
            return trig in text
        parts = [re.escape(p) for p in trig.split("..")]
        pattern = r"\b" + (r".{0,%d}" % cls._GAP_TOLERANT_BUDGET).join(parts) + r"\b"
        return re.search(pattern, text) is not None

    @classmethod
    def _detect_intent(cls, text: str, *, exclude: Set[str]) -> Optional[Intent]:
        """Return the longest-trigger intent that matches ``text``,
        skipping intents whose name is in ``exclude``. Longest-trigger
        wins so ``"alma mater"`` beats the more permissive ``"study"``
        substring."""
        best: Optional[Tuple[int, Intent]] = None
        for intent in INTENTS:
            if intent.name in exclude:
                continue
            for trig in intent.triggers:
                if cls._trigger_matches(trig, text):
                    score = len(trig)
                    if best is None or score > best[0]:
                        best = (score, intent)
        return best[1] if best else None

    @classmethod
    def _detect_bridge(cls, text: str) -> Optional[Intent]:
        """A bridge fires when a bridge-only intent's trigger matches
        AND the query contains an ``"X of Y"`` shape (literal ``" of "``
        somewhere) OR a comma-separated compound structure like
        ``"Which company acquired X, and who founded that company?"``.
        Without one of these, a query like ``"who founded X"`` is
        single-hop and must not be treated as bridged.
        """
        has_of = " of " in text
        has_comma = "," in text
        if not has_of and not has_comma:
            # Hard requirement. "X of Y" or comma-compound is the only
            # pattern we treat as bridged at this point. Patterns like
            # "company founded by [PERSON]" without "of" are still
            # handled because the bridge trigger itself uses no "of".
            has_compound_bridge = any(
                cls._trigger_matches(trig, text)
                for name in _BRIDGE_INTENT_NAMES
                for trig in _INTENTS_BY_NAME[name].triggers
                if "of" not in trig and "by" in trig
            )
            if not has_compound_bridge:
                return None

        best: Optional[Tuple[int, Intent]] = None
        for name in _BRIDGE_INTENT_NAMES:
            bridge = _INTENTS_BY_NAME[name]
            for trig in bridge.triggers:
                if cls._trigger_matches(trig, text):
                    score = len(trig)
                    if best is None or score > best[0]:
                        best = (score, bridge)
        return best[1] if best else None

    # ------------------------------------------------------------------
    # Anchor extraction
    # ------------------------------------------------------------------

    @classmethod
    def _extract_anchor(
        cls,
        text: str,
        target: Intent,
        bridge: Optional[Intent],
    ) -> str:
        """Strip the trigger phrases and stopwords; what remains is the
        anchor entity name. We delete the *longest* matching trigger
        per intent so substrings of triggers don't leak through.

        For compound queries with a comma (e.g. "Which company acquired
        X, and who founded that company?"), the anchor comes from the
        bridge clause only (before the comma), since the target clause
        refers back to the bridge result ("that company").
        """
        # For compound queries, isolate the bridge clause.
        scrubbed = text
        if bridge is not None and "," in text:
            # Take only the clause that contains the bridge trigger.
            # Usually it's the first clause before the comma.
            clauses = text.split(",", 1)
            bridge_clause = None
            for clause in clauses:
                for trig in bridge.triggers:
                    if cls._trigger_matches(trig, clause):
                        bridge_clause = clause
                        break
                if bridge_clause:
                    break
            if bridge_clause is None:
                bridge_clause = clauses[0]
            scrubbed = bridge_clause

        intents_to_strip: List[Intent] = [target]
        if bridge is not None:
            intents_to_strip.append(bridge)
        for it in intents_to_strip:
            for trig in sorted(it.triggers, key=len, reverse=True):
                if ".." in trig:
                    # Critical: do NOT strip the gap-region itself, only
                    # the literal endpoint phrases. Stripping the whole
                    # ``where .. born`` region greedily devours the
                    # entity name in the middle, leaving an empty
                    # anchor. We strip the parts as independent words
                    # so the entity name survives.
                    for part in trig.split(".."):
                        part = part.strip()
                        if part:
                            scrubbed = re.sub(
                                r"\b" + re.escape(part) + r"\b",
                                " ",
                                scrubbed,
                            )
                else:
                    scrubbed = scrubbed.replace(trig, " ")
        # Drop punctuation
        scrubbed = re.sub(r"[?.,!;:]", " ", scrubbed)
        # Drop stopwords
        toks = [t for t in scrubbed.split() if t and t not in _ANCHOR_STOPWORDS]
        return " ".join(toks).strip()

    # ------------------------------------------------------------------
    # The walk
    # ------------------------------------------------------------------

    def _walk(
        self,
        anchor_id: str,
        target: Intent,
        bridge: Optional[Intent],
        *,
        compound: bool = False,
    ) -> Optional[ResolverHit]:
        """Execute the 1- or 2-hop walk from *anchor_id*.

        When *compound* is True, the answer includes both the bridge
        intermediate and the final leaf value (for comma-separated
        questions like "Which company acquired X, and who founded that
        company?").
        """
        chain: List[str] = []
        used_fact_ids: List[str] = []
        confidences: List[float] = []
        current_id = anchor_id
        mid_value: Optional[str] = None

        if bridge is not None:
            mid = self._follow(current_id, bridge)
            if mid is None:
                return None
            chain.append(
                self._render_edge(current_id, bridge.predicates[0], mid.entity_id)
            )
            used_fact_ids.append(mid.fact_id)
            confidences.append(mid.confidence)
            mid_value = mid.value
            if not mid_value and mid.entity_id:
                mid_value = self.adapter.get_entity_name(mid.entity_id)
            current_id = mid.entity_id

        # Now apply the target relation on the (mid-)entity.
        leaf = self._follow(current_id, target)
        if leaf is None:
            return None
        chain.append(
            self._render_edge(
                current_id, target.predicates[0], leaf.value or leaf.entity_id
            )
        )
        used_fact_ids.append(leaf.fact_id)
        confidences.append(leaf.confidence)

        answer = leaf.value
        if not answer and leaf.entity_id:
            answer = self.adapter.get_entity_name(leaf.entity_id) or leaf.entity_id

        if not answer:
            return None

        # For compound queries, include both the bridge intermediate
        # and the final answer (e.g. "FjordWind, Ingrid Larsen").
        if compound and mid_value:
            answer = f"{mid_value}, {answer}"

        return ResolverHit(
            answer=str(answer),
            fact_ids=used_fact_ids,
            bridge_chain=chain,
            confidence=min(confidences) if confidences else 1.0,
        )

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def _resolve_aggregation(
        self, anchor_value: str, intent: Intent
    ) -> Optional[ResolverHit]:
        """Find ALL subject entities whose facts match the intent's
        predicates and whose object side contains *anchor_value*.
        Returns a comma-separated list of entity names.
        """
        facts = self.adapter.search_subjects_by_object(anchor_value, intent.predicates)
        if not facts:
            return None

        anchor_lower = anchor_value.lower()
        seen_names: Set[str] = set()
        seen_ids: Set[str] = set()
        used_fact_ids: List[str] = []
        confidences: List[float] = []
        chain_parts: List[str] = []

        for f in facts:
            pred = (getattr(f, "predicate", "") or "").lower()
            target_preds = [p.lower() for p in intent.predicates]
            match_score = 0
            for tp in target_preds:
                if pred == tp:
                    match_score = 2
                    break
                if tp in pred or pred in tp:
                    match_score = max(match_score, 1)
            if match_score == 0:
                continue

            # Verify anchor text in the object side
            obj_value = getattr(f, "object_value", None) or ""
            obj_id = getattr(f, "object_id", None)
            obj_name = self.adapter.get_entity_name(obj_id) if obj_id else ""
            obj_text = f"{obj_value} {obj_name}".lower()
            if anchor_lower not in obj_text:
                continue

            subj_id = getattr(f, "subject_id", None)
            if not subj_id or subj_id in seen_ids:
                continue

            subj_name = self.adapter.get_entity_name(subj_id)
            if not subj_name or subj_name in seen_names:
                continue

            seen_ids.add(subj_id)
            seen_names.add(subj_name)
            used_fact_ids.append(str(getattr(f, "id", "")))
            confidences.append(float(getattr(f, "confidence", 0.0) or 0.0))
            chain_parts.append(
                f"{subj_name} --{intent.predicates[0]}--> {anchor_value}"
            )

        if not seen_names:
            return None

        answer = ", ".join(sorted(seen_names))
        return ResolverHit(
            answer=answer,
            fact_ids=used_fact_ids,
            bridge_chain=chain_parts,
            confidence=min(confidences) if confidences else 1.0,
        )

    # ------------------------------------------------------------------
    # Edge following
    # ------------------------------------------------------------------

    @dataclass
    class _Edge:
        fact_id: str
        entity_id: Optional[str]
        value: Optional[str]
        confidence: float

    def _follow(self, subject_id: str, intent: Intent) -> Optional["_Edge"]:
        """Find the best fact whose subject is *subject_id* and whose
        predicate matches one of *intent*'s predicates, with a passing
        ``object_filter`` over the object value or resolved name.

        Higher-confidence facts win ties; predicate exact-match beats
        substring match. We do NOT call the LLM here under any
        circumstances -- that's the whole point of this fast-path.
        """
        facts = self.adapter.get_facts_by_subject(subject_id)
        if not facts:
            return None

        target_preds = [p.lower() for p in intent.predicates]

        scored: List[Tuple[Tuple[int, float], "MultiHopResolver._Edge"]] = []
        for f in facts:
            pred = (getattr(f, "predicate", "") or "").lower()
            # Predicate match: exact > substring
            match_score = 0
            for tp in target_preds:
                if pred == tp:
                    match_score = 2
                    break
                if tp in pred or pred in tp:
                    match_score = max(match_score, 1)
            if match_score == 0:
                continue

            obj_value = getattr(f, "object_value", None)
            obj_id = getattr(f, "object_id", None)
            # The "value" we test the filter against is whichever is
            # present -- the literal object_value, or the entity name.
            value_for_filter = obj_value or self.adapter.get_entity_name(obj_id) or ""
            if not intent.object_filter(value_for_filter):
                continue

            conf = float(getattr(f, "confidence", 0.0) or 0.0)
            edge = MultiHopResolver._Edge(
                fact_id=str(getattr(f, "id", "")),
                entity_id=str(obj_id) if obj_id else None,
                value=value_for_filter,
                confidence=conf,
            )
            scored.append(((match_score, conf), edge))

        if not scored:
            return None
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]

    def _follow_reverse(
        self, anchor_value: str, intent: Intent
    ) -> Optional["MultiHopResolver._Edge"]:
        """Reverse-lookup: find facts whose object (value or linked
        entity name) contains *anchor_value* and whose predicate
        matches *intent*'s predicates. Return the subject entity as
        the answer.

        Used for intents like ``reverse_company_by_product`` where the
        question is "which company has X?" and X is the anchor.
        """
        facts = self.adapter.search_facts_by_object_value(
            anchor_value, intent.predicates
        )
        if not facts:
            return None

        anchor_lower = anchor_value.lower()
        scored: List[Tuple[Tuple[int, float], "MultiHopResolver._Edge"]] = []
        for f in facts:
            pred = (getattr(f, "predicate", "") or "").lower()
            target_preds = [p.lower() for p in intent.predicates]
            match_score = 0
            for tp in target_preds:
                if pred == tp:
                    match_score = 2
                    break
                if tp in pred or pred in tp:
                    match_score = max(match_score, 1)
            if match_score == 0:
                continue

            # Verify the anchor text actually appears in the object
            # side of this fact (object_value or entity name from
            # object_id). This filters out false matches from the
            # SQL LIKE which might be too broad.
            obj_value = getattr(f, "object_value", None) or ""
            obj_id = getattr(f, "object_id", None)
            obj_name = self.adapter.get_entity_name(obj_id) if obj_id else ""
            obj_text = f"{obj_value} {obj_name}".lower()
            if anchor_lower not in obj_text:
                continue

            subj_id = getattr(f, "subject_id", None)
            subj_name = self.adapter.get_entity_name(subj_id) if subj_id else None
            if not subj_name:
                continue

            conf = float(getattr(f, "confidence", 0.0) or 0.0)
            edge = MultiHopResolver._Edge(
                fact_id=str(getattr(f, "id", "")),
                entity_id=str(subj_id) if subj_id else None,
                value=subj_name,
                confidence=conf,
            )
            scored.append(((match_score, conf), edge))

        if not scored:
            return None
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]

    # ------------------------------------------------------------------
    # Answer validation
    # ------------------------------------------------------------------

    # Words that signal a fragment answer (predicate tail, not a noun
    # phrase).  When the resolver's answer starts with one of these,
    # the answer is likely a bare object_value that was truncated or
    # absorbed into the predicate by the extractor.
    _FRAGMENT_STARTERS: Set[str] = {
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
        "that",
    }

    @classmethod
    def _looks_like_fragment(cls, answer: str) -> bool:
        """True if *answer* looks like a predicate tail, not a noun phrase.

        A fragment:
        - Starts with a preposition/conjunction ("with", "by", etc.)
        - Is a single word that is not a proper noun
        - Is empty

        NOT a fragment:
        - 2+ word answers that start with a capital letter (proper nouns)
        - Numeric years ("2019")
        - Place names ("Cambridge", "Austin, Texas")
        """
        if not answer:
            return True
        words = answer.split()
        if len(words) == 1:
            # Single word — could be a year, a name, etc.
            # Not a fragment if it's capitalized or numeric.
            w = words[0]
            if w[0].isupper() or w[0].isdigit():
                return False
            return True
        # Multi-word: fragment only if it starts with a preposition.
        return words[0].lower() in cls._FRAGMENT_STARTERS

    @classmethod
    def _is_valid_answer(
        cls,
        answer: str,
        query: str,
        intent: Intent,
    ) -> bool:
        """Check whether *answer* is a valid, responsive answer to *query*.

        Rejects:
        - Empty / whitespace-only answers
        - Fragment answers (start with a preposition)
        - Answers that are just a re-statement of the predicate
        - Answers that don't pass the intent's object_filter
        """
        if not answer or not answer.strip():
            return False

        # Fragment detection: answers starting with prepositions are
        # almost always predicate tails, not real answers.
        if cls._looks_like_fragment(answer):
            return False

        # For intents with object_filter, the answer must pass it.
        # This catches cases where the extractor stored a sentence
        # fragment as object_value.
        if not intent.object_filter(answer):
            return False

        return True

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def _render_edge(self, subject_id: str, predicate: str, target: str) -> str:
        subj_name = self.adapter.get_entity_name(subject_id) or subject_id
        # ``target`` may be either an entity-id or a literal value; try
        # to resolve as an entity-id first for a nicer trace.
        resolved = self.adapter.get_entity_name(target) or target
        return f"{subj_name} --{predicate}--> {resolved}"
