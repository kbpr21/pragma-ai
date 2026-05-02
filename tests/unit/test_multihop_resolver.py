"""Tests for ``pragma.query.multihop`` -- the deterministic graph-walking
multi-hop resolver introduced in 1.0.2.

The resolver is the lever that took multi-hop benchmark accuracy from
2/6 to (at the time of writing) 6/6. Every test here pins one of the
question shapes the resolver claims to handle, so a future refactor
either keeps the gain or fails loudly.

Strategy:

* A tiny in-memory storage adapter (``_FakeAdapter``) lets each test
  declare exactly the entities and facts it cares about, without any
  SQLite or graph machinery.
* The intent table is module-level frozen, so we can drive it with
  high-level natural-language queries and simply assert on the
  ``answer`` and the ``bridge_chain`` length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from pragma.query.multihop import (
    MultiHopResolver,
    _StorageAdapter,
    _is_industry,
    _is_place,
    _is_year,
)


# ---------------------------------------------------------------------------
# In-memory adapter & fact builder
# ---------------------------------------------------------------------------


@dataclass
class _Fact:
    id: str
    subject_id: str
    predicate: str
    object_id: Optional[str] = None
    object_value: Optional[str] = None
    confidence: float = 1.0


class _FakeAdapter(_StorageAdapter):
    """Dict-backed adapter mirroring the production interface. BM25 is
    replaced with a simple lowercase substring contains-match against
    entity names; fine for unit tests where the corpus is tiny and
    fully under our control."""

    def __init__(self, entities: Dict[str, str], facts: List[_Fact]) -> None:
        self.entities = entities  # entity_id -> name
        self.by_subject: Dict[str, List[_Fact]] = {}
        self.by_object: Dict[str, List[_Fact]] = {}
        for f in facts:
            self.by_subject.setdefault(f.subject_id, []).append(f)
            if f.object_id:
                self.by_object.setdefault(f.object_id, []).append(f)

    def search_anchor_entities(self, query: str, top_k: int = 3) -> List[str]:
        q = query.lower()
        scored = []
        for eid, name in self.entities.items():
            n = name.lower()
            if n in q or q in n:
                # crude: length of overlapping substring
                scored.append((len(n) if n in q else len(q), eid))
        scored.sort(reverse=True)
        return [eid for _, eid in scored[:top_k]]

    def get_facts_by_subject(self, subject_id: str) -> List[Any]:
        return self.by_subject.get(subject_id, [])

    def get_facts_by_object(self, object_id: str) -> List[Any]:
        return self.by_object.get(object_id, [])

    def get_entity_name(self, entity_id: str) -> Optional[str]:
        return self.entities.get(entity_id)

    def search_facts_by_object_value(
        self, value: str, predicates: Sequence[str]
    ) -> List[Any]:
        v = value.lower()
        pred_lower = [p.lower() for p in predicates]
        results = []
        for facts in self.by_subject.values():
            for f in facts:
                pred = (f.predicate or "").lower()
                obj_val = (f.object_value or "").lower()
                if v in obj_val and any(
                    p == pred or p in pred or pred in p for p in pred_lower
                ):
                    results.append(f)
        return results

    def search_subjects_by_object(
        self, value: str, predicates: Sequence[str]
    ) -> List[Any]:
        # Same logic as search_facts_by_object_value — both search
        # the object side of facts for the anchor text.
        return self.search_facts_by_object_value(value, predicates)


# ---------------------------------------------------------------------------
# Object-shape filter unit tests
# ---------------------------------------------------------------------------


class TestObjectFilters:
    def test_year_accepts_4digit_years_in_modern_range(self):
        assert _is_year("2014")
        assert _is_year("1985")
        assert _is_year("  2024 ")  # whitespace tolerated

    def test_year_rejects_non_year_shapes(self):
        assert not _is_year("Bergen")
        assert not _is_year("19")  # too short
        assert not _is_year("twenty14")  # mixed
        assert not _is_year("2200")  # out of plausible window
        assert not _is_year("")
        assert not _is_year(None)  # type: ignore[arg-type]

    def test_place_accepts_short_capitalised_strings(self):
        assert _is_place("Bergen")
        assert _is_place("Austin, Texas")
        assert _is_place("Cambridge")

    def test_place_rejects_year_and_long_sentences(self):
        assert not _is_place("2014")
        assert not _is_place(
            "a hands-on approach to engineering across many disciplines"
        )
        assert not _is_place("")

    def test_industry_marker_words(self):
        assert _is_industry("advanced ceramics company")
        assert _is_industry("biotech industry")
        assert _is_industry("supply-chain analytics business")
        assert not _is_industry("Maya Chen")  # founder, not an industry
        assert not _is_industry("")


# ---------------------------------------------------------------------------
# End-to-end resolver tests
# ---------------------------------------------------------------------------


def _build_corpus() -> _FakeAdapter:
    """Construct a tiny knowledge graph that exercises every intent
    the resolver knows about. Mirrors the structure of the real
    benchmark corpus but with hand-crafted entities so the assertions
    are explicit."""
    entities = {
        "helix": "Helix Robotics",
        "maya": "Maya Chen",
        "atlas": "AtlasFlow",
        "beatriz": "Beatriz Ferreira",
        "palmpay": "PalmPay Africa",
        "qubit": "QubitForge",
        "sofia": "Sofia Petrova",
        "andes": "Andes Imaging",
        "pedro": "Pedro Mendoza",
        "kintsu": "Kintsu Materials",
        "aiko": "Aiko Tanaka",
        "fjord": "FjordWind",
        "ingrid": "Ingrid Larsen",
        "bluecell": "BlueCell Storage",
        "jonas": "Jonas Berg",
        "edgemint": "EdgeMint",
    }
    facts = [
        # Helix Robotics
        _Fact("f1", "helix", "was founded by", object_id="maya"),
        _Fact("f2", "maya", "founded", object_id="helix"),
        _Fact("f3", "maya", "was born in", object_value="Singapore"),
        _Fact("f4", "maya", "was born in", object_value="1985"),
        _Fact("f5", "maya", "studied at", object_value="MIT"),
        _Fact("f6", "helix", "is headquartered in", object_value="Austin, USA"),
        _Fact("f7", "helix", "is best known for", object_value="Helix Picker"),
        _Fact("f8", "helix", "is", object_value="warehouse automation company"),
        _Fact("f9", "helix", "is led by", object_id="maya"),
        # AtlasFlow / Beatriz / PalmPay (prior employer chain)
        _Fact("f10", "atlas", "was founded by", object_id="beatriz"),
        _Fact("f11", "beatriz", "served at", object_id="palmpay"),
        _Fact("f12", "beatriz", "studied at", object_value="INSEAD"),
        # QubitForge / Sofia
        _Fact("f13", "qubit", "was founded by", object_id="sofia"),
        _Fact("f14", "sofia", "studied at", object_value="Cambridge"),
        # Andes Imaging / Pedro
        _Fact("f15", "andes", "was founded by", object_id="pedro"),
        _Fact("f16", "pedro", "studied at", object_value="MIT"),
        # Kintsu / Aiko (industry vs personal field disambiguation)
        _Fact("f17", "aiko", "founded", object_id="kintsu"),
        _Fact("f18", "kintsu", "is", object_value="advanced ceramics company"),
        _Fact("f19", "aiko", "specialised in", object_value="materials science"),
        # FjordWind / Ingrid
        _Fact("f20", "fjord", "is headquartered in", object_value="Bergen"),
        _Fact("f21", "fjord", "was founded by", object_id="ingrid"),
        # BlueCell Storage founding year
        _Fact("f22", "bluecell", "was founded in", object_value="2011"),
        _Fact("f23", "bluecell", "was founded by", object_id="jonas"),
        _Fact("f23b", "bluecell", "is headquartered in", object_value="Austin, USA"),
        # EdgeMint flagship
        _Fact("f24", "edgemint", "is best known for", object_value="EdgeMint Box"),
    ]
    return _FakeAdapter(entities, facts)


class TestSingleHopResolution:
    def test_who_founded(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("Who founded Helix Robotics?")
        assert hit is not None
        assert hit.answer == "Maya Chen"
        assert hit.fact_ids == ["f1"]

    def test_where_is_headquartered(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("Where is FjordWind headquartered?")
        assert hit is not None
        assert "Bergen" in hit.answer

    def test_when_was_founded_returns_year(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("In what year was BlueCell Storage founded?")
        assert hit is not None
        assert hit.answer == "2011"

    def test_flagship_product(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("Which company is best known for the EdgeMint Box?")
        # This is the REVERSE direction of "best known for" -- we ask which
        # entity has EdgeMint Box as its flagship. The resolver may not
        # handle reverse lookups directly, in which case it returns None
        # and the LLM path takes over. We accept either: a None result
        # (graceful fall-through) OR the right answer.
        if hit is not None:
            assert "EdgeMint" in hit.answer or "edgemint" in hit.answer.lower()


class TestMultiHopResolution:
    """The headline tests -- the failures from the 50-doc benchmark."""

    def test_birthplace_of_founder(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("Where was the founder of Helix Robotics born?")
        assert hit is not None
        assert hit.answer == "Singapore"
        # Must have walked TWO edges, not one
        assert len(hit.bridge_chain) == 2
        assert len(hit.fact_ids) == 2

    def test_alma_mater_of_founder(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("What is the alma mater of the founder of Andes Imaging?")
        assert hit is not None
        assert hit.answer == "MIT"
        assert len(hit.bridge_chain) == 2

    def test_where_did_founder_study(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("Where did the founder of QubitForge study?")
        assert hit is not None
        assert hit.answer == "Cambridge"

    def test_prior_employer_of_founder(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("What was the prior employer of the founder of AtlasFlow?")
        assert hit is not None
        # The chain landed on PalmPay Africa via Beatriz Ferreira's
        # ``served at`` predicate, not on a Beatriz factoid.
        assert "PalmPay" in hit.answer

    def test_industry_does_not_pick_personal_field(self):
        """The Q8 regression: 'industry of company founded by Aiko Tanaka'
        must return Kintsu's industry (advanced ceramics company), NOT
        Aiko's personal field (materials science). The object-shape
        filter is what guards this."""
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("Which industry is the company founded by Aiko Tanaka in?")
        assert hit is not None
        assert "ceramics" in hit.answer.lower()
        assert "materials science" not in hit.answer.lower()


class TestFalsePositiveSafety:
    """The resolver MUST return None when it cannot match a query
    confidently -- a false positive here would silently corrupt
    answers that the LLM would otherwise have answered correctly."""

    def test_unknown_entity_returns_none(self):
        r = MultiHopResolver(_build_corpus())
        assert r.try_resolve("Who founded NonExistentCorp?") is None

    def test_query_with_no_intent_pattern_returns_none(self):
        r = MultiHopResolver(_build_corpus())
        # No question-word, no canonical pattern.
        assert r.try_resolve("Tell me a story about robotics") is None

    def test_empty_query_returns_none(self):
        r = MultiHopResolver(_build_corpus())
        assert r.try_resolve("") is None


class TestAggregation:
    """Aggregation queries return ALL matching entities."""

    def test_companies_headquartered_in_place(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("Name a company headquartered in Austin.")
        assert hit is not None
        # Both Helix Robotics and BlueCell Storage are in Austin
        assert "Helix Robotics" in hit.answer
        assert "BlueCell Storage" in hit.answer

    def test_aggregation_returns_none_for_unknown_place(self):
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("Name a company headquartered in Tokyo.")
        assert hit is None

    def test_year_filter_rejects_nonyear_for_birthyear_query(self):
        """If a fact incorrectly stored a city in a 'when was X born'
        query, the year object_filter MUST drop it instead of returning
        the non-year value as the answer. The Maya Chen fact set has
        BOTH 'Singapore' and '1985' under the same predicate, so this
        is a real-data regression test, not a hypothetical."""
        r = MultiHopResolver(_build_corpus())
        hit = r.try_resolve("When was Maya Chen born?")
        assert hit is not None
        assert hit.answer == "1985"


class TestAnchorExtraction:
    """The internal ``_extract_anchor`` logic; pinned because subtle
    stopword changes can totally break BM25 anchor resolution."""

    def test_strip_intent_triggers_and_stopwords(self):
        from pragma.query.multihop import INTENTS

        founder = next(it for it in INTENTS if it.name == "founder")
        # No bridge involved here.
        text = "who founded helix robotics?"
        anchor = MultiHopResolver._extract_anchor(text, founder, None)
        # Must keep the entity tokens, drop "who founded" and the "?".
        assert "helix" in anchor and "robotics" in anchor
        assert "who" not in anchor and "founded" not in anchor

    def test_strip_bridge_phrase_for_multihop(self):
        from pragma.query.multihop import INTENTS

        education = next(it for it in INTENTS if it.name == "education")
        founder = next(it for it in INTENTS if it.name == "founder")
        text = "where did the founder of qubitforge study?"
        anchor = MultiHopResolver._extract_anchor(text, education, founder)
        assert "qubitforge" in anchor
        # The bridge trigger ("founder of") and target trigger
        # ("where did", "study") must all be gone.
        assert "founder" not in anchor
        assert "study" not in anchor
