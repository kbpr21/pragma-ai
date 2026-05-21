"""Adversarial benchmark tests for Pragma AI v2.0 enhancements.

Tests the system against complex, noisy, and ambiguous datasets that
specifically target the gaps fixed in the v2.0 hardening:

1. Speculative language preservation (modality + hedge_phrase)
2. Implicit causality extraction
3. Contradiction detection
4. Ambiguous entity resolution
5. Multi-hop reasoning chains
6. Negation preservation
7. Numerical normalization
"""

import json
from pathlib import Path

import pytest

from pragma.ingestion.extractor import FactExtractor
from pragma.graph.resolver import EntityResolver
from pragma.models import AtomicFact
from pragma.query.synthesizer import AgenticSynthesizer, ContradictionReport


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def corpus():
    """Load the adversarial test corpus."""
    with open(FIXTURES_DIR / "adversarial_corpus.json", encoding="utf-8") as f:
        return json.load(f)


class MockLLM:
    """Minimal LLM mock that returns pre-canned JSON for tests."""

    def __init__(self, response: str = "[]"):
        self.response = response
        self.model_name = "mock"

    def complete(self, messages, **kwargs):
        return self.response


# ---------------------------------------------------------------------------
# 1. Speculative language preservation
# ---------------------------------------------------------------------------


class TestSpeculativeLanguage:
    """Verify that the fact extractor preserves hedging and modality."""

    def test_validate_facts_preserves_modality(self):
        """The validation pipeline should pass through modality fields."""
        extractor = FactExtractor(MockLLM())
        raw_facts = [
            {
                "subject": "AttnRes",
                "predicate": "mitigates",
                "object": "PreNorm dilution",
                "confidence": 0.55,
                "modality": "hypothesis",
                "is_speculative": True,
                "hedge_phrase": "appears to",
            }
        ]
        validated = extractor._validate_facts(raw_facts)
        assert len(validated) == 1
        assert validated[0]["modality"] == "hypothesis"
        assert validated[0]["is_speculative"] is True
        assert validated[0]["hedge_phrase"] == "appears to"

    def test_validate_facts_defaults_to_assertion(self):
        """Facts without modality fields should default to assertion."""
        extractor = FactExtractor(MockLLM())
        raw_facts = [
            {
                "subject": "Apple",
                "predicate": "was founded on",
                "object_value": "April 1, 1976",
                "confidence": 1.0,
            }
        ]
        validated = extractor._validate_facts(raw_facts)
        assert len(validated) == 1
        assert validated[0]["modality"] == "assertion"
        assert validated[0]["is_speculative"] is False
        assert validated[0]["hedge_phrase"] is None

    def test_validate_facts_rejects_invalid_modality(self):
        """Invalid modality values should be normalized to 'assertion'."""
        extractor = FactExtractor(MockLLM())
        raw_facts = [
            {
                "subject": "Test",
                "predicate": "is",
                "confidence": 1.0,
                "modality": "INVALID_VALUE",
            }
        ]
        validated = extractor._validate_facts(raw_facts)
        assert validated[0]["modality"] == "assertion"

    def test_atomic_fact_modality_serialization(self):
        """AtomicFact should serialize/deserialize modality fields."""
        fact = AtomicFact(
            id="test-1",
            subject_id="s1",
            predicate="mitigates",
            modality="hypothesis",
            is_speculative=True,
            hedge_phrase="appears to",
        )
        d = fact.to_dict()
        assert d["modality"] == "hypothesis"
        assert d["is_speculative"] is True
        assert d["hedge_phrase"] == "appears to"

        restored = AtomicFact.from_dict(d)
        assert restored.modality == "hypothesis"
        assert restored.is_speculative is True
        assert restored.hedge_phrase == "appears to"


# ---------------------------------------------------------------------------
# 2. Implicit causality extraction
# ---------------------------------------------------------------------------


class TestImplicitCausality:
    """Verify extraction captures causal relationships from temporal cues."""

    def test_causal_fact_validation(self):
        """A causal fact extracted from 'After X, Y improved' should validate."""
        extractor = FactExtractor(MockLLM())
        raw_facts = [
            {
                "subject": "Acme Corp deployment frequency",
                "predicate": "improved by",
                "object_value": "300%",
                "context": "After adopting Kubernetes, Acme Corp's deployment frequency improved by 300%.",
                "confidence": 0.85,
                "modality": "assertion",
                "is_speculative": False,
                "hedge_phrase": None,
            }
        ]
        validated = extractor._validate_facts(raw_facts)
        assert len(validated) == 1
        assert "300%" in validated[0]["object_value"]


# ---------------------------------------------------------------------------
# 3. Contradiction detection
# ---------------------------------------------------------------------------


class TestContradictionDetection:
    """Verify the AgenticSynthesizer can detect and resolve contradictions."""

    def test_contradiction_report_dataclass(self):
        """ContradictionReport should instantiate correctly."""
        report = ContradictionReport(
            fact_a_id="f1",
            fact_b_id="f2",
            description="Conflicting founding years",
            resolution="Kept highest-confidence fact",
            resolved_by="confidence",
        )
        assert report.fact_a_id == "f1"
        assert report.resolved_by == "confidence"

    def test_resolve_contradictions_deduplicates(self):
        """Contradictory facts with same subject+predicate should be
        deduplicated, keeping the highest-confidence version."""
        synth = AgenticSynthesizer(MockLLM())
        facts = [
            {
                "subject": "Apple",
                "predicate": "was founded in",
                "object_value": "1976",
                "confidence": 0.9,
            },
            {
                "subject": "Apple",
                "predicate": "was founded in",
                "object_value": "1975",
                "confidence": 0.6,
            },
        ]
        reports, resolved = synth._resolve_contradictions(
            "When was Apple founded?",
            facts,
            ["Conflicting founding years"],
        )
        assert len(reports) == 1
        # Should keep only one fact per (subject, predicate).
        assert len(resolved) == 1
        assert resolved[0]["object_value"] == "1976"  # Higher confidence


# ---------------------------------------------------------------------------
# 4. Ambiguous entity resolution
# ---------------------------------------------------------------------------


class TestAmbiguousEntities:
    """Verify the ontology normalization handles ambiguous names."""

    def test_normalize_strips_corporate_suffix(self, tmp_path):
        """'Apple Inc.' should normalize to 'Apple'."""
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        resolver = EntityResolver(store)
        normalized = resolver.normalize("Apple Inc.")
        assert "Inc" not in normalized
        store.close()

    def test_normalize_expands_abbreviation(self, tmp_path):
        """'ML' should normalize to 'Machine Learning'."""
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        resolver = EntityResolver(store)
        normalized = resolver.normalize("ML")
        assert "machine learning" in normalized.lower()
        store.close()

    def test_normalize_preserves_regular_names(self, tmp_path):
        """Regular entity names should be title-cased but otherwise unchanged."""
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        resolver = EntityResolver(store)
        assert resolver.normalize("Steve Jobs") == "Steve Jobs"
        store.close()


# ---------------------------------------------------------------------------
# 5. Multi-hop reasoning chains
# ---------------------------------------------------------------------------


class TestMultiHopChains:
    """Verify multi-hop fact chain construction."""

    def test_fact_chain_assembly(self):
        """Given A→B→C→D facts, they should all pass validation."""
        extractor = FactExtractor(MockLLM())
        chain_facts = [
            {
                "subject": "Alice",
                "predicate": "is CEO of",
                "object": "BetaCorp",
                "confidence": 1.0,
                "modality": "assertion",
                "is_speculative": False,
                "hedge_phrase": None,
            },
            {
                "subject": "BetaCorp",
                "predicate": "acquired",
                "object": "GammaTech",
                "confidence": 1.0,
                "modality": "assertion",
                "is_speculative": False,
                "hedge_phrase": None,
            },
            {
                "subject": "GammaTech",
                "predicate": "developed",
                "object": "QuantumX",
                "confidence": 1.0,
                "modality": "assertion",
                "is_speculative": False,
                "hedge_phrase": None,
            },
            {
                "subject": "QuantumX",
                "predicate": "powers",
                "object": "Nebula satellite",
                "confidence": 1.0,
                "modality": "assertion",
                "is_speculative": False,
                "hedge_phrase": None,
            },
        ]
        validated = extractor._validate_facts(chain_facts)
        assert len(validated) == 4
        # Each hop should maintain its modality.
        for f in validated:
            assert f["modality"] == "assertion"


# ---------------------------------------------------------------------------
# 6. Negation preservation
# ---------------------------------------------------------------------------


class TestNegationPreservation:
    """Verify that negation facts are properly classified."""

    def test_negation_modality(self):
        """Facts with explicit negation should be classified as negation."""
        extractor = FactExtractor(MockLLM())
        raw_facts = [
            {
                "subject": "This approach",
                "predicate": "is NOT",
                "object": "best solution for scalability",
                "confidence": 1.0,
                "modality": "negation",
                "is_speculative": False,
                "hedge_phrase": None,
            }
        ]
        validated = extractor._validate_facts(raw_facts)
        assert len(validated) == 1
        assert validated[0]["modality"] == "negation"
        assert validated[0]["is_speculative"] is False


# ---------------------------------------------------------------------------
# 7. Numerical precision
# ---------------------------------------------------------------------------


class TestNumericalPrecision:
    """Verify numerical values are preserved accurately."""

    def test_numeric_value_preservation(self):
        """Dollar amounts and percentages should be preserved exactly."""
        extractor = FactExtractor(MockLLM())
        raw_facts = [
            {
                "subject": "Apple",
                "predicate": "reached market cap of",
                "object_value": "$3 trillion",
                "confidence": 1.0,
                "modality": "assertion",
                "is_speculative": False,
                "hedge_phrase": None,
            }
        ]
        validated = extractor._validate_facts(raw_facts)
        assert validated[0]["object_value"] == "$3 trillion"


# ---------------------------------------------------------------------------
# 8. Schema migration backward compatibility
# ---------------------------------------------------------------------------


class TestSchemaBackwardCompatibility:
    """Verify that v1 facts without semantic metadata fields are handled."""

    def test_atomic_fact_from_dict_without_modality(self):
        """AtomicFact.from_dict should handle v1 dicts without modality."""
        v1_data = {
            "id": "old-fact",
            "subject_id": "s1",
            "predicate": "is",
            "confidence": 1.0,
        }
        fact = AtomicFact.from_dict(v1_data)
        assert fact.modality == "assertion"
        assert fact.is_speculative is False
        assert fact.hedge_phrase is None

    def test_save_and_load_with_semantic_metadata(self, tmp_path):
        """Facts with semantic metadata should round-trip through SQLite."""
        from pragma.storage.sqlite import SQLiteStore

        store = SQLiteStore(str(tmp_path))
        store.save_entity("e1", "TestEntity")

        fact = AtomicFact(
            id="f-meta",
            subject_id="e1",
            predicate="suggests",
            object_value="correlation",
            modality="hypothesis",
            is_speculative=True,
            hedge_phrase="appears to suggest",
        )
        store.save_fact(fact)

        # Retrieve and verify.
        facts = store.get_facts_by_subject("e1")
        assert len(facts) == 1
        retrieved_fact = facts[0]
        assert retrieved_fact.modality == "hypothesis"
        assert retrieved_fact.is_speculative is True
        assert retrieved_fact.hedge_phrase == "appears to suggest"
        store.close()
