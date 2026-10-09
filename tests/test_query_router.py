"""Tests for the intelligent query routing system.

Covers signal extraction, routing strategies, forced tier overrides,
COMPREHENSIVE downgrade, config-disabled routing, FAST overrides,
recursion guard semantics, and cache key correctness.
"""

import pytest
from dataclasses import dataclass

from lasagna.retrieval.routing.models import (
    RoutingDecision,
    RoutingSignals,
    RoutingTier,
)
from lasagna.retrieval.routing.signals import (
    extract_signals,
    _detect_multi_hop,
    _estimate_complexity,
)
from lasagna.retrieval.routing.strategies import (
    AdaptiveStrategy,
    RuleBasedStrategy,
)
from lasagna.retrieval.routing.router import (
    QueryRouter,
    _parse_tier,
)
from lasagna.retrieval.enhancement.classifier import (
    QueryClassification,
    QueryType,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight config stubs
# ---------------------------------------------------------------------------


@dataclass
class _StubConfig:
    """Minimal config stub with defaults matching production."""

    enable_query_routing: bool = True
    routing_strategy: str = "rule_based"
    routing_default_tier: str = "standard"
    routing_confidence_threshold: float = 0.5
    routing_allow_tier_override: bool = True
    routing_log_decisions: bool = False
    use_agentic_rag: bool = True
    enable_query_classification: bool = True


def _make_classification(
    query_type: QueryType,
    confidence: float = 0.8,
) -> QueryClassification:
    return QueryClassification(
        query_type=query_type,
        confidence=confidence,
        recommended_alpha=0.7,
        use_hyde=False,
        hyde_strategy="off",
        use_query_expansion=False,
        rationale="test",
    )


# =========================================================================
# 1. Signal Extraction
# =========================================================================


class TestSignalExtraction:
    """Test extract_signals and its helper functions."""

    def test_factual_short_is_simple(self):
        cls = _make_classification(QueryType.FACTUAL, confidence=0.8)
        signals = extract_signals("What is X?", None, cls)
        assert signals.estimated_complexity == "simple"
        assert signals.query_type == "factual"
        assert signals.word_count == 3

    def test_navigational_short_is_simple(self):
        cls = _make_classification(QueryType.NAVIGATIONAL, confidence=0.7)
        signals = extract_signals("Find section A", None, cls)
        assert signals.estimated_complexity == "simple"

    def test_conceptual_long_is_complex(self):
        cls = _make_classification(QueryType.CONCEPTUAL, confidence=0.8)
        query = " ".join(["word"] * 25)  # 25 words
        signals = extract_signals(query, None, cls)
        assert signals.estimated_complexity == "complex"

    def test_multi_hop_is_complex(self):
        cls = _make_classification(QueryType.FACTUAL, confidence=0.8)
        query = "Summarize all findings and then compare with baseline"
        signals = extract_signals(query, None, cls)
        assert signals.has_multi_hop_indicators is True
        assert signals.estimated_complexity == "complex"

    def test_comparative_moderate_is_moderate(self):
        cls = _make_classification(QueryType.COMPARATIVE, confidence=0.7)
        signals = extract_signals("Compare revenue and expenses", None, cls)
        assert signals.estimated_complexity == "moderate"

    def test_no_classification_defaults_to_unknown(self):
        signals = extract_signals("some query", None, None)
        assert signals.query_type == "unknown"
        assert signals.query_type_confidence == 0.0

    def test_document_ids_sets_filter_flag(self):
        signals = extract_signals("query", ["doc_1", "doc_2"], None)
        assert signals.has_specific_document_filter is True

    def test_no_document_ids(self):
        signals = extract_signals("query", None, None)
        assert signals.has_specific_document_filter is False


class TestMultiHopDetection:
    """Test _detect_multi_hop regex patterns."""

    @pytest.mark.parametrize(
        "query",
        [
            "Summarize all the findings",
            "Find revenue and also list expenses",
            "Based on the data what is the trend",
            "Synthesize information from each document",
            "First explain X then describe Y",
            "Given that revenue grows how does profit change",
        ],
    )
    def test_positive_patterns(self, query):
        assert _detect_multi_hop(query) is True

    @pytest.mark.parametrize(
        "query",
        [
            "What is the revenue?",
            "Find the CEO name",
            "Compare X with Y",
        ],
    )
    def test_negative_patterns(self, query):
        assert _detect_multi_hop(query) is False


class TestComplexityEstimation:
    """Test _estimate_complexity function."""

    def test_simple(self):
        assert _estimate_complexity("factual", 0.8, 5, False) == "simple"

    def test_complex_multi_hop(self):
        assert _estimate_complexity("factual", 0.8, 5, True) == "complex"

    def test_complex_long_conceptual(self):
        assert _estimate_complexity("conceptual", 0.8, 25, False) == "complex"

    def test_moderate_default(self):
        assert _estimate_complexity("comparative", 0.8, 10, False) == "moderate"

    def test_low_confidence_factual_not_simple(self):
        # confidence <= 0.5 should not be "simple"
        assert _estimate_complexity("factual", 0.3, 5, False) == "moderate"


# =========================================================================
# 2. RuleBasedStrategy
# =========================================================================


class TestRuleBasedStrategy:
    """Test the deterministic rule-based routing strategy."""

    def setup_method(self):
        self.strategy = RuleBasedStrategy()
        self.config = _StubConfig()

    def test_navigational_short_to_fast(self):
        cls = _make_classification(QueryType.NAVIGATIONAL, confidence=0.8)
        signals = extract_signals("Find X", None, cls)
        decision = self.strategy.decide(signals, self.config)
        assert decision.tier == RoutingTier.FAST

    def test_factual_confident_short_to_fast(self):
        cls = _make_classification(QueryType.FACTUAL, confidence=0.8)
        signals = extract_signals("What is revenue?", None, cls)
        decision = self.strategy.decide(signals, self.config)
        assert decision.tier == RoutingTier.FAST

    def test_factual_low_confidence_not_fast(self):
        cls = _make_classification(QueryType.FACTUAL, confidence=0.4)
        signals = extract_signals("What is X?", None, cls)
        decision = self.strategy.decide(signals, self.config)
        assert decision.tier != RoutingTier.FAST

    def test_comparative_to_standard(self):
        """LOCKED: COMPARATIVE always routes to STANDARD."""
        cls = _make_classification(QueryType.COMPARATIVE, confidence=0.9)
        signals = extract_signals("Compare A with B", None, cls)
        decision = self.strategy.decide(signals, self.config)
        assert decision.tier == RoutingTier.STANDARD

    def test_multi_hop_to_comprehensive(self):
        cls = _make_classification(QueryType.CONCEPTUAL, confidence=0.5)
        signals = extract_signals(
            "Summarize all findings across documents and then compare the results with the baseline",
            None,
            cls,
        )
        decision = self.strategy.decide(signals, self.config)
        assert decision.tier == RoutingTier.COMPREHENSIVE

    def test_conceptual_confident_to_standard(self):
        cls = _make_classification(QueryType.CONCEPTUAL, confidence=0.7)
        signals = extract_signals("Explain the concept of RAG", None, cls)
        decision = self.strategy.decide(signals, self.config)
        assert decision.tier == RoutingTier.STANDARD

    def test_default_to_standard(self):
        cls = _make_classification(QueryType.UNKNOWN, confidence=0.3)
        signals = extract_signals("random query text here", None, cls)
        decision = self.strategy.decide(signals, self.config)
        assert decision.tier == RoutingTier.STANDARD


# =========================================================================
# 3. AdaptiveStrategy
# =========================================================================


class TestAdaptiveStrategy:
    """Test confidence-aware tier escalation."""

    def setup_method(self):
        self.strategy = AdaptiveStrategy()

    def test_low_confidence_escalates_fast_to_standard(self):
        config = _StubConfig(routing_confidence_threshold=0.75)
        cls = _make_classification(QueryType.NAVIGATIONAL, confidence=0.8)
        signals = extract_signals("Find X", None, cls)
        # RuleBasedStrategy would give FAST with confidence=0.8
        # But confidence 0.8 >= 0.75 threshold, so no escalation
        decision = self.strategy.decide(signals, config)
        assert decision.tier == RoutingTier.FAST

    def test_escalation_below_threshold(self):
        # Set threshold high so default confidence of 0.5 triggers escalation
        config = _StubConfig(routing_confidence_threshold=0.9)
        cls = _make_classification(QueryType.UNKNOWN, confidence=0.3)
        signals = extract_signals("some query", None, cls)
        # RuleBasedStrategy gives STANDARD with confidence 0.5 < 0.9
        decision = self.strategy.decide(signals, config)
        assert decision.tier == RoutingTier.COMPREHENSIVE
        assert "escalated" in decision.rationale

    def test_comprehensive_stays_comprehensive(self):
        config = _StubConfig(routing_confidence_threshold=0.9)
        cls = _make_classification(QueryType.FACTUAL, confidence=0.8)
        query = "Summarize all findings and also synthesize insights across documents"
        signals = extract_signals(query, None, cls)
        decision = self.strategy.decide(signals, config)
        # COMPREHENSIVE doesn't escalate further
        assert decision.tier == RoutingTier.COMPREHENSIVE


# =========================================================================
# 4. Forced Tier Override
# =========================================================================


class TestForcedTier:
    """Test forced tier ignores query complexity."""

    def setup_method(self):
        self.router = QueryRouter()

    def test_force_fast_on_complex_query(self):
        config = _StubConfig()
        cls = _make_classification(QueryType.CONCEPTUAL, confidence=0.8)
        query = "Summarize all findings across documents"
        decision = self.router.route(query, None, cls, config, forced_tier="fast")
        assert decision.tier == RoutingTier.FAST
        assert decision.strategy_used == "forced"

    def test_force_comprehensive(self):
        config = _StubConfig()
        cls = _make_classification(QueryType.FACTUAL, confidence=0.9)
        decision = self.router.route("What is X?", None, cls, config, forced_tier="comprehensive")
        assert decision.tier == RoutingTier.COMPREHENSIVE

    def test_force_denied_when_override_disabled(self):
        config = _StubConfig(routing_allow_tier_override=False)
        cls = _make_classification(QueryType.FACTUAL, confidence=0.9)
        decision = self.router.route("What is X?", None, cls, config, forced_tier="fast")
        # Should ignore forced_tier and route normally
        assert decision.strategy_used != "forced"


# =========================================================================
# 5. COMPREHENSIVE Downgrade
# =========================================================================


class TestComprehensiveDowngrade:
    """Test COMPREHENSIVE → STANDARD when use_agentic_rag=false."""

    def setup_method(self):
        self.router = QueryRouter()

    def test_downgrade_when_agentic_disabled(self):
        config = _StubConfig(use_agentic_rag=False)
        decision = self.router.route(
            "Summarize all findings across documents",
            None,
            None,
            config,
            forced_tier="comprehensive",
        )
        assert decision.tier == RoutingTier.STANDARD
        assert "downgraded" in decision.rationale
        assert "use_agentic_rag" in decision.rationale

    def test_no_downgrade_when_agentic_enabled(self):
        config = _StubConfig(use_agentic_rag=True)
        decision = self.router.route(
            "query",
            None,
            None,
            config,
            forced_tier="comprehensive",
        )
        assert decision.tier == RoutingTier.COMPREHENSIVE


# =========================================================================
# 6. Config: enable_query_routing=False
# =========================================================================


class TestRoutingDisabled:
    """Test that disabling routing returns the default tier."""

    def setup_method(self):
        self.router = QueryRouter()

    def test_disabled_returns_default_tier(self):
        config = _StubConfig(enable_query_routing=False, routing_default_tier="standard")
        cls = _make_classification(QueryType.FACTUAL, confidence=0.9)
        decision = self.router.route("What is X?", None, cls, config)
        assert decision.tier == RoutingTier.STANDARD
        assert decision.strategy_used == "disabled"

    def test_disabled_with_fast_default(self):
        config = _StubConfig(enable_query_routing=False, routing_default_tier="fast")
        decision = self.router.route("complex query", None, None, config)
        assert decision.tier == RoutingTier.FAST

    def test_disabled_comprehensive_downgraded(self):
        config = _StubConfig(
            enable_query_routing=False,
            routing_default_tier="comprehensive",
            use_agentic_rag=False,
        )
        decision = self.router.route("query", None, None, config)
        assert decision.tier == RoutingTier.STANDARD
        assert "downgraded" in decision.rationale


# =========================================================================
# 7. Override Application
# =========================================================================


class TestTierOverrides:
    """Test that FAST disables features and STANDARD doesn't."""

    def test_fast_overrides(self):
        overrides = QueryRouter.get_tier_overrides(RoutingTier.FAST)
        assert overrides["use_hyde"] is False
        assert overrides["use_reranking"] is False
        assert overrides["enable_hybrid_search"] is False
        assert overrides["enable_mmr"] is False
        assert overrides["enable_context_reordering"] is False
        assert overrides["enable_hierarchical"] is False

    def test_standard_no_overrides(self):
        overrides = QueryRouter.get_tier_overrides(RoutingTier.STANDARD)
        assert overrides == {}

    def test_comprehensive_no_overrides(self):
        overrides = QueryRouter.get_tier_overrides(RoutingTier.COMPREHENSIVE)
        assert overrides == {}

    def test_fast_decision_has_overrides(self):
        router = QueryRouter()
        config = _StubConfig()
        cls = _make_classification(QueryType.NAVIGATIONAL, confidence=0.9)
        decision = router.route("Find X", None, cls, config)
        assert decision.tier == RoutingTier.FAST
        assert decision.overrides.get("use_hyde") is False
        assert decision.overrides.get("use_reranking") is False


# =========================================================================
# 8. Recursion Guard (semantic test)
# =========================================================================


class TestRecursionGuard:
    """Verify the _from_routing=True pattern prevents routing recursion.

    This is a design-level test: when _from_routing=True is passed,
    the SearchOrchestrator's routing block is skipped. We verify this
    by testing that the router itself is deterministic and produces
    consistent decisions (so re-routing would be redundant).
    """

    def test_same_query_same_decision(self):
        router = QueryRouter()
        config = _StubConfig()
        cls = _make_classification(QueryType.FACTUAL, confidence=0.8)
        d1 = router.route("What is X?", None, cls, config)
        d2 = router.route("What is X?", None, cls, config)
        assert d1.tier == d2.tier
        assert d1.confidence == d2.confidence


# =========================================================================
# 9. Cache Key Correctness
# =========================================================================


class TestCacheKeyCorrectness:
    """Verify FAST and STANDARD produce different cache keys for same query."""

    def test_different_tiers_different_cache_keys(self):
        from lasagna.infrastructure.cache.search_cache import SearchCache

        cache = SearchCache(max_size=100, ttl_seconds=60)

        # Compute keys with FAST settings (hybrid=False, mmr=False)
        fast_key = cache._compute_key(
            query="What is X?",
            session_id="sess_1",
            top_k=10,
            use_hyde=False,
            use_reranking=False,
            enable_hybrid_search=False,
            enable_mmr=False,
        )

        # Compute keys with STANDARD settings (hybrid=True, mmr=True)
        standard_key = cache._compute_key(
            query="What is X?",
            session_id="sess_1",
            top_k=10,
            use_hyde=True,
            use_reranking=True,
            enable_hybrid_search=True,
            enable_mmr=True,
        )

        assert fast_key != standard_key, (
            "FAST and STANDARD must produce different cache keys to prevent "
            "cross-tier cache collisions"
        )

    def test_same_params_same_key(self):
        from lasagna.infrastructure.cache.search_cache import SearchCache

        cache = SearchCache(max_size=100, ttl_seconds=60)
        k1 = cache._compute_key("q", "s", top_k=10)
        k2 = cache._compute_key("q", "s", top_k=10)
        assert k1 == k2


# =========================================================================
# 10. RoutingDecision serialization
# =========================================================================


class TestRoutingDecisionSerialization:
    """Test to_dict() for API/log serialization."""

    def test_to_dict_with_signals(self):
        signals = RoutingSignals(
            query_text="test",
            word_count=1,
            query_type="factual",
            query_type_confidence=0.8,
            has_multi_hop_indicators=False,
            has_specific_document_filter=False,
            estimated_complexity="simple",
        )
        decision = RoutingDecision(
            tier=RoutingTier.FAST,
            signals=signals,
            confidence=0.8,
            rationale="test rationale",
            strategy_used="rule_based",
        )
        d = decision.to_dict()
        assert d["tier"] == "fast"
        assert d["confidence"] == 0.8
        assert d["signals"]["query_type"] == "factual"
        assert d["strategy_used"] == "rule_based"

    def test_to_dict_without_signals(self):
        decision = RoutingDecision(
            tier=RoutingTier.STANDARD,
            signals=None,
            confidence=1.0,
            rationale="disabled",
            strategy_used="disabled",
        )
        d = decision.to_dict()
        assert d["signals"] is None


# =========================================================================
# 11. _parse_tier helper
# =========================================================================


class TestParseTier:
    """Test tier string parsing with fallback."""

    def test_valid_tiers(self):
        assert _parse_tier("fast") == RoutingTier.FAST
        assert _parse_tier("standard") == RoutingTier.STANDARD
        assert _parse_tier("comprehensive") == RoutingTier.COMPREHENSIVE

    def test_case_insensitive(self):
        assert _parse_tier("FAST") == RoutingTier.FAST
        assert _parse_tier("Standard") == RoutingTier.STANDARD

    def test_unknown_falls_back_to_standard(self):
        assert _parse_tier("invalid") == RoutingTier.STANDARD
        assert _parse_tier("") == RoutingTier.STANDARD

    def test_none_falls_back_to_standard(self):
        """_parse_tier(None) must not crash with AttributeError."""
        assert _parse_tier(None) == RoutingTier.STANDARD


# =========================================================================
# 12. Strategy Caching (Fix 3)
# =========================================================================


class TestStrategyCaching:
    """Verify strategy instances are cached and reused."""

    def setup_method(self):
        # Clear cache before each test
        QueryRouter._strategy_cache.clear()

    def test_same_strategy_returns_same_instance(self):
        config = _StubConfig(routing_strategy="rule_based")
        s1 = QueryRouter._get_strategy(config)
        s2 = QueryRouter._get_strategy(config)
        assert s1 is s2, "Strategy instances should be cached (same object)"

    def test_different_strategies_cached_separately(self):
        config_rb = _StubConfig(routing_strategy="rule_based")
        config_ad = _StubConfig(routing_strategy="adaptive")
        s_rb = QueryRouter._get_strategy(config_rb)
        s_ad = QueryRouter._get_strategy(config_ad)
        assert s_rb is not s_ad
        assert isinstance(s_rb, RuleBasedStrategy)
        assert isinstance(s_ad, AdaptiveStrategy)

    def test_unknown_strategy_defaults_to_rule_based(self):
        config = _StubConfig(routing_strategy="nonexistent")
        s = QueryRouter._get_strategy(config)
        assert isinstance(s, RuleBasedStrategy)


# =========================================================================
# 13. Invalid Forced Tier Validation (Fix 5)
# =========================================================================


class TestInvalidForcedTier:
    """Verify invalid forced_tier is rejected, not silently defaulted."""

    def setup_method(self):
        self.router = QueryRouter()

    def test_typo_tier_falls_through_to_strategy(self):
        """An invalid forced_tier should be ignored and strategy-based
        routing should decide instead."""
        config = _StubConfig()
        cls = _make_classification(QueryType.NAVIGATIONAL, confidence=0.9)
        decision = self.router.route("Find X", None, cls, config, forced_tier="fst")
        # Should NOT be "forced" — invalid tier falls through to strategy
        assert decision.strategy_used != "forced"
        # Strategy should route this to FAST based on NAVIGATIONAL+short
        assert decision.tier == RoutingTier.FAST
        assert decision.strategy_used == "rule_based"

    def test_empty_forced_tier_ignored(self):
        """Empty string forced_tier should be treated as no override."""
        config = _StubConfig()
        cls = _make_classification(QueryType.FACTUAL, confidence=0.8)
        decision = self.router.route("What is X?", None, cls, config, forced_tier="")
        # Empty string is falsy, so forced_tier check is skipped entirely
        assert decision.strategy_used == "rule_based"

    def test_valid_forced_tier_still_works(self):
        """Valid forced tiers should still be honored after the validation change."""
        config = _StubConfig()
        cls = _make_classification(QueryType.FACTUAL, confidence=0.8)
        for tier_str in ("fast", "standard", "comprehensive"):
            decision = self.router.route("test", None, cls, config, forced_tier=tier_str)
            assert decision.tier == RoutingTier(tier_str)
            assert decision.strategy_used == "forced"


# =========================================================================
# 14. Immutable Constants (Fix 6)
# =========================================================================


class TestImmutableConstants:
    """Verify module-level constants are immutable."""

    def test_multi_hop_patterns_is_tuple(self):
        from lasagna.retrieval.routing.signals import _MULTI_HOP_PATTERNS

        assert isinstance(_MULTI_HOP_PATTERNS, tuple)

    def test_fast_overrides_is_immutable(self):
        from lasagna.retrieval.routing.router import _FAST_OVERRIDES

        with pytest.raises(TypeError):
            _FAST_OVERRIDES["new_key"] = True


# =========================================================================
# 15. Async Routing Decision Relay
# =========================================================================


class TestAsyncRoutingRelay:
    """Verify routing decision flows from executor thread to event loop.

    The async_search methods must relay the routing decision written
    by search() in the executor thread back to an instance attribute
    readable in the event loop thread.
    """

    def test_routing_local_is_thread_specific(self):
        """Confirm threading.local() isolates per-thread — proves why
        the relay pattern is needed."""
        import threading
        from lasagna.api._search_orchestrator import _routing_local

        _routing_local.last_decision = "main_thread_value"
        seen_in_child = {}

        def _child():
            seen_in_child["value"] = getattr(_routing_local, "last_decision", None)

        t = threading.Thread(target=_child)
        t.start()
        t.join()

        # Child thread must NOT see the main thread's value
        assert seen_in_child["value"] is None, (
            "threading.local() must be per-thread; child should see None"
        )
        # Main thread still has its value
        assert _routing_local.last_decision == "main_thread_value"
