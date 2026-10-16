"""Query Router — selects the optimal retrieval tier per query.

Thread-safe singleton that integrates signal extraction, strategy selection,
and tier override/downgrade logic.
"""

from __future__ import annotations

import threading
from types import MappingProxyType
from typing import Any, Dict, Optional, Sequence

from loguru import logger

from ..enhancement.classifier import QueryClassification
from .models import RoutingDecision, RoutingTier
from .signals import extract_signals
from .strategies import AdaptiveStrategy, BaseRoutingStrategy, RuleBasedStrategy


# ---------------------------------------------------------------------------
# Tier → override map (immutable to prevent accidental mutation)
# ---------------------------------------------------------------------------

_FAST_OVERRIDES: MappingProxyType = MappingProxyType(
    {
        "use_hyde": False,
        "use_reranking": False,
        "enable_hybrid_search": False,
        "enable_mmr": False,
        "enable_context_reordering": False,
        "enable_hierarchical": False,
    }
)

_STRATEGY_MAP: Dict[str, type[BaseRoutingStrategy]] = {
    "rule_based": RuleBasedStrategy,
    "adaptive": AdaptiveStrategy,
}

_VALID_TIERS = frozenset(t.value for t in RoutingTier)


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------


class QueryRouter:
    """Selects a retrieval tier for each incoming query.

    Lifecycle:
        1. Checks config flags (``enable_query_routing``, forced tier).
        2. Extracts lightweight signals from the query + classification.
        3. Delegates to the configured strategy (rule-based or adaptive).
        4. Applies COMPREHENSIVE downgrade when agentic RAG is disabled.
        5. Returns a ``RoutingDecision`` with the tier and any overrides.
    """

    # Strategies are stateless — pre-populated to avoid unsynchronized writes.
    _strategy_cache: Dict[str, BaseRoutingStrategy] = {
        "rule_based": RuleBasedStrategy(),
        "adaptive": AdaptiveStrategy(),
    }

    def route(
        self,
        query: str,
        document_ids: Optional[Sequence[str]],
        classification: Optional[QueryClassification],
        config: Any,
        forced_tier: Optional[str] = None,
    ) -> RoutingDecision:
        """Route a query to a retrieval tier.

        Args:
            query: The raw search query.
            document_ids: Optional document filter.
            classification: Pre-computed QueryClassification (may be None).
            config: RAGConfig (session-scoped).
            forced_tier: Tier forced by the API caller (e.g. "fast").

        Returns:
            RoutingDecision with the selected tier, rationale, and overrides.
        """
        enable = getattr(config, "enable_query_routing", True)
        default_tier_str = getattr(config, "routing_default_tier", "standard")

        # --- Routing disabled: return default tier ---
        if not enable:
            default_tier = _parse_tier(default_tier_str)
            decision = RoutingDecision(
                tier=default_tier,
                signals=None,
                confidence=1.0,
                rationale="Query routing disabled: using default tier",
                strategy_used="disabled",
            )
            decision.overrides = self.get_tier_overrides(default_tier)
            return self._apply_downgrade(decision, config)

        # --- Forced tier from API ---
        allow_override = getattr(config, "routing_allow_tier_override", True)
        if forced_tier and allow_override:
            forced_lower = forced_tier.strip().lower()
            if forced_lower not in _VALID_TIERS:
                logger.warning(
                    f"Invalid routing_tier '{forced_tier}', ignoring. Valid: {sorted(_VALID_TIERS)}"
                )
                # Fall through to strategy-based routing
            else:
                tier = RoutingTier(forced_lower)
                signals = extract_signals(query, document_ids, classification)
                decision = RoutingDecision(
                    tier=tier,
                    signals=signals,
                    confidence=1.0,
                    rationale=f"Tier forced to '{tier.value}' by caller",
                    strategy_used="forced",
                )
                decision.overrides = self.get_tier_overrides(tier)
                return self._apply_downgrade(decision, config)

        # --- Strategy-based routing ---
        signals = extract_signals(query, document_ids, classification)
        strategy = self._get_strategy(config)
        decision = strategy.decide(signals, config)
        decision.overrides = self.get_tier_overrides(decision.tier)
        return self._apply_downgrade(decision, config)

    # -- helpers ---------------------------------------------------------------

    @staticmethod
    def get_tier_overrides(tier: RoutingTier) -> Dict[str, Any]:
        """Return config overrides for a given tier.

        FAST disables heavyweight features; STANDARD and COMPREHENSIVE
        use the existing pipeline config as-is.
        """
        if tier == RoutingTier.FAST:
            return dict(_FAST_OVERRIDES)
        return {}

    @staticmethod
    def _apply_downgrade(decision: RoutingDecision, config: Any) -> RoutingDecision:
        """Downgrade COMPREHENSIVE → STANDARD when agentic RAG is disabled."""
        if decision.tier != RoutingTier.COMPREHENSIVE:
            return decision

        use_agentic = getattr(config, "use_agentic_rag", False)
        if not use_agentic:
            original_rationale = decision.rationale
            decision.tier = RoutingTier.STANDARD
            decision.overrides = QueryRouter.get_tier_overrides(RoutingTier.STANDARD)
            decision.rationale = (
                f"{original_rationale}: downgraded to standard "
                f"(use_agentic_rag is disabled in config)"
            )
            logger.debug(
                "Routing tier downgraded from comprehensive to standard: use_agentic_rag=false"
            )
        return decision

    @classmethod
    def _get_strategy(cls, config: Any) -> BaseRoutingStrategy:
        strategy_name = getattr(config, "routing_strategy", "rule_based")
        if strategy_name not in cls._strategy_cache:
            strategy_cls = _STRATEGY_MAP.get(strategy_name, RuleBasedStrategy)
            cls._strategy_cache[strategy_name] = strategy_cls()
        return cls._strategy_cache[strategy_name]


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------

_router: Optional[QueryRouter] = None
_router_lock = threading.Lock()


def get_query_router() -> QueryRouter:
    """Get or create the query router singleton (thread-safe)."""
    global _router
    if _router is None:
        with _router_lock:
            if _router is None:
                _router = QueryRouter()
    return _router


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_tier(value: Optional[str]) -> RoutingTier:
    """Parse a tier string, falling back to STANDARD for unknown/None values."""
    if not value:
        logger.warning("Routing tier is empty/None, defaulting to STANDARD")
        return RoutingTier.STANDARD
    try:
        return RoutingTier(value.lower())
    except ValueError:
        logger.warning(f"Unknown routing tier '{value}', defaulting to STANDARD")
        return RoutingTier.STANDARD
