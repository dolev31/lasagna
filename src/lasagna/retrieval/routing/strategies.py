"""Routing strategies for selecting retrieval tiers.

Provides a rule-based strategy (deterministic, zero LLM overhead) and an
adaptive strategy that escalates the tier when classification confidence
is low.  Both strategies are informed by Adaptive-RAG (NAACL 2024).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Protocol

from ..enhancement.classifier import QueryType
from .models import RoutingDecision, RoutingSignals, RoutingTier


# ---------------------------------------------------------------------------
# Config protocol — only the properties we actually need
# ---------------------------------------------------------------------------


class _RoutingConfig(Protocol):
    """Minimal config surface used by strategies."""

    routing_confidence_threshold: float


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


class BaseRoutingStrategy(ABC):
    """Abstract base for routing strategies."""

    @abstractmethod
    def decide(self, signals: RoutingSignals, config: Any) -> RoutingDecision:
        """Select a routing tier based on extracted signals."""
        ...


# ---------------------------------------------------------------------------
# Rule-Based Strategy
# ---------------------------------------------------------------------------


class RuleBasedStrategy(BaseRoutingStrategy):
    """Deterministic decision tree — zero LLM overhead, <1 ms.

    Decision order:
        1. NAVIGATIONAL + short  → FAST
        2. FACTUAL + confident + short → FAST
        3. multi-hop OR complex  → COMPREHENSIVE
        4. COMPARATIVE           → STANDARD  (locked design decision)
        5. CONCEPTUAL + confident → STANDARD
        6. default               → STANDARD
    """

    strategy_name = "rule_based"

    def decide(self, signals: RoutingSignals, config: Any) -> RoutingDecision:
        qt = signals.query_type
        conf = signals.query_type_confidence
        wc = signals.word_count

        # 1. NAVIGATIONAL short queries — simple lookup
        if qt == QueryType.NAVIGATIONAL.value and wc <= 5:
            return self._make(
                RoutingTier.FAST, signals, 0.8, "Short navigational query — vector-only sufficient"
            )

        # 2. FACTUAL confident short queries
        if qt == QueryType.FACTUAL.value and conf > 0.6 and wc <= 8:
            return self._make(
                RoutingTier.FAST,
                signals,
                0.7,
                "Confident factual query — keyword/vector sufficient",
            )

        # 3. Multi-hop or complex
        if signals.has_multi_hop_indicators or signals.estimated_complexity == "complex":
            return self._make(
                RoutingTier.COMPREHENSIVE,
                signals,
                0.7,
                "Complex or multi-hop query — agentic retrieval needed",
            )

        # 4. COMPARATIVE → STANDARD (locked: needs reranking, not agentic)
        if qt == QueryType.COMPARATIVE.value:
            return self._make(
                RoutingTier.STANDARD, signals, 0.7, "Comparative query — hybrid + reranking optimal"
            )

        # 5. CONCEPTUAL with confidence
        if qt == QueryType.CONCEPTUAL.value and conf > 0.5:
            return self._make(
                RoutingTier.STANDARD, signals, 0.6, "Conceptual query — HyDE + hybrid beneficial"
            )

        # 6. Default
        return self._make(RoutingTier.STANDARD, signals, 0.5, "Default — standard pipeline")

    def _make(
        self, tier: RoutingTier, signals: RoutingSignals, confidence: float, rationale: str
    ) -> RoutingDecision:
        return RoutingDecision(
            tier=tier,
            signals=signals,
            confidence=confidence,
            rationale=rationale,
            strategy_used=self.strategy_name,
        )


# ---------------------------------------------------------------------------
# Adaptive Strategy
# ---------------------------------------------------------------------------


class AdaptiveStrategy(BaseRoutingStrategy):
    """Extends rule-based with confidence-aware tier escalation.

    If the rule-based decision's confidence is below
    ``routing_confidence_threshold``, the tier is escalated one level
    (FAST → STANDARD, STANDARD → COMPREHENSIVE).
    """

    strategy_name = "adaptive"

    def __init__(self) -> None:
        self._rule_based = RuleBasedStrategy()

    def decide(self, signals: RoutingSignals, config: Any) -> RoutingDecision:
        decision = self._rule_based.decide(signals, config)

        threshold = getattr(config, "routing_confidence_threshold", 0.5)
        if decision.confidence < threshold:
            escalated = self._escalate(decision.tier)
            if escalated != decision.tier:
                return RoutingDecision(
                    tier=escalated,
                    signals=signals,
                    confidence=decision.confidence,
                    rationale=(
                        f"{decision.rationale} — escalated from {decision.tier.value} "
                        f"to {escalated.value} (confidence {decision.confidence:.2f} "
                        f"< threshold {threshold:.2f})"
                    ),
                    strategy_used=self.strategy_name,
                )

        # Confidence is acceptable — keep the rule-based decision
        decision.strategy_used = self.strategy_name
        return decision

    @staticmethod
    def _escalate(tier: RoutingTier) -> RoutingTier:
        if tier == RoutingTier.FAST:
            return RoutingTier.STANDARD
        if tier == RoutingTier.STANDARD:
            return RoutingTier.COMPREHENSIVE
        return tier  # COMPREHENSIVE stays
