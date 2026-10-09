"""Data models for the query routing system.

Defines the routing tiers, signal extraction results, and routing decisions
used by the intelligent query router to select optimal retrieval strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class RoutingTier(Enum):
    """Retrieval complexity tiers.

    FAST: Vector-only search, no HyDE/reranking/hybrid. ~50-100ms.
    STANDARD: Full pipeline (hybrid + reranking + HyDE). ~200-400ms.
    COMPREHENSIVE: Agentic multi-hop retrieval. ~1-5s.
    """

    FAST = "fast"
    STANDARD = "standard"
    COMPREHENSIVE = "comprehensive"


@dataclass
class RoutingSignals:
    """Extracted signals used by routing strategies to make tier decisions."""

    query_text: str
    word_count: int
    query_type: str  # QueryType.value from classifier
    query_type_confidence: float
    has_multi_hop_indicators: bool
    has_specific_document_filter: bool
    estimated_complexity: str  # "simple", "moderate", "complex"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word_count": self.word_count,
            "query_type": self.query_type,
            "query_type_confidence": self.query_type_confidence,
            "has_multi_hop_indicators": self.has_multi_hop_indicators,
            "has_specific_document_filter": self.has_specific_document_filter,
            "estimated_complexity": self.estimated_complexity,
        }


@dataclass
class RoutingDecision:
    """Result of query routing with the selected tier and metadata."""

    tier: RoutingTier
    signals: Optional[RoutingSignals]
    confidence: float
    rationale: str
    overrides: Dict[str, Any] = field(default_factory=dict)
    strategy_used: str = "none"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier.value,
            "confidence": round(self.confidence, 3),
            "rationale": self.rationale,
            "strategy_used": self.strategy_used,
            "overrides": self.overrides,
            "signals": self.signals.to_dict() if self.signals else None,
        }
