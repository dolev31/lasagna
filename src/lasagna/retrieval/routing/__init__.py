"""Query routing: each query goes to a retrieval tier by its estimated complexity.

Routes each query to the optimal retrieval tier (FAST, STANDARD, or
COMPREHENSIVE) based on query complexity signals, informed by
Adaptive-RAG (NAACL 2024).
"""

from .models import RoutingDecision, RoutingSignals, RoutingTier
from .router import QueryRouter, get_query_router
from .signals import extract_signals
from .strategies import AdaptiveStrategy, RuleBasedStrategy

__all__ = [
    "QueryRouter",
    "get_query_router",
    "RoutingTier",
    "RoutingDecision",
    "RoutingSignals",
    "extract_signals",
    "RuleBasedStrategy",
    "AdaptiveStrategy",
]
