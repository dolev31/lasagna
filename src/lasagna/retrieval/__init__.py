"""Retrieval layer for Lasagna.

This layer provides retrieval and ranking capabilities:
- Search: Hybrid search (dense + sparse) with reranking
- Enhancement: Query enhancement, classification, coverage
- Routing: send each query to a retrieval tier by its estimated complexity

This layer depends on Infrastructure for embedding and storage.
"""

# Search subsystem
from .search import (
    HybridSearcher,
    HybridSearchResult,
    get_hybrid_searcher,
    CombinedReranker,
    EnsembleReranker,
    RerankedResult,
    get_reranker,
)

# Enhancement subsystem
from .enhancement import (
    QueryEnhancer,
    HyDEGenerator,
    EnhancedQuery,
    get_query_enhancer,
    QueryClassifier,
    QueryClassification,
    QueryType,
    get_query_classifier,
    CoverageScorer,
    CoverageScore,
    QueryAspect,
    LLMCoverageScorer,
)

# Routing subsystem
from .routing import (
    QueryRouter,
    RoutingTier,
    RoutingDecision,
    RoutingSignals,
    get_query_router,
    extract_signals,
    RuleBasedStrategy,
    AdaptiveStrategy,
)

__all__ = [
    # Search
    "HybridSearcher",
    "HybridSearchResult",
    "get_hybrid_searcher",
    "CombinedReranker",
    "EnsembleReranker",
    "RerankedResult",
    "get_reranker",
    # Enhancement
    "QueryEnhancer",
    "HyDEGenerator",
    "EnhancedQuery",
    "get_query_enhancer",
    "QueryClassifier",
    "QueryClassification",
    "QueryType",
    "get_query_classifier",
    "CoverageScorer",
    "CoverageScore",
    "QueryAspect",
    "LLMCoverageScorer",
    # Routing
    "QueryRouter",
    "RoutingTier",
    "RoutingDecision",
    "RoutingSignals",
    "get_query_router",
    "extract_signals",
    "RuleBasedStrategy",
    "AdaptiveStrategy",
]
