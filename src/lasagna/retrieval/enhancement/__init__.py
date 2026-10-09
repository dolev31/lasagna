"""Query enhancement subsystem for RAG retrieval.

This module provides query enhancement capabilities:
- HyDE (Hypothetical Document Embeddings)
- Query expansion and decomposition
- Query classification for adaptive retrieval
- Coverage scoring for completeness analysis
"""

from .enhancer import (
    QueryEnhancer,
    HyDEGenerator,
    EnhancedQuery,
    get_query_enhancer,
)
from .classifier import (
    QueryClassifier,
    QueryClassification,
    QueryType,
    get_query_classifier,
    classify_query,
)
from .coverage import (
    CoverageScorer,
    CoverageScore,
    QueryAspect,
    get_coverage_scorer,
)
from .llm_coverage import LLMCoverageScorer

__all__ = [
    # Query enhancement
    "QueryEnhancer",
    "HyDEGenerator",
    "EnhancedQuery",
    "get_query_enhancer",
    # Classification
    "QueryClassifier",
    "QueryClassification",
    "QueryType",
    "get_query_classifier",
    "classify_query",
    # Coverage
    "CoverageScorer",
    "CoverageScore",
    "QueryAspect",
    "get_coverage_scorer",
    "LLMCoverageScorer",
]
