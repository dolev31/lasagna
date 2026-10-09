"""Search subsystem for RAG retrieval.

This module provides search and ranking capabilities:
- Hybrid search (dense + BM25)
- Cross-encoder reranking with MMR
"""

from .hybrid import (
    HybridSearcher,
    HybridSearchResult,
    MilvusNativeHybridSearcher,
    BM25FallbackIndex,
    get_hybrid_searcher,
)
from .ranker import (
    CombinedReranker,
    EnsembleReranker,
    RerankedResult,
    get_reranker,
    reset_reranker,
)

__all__ = [
    # Hybrid search
    "HybridSearcher",
    "HybridSearchResult",
    "MilvusNativeHybridSearcher",
    "BM25FallbackIndex",
    "get_hybrid_searcher",
    # Reranking
    "CombinedReranker",
    "EnsembleReranker",
    "RerankedResult",
    "get_reranker",
    "reset_reranker",
]
