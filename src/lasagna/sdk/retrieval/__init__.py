"""Retrieval Pipeline Components.

Provides embedding generation, search, reranking, and query enhancement.
Wraps the existing retrieval module.

Example:
    ```python
    from lasagna.sdk.retrieval import (
        EmbeddingModel,
        HybridSearcher,
        Reranker,
        QueryEnhancer,
    )

    # Generate embeddings
    model = EmbeddingModel()
    embeddings = model.encode(["text1", "text2"])

    # Search
    searcher = HybridSearcher()
    results = searcher.search(query_embedding, collection)

    # Rerank
    reranker = Reranker()
    reranked = reranker.rerank(query, results)

    # Enhance query
    enhancer = QueryEnhancer()
    enhanced = enhancer.enhance(query)
    ```
"""

# Re-export from existing retrieval module
from ...infrastructure.embedding.manager import (
    EmbeddingManager as EmbeddingModel,
    get_embedding_manager,
)
from ...retrieval.search.hybrid import (
    HybridSearcher,
    get_hybrid_searcher,
)
from ...retrieval.search.ranker import (
    CombinedReranker as Reranker,
    get_reranker,
    RerankedResult,
)
from ...retrieval.enhancement.enhancer import (
    QueryEnhancer,
    get_query_enhancer,
)
from ...retrieval.enhancement.classifier import (
    QueryClassifier,
    get_query_classifier,
)
from ...retrieval.enhancement.coverage import (
    CoverageScorer,
    get_coverage_scorer,
)

__all__ = [
    # Embeddings
    "EmbeddingModel",
    "get_embedding_manager",
    # Search
    "HybridSearcher",
    "get_hybrid_searcher",
    # Reranking
    "Reranker",
    "get_reranker",
    "RerankedResult",
    # Enhancement
    "QueryEnhancer",
    "get_query_enhancer",
    "QueryClassifier",
    "get_query_classifier",
    "CoverageScorer",
    "get_coverage_scorer",
]
