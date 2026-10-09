"""Component Registry for RAG retrieval pipeline.

This module re-exports the unified registry from ``core.registry``.
All registrations are shared: components registered via this module
are visible from ``core.registry.registry`` and vice versa.

Usage:
    # Register a custom reranker
    from lasagna.retrieval.registry import component_registry

    component_registry.register_reranker("my_colbert", MyColBERTReranker)

    # Use via config
    config = KnowledgeConfig(reranker="my_colbert")

    # List available components
    component_registry.list_rerankers()
    # ["cross_encoder", "ensemble", "my_colbert"]

    # Register a custom chunker
    component_registry.register_chunker("my_chunker", MySemanticChunker)
"""

from lasagna.core.registry import (
    ComponentRegistry,
    _Registration,
    component_registry,
    get_component_registry,
    registry,
)

__all__ = [
    "ComponentRegistry",
    "_Registration",
    "component_registry",
    "get_component_registry",
    "registry",
]
