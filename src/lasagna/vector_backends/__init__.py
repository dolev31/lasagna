"""Vector Database Backend Abstraction Layer.

This module provides a clean abstraction over vector database backends,
enabling Lasagna's RAG system to work with multiple vector databases
(Milvus, Pinecone, Qdrant, Weaviate, etc.) without changing core logic.

Architecture:
    - VectorStoreAdapter: Abstract base class defining the interface
    - FilterExpression: Backend-agnostic filter representation
    - BackendRegistry: Factory for creating backend instances
    - MilvusBackend: Default implementation wrapping current Milvus code

Usage:
    from lasagna.vector_backends import get_backend

    # Get the configured backend (default: Milvus)
    backend = get_backend()

    # Insert documents
    backend.insert(collection_name, documents)

    # Search
    results = backend.search(
        collection_name=collection_name,
        query_embedding=embedding,
        filter_expr=FilterExpression.eq("session_id", "user123"),
        limit=10
    )

Design Principles:
    1. Minimal Interface: Only expose methods needed by RAG
    2. Backend-Agnostic Filters: No Milvus-specific syntax leaks
    3. Capability Detection: Runtime detection of backend features
    4. Graceful Degradation: Fallback when features unavailable
    5. Zero Breaking Changes: Existing code works without modification
"""

from .base import (
    VectorStoreAdapter,
    VectorSearchResult,
    CollectionInfo,
    IndexConfig,
    SearchParams,
    BackendCapabilities,
    Fields,
    LASAGNA_UUID_NAMESPACE,
)
from .filters import (
    FilterExpression,
    FilterOperator,
    UnsupportedFilterError,
)
from .registry import (
    BackendRegistry,
    get_backend,
    register_backend,
    get_available_backends,
)
from .milvus_backend import MilvusBackend

# Optional imports for backends with external dependencies
try:
    from .pinecone_backend import PineconeBackend
except ImportError:
    PineconeBackend = None  # type: ignore

try:
    from .qdrant_backend import QdrantBackend
except ImportError:
    QdrantBackend = None  # type: ignore

try:
    from .weaviate_backend import WeaviateBackend
except ImportError:
    WeaviateBackend = None  # type: ignore

try:
    from .pgvector_backend import PgVectorBackend
except ImportError:
    PgVectorBackend = None  # type: ignore

try:
    from .opensearch_backend import OpenSearchBackend
except ImportError:
    OpenSearchBackend = None  # type: ignore

__all__ = [
    # Base classes
    "VectorStoreAdapter",
    "VectorSearchResult",
    "CollectionInfo",
    "IndexConfig",
    "SearchParams",
    "BackendCapabilities",
    "Fields",
    "LASAGNA_UUID_NAMESPACE",
    # Filters
    "FilterExpression",
    "FilterOperator",
    "UnsupportedFilterError",
    # Registry
    "BackendRegistry",
    "get_backend",
    "register_backend",
    "get_available_backends",
    # Implementations
    "MilvusBackend",
    "PineconeBackend",
    "QdrantBackend",
    "WeaviateBackend",
    "PgVectorBackend",
    "OpenSearchBackend",
]
