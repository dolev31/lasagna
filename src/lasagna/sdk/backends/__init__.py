"""Vector Database Backend Abstraction (BYOVDB).

Supports multiple vector databases through a unified interface.
The SDK wraps the existing vector_backends module.

Example:
    ```python
    from lasagna.sdk.backends import (
        get_backend,
        MilvusBackend,
        get_available_backends,
    )

    # Get default backend
    backend = get_backend("milvus")

    # List available backends
    backends = get_available_backends()
    print(backends)  # ['milvus', 'pinecone', 'qdrant', 'weaviate', 'pgvector']

    # Use specific backend
    backend = MilvusBackend(db_path="./my_vectors.db")
    ```
"""

from typing import Dict, Any, Optional, List, Type

# Re-export from existing vector_backends
from ...vector_backends.base import (
    VectorStoreAdapter as VectorBackendBase,
    BackendCapabilities,
    VectorSearchResult,
    CollectionSchema,
    FieldSchema,
    IndexConfig,
    IndexType,
    MetricType,
)
from ...vector_backends.filters import FilterExpression
from ...vector_backends.registry import (
    get_backend as _get_backend,
    get_available_backends,
    register_backend,
)
from ...vector_backends.milvus_backend import MilvusBackend

# Optional backends (may not be installed)
try:
    from ...vector_backends.pinecone_backend import PineconeBackend
except ImportError:
    PineconeBackend = None  # type: ignore

try:
    from ...vector_backends.qdrant_backend import QdrantBackend
except ImportError:
    QdrantBackend = None  # type: ignore

try:
    from ...vector_backends.weaviate_backend import WeaviateBackend
except ImportError:
    WeaviateBackend = None  # type: ignore

try:
    from ...vector_backends.pgvector_backend import PgVectorBackend
except ImportError:
    PgVectorBackend = None  # type: ignore


def get_backend(
    backend_name: str = "milvus",
    **kwargs,
) -> VectorBackendBase:
    """Get a vector database backend instance.

    Factory function to create backend instances by name.

    Args:
        backend_name: Name of backend ('milvus', 'pinecone', 'qdrant', 'weaviate', 'pgvector')
        **kwargs: Backend-specific configuration

    Returns:
        VectorBackendBase instance

    Raises:
        ValueError: If backend not found or not installed

    Example:
        ```python
        # Default Milvus backend
        backend = get_backend("milvus")

        # Milvus with custom path
        backend = get_backend("milvus", db_path="./my_vectors.db")

        # Pinecone (requires API key)
        backend = get_backend(
            "pinecone",
            api_key="your-key",
            environment="us-east-1",
            index_name="my-index",
        )

        # Qdrant
        backend = get_backend(
            "qdrant",
            url="http://localhost:6333",
        )

        # PgVector (PostgreSQL)
        backend = get_backend(
            "pgvector",
            dsn="postgresql://user:pass@localhost:5432/db",
        )
        ```
    """
    return _get_backend(backend_name, **kwargs)


def list_backends() -> List[str]:
    """List all available vector backends.

    Returns:
        List of backend names that can be used with get_backend()

    Example:
        ```python
        backends = list_backends()
        # ['milvus', 'pinecone', 'qdrant', 'weaviate', 'pgvector']
        ```
    """
    return get_available_backends()


__all__ = [
    # Factory
    "get_backend",
    "list_backends",
    "get_available_backends",
    "register_backend",
    # Base types
    "VectorBackendBase",
    "BackendCapabilities",
    "VectorSearchResult",
    "CollectionSchema",
    "FieldSchema",
    "IndexConfig",
    "IndexType",
    "MetricType",
    "FilterExpression",
    # Backends
    "MilvusBackend",
    "PineconeBackend",
    "QdrantBackend",
    "WeaviateBackend",
    "PgVectorBackend",
]
