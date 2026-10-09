"""Infrastructure layer for RAG Knowledge Base system.

This layer provides core infrastructure components:

- **Embedding**: Vector generation with caching (BGE-M3, sentence-transformers)
- **Storage**: Vector database abstraction (BYOVDB)
- **Cache**: Search result caching + low-level key-value cache backend
- **Session Store**: Shared session state (in-memory or Redis)

LLMManager is intentionally NOT exported here because it imports
``langchain_openai`` and ``langchain_ibm`` at module level — eagerly
importing it would break environments where those packages are not
installed.  Use the explicit import path instead::

    from lasagna.infrastructure.llm_manager import LLMManager

This layer is independent of retrieval logic and can be used
by any component that needs embedding, caching, or session services.
"""

# Embedding subsystem
from .embedding import (
    EmbeddingManager,
    EmbeddingCache,
    get_embedding_manager,
)

# Storage subsystem (BYOVDB) - import from canonical vector_backends
from lasagna.vector_backends import (
    VectorStoreAdapter,
    VectorSearchResult,
    CollectionInfo,
    IndexConfig,
    SearchParams,
    BackendCapabilities,
    FilterExpression,
    FilterOperator,
    get_backend,
    register_backend,
    get_available_backends,
)

# Cache subsystem
from .cache import (
    SearchCache,
    CachedSearchResult,
    get_search_cache,
    CacheBackend,
    InMemoryCacheBackend,
    RedisCacheBackend,
    get_cache_backend,
    configure_cache_backend,
)

# Session store
from .session_store import (
    SessionStore,
    InMemorySessionStore,
    RedisSessionStore,
    get_session_store,
    configure_session_store,
)

__all__ = [
    # Embedding
    "EmbeddingManager",
    "EmbeddingCache",
    "get_embedding_manager",
    # Storage
    "VectorStoreAdapter",
    "VectorSearchResult",
    "CollectionInfo",
    "IndexConfig",
    "SearchParams",
    "BackendCapabilities",
    "FilterExpression",
    "FilterOperator",
    "get_backend",
    "register_backend",
    "get_available_backends",
    # Cache
    "SearchCache",
    "CachedSearchResult",
    "get_search_cache",
    "CacheBackend",
    "InMemoryCacheBackend",
    "RedisCacheBackend",
    "get_cache_backend",
    "configure_cache_backend",
    # Session store
    "SessionStore",
    "InMemorySessionStore",
    "RedisSessionStore",
    "get_session_store",
    "configure_session_store",
]
