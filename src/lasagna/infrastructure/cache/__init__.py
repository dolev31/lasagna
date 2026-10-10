"""Caching subsystem for Lasagna.

This module provides:
- **SearchCache**: TTL-based LRU cache for search results with session-aware invalidation
- **CacheBackend**: Low-level key-value cache abstraction (in-memory or Redis)
"""

from .search_cache import (
    SearchCache,
    CachedSearchResult,
    get_search_cache,
    reset_search_cache,
)

from .backend import (
    CacheBackend,
    InMemoryCacheBackend,
    RedisCacheBackend,
    get_cache_backend,
    configure_cache_backend,
)

__all__ = [
    "SearchCache",
    "CachedSearchResult",
    "get_search_cache",
    "reset_search_cache",
    "CacheBackend",
    "InMemoryCacheBackend",
    "RedisCacheBackend",
    "get_cache_backend",
    "configure_cache_backend",
]
