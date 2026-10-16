"""Search Result Caching for RAG System.

This module implements TTL-based caching for search results to avoid
redundant Milvus queries and embedding computations. This is a
optimization that significantly reduces latency for repeated queries.

Features:
- TTL-based expiration (default 5 minutes)
- LRU eviction when cache is full
- Thread-safe implementation
- Session-aware cache keys
- Cache invalidation on document changes
- Statistics tracking for monitoring

The config declares enable_search_cache=True and search_cache_ttl=300,
but the cache was never implemented. This module fills that gap.
"""

import hashlib
import json
import time
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger


@dataclass
class CachedSearchResult:
    """Cached search result with metadata."""

    results: List[Dict[str, Any]]  # Serializable search results
    query_embedding: List[float]  # For potential reuse
    created_at: float
    expires_at: float
    hit_count: int = 0
    query_hash: str = ""
    session_id: str = ""
    query_text: str = ""  # Original query for targeted invalidation


class SearchCache:
    """TTL-based LRU cache for search results.

    Thread-safe implementation with automatic expiration and LRU eviction.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: int = 300,  # 5 minutes default
        cleanup_interval: int = 60,
    ):
        """Initialize search cache.

        Args:
            max_size: Maximum cached entries before LRU eviction
            ttl_seconds: Time-to-live for cache entries
            cleanup_interval: Seconds between automatic cleanup runs
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cleanup_interval = cleanup_interval

        self._cache: OrderedDict[str, CachedSearchResult] = OrderedDict()
        self._lock = threading.RLock()
        self._last_cleanup = time.time()

        # Track session -> keys mapping for efficient invalidation
        # This enables O(1) session invalidation instead of iterating all entries
        self._session_keys: Dict[str, set] = {}

        # Statistics
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._expirations = 0

    def _compute_key(
        self,
        query: str,
        session_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        use_hyde: bool = True,
        use_reranking: bool = True,
        hybrid_alpha: float = 0.7,
        mmr_lambda: float = 0.7,
        hyde_n_docs: int = 3,
        enable_hybrid_search: bool = True,
        enable_mmr: bool = True,
        component_fingerprint: str = "",
    ) -> str:
        """Compute deterministic cache key from search parameters.

        IMPORTANT: This key must include ALL parameters that affect search results.
        Missing parameters can cause stale cache hits with incorrect results.

        Args:
            query: Search query
            session_id: Session ID
            document_ids: Document filter
            top_k: Number of results
            use_hyde: Whether HyDE is enabled
            use_reranking: Whether reranking is enabled
            hybrid_alpha: Dense/sparse weight for hybrid search
            mmr_lambda: Diversity parameter for MMR
            hyde_n_docs: Number of hypothetical documents for HyDE
            enable_hybrid_search: Whether hybrid search (BM25) is enabled
            enable_mmr: Whether MMR diversity is enabled
            component_fingerprint: Identity string for active pipeline components
                (reranker, embedder, query enhancer). Ensures different component
                configurations produce different cache keys.

        Returns:
            SHA256 hash of parameters (truncated to 32 chars)
        """
        # Include ALL parameters that affect results to prevent stale cache hits
        key_parts = {
            "query": query,
            "session_id": session_id,
            "document_ids": sorted(document_ids) if document_ids else [],
            "top_k": top_k,
            "use_hyde": use_hyde,
            "use_reranking": use_reranking,
            # Include config parameters that affect search results
            "hybrid_alpha": round(hybrid_alpha, 3),  # Round to avoid float precision issues
            "mmr_lambda": round(mmr_lambda, 3),
            "hyde_n_docs": hyde_n_docs if use_hyde else 0,
            # Routing-aware: FAST tier disables these, producing different results
            "enable_hybrid_search": enable_hybrid_search,
            "enable_mmr": enable_mmr,
            # Component identity: ensures swapped components get fresh cache entries
            "component_fingerprint": component_fingerprint,
        }
        key_str = json.dumps(key_parts, sort_keys=True)
        return hashlib.sha256(key_str.encode()).hexdigest()[:32]

    def get(
        self,
        query: str,
        session_id: str,
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        use_hyde: bool = True,
        use_reranking: bool = True,
        hybrid_alpha: float = 0.7,
        mmr_lambda: float = 0.7,
        hyde_n_docs: int = 3,
        enable_hybrid_search: bool = True,
        enable_mmr: bool = True,
        component_fingerprint: str = "",
    ) -> Optional[Tuple[List[Dict[str, Any]], List[float]]]:
        """Get cached search results.

        Args:
            query: Search query
            session_id: Session ID
            document_ids: Document filter
            top_k: Number of results
            use_hyde: Whether HyDE is enabled
            use_reranking: Whether reranking is enabled
            hybrid_alpha: Dense/sparse weight for hybrid search
            mmr_lambda: Diversity parameter for MMR
            hyde_n_docs: Number of hypothetical documents for HyDE
            enable_hybrid_search: Whether hybrid search is enabled
            enable_mmr: Whether MMR diversity is enabled
            component_fingerprint: Identity string for active pipeline
                components (reranker, embedder, query enhancer).

        Returns:
            Tuple of (results, query_embedding) or None if not cached
        """
        key = self._compute_key(
            query,
            session_id,
            document_ids,
            top_k,
            use_hyde,
            use_reranking,
            hybrid_alpha,
            mmr_lambda,
            hyde_n_docs,
            enable_hybrid_search,
            enable_mmr,
            component_fingerprint,
        )

        with self._lock:
            self._maybe_cleanup()

            if key not in self._cache:
                self._misses += 1
                return None

            entry = self._cache[key]

            # Check expiration
            if time.time() > entry.expires_at:
                del self._cache[key]
                self._misses += 1
                self._expirations += 1
                return None

            # Move to end (LRU) and update hit count
            self._cache.move_to_end(key)
            entry.hit_count += 1
            self._hits += 1

            logger.debug(f"Search cache hit (key={key[:8]}..., hits={entry.hit_count})")
            return entry.results, entry.query_embedding

    def put(
        self,
        query: str,
        session_id: str,
        results: List[Dict[str, Any]],
        query_embedding: List[float],
        document_ids: Optional[List[str]] = None,
        top_k: int = 10,
        use_hyde: bool = True,
        use_reranking: bool = True,
        hybrid_alpha: float = 0.7,
        mmr_lambda: float = 0.7,
        hyde_n_docs: int = 3,
        enable_hybrid_search: bool = True,
        enable_mmr: bool = True,
        component_fingerprint: str = "",
    ) -> None:
        """Cache search results.

        Args:
            query: Search query
            session_id: Session ID
            results: Search results to cache (must be serializable)
            query_embedding: Query embedding for potential reuse
            document_ids: Document filter
            top_k: Number of results
            use_hyde: Whether HyDE is enabled
            use_reranking: Whether reranking is enabled
            hybrid_alpha: Dense/sparse weight for hybrid search
            mmr_lambda: Diversity parameter for MMR
            hyde_n_docs: Number of hypothetical documents for HyDE
            enable_hybrid_search: Whether hybrid search is enabled
            enable_mmr: Whether MMR diversity is enabled
            component_fingerprint: Identity string for active pipeline
                components (reranker, embedder, query enhancer).
        """
        key = self._compute_key(
            query,
            session_id,
            document_ids,
            top_k,
            use_hyde,
            use_reranking,
            hybrid_alpha,
            mmr_lambda,
            hyde_n_docs,
            enable_hybrid_search,
            enable_mmr,
            component_fingerprint,
        )
        now = time.time()

        with self._lock:
            # Evict if at capacity
            while len(self._cache) >= self.max_size:
                evicted_key, evicted_entry = self._cache.popitem(last=False)
                self._evictions += 1
                # Clean up session -> keys mapping on eviction
                if evicted_entry.session_id and evicted_entry.session_id in self._session_keys:
                    self._session_keys[evicted_entry.session_id].discard(evicted_key)
                    if not self._session_keys[evicted_entry.session_id]:
                        del self._session_keys[evicted_entry.session_id]
                logger.debug(f"Search cache eviction (LRU): {evicted_key[:8]}...")

            self._cache[key] = CachedSearchResult(
                results=results,
                query_embedding=query_embedding,
                created_at=now,
                expires_at=now + self.ttl_seconds,
                query_hash=key,
                session_id=session_id,
                query_text=query,
            )

            # Track session -> keys mapping for efficient invalidation
            if session_id not in self._session_keys:
                self._session_keys[session_id] = set()
            self._session_keys[session_id].add(key)

            logger.debug(
                f"Search cache put (key={key[:8]}..., ttl={self.ttl_seconds}s, session={session_id[:8] if session_id else 'none'}...)"
            )

    def invalidate_session(self, session_id: str) -> int:
        """Invalidate all cached entries for a session.

        Called when documents are added/removed to ensure stale results
        are not returned.

        Now uses O(1) session -> keys mapping instead of
        clearing the entire cache, preserving entries from other sessions.

        Args:
            session_id: Session ID to invalidate

        Returns:
            Number of entries invalidated
        """
        with self._lock:
            # Use the session -> keys mapping for efficient invalidation
            if session_id not in self._session_keys:
                logger.debug(
                    f"No cached entries for session {session_id[:8] if session_id else 'none'}..."
                )
                return 0

            keys_to_remove = self._session_keys[session_id].copy()
            count = 0

            for key in keys_to_remove:
                if key in self._cache:
                    del self._cache[key]
                    count += 1

            # Clean up the session mapping
            del self._session_keys[session_id]

            logger.info(
                f"Search cache invalidated for session {session_id[:8] if session_id else 'none'}... ({count} entries)"
            )
            return count

    def invalidate_query(
        self,
        query: str,
        session_id: str,
        document_ids: Optional[List[str]] = None,
    ) -> bool:
        """Invalidate all cached entries for a specific query within a session.

        Removes every cache entry whose original query text matches, regardless
        of other search parameters (top_k, component fingerprint, etc.).

        For most use cases, prefer ``invalidate_session()`` which clears all
        entries for a session when documents change.

        Args:
            query: Search query to invalidate
            session_id: Session ID
            document_ids: Unused, kept for API compatibility

        Returns:
            True if any entries were found and removed
        """
        removed = False
        with self._lock:
            if session_id not in self._session_keys:
                return False

            keys_to_remove = []
            for key in self._session_keys[session_id]:
                if key in self._cache:
                    entry = self._cache[key]
                    if entry.query_text == query:
                        keys_to_remove.append(key)

            for key in keys_to_remove:
                if key in self._cache:
                    del self._cache[key]
                    self._session_keys[session_id].discard(key)
                    removed = True

            # Clean up empty session mapping
            if session_id in self._session_keys and not self._session_keys[session_id]:
                del self._session_keys[session_id]

        return removed

    def _maybe_cleanup(self) -> None:
        """Run periodic cleanup if interval elapsed.

        Now properly maintains session -> keys mapping when
        removing expired entries.
        """
        now = time.time()
        if now - self._last_cleanup < self.cleanup_interval:
            return

        self._last_cleanup = now
        expired = []

        for key, entry in self._cache.items():
            if now > entry.expires_at:
                expired.append((key, entry.session_id))

        for key, session_id in expired:
            del self._cache[key]
            self._expirations += 1
            # Clean up session -> keys mapping on expiration
            if session_id and session_id in self._session_keys:
                self._session_keys[session_id].discard(key)
                if not self._session_keys[session_id]:
                    del self._session_keys[session_id]

        if expired:
            logger.debug(f"Search cache cleanup: {len(expired)} expired entries removed")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        with self._lock:
            total_requests = self._hits + self._misses
            return {
                "size": len(self._cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total_requests if total_requests > 0 else 0.0,
                "evictions": self._evictions,
                "expirations": self._expirations,
                "ttl_seconds": self.ttl_seconds,
            }

    def clear(self) -> None:
        """Clear the cache and reset statistics."""
        with self._lock:
            self._cache.clear()
            self._session_keys.clear()  # Also clear session mapping
            self._hits = 0
            self._misses = 0
            self._evictions = 0
            self._expirations = 0
            logger.info("Search cache cleared")


# =============================================================================
# Module Singleton
# =============================================================================

_search_cache: Optional[SearchCache] = None
_cache_lock = threading.Lock()


def get_search_cache(config=None) -> SearchCache:
    """Get or create the search cache singleton.

    First-config-wins: the cache is initialized with the first ``config``
    passed.  Subsequent calls with a different ``config`` log a warning
    but retain the original cache to avoid dropping warm entries.

    Args:
        config: Optional RAG config with cache settings

    Returns:
        SearchCache instance
    """
    global _search_cache

    # Capture local snapshot to avoid race with concurrent reset_search_cache().
    cache = _search_cache
    if cache is not None:
        # Fast path — already initialised.
        if config is not None:
            desired_max = getattr(config, "search_cache_max_size", 1000)
            desired_ttl = getattr(config, "search_cache_ttl", 300)
            if desired_max != cache.max_size or desired_ttl != cache.ttl_seconds:
                logger.warning(
                    f"get_search_cache() called with different config "
                    f"(max_size={desired_max}, ttl={desired_ttl}) after initialization "
                    f"(max_size={cache.max_size}, ttl={cache.ttl_seconds}); "
                    f"original config retained"
                )
        return cache

    with _cache_lock:
        if _search_cache is None:
            max_size = 1000
            ttl_seconds = 300

            if config:
                max_size = getattr(config, "search_cache_max_size", max_size)
                ttl_seconds = getattr(config, "search_cache_ttl", ttl_seconds)

            _search_cache = SearchCache(
                max_size=max_size,
                ttl_seconds=ttl_seconds,
            )
            logger.info(f"Search cache initialized (max_size={max_size}, ttl={ttl_seconds}s)")
        return _search_cache


def reset_search_cache():
    """Reset the search cache singleton."""
    global _search_cache
    with _cache_lock:
        if _search_cache:
            _search_cache.clear()
        _search_cache = None
