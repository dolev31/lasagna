"""Cache backend abstraction with in-memory and Redis implementations.

Provides a simple key-value cache interface used by the search cache and
embedding cache.  The in-memory backend is the default (single-user mode);
the Redis backend is opt-in for multi-instance production deployments.

Dual-mode:
    - ``cache_backend = "memory"`` (default): In-process dict.
    - ``cache_backend = "redis"``: Shared Redis cache across workers.

Usage::

    from lasagna.infrastructure.cache.backend import get_cache_backend

    cache = get_cache_backend()
    cache.set("key", serialized_value, ttl=300)
    value = cache.get("key")
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Optional, Protocol, runtime_checkable

from loguru import logger


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheBackend(Protocol):
    """Abstract cache backend interface."""

    def get(self, key: str) -> Optional[bytes]:
        """Get a value by key.  Returns None on miss or expiry."""
        ...

    def set(self, key: str, value: bytes, ttl: int = 300) -> None:
        """Set a value with TTL (seconds).  0 = no expiry."""
        ...

    def delete(self, key: str) -> bool:
        """Delete a key.  Returns True if existed."""
        ...

    def delete_pattern(self, pattern: str) -> int:
        """Delete keys matching a glob pattern.  Returns count deleted."""
        ...

    def exists(self, key: str) -> bool:
        """Check if a key exists (and is not expired)."""
        ...

    def flush(self) -> None:
        """Clear all keys."""
        ...


# ---------------------------------------------------------------------------
# In-Memory Backend
# ---------------------------------------------------------------------------


class InMemoryCacheBackend:
    """Thread-safe in-memory LRU cache with TTL.

    Suitable for single-process deployments.  Keys are automatically
    evicted on access if expired.  Eviction policy: purge expired entries
    first, then LRU-evict the least recently accessed entry.
    """

    def __init__(self, max_size: int = 10000) -> None:
        self._store: OrderedDict[str, tuple[bytes, float]] = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[bytes]:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expires_at = entry
            if expires_at > 0 and time.monotonic() > expires_at:
                del self._store[key]
                return None
            # LRU: move to end on access
            self._store.move_to_end(key)
            return value

    def set(self, key: str, value: bytes, ttl: int = 300) -> None:
        expires_at = time.monotonic() + ttl if ttl > 0 else 0
        with self._lock:
            if key in self._store:
                # Update existing — move to end (most recently used)
                self._store[key] = (value, expires_at)
                self._store.move_to_end(key)
                return

            if len(self._store) >= self._max_size:
                # First try to purge expired entries
                now = time.monotonic()
                expired = [k for k, (_, exp) in self._store.items() if exp > 0 and now > exp]
                for k in expired:
                    del self._store[k]

                # If still at capacity, LRU-evict the oldest entry
                if len(self._store) >= self._max_size:
                    self._store.popitem(last=False)

            self._store[key] = (value, expires_at)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def delete_pattern(self, pattern: str) -> int:
        import fnmatch

        with self._lock:
            keys = [k for k in self._store if fnmatch.fnmatch(k, pattern)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def exists(self, key: str) -> bool:
        return self.get(key) is not None

    def flush(self) -> None:
        with self._lock:
            self._store.clear()


# ---------------------------------------------------------------------------
# Redis Backend
# ---------------------------------------------------------------------------


class RedisCacheBackend:
    """Redis-backed cache for multi-instance deployments.

    Requires ``redis>=5.0.0``.  Optionally uses ``hiredis`` for
    higher performance: ``pip install redis[hiredis]``.
    """

    def __init__(
        self, url: str = "redis://localhost:6379", pool_size: int = 10, prefix: str = "rag:"
    ) -> None:
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis cache backend requires the redis package. "
                "Install with: pip install redis[hiredis]>=5.0.0"
            )
        self._prefix = prefix
        self._client = redis.Redis.from_url(
            url,
            max_connections=pool_size,
            decode_responses=False,  # We store bytes
        )
        # Verify connectivity
        try:
            self._client.ping()
            logger.info(f"Redis cache connected: {url}")
        except Exception as exc:
            logger.warning(
                f"Redis cache connection failed: {exc}. Operations will retry on demand."
            )

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Optional[bytes]:
        try:
            return self._client.get(self._key(key))
        except Exception as exc:
            logger.warning(f"Redis GET failed: {exc}")
            return None

    def set(self, key: str, value: bytes, ttl: int = 300) -> None:
        try:
            if ttl > 0:
                self._client.setex(self._key(key), ttl, value)
            else:
                self._client.set(self._key(key), value)
        except Exception as exc:
            logger.warning(f"Redis SET failed: {exc}")

    def delete(self, key: str) -> bool:
        try:
            return self._client.delete(self._key(key)) > 0
        except Exception as exc:
            logger.warning(f"Redis DELETE failed: {exc}")
            return False

    def delete_pattern(self, pattern: str) -> int:
        try:
            full_pattern = self._key(pattern)
            keys = list(self._client.scan_iter(match=full_pattern, count=1000))
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as exc:
            logger.warning(f"Redis DELETE pattern failed: {exc}")
            return 0

    def exists(self, key: str) -> bool:
        try:
            return self._client.exists(self._key(key)) > 0
        except Exception as exc:
            logger.warning(f"Redis EXISTS failed: {exc}")
            return False

    def flush(self) -> None:
        try:
            keys = list(self._client.scan_iter(match=f"{self._prefix}*", count=1000))
            if keys:
                self._client.delete(*keys)
        except Exception as exc:
            logger.warning(f"Redis FLUSH failed: {exc}")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_instance: Optional[CacheBackend] = None
_instance_lock = threading.Lock()


def get_cache_backend() -> CacheBackend:
    """Get the global cache backend singleton."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                from lasagna.core.config import rag_config

                backend_type = getattr(rag_config, "cache_backend", "memory")
                if backend_type == "redis":
                    redis_url = getattr(rag_config, "redis_url", "redis://localhost:6379")
                    _instance = RedisCacheBackend(url=redis_url)
                else:
                    _instance = InMemoryCacheBackend()
                logger.debug(f"Cache backend: {backend_type}")
    return _instance


def configure_cache_backend(backend: CacheBackend) -> None:
    """Replace the global cache backend (for testing or explicit setup)."""
    global _instance
    with _instance_lock:
        _instance = backend
