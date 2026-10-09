"""Shared session state store for multi-instance deployments.

Provides a SessionStore protocol with in-memory (default) and Redis
implementations.  When multiple API workers run behind a load balancer,
session state (config overrides, focused documents, RAG toggle) must be
shared across processes.

Dual-mode:
    - ``session_store = "memory"`` (default): In-process dict — zero dependencies.
    - ``session_store = "redis"``: Shared via Redis — required for horizontal scaling.

Usage::

    from lasagna.infrastructure.session_store import get_session_store

    store = get_session_store()
    store.save("session_123", {"active_profile": "balanced", "rag_enabled": True})
    state = store.load("session_123")
"""

from __future__ import annotations

import json
import threading
import time
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

from loguru import logger


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class SessionStore(Protocol):
    """Abstract session state store."""

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session state.  Returns None if not found or expired."""
        ...

    def save(self, session_id: str, state: Dict[str, Any]) -> None:
        """Persist session state."""
        ...

    def delete(self, session_id: str) -> bool:
        """Delete session state.  Returns True if it existed."""
        ...

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        ...

    def cleanup_expired(self, max_age_seconds: float = 86400) -> int:
        """Remove sessions older than *max_age_seconds*.  Returns count deleted."""
        ...


# ---------------------------------------------------------------------------
# In-Memory Implementation
# ---------------------------------------------------------------------------


class InMemorySessionStore:
    """Thread-safe in-memory session store.

    Suitable for single-process deployments.  Session state is lost on
    process restart (acceptable for single-user mode where the persistence
    mixin handles file-based backup).
    """

    def __init__(self) -> None:
        self._store: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            entry = self._store.get(session_id)
            if entry is None:
                return None
            # Touch last_accessed
            entry["last_accessed"] = time.time()
            return entry

    def save(self, session_id: str, state: Dict[str, Any]) -> None:
        with self._lock:
            state["last_accessed"] = time.time()
            self._store[session_id] = state

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._store.pop(session_id, None) is not None

    def list_sessions(self) -> List[str]:
        with self._lock:
            return list(self._store.keys())

    def cleanup_expired(self, max_age_seconds: float = 86400) -> int:
        cutoff = time.time() - max_age_seconds
        with self._lock:
            expired = [
                sid for sid, state in self._store.items() if state.get("last_accessed", 0) < cutoff
            ]
            for sid in expired:
                del self._store[sid]
            return len(expired)


# ---------------------------------------------------------------------------
# Redis Implementation
# ---------------------------------------------------------------------------


class RedisSessionStore:
    """Redis-backed session store for multi-instance deployments.

    Each session is stored as a JSON-encoded Redis key with a configurable
    TTL.  All workers share the same Redis instance so session state is
    consistent regardless of which worker handles the request.

    Requires ``redis>=5.0.0``.
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        prefix: str = "rag:session:",
        ttl: int = 86400,
    ) -> None:
        try:
            import redis
        except ImportError:
            raise ImportError(
                "Redis session store requires the redis package. "
                "Install with: pip install redis[hiredis]>=5.0.0"
            )
        self._prefix = prefix
        self._ttl = ttl
        self._client = redis.Redis.from_url(url, decode_responses=True)
        try:
            self._client.ping()
            logger.info(f"Redis session store connected: {url}")
        except Exception as exc:
            logger.warning(f"Redis session store connection failed: {exc}")

    def _key(self, session_id: str) -> str:
        return f"{self._prefix}{session_id}"

    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            raw = self._client.get(self._key(session_id))
            if raw is None:
                return None
            state = json.loads(raw)
            # Touch TTL on access
            self._client.expire(self._key(session_id), self._ttl)
            state["last_accessed"] = time.time()
            return state
        except Exception as exc:
            logger.warning(f"Redis session load failed for {session_id}: {exc}")
            return None

    def save(self, session_id: str, state: Dict[str, Any]) -> None:
        try:
            state["last_accessed"] = time.time()
            self._client.setex(
                self._key(session_id),
                self._ttl,
                json.dumps(state, default=str),
            )
        except Exception as exc:
            logger.warning(f"Redis session save failed for {session_id}: {exc}")

    def delete(self, session_id: str) -> bool:
        try:
            return self._client.delete(self._key(session_id)) > 0
        except Exception as exc:
            logger.warning(f"Redis session delete failed for {session_id}: {exc}")
            return False

    def list_sessions(self) -> List[str]:
        try:
            keys = list(self._client.scan_iter(match=f"{self._prefix}*", count=1000))
            prefix_len = len(self._prefix)
            return [k[prefix_len:] for k in keys]
        except Exception as exc:
            logger.warning(f"Redis session list failed: {exc}")
            return []

    def cleanup_expired(self, max_age_seconds: float = 86400) -> int:
        # Redis TTL handles expiry natively; this is a no-op
        return 0


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_instance: Optional[SessionStore] = None
_instance_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """Get the global session store singleton.

    Uses ``cache_backend`` config to determine the implementation:
    - ``"memory"`` (default): InMemorySessionStore
    - ``"redis"``: RedisSessionStore (shared across workers)
    """
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                try:
                    from lasagna.core.config import rag_config

                    backend_type = getattr(rag_config, "cache_backend", "memory")
                    if backend_type == "redis":
                        redis_url = getattr(rag_config, "redis_url", "redis://localhost:6379")
                        _instance = RedisSessionStore(url=redis_url)
                    else:
                        _instance = InMemorySessionStore()
                except Exception:
                    _instance = InMemorySessionStore()
                logger.debug(f"Session store: {type(_instance).__name__}")
    return _instance


def configure_session_store(store: SessionStore) -> None:
    """Replace the global session store (for testing or explicit setup)."""
    global _instance
    with _instance_lock:
        _instance = store
