"""PostgreSQL + Redis persistence mixin for session configuration.

Drop-in replacement for ``PersistenceMixin`` when ``RAG_DATABASE_BACKEND=postgresql``.
Stores profiles and session state in PostgreSQL with optional Redis write-through
caching for sub-millisecond reads across workers.

Read path:  Redis (hot, <1ms) → PostgreSQL (cold, ~5ms)
Write path: PostgreSQL (durable) → Redis update (cache invalidation/refresh)
"""

import threading
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from ._data_models import (
    ANONYMOUS,
    SESSION_TTL_SECONDS,
    RAGProfile,
    KnowledgeConfigState,
    kept_in_process,
    not_saved,
    session_idle_seconds,
)


# Module-level cleanup throttle.  Shared across all request-scoped instances
# within a worker so that session cleanup runs at most once per 60 seconds,
# not on every request.
_cleanup_last_run: float = 0.0
_cleanup_lock = threading.Lock()


_pg_db_lock = threading.Lock()


def _get_pg_db():
    """Get the PostgreSQL database singleton.

    The metadata store's own (``get_rag_database``), so a process keeps one
    connection pool: this module opened a second one, of up to
    ``database_pool_max`` connections more per process. Its own instance only
    when the metadata store is not PostgreSQL (``RAG_DATABASE_BACKEND`` set
    without ``database_backend``). Uses double-check locking to prevent double
    initialization of the PG connection pool when called concurrently.
    """
    from lasagna.core.database import get_rag_database
    from lasagna.core.database_pg import PostgresRAGDatabase

    shared = get_rag_database()
    if isinstance(shared, PostgresRAGDatabase):
        return shared
    if not hasattr(_get_pg_db, "_instance"):
        with _pg_db_lock:
            if not hasattr(_get_pg_db, "_instance"):
                from lasagna.core.config import rag_config
                from lasagna.core.database_pg import PostgresRAGDatabase

                _get_pg_db._instance = PostgresRAGDatabase(
                    dsn=rag_config.database_dsn,
                    pool_min=getattr(rag_config, "database_pool_min", 2),
                    pool_max=getattr(rag_config, "database_pool_max", 10),
                )
    return _get_pg_db._instance


def _get_cache():
    """The session cache every worker shares (Redis), or None.

    Not a store of this process alone (``RAG_CACHE_BACKEND=memory``, the
    default): nothing tells it that another worker changed a session, so a
    worker that had served a session kept that copy for the life of the
    process. PostgreSQL is read instead, at most once a second per session
    (``_adopt_persisted_session``).
    """
    try:
        from lasagna.infrastructure.session_store import InMemorySessionStore, get_session_store

        store = get_session_store()
    except Exception:
        return None
    return None if isinstance(store, InMemorySessionStore) else store


class DatabasePersistenceMixin:
    """Mixin for PostgreSQL-backed persistence of profiles and session states.

    Uses the existing ``RedisSessionStore`` (from ``session_store.py``) as an
    L1 write-through cache when ``RAG_CACHE_BACKEND=redis``.

    Expects the host class to provide the same attributes as ``PersistenceMixin``:
    - self._lock: RLock
    - self._session_states: Dict[str, KnowledgeConfigState]
    - self._global_custom_profiles: List[RAGProfile]
    - self._global_profile_names: set
    - self._default_profiles: List[RAGProfile]
    - self._state_from_dict(): method
    """

    # Cache key prefixes for Redis
    _CACHE_PREFIX_SESSION = "rag:config:session:"
    _CACHE_PREFIX_PROFILES = "rag:config:profiles:"

    def _migrate_legacy_data(self) -> None:
        """No-op for PostgreSQL backend (no legacy JSON files to migrate)."""
        pass

    def _read_saved_profiles(self) -> Optional[List[RAGProfile]]:
        """Every saved profile in PostgreSQL, of every owner; None when it cannot be read."""
        try:
            db = _get_pg_db()
            reader = getattr(db, "get_saved_profiles", None)
            rows = reader() if reader else db.get_profiles(getattr(self, "_user_id", ANONYMOUS))
        except Exception as exc:
            logger.warning(f"Failed to load profiles from PostgreSQL: {exc}")
            return None
        if not isinstance(rows, list):
            return None  # a stand-in store: keep what this process has
        return [
            RAGProfile(
                name=row["profile_name"],
                description=row.get("description", ""),
                config=row.get("config", {}),
                user_id=row.get("user_id") or ANONYMOUS,
            )
            for row in rows
            if not row.get("is_builtin")  # Built-in profiles come from _load_defaults()
        ]

    def _load_global_profiles(self) -> None:
        """Load global custom profiles from PostgreSQL."""
        profiles = self._read_saved_profiles()
        if profiles is None:
            return
        self._global_custom_profiles = profiles
        self._global_profile_names = {p.name for p in profiles}

    def _save_global_profile(self, profile: RAGProfile) -> None:
        """Store one saved profile, under its owner, leaving every other row as it is.

        This deleted every profile of this manager's user and wrote back the
        ones it had read: a profile another worker had saved since was gone.
        """
        try:
            _get_pg_db().save_profile(
                user_id=profile.user_id or ANONYMOUS,
                profile_name=profile.name,
                config=profile.config,
                description=profile.description,
                is_builtin=False,
            )
            self._cache_profiles(profile.user_id or ANONYMOUS)
        except Exception as exc:
            raise not_saved(f"profile {profile.name!r}", exc) from exc

    def _delete_global_profile(self, profile: RAGProfile) -> None:
        """Delete one saved profile's row."""
        try:
            _get_pg_db().delete_profile(profile.user_id or ANONYMOUS, profile.name)
            self._cache_profiles(profile.user_id or ANONYMOUS)
        except Exception as exc:
            raise not_saved(f"the deletion of profile {profile.name!r}", exc) from exc

    def _load_session_states(self) -> None:
        """Load session states from PostgreSQL.

        When ``self._session_id`` is set (request-scoped mode), only loads that
        single session — first from Redis cache, then PostgreSQL.  This avoids
        pulling all sessions into memory on every request.

        When ``self._session_id`` is None (startup / singleton mode), loads all
        active sessions within the TTL window.
        """
        target_session = getattr(self, "_session_id", None)

        if target_session:
            # Request-scoped: load only the target session
            self._load_single_session(target_session)
            return

        # Startup / singleton mode: load all active sessions
        try:
            db = _get_pg_db()
            rows = db.get_active_sessions(max_age_seconds=SESSION_TTL_SECONDS)

            loaded_count = 0
            for row in rows:
                session_id = row.get("session_id")
                if not session_id:
                    continue
                state_data = self._row_to_state_data(row)
                self._session_states[session_id] = self._state_from_dict(state_data)
                loaded_count += 1

            if loaded_count:
                logger.info(f"Loaded {loaded_count} active session(s) from PostgreSQL")
        except Exception as exc:
            logger.warning(f"Failed to load session states from PostgreSQL: {exc}")

    def _load_single_session(self, session_id: str) -> None:
        """Load a single session from Redis cache → PostgreSQL."""
        # L1: Try Redis cache first
        cached = self._load_session_from_cache(session_id)
        if cached:
            self._session_states[session_id] = self._state_from_dict(cached)
            return

        # L2: Fall back to PostgreSQL
        try:
            db = _get_pg_db()
            row = db.get_session(session_id)
            if row:
                state_data = self._row_to_state_data(row)
                state = self._state_from_dict(state_data)
                self._session_states[session_id] = state
                # Only where Redis holds nothing: a worker that saved a change
                # since this row was read has written Redis after PostgreSQL,
                # and this older row put over it stayed there, the TTL sliding
                # on every read.
                self._cache_session(session_id, state, only_if_absent=True)
        except Exception as exc:
            logger.debug(f"Failed to load session {session_id} from PostgreSQL: {exc}")

    @staticmethod
    def _row_to_state_data(row: dict) -> dict:
        """Convert a PostgreSQL row dict to a state data dict."""
        return {
            "active_profile": row.get("active_profile", "default"),
            "session_overrides": row.get("session_overrides", {}),
            "rag_enabled": row.get("rag_enabled", True),
            "focused_document_ids": row.get("focused_document_ids", []),
            "last_accessed": (
                row["last_accessed"].timestamp()
                if hasattr(row.get("last_accessed"), "timestamp")
                else time.time()
            ),
        }

    def _reload_if_changed(self) -> None:
        """No-op for PostgreSQL backend.

        Unlike file-based persistence, PostgreSQL reads are always fresh.
        Redis caching handles the read-hot path.
        """
        pass

    def _delete_persisted_session(self, session_id: str) -> None:
        """Delete a single session from PostgreSQL + Redis cache."""
        try:
            db = _get_pg_db()
            db.delete_session(session_id)
            # Invalidate Redis cache
            cache = _get_cache()
            if cache:
                try:
                    cache.delete(f"{self._CACHE_PREFIX_SESSION}{session_id}")
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"Failed to delete session {session_id} from PostgreSQL: {exc}")

    def _persist_session_states(self) -> None:
        """Persist session states to PostgreSQL + Redis cache."""
        try:
            db = _get_pg_db()

            for session_id, state in self._session_states.items():
                state_dict = {
                    "user_id": getattr(self, "_user_id", "__anonymous__"),
                    "active_profile": state.active_profile,
                    "session_overrides": state.session_overrides,
                    "rag_enabled": state.rag_enabled,
                    "focused_document_ids": state.focused_document_ids,
                }
                db.save_session(session_id, state_dict)

                # Update Redis cache for this session
                self._cache_session(session_id, state)

        except Exception as exc:
            logger.warning(f"Failed to persist session states to PostgreSQL: {exc}")

    def _read_persisted_session(
        self, session_id: str, *, fresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """The session as stored, Redis first (unless ``fresh``), then PostgreSQL; None when absent.

        A manager that outlives a request re-reads its sessions through this
        once a second: the one ``PersistentSessionProvider`` holds for the
        whole process does. Returning None here (on the theory that every
        manager is request-scoped) left each process with the first copy of a
        session it read, so a profile applied through one api worker never
        reached the other workers, the Celery worker or the MCP server.
        """
        cached = None if fresh else self._load_session_from_cache(session_id)
        if isinstance(cached, dict) and cached:
            return cached
        try:
            row = _get_pg_db().get_session(session_id)
        except Exception as exc:  # noqa: BLE001 - the cached copy is kept
            logger.debug(f"Could not read session {session_id} from PostgreSQL: {exc}")
            return None
        return self._row_to_state_data(row) if isinstance(row, dict) and row else None

    def _touch_persisted_session(self, session_id: str) -> None:
        """Record in PostgreSQL that the session is in use."""
        try:
            _get_pg_db().touch_session(session_id)
        except Exception as exc:
            logger.debug(f"Could not record the use of session {session_id}: {exc}")

    def _persist_single_session(self, session_id: str) -> None:
        """Persist one session; ``SettingsNotSaved`` when PostgreSQL does not take it.

        A failure was logged, and the change answered as made.
        """
        if kept_in_process(session_id):
            return  # the evaluation harness's: this process's only
        state = self._session_states.get(session_id)
        if state is None:
            return
        # Redis's copy goes first: if writing the new one fails after the
        # PostgreSQL write, readers read PostgreSQL rather than the old copy.
        self._forget_cached_session(session_id)
        try:
            db = _get_pg_db()
            state_dict = {
                "user_id": getattr(self, "_user_id", "__anonymous__"),
                "active_profile": state.active_profile,
                "session_overrides": state.session_overrides,
                "rag_enabled": state.rag_enabled,
                "focused_document_ids": state.focused_document_ids,
            }
            db.save_session(session_id, state_dict)
        except Exception as exc:
            raise not_saved(f"the settings of session {session_id!r}", exc) from exc
        self._cache_session(session_id, state)

    def _cleanup_old_sessions(self) -> None:
        """Remove expired sessions from PostgreSQL.

        Uses a module-level timestamp so that request-scoped instances
        (PostgreSQL mode) share the throttle across all requests within
        a worker, preventing a DELETE query on every request. The store keeps
        a session's settings while it holds documents, and otherwise for the
        GC policy's ``session_idle_days`` after its last use.
        """
        global _cleanup_last_run
        now = time.time()

        # Fast path: skip if another instance ran cleanup recently
        if now - _cleanup_last_run < 60:
            return

        # Double-check after acquiring lock to prevent concurrent cleanup
        with _cleanup_lock:
            if now - _cleanup_last_run < 60:
                return
            _cleanup_last_run = now

        try:
            db = _get_pg_db()
            count = db.cleanup_expired_sessions(max_age_seconds=session_idle_seconds())

            # Also clean up in-memory cache
            expired = []
            for session_id, state in self._session_states.items():
                if now - state.last_accessed > SESSION_TTL_SECONDS:
                    expired.append(session_id)

            for session_id in expired:
                del self._session_states[session_id]
                # Invalidate Redis cache
                cache = _get_cache()
                if cache:
                    try:
                        cache.delete(f"{self._CACHE_PREFIX_SESSION}{session_id}")
                    except Exception:
                        pass

            if expired or count:
                logger.debug(
                    f"Session cleanup: {len(expired)} from memory, {count} from PostgreSQL"
                )
        except Exception as exc:
            logger.warning(f"Session cleanup failed: {exc}")

    # ------------------------------------------------------------------
    # Redis cache helpers
    # ------------------------------------------------------------------

    def _cache_session(
        self, session_id: str, state: KnowledgeConfigState, *, only_if_absent: bool = False
    ) -> None:
        """Write session state to Redis; ``only_if_absent`` (a refill) never replaces a copy."""
        cache = _get_cache()
        if cache is None:
            return
        try:
            cache_data = {
                "active_profile": state.active_profile,
                "session_overrides": state.session_overrides,
                "rag_enabled": state.rag_enabled,
                "focused_document_ids": state.focused_document_ids,
                "last_accessed": state.last_accessed,
            }
            key = f"{self._CACHE_PREFIX_SESSION}{session_id}"
            if not only_if_absent:
                cache.save(key, cache_data)
            elif hasattr(cache, "save_if_absent"):
                cache.save_if_absent(key, cache_data)
        except Exception as exc:
            logger.debug(f"Failed to cache session {session_id}: {exc}")

    def _forget_cached_session(self, session_id: str) -> None:
        """Drop Redis's copy of the session, if there is a Redis."""
        cache = _get_cache()
        if cache is None:
            return
        try:
            cache.delete(f"{self._CACHE_PREFIX_SESSION}{session_id}")
        except Exception as exc:
            logger.debug(f"Failed to drop the cached session {session_id}: {exc}")

    def _cache_profiles(self, user_id: str) -> None:
        """Write profile list to Redis cache."""
        cache = _get_cache()
        if cache is None:
            return
        try:
            profiles_data = {
                "profiles": [p.to_dict() for p in self._global_custom_profiles],
                "cached_at": time.time(),
            }
            cache.save(f"{self._CACHE_PREFIX_PROFILES}{user_id}", profiles_data)
        except Exception as exc:
            logger.debug(f"Failed to cache profiles for {user_id}: {exc}")

    def _load_session_from_cache(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Try to load session state from Redis cache."""
        cache = _get_cache()
        if cache is None:
            return None
        try:
            return cache.load(f"{self._CACHE_PREFIX_SESSION}{session_id}")
        except Exception:
            return None
