"""PostgreSQL persistence mixin for session configuration.

Drop-in replacement for ``PersistenceMixin`` when ``RAG_DATABASE_BACKEND=postgresql``.
Stores profiles and session state in PostgreSQL, the one copy every worker reads.

Session settings had a Redis copy in front (``RAG_CACHE_BACKEND=redis``, the
compose default), which readers took first and every read kept alive. A Redis
restored from an older snapshot, or one that dropped during a write (its errors
were logged and the write answered as saved), served the older settings to
every worker, the Celery worker and MCP until the session's next write: the old
profile or model, uploads ingested into the old collection. PostgreSQL is read
at most once a second per session per process (``_adopt_persisted_session``),
which costs about a millisecond.
"""

import threading
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from ._data_models import (
    ANONYMOUS,
    MAX_CACHED_SESSIONS,
    SESSION_TTL_SECONDS,
    RAGProfile,
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


class DatabasePersistenceMixin:
    """Mixin for PostgreSQL-backed persistence of profiles and session states.

    Expects the host class to provide the same attributes as ``PersistenceMixin``:
    - self._lock: RLock
    - self._session_states: Dict[str, KnowledgeConfigState]
    - self._global_custom_profiles: List[RAGProfile]
    - self._global_profile_names: set
    - self._default_profiles: List[RAGProfile]
    - self._state_from_dict(): method
    """

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
        except Exception as exc:
            raise not_saved(f"profile {profile.name!r}", exc) from exc

    def _delete_global_profile(self, profile: RAGProfile) -> None:
        """Delete one saved profile's row."""
        try:
            _get_pg_db().delete_profile(profile.user_id or ANONYMOUS, profile.name)
        except Exception as exc:
            raise not_saved(f"the deletion of profile {profile.name!r}", exc) from exc

    def _load_session_states(self) -> None:
        """Load session states from PostgreSQL.

        When ``self._session_id`` is set (request-scoped mode), only loads that
        single session. This avoids pulling all sessions into memory on every
        request.

        Without one (the process's own manager, which the session provider
        keeps) nothing is read now: each session is read when it is first
        asked for, and at most ``MAX_CACHED_SESSIONS`` are kept. Every session
        used in the last day was read at startup and kept, with no limit.
        """
        target_session = getattr(self, "_session_id", None)
        if target_session:
            self._load_single_session(target_session)

    def _load_single_session(self, session_id: str) -> None:
        """Load a single session from PostgreSQL."""
        try:
            row = _get_pg_db().get_session(session_id)
            if row:
                self._session_states[session_id] = self._state_from_dict(
                    self._row_to_state_data(row)
                )
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
            "user_id": row.get("user_id") or ANONYMOUS,
            "last_accessed": (
                row["last_accessed"].timestamp()
                if hasattr(row.get("last_accessed"), "timestamp")
                else time.time()
            ),
        }

    def _reload_if_changed(self) -> None:
        """No-op for PostgreSQL backend: PostgreSQL reads are always fresh."""
        pass

    def _delete_persisted_session(self, session_id: str) -> bool:
        """Delete a single session from PostgreSQL; whether a row was there."""
        try:
            return bool(_get_pg_db().delete_session(session_id))
        except Exception as exc:
            raise not_saved(f"the deletion of session {session_id!r}'s settings", exc) from exc

    def _persist_session_states(self) -> None:
        """Persist session states to PostgreSQL."""
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

        except Exception as exc:
            logger.warning(f"Failed to persist session states to PostgreSQL: {exc}")

    def _read_persisted_session(
        self, session_id: str, *, fresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """The session as stored in PostgreSQL; None when absent or unreadable.

        Every read is ``fresh``: no copy sits in front of PostgreSQL. A
        manager that outlives a request re-reads its sessions through this
        once a second: the one ``PersistentSessionProvider`` holds for the
        whole process does. Returning None here (on the theory that every
        manager is request-scoped) left each process with the first copy of a
        session it read, so a profile applied through one api worker never
        reached the other workers, the Celery worker or the MCP server.
        """
        try:
            row = _get_pg_db().get_session(session_id)
        except Exception as exc:  # noqa: BLE001 - the cached copy is kept
            logger.debug(f"Could not read session {session_id} from PostgreSQL: {exc}")
            return None
        return self._row_to_state_data(row) if isinstance(row, dict) and row else None

    def _sessions_on_profile(self, profile_name: str) -> Optional[List[Dict[str, Any]]]:
        """The stored sessions on ``profile_name`` (id, owner, overrides); None when unknown."""
        lister = getattr(_get_pg_db(), "list_sessions_on_profile", None)
        if lister is None:
            return None  # a stand-in store
        rows = lister(profile_name)
        return rows if isinstance(rows, list) else None

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
        try:
            db = _get_pg_db()
            state_dict = {
                # Whose profile of that name the session is on, as SQLite keeps it.
                "user_id": getattr(state, "user_id", None) or ANONYMOUS,
                "active_profile": state.active_profile,
                "session_overrides": state.session_overrides,
                "rag_enabled": state.rag_enabled,
                "focused_document_ids": state.focused_document_ids,
            }
            db.save_session(session_id, state_dict)
        except Exception as exc:
            raise not_saved(f"the settings of session {session_id!r}", exc) from exc

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
            idle = session_idle_seconds()  # None: session_idle_days = 0 keeps them
            count = db.cleanup_expired_sessions(max_age_seconds=idle) if idle is not None else 0

            # Also clean up in-memory cache, and keep at most
            # MAX_CACHED_SESSIONS of it (the least recently used go), as SQLite does.
            expired = [
                session_id
                for session_id, state in self._session_states.items()
                if now - state.last_accessed > SESSION_TTL_SECONDS
            ]
            excess = len(self._session_states) - len(expired) - MAX_CACHED_SESSIONS
            if excess > 0:
                kept = sorted(
                    (item for item in self._session_states.items() if item[0] not in expired),
                    key=lambda item: item[1].last_accessed,
                )
                expired += [session_id for session_id, _ in kept[:excess]]

            for session_id in expired:
                del self._session_states[session_id]
                self._session_checked_at.pop(session_id, None)

            if expired or count:
                logger.debug(
                    f"Session cleanup: {len(expired)} from memory, {count} from PostgreSQL"
                )
        except Exception as exc:
            logger.warning(f"Session cleanup failed: {exc}")
