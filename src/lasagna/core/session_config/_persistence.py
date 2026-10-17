"""Persistence mixin for session configuration (SQLite backend).

Stores profiles and sessions in the rag.db SQLite database via RAGDatabase,
replacing the legacy JSON file persistence.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional

from loguru import logger

from ._data_models import (
    ANONYMOUS,
    SESSION_TTL_SECONDS,
    MAX_CACHED_SESSIONS,
    RAGProfile,
    kept_in_process,
    not_saved,
    session_idle_seconds,
)


def _get_db():
    """Lazy accessor for the RAGDatabase singleton."""
    from lasagna.core.database import get_rag_database

    return get_rag_database()


class PersistenceMixin:
    """Mixin for SQLite-backed persistence of profiles and session states.

    Expects the host class to provide:
    - self._lock: RLock
    - self._data_dir: str
    - self._profiles_path: str   (used only for JSON migration)
    - self._sessions_path: str   (used only for JSON migration)
    - self._legacy_persist_path: str  (used only for JSON migration)
    - self._last_cleanup_time: float
    - self._session_states: Dict[str, KnowledgeConfigState]
    - self._global_custom_profiles: List[RAGProfile]
    - self._global_profile_names: set
    - self._default_profiles: List[RAGProfile]
    - self._state_from_dict(): method
    """

    def _migrate_legacy_data(self) -> None:
        """Migrate data from legacy JSON files into SQLite tables.

        Handles three file formats:
        1. Legacy single-file: knowledge_config_state.json
        2. Split profiles: knowledge_profiles.json
        3. Split sessions: knowledge_sessions.json

        Files are renamed to .migrated after successful import.
        A schema_meta flag prevents re-running.
        """
        db = _get_db()

        # Check if migration already done
        try:
            with db.connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM schema_meta WHERE key = 'json_migration_done'")
                row = cursor.fetchone()
                if row and row[0] == "1":
                    return
        except Exception:
            return  # Table may not exist yet during early init

        migrated_any = False

        # 1. Migrate legacy single-file format
        if os.path.exists(self._legacy_persist_path):
            try:
                with open(self._legacy_persist_path, "r") as f:
                    data = json.load(f)

                # Import profiles
                for p in data.get("global_custom_profiles", []):
                    if isinstance(p, dict) and p.get("name"):
                        db.save_profile(
                            user_id=p.get("user_id", "__global__"),
                            profile_name=p["name"],
                            config=p.get("config", {}),
                            description=p.get("description", ""),
                        )

                # Import sessions
                now = time.time()
                for sid, sdata in data.get("sessions", {}).items():
                    if not isinstance(sdata, dict):
                        continue
                    if now - sdata.get("last_accessed", 0) > SESSION_TTL_SECONDS:
                        continue
                    db.save_session(sid, sdata)

                os.rename(
                    self._legacy_persist_path,
                    self._legacy_persist_path + ".migrated",
                )
                migrated_any = True
                logger.info("Migrated legacy knowledge_config_state.json to SQLite")
            except Exception as exc:
                logger.warning(f"Failed to migrate legacy config: {exc}")

        # 2. Migrate profiles JSON
        if os.path.exists(self._profiles_path):
            try:
                with open(self._profiles_path, "r") as f:
                    data = json.load(f)
                profiles = data.get("profiles", [])
                count = 0
                for p in profiles:
                    if isinstance(p, dict) and p.get("name"):
                        db.save_profile(
                            user_id=p.get("user_id", "__global__"),
                            profile_name=p["name"],
                            config=p.get("config", {}),
                            description=p.get("description", ""),
                        )
                        count += 1
                os.rename(
                    self._profiles_path,
                    self._profiles_path + ".migrated",
                )
                migrated_any = True
                logger.info(f"Migrated {count} profiles from JSON to SQLite")
            except Exception as exc:
                logger.warning(f"Failed to migrate profiles JSON: {exc}")

        # 3. Migrate sessions JSON
        if os.path.exists(self._sessions_path):
            try:
                with open(self._sessions_path, "r") as f:
                    data = json.load(f)
                sessions = data.get("sessions", {})
                now = time.time()
                count = 0
                for sid, sdata in sessions.items():
                    if not isinstance(sdata, dict):
                        continue
                    if now - sdata.get("last_accessed", 0) > SESSION_TTL_SECONDS:
                        continue
                    db.save_session(sid, sdata)
                    count += 1
                os.rename(
                    self._sessions_path,
                    self._sessions_path + ".migrated",
                )
                migrated_any = True
                logger.info(f"Migrated {count} sessions from JSON to SQLite")
            except Exception as exc:
                logger.warning(f"Failed to migrate sessions JSON: {exc}")

        # Mark migration complete
        if migrated_any:
            try:
                with db.transaction() as conn:
                    conn.cursor().execute(
                        "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
                        ("json_migration_done", "1"),
                    )
            except Exception:
                pass

    def _read_saved_profiles(self) -> Optional[List[RAGProfile]]:
        """Every saved profile in SQLite, of every owner; None when it cannot be read.

        Of every owner: only the shared ones were read, so a profile a tenant
        had saved was gone after a restart while its row stayed.
        """
        try:
            db = _get_db()
            reader = getattr(db, "get_saved_profiles", None)
            rows = reader() if reader else db.get_profiles(getattr(self, "_user_id", ANONYMOUS))
        except Exception as exc:
            logger.warning(f"Failed to load profiles from database: {exc}")
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
            if not row.get("is_builtin")
        ]

    def _load_global_profiles(self) -> None:
        """Load global custom profiles from SQLite."""
        profiles = self._read_saved_profiles()
        if profiles is None:
            return
        self._global_custom_profiles = profiles
        self._global_profile_names = {p.name for p in profiles}

    def _save_global_profile(self, profile: RAGProfile) -> None:
        """Store one saved profile, under its owner, leaving every other row as it is.

        This wrote back every profile this process had read at startup, after
        deleting all of them: a profile another process had saved since was
        deleted, and its sessions fell back to default.
        """
        try:
            _get_db().save_profile(
                user_id=profile.user_id or ANONYMOUS,
                profile_name=profile.name,
                config=profile.config,
                description=profile.description,
            )
        except Exception as exc:
            raise not_saved(f"profile {profile.name!r}", exc) from exc

    def _delete_global_profile(self, profile: RAGProfile) -> None:
        """Delete one saved profile's row."""
        try:
            _get_db().delete_profile(profile.user_id or ANONYMOUS, profile.name)
        except Exception as exc:
            raise not_saved(f"the deletion of profile {profile.name!r}", exc) from exc

    def _load_session_states(self) -> None:
        """Load session states from SQLite."""
        try:
            db = _get_db()
            rows = db.get_active_sessions(max_age_seconds=SESSION_TTL_SECONDS)

            loaded_count = 0
            for row in rows:
                session_id = row.get("session_id")
                if not session_id:
                    continue
                state_data = {
                    "active_profile": row.get("active_profile", "default"),
                    "session_overrides": row.get("session_overrides", {}),
                    "rag_enabled": row.get("rag_enabled", True),
                    "focused_document_ids": row.get("focused_document_ids", []),
                    "last_accessed": row.get("last_accessed", time.time()),
                    "user_id": row.get("user_id", "__anonymous__"),
                }
                self._session_states[session_id] = self._state_from_dict(state_data)
                loaded_count += 1

            if loaded_count:
                logger.info(f"Loaded {loaded_count} active session(s) from database")
        except Exception as exc:
            logger.warning(f"Failed to load session states from database: {exc}")

    def _reload_if_changed(self) -> None:
        """No-op for SQLite backend.

        Unlike file-based persistence, SQLite reads are always fresh.
        The singleton pattern ensures all writes go through this instance.
        """
        pass

    def _delete_persisted_session(self, session_id: str) -> bool:
        """Delete a single session from SQLite; whether a row was there."""
        try:
            return bool(_get_db().delete_session(session_id))
        except Exception as exc:
            raise not_saved(f"the deletion of session {session_id!r}'s settings", exc) from exc

    def _read_persisted_session(
        self, session_id: str, *, fresh: bool = False
    ) -> Optional[Dict[str, Any]]:
        """The session as stored, or None when it is not stored or cannot be read.

        SQLite has no cache in front of it, so every read is ``fresh``.
        """
        try:
            row = _get_db().get_session(session_id)
        except Exception as exc:
            logger.debug(f"Could not read session {session_id} from database: {exc}")
            return None
        if not isinstance(row, dict) or not row:
            return None  # none stored (or a stand-in store, which has no rows)
        return {
            "active_profile": row.get("active_profile", "default"),
            "session_overrides": row.get("session_overrides", {}),
            "rag_enabled": row.get("rag_enabled", True),
            "focused_document_ids": row.get("focused_document_ids", []),
            "last_accessed": row.get("last_accessed", time.time()),
            "user_id": row.get("user_id", "__anonymous__"),
        }

    def _sessions_on_profile(self, profile_name: str) -> Optional[List[Dict[str, Any]]]:
        """The stored sessions on ``profile_name`` (id, owner, overrides); None when unknown."""
        lister = getattr(_get_db(), "list_sessions_on_profile", None)
        if lister is None:
            return None  # a stand-in store
        rows = lister(profile_name)
        return rows if isinstance(rows, list) else None

    def _touch_persisted_session(self, session_id: str) -> None:
        """Record in SQLite that the session is in use."""
        try:
            _get_db().touch_session(session_id)
        except Exception as exc:
            logger.debug(f"Could not record the use of session {session_id}: {exc}")

    def _persist_single_session(self, session_id: str) -> None:
        """Persist one session; ``SettingsNotSaved`` when the store does not take it.

        Writing every cached session on each change put this process's stale
        copies of sessions another process had changed back over them. A
        failure was logged, and the change answered as made.
        """
        if kept_in_process(session_id):
            return  # the evaluation harness's: this process's only
        state = self._session_states.get(session_id)
        if state is None:
            return
        try:
            _get_db().save_session(
                session_id,
                {
                    "user_id": getattr(
                        state, "user_id", getattr(self, "_user_id", "__anonymous__")
                    ),
                    "active_profile": state.active_profile,
                    "session_overrides": state.session_overrides,
                    "rag_enabled": state.rag_enabled,
                    "focused_document_ids": state.focused_document_ids,
                    "last_accessed": state.last_accessed,
                },
            )
        except Exception as exc:
            raise not_saved(f"the settings of session {session_id!r}", exc) from exc

    def _persist_session_states(self) -> None:
        """Persist session states to SQLite."""
        try:
            db = _get_db()
            user_id = getattr(self, "_user_id", "__anonymous__")
            for session_id, state in self._session_states.items():
                db.save_session(
                    session_id,
                    {
                        "user_id": getattr(state, "user_id", user_id),
                        "active_profile": state.active_profile,
                        "session_overrides": state.session_overrides,
                        "rag_enabled": state.rag_enabled,
                        "focused_document_ids": state.focused_document_ids,
                        "last_accessed": state.last_accessed,
                    },
                )
        except Exception as exc:
            logger.warning(f"Failed to persist session states to database: {exc}")

    def _cleanup_old_sessions(self) -> None:
        """Remove expired sessions from SQLite and in-memory cache.

        Called periodically (not on every request) to balance performance.
        The store keeps a session's settings while it holds documents, and
        otherwise for the GC policy's ``session_idle_days`` after its last use.
        """
        now = time.time()

        # Only run cleanup every 60 seconds
        if now - self._last_cleanup_time < 60:
            return

        self._last_cleanup_time = now

        # Clean up SQLite
        db_count = 0
        try:
            db = _get_db()
            db_count = db.cleanup_expired_sessions(max_age_seconds=session_idle_seconds())
        except Exception as exc:
            logger.warning(f"SQLite session cleanup failed: {exc}")

        # Clean up in-memory cache
        expired = [
            sid
            for sid, state in self._session_states.items()
            if now - state.last_accessed > SESSION_TTL_SECONDS
        ]

        # Also enforce max cached sessions (LRU eviction)
        if len(self._session_states) > MAX_CACHED_SESSIONS:
            sorted_sessions = sorted(
                self._session_states.items(),
                key=lambda x: x[1].last_accessed,
            )
            excess = len(self._session_states) - MAX_CACHED_SESSIONS
            for sid, _ in sorted_sessions[:excess]:
                if sid not in expired:
                    expired.append(sid)

        if expired:
            for session_id in expired:
                del self._session_states[session_id]
                # Clean up orphaned citations files
                try:
                    citations_file = self._get_citations_file_path(session_id)
                    if os.path.exists(citations_file):
                        os.remove(citations_file)
                except Exception:
                    pass

        if expired or db_count:
            logger.debug(f"Session cleanup: {len(expired)} from memory, {db_count} from database")
