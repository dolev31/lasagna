"""Persistence mixin for session configuration (SQLite backend).

Stores profiles and sessions in the rag.db SQLite database via RAGDatabase,
replacing the legacy JSON file persistence.
"""

import json
import os
import time
from typing import Any, Dict, Optional

from loguru import logger

from ._data_models import (
    SESSION_TTL_SECONDS,
    MAX_CACHED_SESSIONS,
    RAGProfile,
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

    def _load_global_profiles(self) -> None:
        """Load global custom profiles from SQLite."""
        try:
            db = _get_db()
            user_id = getattr(self, "_user_id", "__global__")
            rows = db.get_profiles(user_id)

            self._global_custom_profiles = []
            self._global_profile_names = set()

            for row in rows:
                if row.get("is_builtin"):
                    continue
                profile = RAGProfile(
                    name=row["profile_name"],
                    description=row.get("description", ""),
                    config=row.get("config", {}),
                    user_id=row.get("user_id", "__anonymous__"),
                )
                self._global_custom_profiles.append(profile)
                self._global_profile_names.add(profile.name)

            if self._global_custom_profiles:
                logger.info(
                    f"Loaded {len(self._global_custom_profiles)} custom profile(s) from database"
                )
        except Exception as exc:
            logger.warning(f"Failed to load profiles from database: {exc}")

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

    def _persist_global_profiles(self) -> None:
        """Persist global custom profiles to SQLite.

        Uses snapshot-style persist: delete all non-builtin profiles for this
        user, then re-insert the current in-memory state. This ensures deleted
        profiles are removed from the database (not just from memory).
        """
        try:
            db = _get_db()
            user_id = getattr(self, "_user_id", "__global__")
            # Delete all existing custom profiles, then re-insert current state
            db.delete_all_user_profiles(user_id)
            for profile in self._global_custom_profiles:
                db.save_profile(
                    user_id=profile.user_id if profile.user_id != "__anonymous__" else user_id,
                    profile_name=profile.name,
                    config=profile.config,
                    description=profile.description,
                )
        except Exception as exc:
            logger.warning(f"Failed to persist profiles to database: {exc}")

    def _delete_persisted_session(self, session_id: str) -> None:
        """Delete a single session from SQLite."""
        try:
            db = _get_db()
            db.delete_session(session_id)
        except Exception as exc:
            logger.warning(f"Failed to delete session {session_id} from database: {exc}")

    def _read_persisted_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """The session as stored, or None when it is not stored or cannot be read."""
        try:
            row = _get_db().get_session(session_id)
        except Exception as exc:
            logger.debug(f"Could not read session {session_id} from database: {exc}")
            return None
        if not row:
            return None
        return {
            "active_profile": row.get("active_profile", "default"),
            "session_overrides": row.get("session_overrides", {}),
            "rag_enabled": row.get("rag_enabled", True),
            "focused_document_ids": row.get("focused_document_ids", []),
            "last_accessed": row.get("last_accessed", time.time()),
            "user_id": row.get("user_id", "__anonymous__"),
        }

    def _persist_single_session(self, session_id: str) -> None:
        """Persist one session.

        Writing every cached session on each change put this process's stale
        copies of sessions another process had changed back over them.
        """
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
            logger.warning(f"Failed to persist session {session_id} to database: {exc}")

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
            db_count = db.cleanup_expired_sessions(max_age_seconds=SESSION_TTL_SECONDS)
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
