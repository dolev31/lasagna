"""Session-scoped Knowledge/RAG configuration management.

This package provides session-scoped configuration management for the RAG system,
allowing users to modify RAG settings via the UI without permanently altering
the settings.lasagna.toml file.

Architecture (Dual-Mode):
-------------------------
**SQLite (default, RAG_DATABASE_BACKEND=sqlite)**:
    Singleton ``KnowledgeConfigManager`` with SQLite persistence (rag.db).
    Profiles and sessions stored in ``rag_profiles`` and ``rag_sessions`` tables.
    Optimised for local single-user deployment where simplicity is preferred.

**PostgreSQL + Redis (RAG_DATABASE_BACKEND=postgresql)**:
    Request-scoped ``KnowledgeConfigManager`` instances.  Each request gets a
    lightweight manager that reads from Redis (L1, <1 ms) → PostgreSQL (L2, ~5 ms).
    Supports horizontal scaling across N Gunicorn/uvicorn workers with shared
    state, per-user profile namespacing, and TTL-based session cleanup.

    Workers share state via:
        - **PostgreSQL**: durable storage for profiles (``rag_profiles``),
          sessions (``rag_sessions``), and quotas (``rag_quotas``)
        - **Redis**: write-through cache for sub-millisecond reads, also used
          for distributed rate limiting and quota enforcement
"""

import os
from pathlib import Path
from typing import Dict, List, Optional
from threading import Lock, RLock

from loguru import logger

from ._data_models import (
    SCHEMA_VERSION,
    SESSION_TTL_SECONDS,
    MAX_CACHED_SESSIONS,
    MAX_GLOBAL_PROFILES,
    MAX_CITATIONS_PER_SESSION,
    MAX_CHUNKS_PER_SESSION,
    MAX_CHUNK_CONTENT_LENGTH,
    RAGProfile,
    KnowledgeConfigState,
    UnknownProfileError,
    _deep_merge,
)
from ._profiles import ProfileMixin
from ._session_state import SessionStateMixin
from ._citations_store import CitationStoreMixin
from ._backend_config import BackendConfigMixin
from ._validation_mixin import ValidationMixin

# Dynamic persistence mixin selection based on database backend.
# PostgreSQL path uses database + Redis cache; SQLite path uses rag.db tables.
_db_backend = os.environ.get("RAG_DATABASE_BACKEND", "sqlite")
if _db_backend == "postgresql":
    from ._persistence_db import DatabasePersistenceMixin as _PersistenceBase
else:
    from ._persistence import PersistenceMixin as _PersistenceBase


class KnowledgeConfigManager(
    ProfileMixin,
    SessionStateMixin,
    _PersistenceBase,
    CitationStoreMixin,
    BackendConfigMixin,
    ValidationMixin,
):
    """Manages session-scoped knowledge/RAG configuration.

    This class:
    - Loads default profiles from settings.lasagna.toml
    - Maintains per-session configuration state with TTL-based cleanup
    - Provides APIs for getting, setting, and resetting config
    - Supports import/export of configurations
    - Persists global custom profiles separately from session state

    Thread Safety:
    - Uses RLock for reentrant locking (methods can call each other)
    - Session cleanup runs periodically to prevent memory bloat
    """

    @staticmethod
    def _resolve_user_data_path() -> str:
        """Resolve user data path using the config provider."""
        try:
            from lasagna.providers import get_config_provider

            provider = get_config_provider()
            return str(provider.data_dir)
        except (RuntimeError, Exception):
            pass

        return str(Path.home() / ".rag_knowledge" / "dbs")

    def __init__(
        self,
        user_id: str = "__anonymous__",
        session_id: Optional[str] = None,
    ):
        self._lock = RLock()  # Reentrant lock for nested calls
        self._session_states: Dict[str, KnowledgeConfigState] = {}
        self._default_profiles: List[RAGProfile] = []
        # Global custom profiles - shared across all sessions (persisted separately)
        self._global_custom_profiles: List[RAGProfile] = []
        # Track profile names as a set for O(1) lookup
        self._global_profile_names: set = set()

        # Per-user/session identity (used by DatabasePersistenceMixin)
        self._user_id: str = user_id
        self._session_id: Optional[str] = session_id

        user_data_path = self._resolve_user_data_path()
        self._data_dir = user_data_path

        # Separate files for global profiles vs session state
        self._profiles_path = os.path.join(user_data_path, "knowledge_profiles.json")
        self._sessions_path = os.path.join(user_data_path, "knowledge_sessions.json")

        # Legacy path for migration
        self._legacy_persist_path = os.path.join(user_data_path, "knowledge_config_state.json")

        self._profiles_mtime: Optional[float] = None  # unused (kept for compat)
        self._sessions_mtime: Optional[float] = None  # unused (kept for compat)
        self._last_cleanup_time: float = 0

        self._load_defaults()
        # Migration chain: move straggler JSON files first, then import into SQLite
        self._migrate_session_files_from_parent()
        self._migrate_legacy_data()
        self._load_global_profiles()
        self._load_session_states()

    def _migrate_session_files_from_parent(self):
        """One-time migration: move session files from old db_dir.parent to data_dir."""
        try:
            from lasagna.providers import get_config_provider

            provider = get_config_provider()
            old_dir = str(provider.db_dir.parent)
            new_dir = self._data_dir
            if old_dir == new_dir:
                return
            for fname in ["knowledge_profiles.json", "knowledge_sessions.json"]:
                old_path = os.path.join(old_dir, fname)
                new_path = os.path.join(new_dir, fname)
                if os.path.exists(old_path) and not os.path.exists(new_path):
                    import shutil

                    shutil.move(old_path, new_path)
                    logger.info(f"Migrated {fname} from {old_dir} to {new_dir}")
        except Exception as e:
            logger.debug(f"Session file migration skipped: {e}")


# =============================================================================
# Global Instance
# =============================================================================

# Global singleton instance (file-based backend only)
_knowledge_config_manager: Optional[KnowledgeConfigManager] = None
# CRITICAL FIX: Lock for thread-safe singleton initialization
# Without this, multiple threads could create multiple instances during startup
_knowledge_config_manager_lock = Lock()


def _get_singleton() -> KnowledgeConfigManager:
    """Get the file-based singleton (double-check locking)."""
    global _knowledge_config_manager
    if _knowledge_config_manager is None:
        with _knowledge_config_manager_lock:
            if _knowledge_config_manager is None:
                _knowledge_config_manager = KnowledgeConfigManager()
    return _knowledge_config_manager


def get_knowledge_config_manager(
    user_id: str = "__anonymous__",
    session_id: Optional[str] = None,
) -> KnowledgeConfigManager:
    """Get or create a knowledge config manager.

    **SQLite backend** (default):
        Returns a global singleton.  ``user_id`` and ``session_id`` are ignored
        — all callers share the same in-memory state backed by SQLite tables.

    **PostgreSQL backend** (``RAG_DATABASE_BACKEND=postgresql``):
        Returns a request-scoped instance scoped to the given user/session.
        Each instance reads from Redis (L1) → PostgreSQL (L2) and only loads
        the data it needs, enabling horizontal scaling across N workers.
    """
    if _db_backend == "postgresql":
        # Request-scoped: each request gets a fresh, lightweight manager
        # that reads only the target session from Redis → PostgreSQL.
        return KnowledgeConfigManager(user_id=user_id, session_id=session_id)
    else:
        # Legacy singleton for local/single-user mode
        return _get_singleton()


# Re-export public API
__all__ = [
    "KnowledgeConfigManager",
    "KnowledgeConfigState",
    "RAGProfile",
    "UnknownProfileError",
    "get_knowledge_config_manager",
    # Constants (for consumers that import them directly)
    "SCHEMA_VERSION",
    "SESSION_TTL_SECONDS",
    "MAX_CACHED_SESSIONS",
    "MAX_GLOBAL_PROFILES",
    "MAX_CITATIONS_PER_SESSION",
    "MAX_CHUNKS_PER_SESSION",
    "MAX_CHUNK_CONTENT_LENGTH",
]
