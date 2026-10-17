"""A ``KnowledgeConfigManager`` on the PostgreSQL persistence, over a store in memory.

``RAG_DATABASE_BACKEND=postgresql`` picks the persistence when the package is
first imported; these build the same manager on it for one test, with the rows
PostgreSQL would keep in a dict that several managers (processes) share.
"""

from __future__ import annotations

import copy
import time
from datetime import datetime, timezone


class FakePostgres:
    """What the PostgreSQL persistence reads and writes: sessions and saved profiles."""

    def __init__(self):
        self.sessions, self.profiles, self.fail = {}, {}, False
        self.reads = 0

    def get_session(self, sid):
        self.reads += 1
        row = self.sessions.get(sid)
        return copy.deepcopy(row) if row else None

    def save_session(self, sid, state):
        if self.fail:
            raise ConnectionError('connection to server at "db" failed: Connection refused')
        self.sessions[sid] = {
            **copy.deepcopy(state),
            "session_id": sid,
            "last_accessed": datetime.fromtimestamp(time.time(), tz=timezone.utc),
        }
        return True

    def touch_session(self, sid):
        return sid in self.sessions

    def delete_session(self, sid):
        return self.sessions.pop(sid, None) is not None

    def get_active_sessions(self, max_age_seconds=0):
        return [copy.deepcopy(r) for r in self.sessions.values()]

    def cleanup_expired_sessions(self, max_age_seconds=0):
        return 0

    def get_saved_profiles(self):
        return [
            {"user_id": o, "profile_name": n, "description": "", "config": copy.deepcopy(c)}
            for (o, n), c in self.profiles.items()
        ]

    def save_profile(self, user_id, profile_name, config, description="", is_builtin=False):
        if self.fail:
            raise ConnectionError("connection refused")
        self.profiles[(user_id, profile_name)] = copy.deepcopy(config)
        return True

    def delete_profile(self, user_id, profile_name):
        return self.profiles.pop((user_id, profile_name), None) is not None


def manager_class_on(monkeypatch, store, cache=None):
    """``KnowledgeConfigManager`` with the PostgreSQL persistence over ``store`` (and ``cache``)."""
    from lasagna.core.session_config import KnowledgeConfigManager, _persistence_db
    from lasagna.core.session_config._persistence_db import DatabasePersistenceMixin

    monkeypatch.setattr(_persistence_db, "_get_pg_db", lambda: store)
    monkeypatch.setattr(_persistence_db, "_get_cache", lambda: cache)

    class OnPostgres(DatabasePersistenceMixin, KnowledgeConfigManager):
        """Its persistence methods come first, over the SQLite ones."""

        def _migrate_session_files_from_parent(self):
            pass

    return OnPostgres
