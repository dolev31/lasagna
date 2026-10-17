"""With PostgreSQL, a process reads a session when it is asked for, and keeps at most 100.

The process's long-lived manager (the one the session provider keeps) read
every session used in the last day when it started, about 25 KB each, and kept
them all with no limit; SQLite keeps 100.
"""

from __future__ import annotations

from tests._postgres_store import FakePostgres, manager_class_on


def test_the_process_manager_loads_sessions_when_asked_and_keeps_at_most_the_cap(monkeypatch):
    from lasagna.core.session_config import MAX_CACHED_SESSIONS, _persistence_db

    store = FakePostgres()
    on_postgres = manager_class_on(monkeypatch, store)
    for i in range(MAX_CACHED_SESSIONS + 50):
        store.save_session(f"many-{i}", {"active_profile": "default", "session_overrides": {}})
    manager = on_postgres()
    assert manager._session_states == {}  # none read at startup
    for i in range(MAX_CACHED_SESSIONS + 50):
        manager.get_session_state(f"many-{i}")
    monkeypatch.setattr(_persistence_db, "_cleanup_last_run", 0.0)
    manager._cleanup_old_sessions()  # at most once a minute, from get_session_state
    assert len(manager._session_states) == MAX_CACHED_SESSIONS
    assert "many-0" not in manager._session_states  # the least recently used went
    assert manager.get_session_state("many-0").active_profile == "default"  # read again
