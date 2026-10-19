"""A deleted session stays deleted in every process, and a delete that fails says so.

``delete_session_state`` removed this process's copy and the stored row, with no
session lock and no check for a switch, and a process holding its own copy
(an api worker beside the SDK that deleted the session) kept it and wrote it
back, row included, with its next change. ``clear_session_state`` swallowed a
failed deletion. A deletion now runs under the session's lock, is refused during
a switch, and leaves a stored session with no row, which every process reads as
"deleted" and drops its copy for.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.core.session_config import _session_state
from lasagna.core.session_config._data_models import SettingsNotSaved
from tests._postgres_store import FakePostgres, manager_class_on


@pytest.fixture
def stack(monkeypatch):
    monkeypatch.setattr(_session_state, "_RECHECK_SECONDS", 0.0)
    store = FakePostgres()
    return manager_class_on(monkeypatch, store), store


def test_a_copy_in_another_process_is_dropped_not_written_back(stack):
    on_postgres, store = stack
    sid = f"d-{uuid.uuid4().hex[:8]}"
    server = on_postgres()  # `lasagna serve`: one manager for the process
    server.set_active_profile("speed", session_id=sid)
    assert server.get_active_profile(sid) == "speed"

    on_postgres().delete_session_state(sid)  # the SDK, in another process
    assert sid not in store.sessions

    assert server.get_active_profile(sid) == "default"  # the copy was dropped
    server.patch_config({"rerank_top_k": 7}, session_id=sid)
    assert store.sessions[sid]["active_profile"] == "default"  # not "speed" again


def test_a_deletion_the_store_does_not_take_is_raised_and_keeps_the_session(stack):
    on_postgres, store = stack
    sid = f"d-{uuid.uuid4().hex[:8]}"
    manager = on_postgres()
    manager.set_active_profile("speed", session_id=sid)

    def refuse(_sid):
        raise ConnectionError("connection refused")

    store.delete_session = refuse
    with pytest.raises(SettingsNotSaved):
        manager.delete_session_state(sid)
    assert store.sessions[sid]["active_profile"] == "speed"


def test_a_deletion_during_a_switch_is_refused(stack, monkeypatch):
    from lasagna.api import _profile_guard

    on_postgres, store = stack
    sid = f"d-{uuid.uuid4().hex[:8]}"
    manager = on_postgres()
    manager.set_active_profile("speed", session_id=sid)
    monkeypatch.setattr(_profile_guard, "unfinished_switch", lambda s: {"id": "x"})
    monkeypatch.setattr(_profile_guard, "switch_in_progress_message", lambda r, c: f"{c} refused")
    with pytest.raises(_profile_guard.ProfileSwitchInProgress):
        manager.delete_session_state(sid)
    assert sid in store.sessions


def test_the_backend_lets_a_failed_deletion_through(stack, monkeypatch):
    from lasagna.api.rag_backend import RAGBackend

    on_postgres, store = stack
    sid = f"d-{uuid.uuid4().hex[:8]}"
    manager = on_postgres()
    manager.set_active_profile("speed", session_id=sid)
    store.delete_session = lambda _sid: (_ for _ in ()).throw(ConnectionError("down"))
    monkeypatch.setattr("lasagna.core.session_config.get_knowledge_config_manager", lambda: manager)
    backend = RAGBackend.__new__(RAGBackend)
    backend.hybrid_searcher = None
    with pytest.raises(SettingsNotSaved):
        backend.clear_session_state(sid)
