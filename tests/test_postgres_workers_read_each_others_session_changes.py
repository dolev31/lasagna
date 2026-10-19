"""With PostgreSQL, every worker reads a session as another worker last saved it.

Without Redis (``RAG_CACHE_BACKEND=memory``, the default) each process kept a
cache of the sessions it had read that nothing invalidated: with four gunicorn
workers, a worker that had served a session kept that copy after another worker
patched or switched it, for the life of the process. A worker that read the row
while another saved a change must not put the older row back over it.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.core.session_config import _session_state
from tests._postgres_store import FakePostgres, manager_class_on


@pytest.fixture
def recheck_now(monkeypatch):
    """No wait for the recheck interval: the next read of a cached session asks the store."""
    monkeypatch.setattr(_session_state, "_RECHECK_SECONDS", 0.0)


def _sid() -> str:
    return f"pg-{uuid.uuid4().hex[:8]}"


def test_a_worker_reads_another_workers_patch(monkeypatch, recheck_now):
    store = FakePostgres()
    worker = {"now": "b"}
    on_postgres = manager_class_on(monkeypatch, store)

    sid = _sid()

    def request(name, call):
        worker["now"] = name
        return call(on_postgres(session_id=sid))

    request("b", lambda m: m.patch_config({"rerank_top_k": 7}, session_id=sid))
    assert request("a", lambda m: m.get_resolved_config(sid)["rerank_top_k"]) == 7
    worker["now"] = "a"
    long_lived = on_postgres()  # the session provider's, for the whole process
    assert long_lived.get_resolved_config(sid)["rerank_top_k"] == 7

    request("b", lambda m: m.patch_config({"rerank_top_k": 11}, session_id=sid))
    request("b", lambda m: m.set_active_profile("speed", session_id=sid))
    assert request("a", lambda m: m.get_resolved_config(sid)["rerank_top_k"]) == 11
    worker["now"] = "a"
    assert long_lived.get_resolved_config(sid)["rerank_top_k"] == 11
    assert long_lived.get_active_profile(sid) == "speed"


def test_a_read_never_puts_an_older_row_over_a_change(monkeypatch, recheck_now):
    store = FakePostgres()
    on_postgres = manager_class_on(monkeypatch, store)
    sid = _sid()
    on_postgres(session_id=sid).patch_config({"rerank_top_k": 7}, session_id=sid)

    # Worker A reads the row; before it is done, worker B saves a change.
    read = store.get_session

    def slow_read(session_id):
        row = read(session_id)
        store.get_session = read
        on_postgres(session_id=sid).patch_config({"rerank_top_k": 11}, session_id=sid)
        return row

    store.get_session = slow_read
    on_postgres(session_id=sid).get_resolved_config(sid)

    assert on_postgres(session_id=sid).get_resolved_config(sid)["rerank_top_k"] == 11
    assert store.sessions[sid]["session_overrides"]["rerank_top_k"] == 11
