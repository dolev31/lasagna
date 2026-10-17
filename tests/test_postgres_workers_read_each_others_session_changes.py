"""With PostgreSQL, every worker reads a session as another worker last saved it.

Without Redis (``RAG_CACHE_BACKEND=memory``, the default) each process kept a
cache of the sessions it had read that nothing invalidated: with four gunicorn
workers, a worker that had served a session kept that copy after another worker
patched or switched it, for the life of the process. With Redis, a worker that
missed the cache read the row and wrote it back to Redis outside any lock, so a
change saved meanwhile was put back to the older row, which the sliding TTL kept
alive.
"""

from __future__ import annotations

import copy
import uuid

import pytest

from lasagna.core.session_config import _session_state
from lasagna.infrastructure.session_store import InMemorySessionStore
from tests._postgres_store import FakePostgres, manager_class_on


class _Redis:
    """The Redis session store's behavior: one copy every worker shares."""

    def __init__(self):
        self.data = {}

    def load(self, key):
        value = self.data.get(key)
        return copy.deepcopy(value) if value is not None else None

    def save(self, key, value):
        self.data[key] = copy.deepcopy(value)

    def save_if_absent(self, key, value):
        if key in self.data:
            return False
        self.data[key] = copy.deepcopy(value)
        return True

    def delete(self, key):
        return self.data.pop(key, None) is not None


@pytest.fixture
def recheck_now(monkeypatch):
    """No wait for the recheck interval: the next read of a cached session asks the store."""
    monkeypatch.setattr(_session_state, "_RECHECK_SECONDS", 0.0)


def _sid() -> str:
    return f"pg-{uuid.uuid4().hex[:8]}"


def test_without_redis_a_worker_reads_another_workers_patch(monkeypatch, recheck_now):
    store = FakePostgres()
    caches = {"a": InMemorySessionStore(), "b": InMemorySessionStore()}
    worker = {"now": "b"}
    from lasagna.core.session_config import _persistence_db

    on_postgres = manager_class_on(monkeypatch, store)
    # Each worker process has an in-memory store of its own.
    monkeypatch.setattr(_persistence_db, "_get_cache", lambda: caches[worker["now"]])

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


def test_with_redis_a_refill_never_puts_an_older_row_over_a_change(monkeypatch, recheck_now):
    store = FakePostgres()
    redis = _Redis()
    on_postgres = manager_class_on(monkeypatch, store, cache=redis)
    sid = _sid()
    on_postgres(session_id=sid).patch_config({"rerank_top_k": 7}, session_id=sid)
    redis.data.clear()  # expired from Redis; the row is in PostgreSQL

    # Worker A misses Redis and reads the row; before it writes Redis back,
    # worker B saves a change.
    read = store.get_session

    def slow_read(session_id):
        row = read(session_id)
        store.get_session = read
        on_postgres(session_id=sid).patch_config({"rerank_top_k": 11}, session_id=sid)
        return row

    store.get_session = slow_read
    on_postgres(session_id=sid).get_resolved_config(sid)

    assert on_postgres(session_id=sid).get_resolved_config(sid)["rerank_top_k"] == 11
    cached = redis.load(f"rag:config:session:{sid}")
    assert cached is None or cached["session_overrides"]["rerank_top_k"] == 11
