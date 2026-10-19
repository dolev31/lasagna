"""Session settings are read from PostgreSQL, whatever Redis holds or does.

The compose stack sets ``RAG_CACHE_BACKEND=redis``, and session settings had a
Redis copy that readers took first, with a TTL every read extended. A write
dropped Redis's copy, wrote PostgreSQL and wrote Redis again, and both Redis
calls only logged their errors: with Redis down during a patch the patch
answered 200, PostgreSQL held it, and every worker, the Celery worker and MCP
went on reading the older copy until the session's next write. A Redis restored
from an older snapshot did the same. Redis still holds rate limits and the
Celery queue; it holds no session settings.
"""

from __future__ import annotations

import copy
import uuid

import pytest

from lasagna.core.session_config import _session_state
from lasagna.infrastructure import session_store
from tests._postgres_store import FakePostgres, manager_class_on


class _Redis:
    """A shared Redis session store that can go down, as the real one's calls do."""

    def __init__(self):
        self.data, self.down = {}, False

    def _check(self):
        if self.down:
            raise ConnectionError("Error 111 connecting to redis:6379. Connection refused.")

    def load(self, key):
        self._check()
        value = self.data.get(key)
        return copy.deepcopy(value) if value is not None else None

    def save(self, key, value):
        self._check()
        self.data[key] = copy.deepcopy(value)

    def save_if_absent(self, key, value):
        self._check()
        return self.data.setdefault(key, copy.deepcopy(value)) == value

    def delete(self, key):
        self._check()
        return self.data.pop(key, None) is not None

    def list_sessions(self):
        return list(self.data)

    def cleanup_expired(self, max_age_seconds=86400):
        return 0


@pytest.fixture
def stack(monkeypatch):
    """PostgreSQL (a fake), a shared Redis session store, and no wait for the recheck."""
    redis = _Redis()
    monkeypatch.setattr(session_store, "_instance", redis)
    monkeypatch.setattr(_session_state, "_RECHECK_SECONDS", 0.0)
    store = FakePostgres()
    return manager_class_on(monkeypatch, store), store, redis


def test_a_patch_made_while_redis_was_down_reaches_every_reader(stack):
    on_postgres, store, redis = stack
    sid = f"r-{uuid.uuid4().hex[:8]}"
    search = on_postgres()  # the session provider's manager, for the whole process
    on_postgres(session_id=sid).patch_config({"rerank_top_k": 7}, session_id=sid)
    assert search.get_resolved_config(sid)["rerank_top_k"] == 7

    redis.down = True
    on_postgres(session_id=sid).patch_config({"rerank_top_k": 11}, session_id=sid)
    redis.down = False

    assert store.sessions[sid]["session_overrides"] == {"rerank_top_k": 11}
    assert on_postgres(session_id=sid).get_resolved_config(sid)["rerank_top_k"] == 11
    assert search.get_resolved_config(sid)["rerank_top_k"] == 11
    assert on_postgres().get_resolved_config(sid)["rerank_top_k"] == 11


def test_a_redis_restored_from_an_older_snapshot_is_not_read(stack):
    on_postgres, store, redis = stack
    sid = f"r-{uuid.uuid4().hex[:8]}"
    on_postgres(session_id=sid).set_active_profile("speed", session_id=sid)
    older = {
        "active_profile": "default",
        "session_overrides": {"rerank_top_k": 3},
        "rag_enabled": True,
        "focused_document_ids": [],
        "user_id": "__anonymous__",
    }
    for key in [f"rag:config:session:{sid}", *list(redis.data)]:
        redis.data[key] = copy.deepcopy(older)

    for manager in (on_postgres(session_id=sid), on_postgres()):
        assert manager.get_active_profile(sid) == "speed"
        assert "rerank_top_k" not in manager.get_session_state(sid).session_overrides
    assert store.sessions[sid]["active_profile"] == "speed"
