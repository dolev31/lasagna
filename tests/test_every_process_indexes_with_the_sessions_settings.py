"""Every process reads a session's current settings and indexes with them.

Found on the compose stack, with session settings in Postgres: a profile
applied through one process never reached a process that had read the session
before. ``PersistentSessionProvider`` keeps one config manager for the whole
process and re-reads each session once a second through
``_read_persisted_session``, which the Postgres persistence returned as None
("each request gets a manager that loads its session fresh"). Then the Celery
worker, a separate process, indexed with whatever embedding model it had
loaded: the upload route makes the embedder and the collection current for the
session, but only in the process that took the upload. Its chunks went to a
collection that did not exist (Qdrant 404) or to another model's.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, call

from lasagna.core.session_config import _persistence_db as pg
from lasagna.infrastructure.tasks import document_tasks


class _Redis:
    def __init__(self):
        self.data = {}

    def load(self, key):
        return self.data.get(key)

    def save(self, key, value):
        self.data[key] = value


class _Postgres:
    def __init__(self):
        self.rows = {}

    def get_session(self, session_id):
        return self.rows.get(session_id)


def _mixin(monkeypatch):
    redis, postgres = _Redis(), _Postgres()
    monkeypatch.setattr(pg, "_get_cache", lambda: redis)
    monkeypatch.setattr(pg, "_get_pg_db", lambda: postgres)
    return pg.DatabasePersistenceMixin.__new__(pg.DatabasePersistenceMixin), redis, postgres


def test_a_session_changed_elsewhere_is_read_from_redis(monkeypatch):
    mixin, redis, _ = _mixin(monkeypatch)
    assert mixin._read_persisted_session("s1") is None

    # Another process applies a profile: it writes Postgres, then Redis.
    redis.save(
        f"{mixin._CACHE_PREFIX_SESSION}s1",
        {"active_profile": "speed", "session_overrides": {}, "rag_enabled": True},
    )

    assert mixin._read_persisted_session("s1")["active_profile"] == "speed"


def test_without_redis_the_session_is_read_from_postgres(monkeypatch):
    mixin, _, postgres = _mixin(monkeypatch)
    postgres.rows["s2"] = {
        "active_profile": "speed",
        "session_overrides": {"rerank_top_k": 7},
        "rag_enabled": True,
        "focused_document_ids": [],
        "last_accessed": datetime.now(timezone.utc),
    }

    stored = mixin._read_persisted_session("s2")

    assert stored["active_profile"] == "speed"
    assert stored["session_overrides"] == {"rerank_top_k": 7}


def test_the_task_makes_the_embedder_current_before_indexing(monkeypatch):
    backend = MagicMock()
    config = backend.get_session_config.return_value
    backend._get_document_metadata.return_value.chunks_count = 1
    monkeypatch.setattr("lasagna.api.rag_backend.get_rag_backend", lambda: backend)

    document_tasks._process_document_impl("doc1", "session-1", "/data/uploads/doc1/a.txt")

    pipeline_calls = [
        c
        for c in backend._document_pipeline.mock_calls
        if c[0] in ("use_embedder_for", "process_document")
    ]
    assert pipeline_calls[0] == call.use_embedder_for(config)
    assert pipeline_calls[1][0] == "process_document"


def test_the_pipeline_switches_to_the_session_model():
    from lasagna.api._document_pipeline import DocumentPipeline

    pipeline = DocumentPipeline.__new__(DocumentPipeline)
    pipeline._collection_manager = MagicMock()
    pipeline.embedding_manager = "the model the process started with"
    config = MagicMock()

    pipeline.use_embedder_for(config)

    pipeline._collection_manager.maybe_reload_embedding_manager.assert_called_once_with(config)
    assert pipeline.embedding_manager is pipeline._collection_manager.embedding_manager
