"""A session's settings stay while it is in use, and while it holds documents.

Every process deleted, once a minute, the settings of each session not written
for 24 hours, and a read never counted as a use. So after `lasagna configure
--profile speed -s r1` and an upload, the next day every surface searched r1
with the default model: empty results, or "bge-m3 is not downloaded". It also
contradicted the GC policy, whose ``session_idle_days`` is 30. The store now
keeps a session's settings while the session holds documents, deletes those of
a session without documents only after ``session_idle_days`` unused, and a read
records a use (at most once an hour per process).
"""

from __future__ import annotations

import time
import uuid

import pytest

from lasagna.core.database import get_rag_database
from lasagna.core.session_config import KnowledgeConfigManager, _session_state

DAY = 86400


@pytest.fixture(autouse=True)
def _fresh_use_records(monkeypatch):
    monkeypatch.setattr(_session_state, "_use_recorded_at", {}, raising=False)
    monkeypatch.setattr(_session_state, "_RECHECK_SECONDS", 0.0)
    monkeypatch.delenv("RAG_GC_SESSION_IDLE_DAYS", raising=False)


@pytest.fixture
def sid():
    return f"idle-{uuid.uuid4().hex[:8]}"


def _age(session_id: str, seconds: float) -> None:
    with get_rag_database().transaction() as conn:
        conn.execute(
            "UPDATE rag_sessions SET last_accessed = ? WHERE session_id = ?",
            (time.time() - seconds, session_id),
        )


def _last_accessed(session_id: str) -> float:
    row = get_rag_database().get_session(session_id)
    assert row is not None, f"session {session_id} was deleted"
    return float(row["last_accessed"])


def _add_document(session_id: str) -> None:
    get_rag_database().save_document(
        {
            "document_id": f"doc_{uuid.uuid4().hex[:12]}",
            "session_id": session_id,
            "filename": "a.txt",
            "file_type": ".txt",
            "file_size": 1,
            "status": "ready",
        },
        "/tmp/a.txt",
    )


def _next_process() -> KnowledgeConfigManager:
    """A manager that has not swept or read yet, as a process starting the next day."""
    getattr(_session_state, "_use_recorded_at", {}).clear()
    return KnowledgeConfigManager()


def test_a_session_with_documents_keeps_its_profile_however_long_unused(sid):
    KnowledgeConfigManager().set_active_profile("speed", session_id=sid, force=True)
    _add_document(sid)
    _age(sid, 400 * DAY)

    assert _next_process().get_active_profile(sid) == "speed"
    # The GC's own sweep (phase 5) keeps it too.
    get_rag_database().cleanup_expired_sessions(max_age_seconds=30 * DAY)
    assert get_rag_database().get_session(sid)["active_profile"] == "speed"


def test_settings_unused_for_a_day_are_kept(sid):
    KnowledgeConfigManager().set_active_profile("speed", session_id=sid, force=True)
    _age(sid, 25 * 3600)
    assert _next_process().get_active_profile(sid) == "speed"


def test_settings_of_a_session_without_documents_expire_after_the_policy_idle_days(sid):
    KnowledgeConfigManager().set_active_profile("speed", session_id=sid, force=True)
    _age(sid, 31 * DAY)
    assert _next_process().get_active_profile(sid) == "default"
    assert get_rag_database().get_session(sid) is None


def test_the_idle_days_come_from_the_gc_policy(sid, monkeypatch):
    from lasagna.lifecycle.policy import LifecyclePolicy

    monkeypatch.setenv("RAG_GC_SESSION_IDLE_DAYS", "2")
    KnowledgeConfigManager().set_active_profile("speed", session_id=sid, force=True)
    _age(sid, 3 * DAY)
    assert _next_process().get_active_profile(sid) == "default"
    from lasagna.core.session_config._data_models import session_idle_seconds

    assert session_idle_seconds() == LifecyclePolicy.load().session_idle_days * DAY
    monkeypatch.delenv("RAG_GC_SESSION_IDLE_DAYS")
    assert session_idle_seconds() == LifecyclePolicy.load().session_idle_days * DAY == 30 * DAY


def test_a_read_records_a_use_in_the_store(sid, monkeypatch):
    KnowledgeConfigManager().set_active_profile("speed", session_id=sid, force=True)
    _age(sid, 10 * DAY)
    reader = _next_process()
    assert reader.get_active_profile(sid) == "speed"
    assert time.time() - _last_accessed(sid) < 60

    # At most once an hour per process: the next reads write nothing.
    _age(sid, 10 * DAY)
    writes = []
    monkeypatch.setattr(
        type(get_rag_database()),
        "touch_session",
        lambda self, session_id: writes.append(session_id) or True,
    )
    for _ in range(5):
        reader.get_resolved_config(sid)
    assert writes == []
