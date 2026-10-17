"""``session_idle_days = 0`` means sessions are never deleted for being idle, like the TTL.

The GC admin route accepted 0, and the session store then deleted the settings
of every session without documents within a minute, sessions in use included,
while ``document_ttl_days = 0`` means "disabled". Now 0 disables both.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

from lasagna.core.database import get_rag_database
from lasagna.core.session_config import KnowledgeConfigManager
from lasagna.core.session_config._data_models import session_idle_seconds


def test_zero_disables_the_idle_cleanup_of_the_session_store(monkeypatch):
    monkeypatch.setenv("RAG_GC_SESSION_IDLE_DAYS", "0")
    assert session_idle_seconds() is None
    sid = f"idle-{uuid.uuid4().hex[:8]}"
    manager = KnowledgeConfigManager()
    manager.set_active_profile("speed", session_id=sid)
    with get_rag_database().transaction() as conn:  # last used two days ago
        conn.execute(
            "UPDATE rag_sessions SET last_accessed = ? WHERE session_id = ?",
            (time.time() - 2 * 86400, sid),
        )
    manager._last_cleanup_time = 0
    manager._cleanup_old_sessions()
    assert get_rag_database().get_session(sid) is not None


def test_zero_skips_the_gc_phase(monkeypatch):
    from lasagna.lifecycle.gc import GarbageCollector
    from lasagna.lifecycle.policy import LifecyclePolicy

    calls = []

    class _Db:
        def cleanup_expired_sessions(self, max_age_seconds=0):
            calls.append(max_age_seconds)
            return 0

    gc = GarbageCollector.__new__(GarbageCollector)
    gc._db, gc._policy = _Db(), LifecyclePolicy(session_idle_days=0)
    result = gc._phase5_cleanup_idle_sessions()
    assert calls == [] and result.details.get("skipped")
    assert result.completed_at <= datetime.now(timezone.utc)
