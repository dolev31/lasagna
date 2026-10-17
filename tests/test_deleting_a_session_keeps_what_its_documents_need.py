"""Deleting a session's settings deletes the stored row, and never while its documents remain.

``delete_session_state`` deleted the stored row only when this process had the
session cached: a session read by another worker, or evicted from the cache,
kept its row and came back. The SDK's ``delete_session`` cleared the settings
even when some documents could not be deleted, so the ones left were searched
with the default model, which had not indexed them.
"""

from __future__ import annotations

import asyncio
import copy
import uuid

from lasagna.core.database import get_rag_database
from lasagna.core.session_config import KnowledgeConfigManager


def _sid() -> str:
    return f"del-{uuid.uuid4().hex[:8]}"


def test_the_stored_row_goes_even_when_this_process_never_read_the_session():
    sid = _sid()
    KnowledgeConfigManager().set_active_profile("speed", session_id=sid)
    other = KnowledgeConfigManager()
    other._session_states.pop(sid, None)  # another worker's session, or evicted
    assert other.delete_session_state(sid) is True
    assert get_rag_database().get_session(sid) is None
    assert KnowledgeConfigManager().get_active_profile(sid) == "default"


def test_the_sdk_keeps_the_settings_while_documents_remain(monkeypatch):
    import lasagna.api.rag_backend as rag_backend
    from lasagna.core.config import rag_config
    from lasagna.sdk import KnowledgeClient
    from lasagna.sdk.exceptions import DocumentError

    cleared = []

    class _Backend:
        def clear_session_state(self, session_id):
            cleared.append(session_id)
            return {"cache_invalidated": True, "session_cleared": True}

    monkeypatch.setattr(rag_backend, "get_rag_backend", lambda: _Backend())
    monkeypatch.setattr(rag_config, "_env_overrides", copy.deepcopy(rag_config._env_overrides))
    sid = _sid()
    KnowledgeConfigManager().set_active_profile("speed", session_id=sid)
    client = KnowledgeClient(session_id=sid)

    async def one_failed(confirm=False):
        raise DocumentError("Failed to delete 1 document(s): d1: the vector store went away")

    async def no_citations(confirm=False):
        return 0

    monkeypatch.setattr(client.documents, "delete_all", one_failed)
    monkeypatch.setattr(client.citations, "clear", no_citations)
    result = asyncio.run(client.delete_session(confirm=True))
    assert cleared == []
    assert result["session_cleared"] is False
    assert any("settings were kept" in e for e in result["errors"]), result["errors"]
    assert KnowledgeConfigManager().get_active_profile(sid) == "speed"
