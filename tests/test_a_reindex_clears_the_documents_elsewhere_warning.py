"""A search right after a successful reindex no longer warns about documents in another collection.

A search remembers where a session's documents are for 3 s. After
``reindex_document`` moved the last document into the session's collection, the
next search still said "1 of 1 document ... is in all-MiniLM-L6-v2's collection".
Every reindex entry point (MCP tool, REST routes, SDK) now drops the session's
remembered answer when it succeeds.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace


from lasagna.core.config import get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager
from tests.test_a_search_says_when_documents_are_in_another_models_collection import (
    _moved_to_small,
    _speed_session_with_a_document,
)


def _warm_warning(sid: str) -> str:
    from lasagna.api._profile_guard import documents_elsewhere, forget_documents_elsewhere

    forget_documents_elsewhere(sid)
    config = get_rag_config().with_session(sid)
    message = documents_elsewhere(sid, config.collection_name, config.embedding_model)
    assert message and "1 of 1 document" in message
    return message


def _the_document_moves_to_the_sessions_collection(sid: str) -> None:
    """What a successful reindex leaves in the store."""
    config = get_knowledge_config_manager(session_id=sid).get_resolved_config(sid)
    from lasagna.core.config import collection_name_for_settings

    get_rag_database().save_document(
        {
            "document_id": f"doc_{sid}",
            "session_id": sid,
            "filename": "a.md",
            "file_size": 10,
            "file_type": "md",
            "status": "ready",
            "chunks_count": 1,
        },
        "/nonexistent/a.md",
        collection_name=collection_name_for_settings(config),
    )


def _after_the_reindex_nothing_is_elsewhere(sid: str) -> None:
    from lasagna.api._profile_guard import documents_elsewhere

    config = get_rag_config().with_session(sid)
    assert documents_elsewhere(sid, config.collection_name, config.embedding_model) is None


class _Backend:
    """A backend whose reindex moves the document, as the real one does."""

    def __init__(self, sid: str):
        self.sid = sid
        self.doc = SimpleNamespace(
            document_id=f"doc_{sid}",
            filename="a.md",
            file_type="md",
            file_size=10,
            chunks_count=1,
            status=SimpleNamespace(value="ready"),
        )

    def get_documents(self, session_id):
        return [self.doc]

    def get_document_by_id(self, document_id, session_id):
        return self.doc

    def get_session_config(self, session_id):
        return get_knowledge_config_manager(session_id=session_id).get_resolved_config(session_id)

    def reindex_document(self, document_id, session_id):
        _the_document_moves_to_the_sessions_collection(session_id)
        return self.doc


def test_the_mcp_reindex_clears_the_warning(monkeypatch):
    from lasagna.mcp import server

    sid = _speed_session_with_a_document()
    _moved_to_small(sid)
    _warm_warning(sid)
    monkeypatch.setattr(server, "_get_backend", lambda: _Backend(sid))
    monkeypatch.setattr(server, "_resolve_session", lambda session_id, token=None: (sid, None))
    assert "reindexed successfully" in server.reindex_document(f"doc_{sid}", sid)
    _after_the_reindex_nothing_is_elsewhere(sid)


def test_the_rest_reindex_clears_the_warning():
    from lasagna.api.rest.routes.documents import reindex_document

    sid = _speed_session_with_a_document()
    _moved_to_small(sid)
    _warm_warning(sid)
    asyncio.run(reindex_document(f"doc_{sid}", session_id=sid, backend=_Backend(sid)))
    _after_the_reindex_nothing_is_elsewhere(sid)


def test_the_rest_reindex_of_every_document_clears_the_warning():
    from lasagna.api.rest.routes.reindex import _reindex_sync

    sid = _speed_session_with_a_document()
    _moved_to_small(sid)
    _warm_warning(sid)
    _reindex_sync(_Backend(sid), sid)
    _after_the_reindex_nothing_is_elsewhere(sid)


def test_the_sdk_reindex_clears_the_warning(monkeypatch):
    from lasagna.sdk import KnowledgeClient

    sid = _speed_session_with_a_document()
    _moved_to_small(sid)
    _warm_warning(sid)
    client = KnowledgeClient(session_id=sid)

    async def backend():
        return _Backend(sid)

    monkeypatch.setattr(client, "_get_backend", backend)
    monkeypatch.setattr(client, "_record_reindexed_model", lambda: None)
    asyncio.run(client.documents.reindex(f"doc_{sid}"))
    _after_the_reindex_nothing_is_elsewhere(sid)


def test_a_failed_reindex_keeps_what_the_search_remembered(monkeypatch):
    from lasagna.api._profile_guard import _elsewhere
    from lasagna.mcp import server

    sid = _speed_session_with_a_document()
    _moved_to_small(sid)
    _warm_warning(sid)
    backend = _Backend(sid)

    def fails(document_id, session_id):
        raise RuntimeError("no space left")

    backend.reindex_document = fails
    monkeypatch.setattr(server, "_get_backend", lambda: backend)
    monkeypatch.setattr(server, "_resolve_session", lambda session_id, token=None: (sid, None))
    assert server.reindex_document(f"doc_{sid}", sid).startswith("Failed to reindex")
    assert any(key[0] == sid for key in _elsewhere)
