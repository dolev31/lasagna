"""The MCP search tool offers the session's own fix, made where the CLI's and doctor's are.

A search that misses the session's documents (they sit in another model's
collection) reports ``documents_elsewhere``. Its reason names the CLI's fix,
made by ``fixes_for_documents_elsewhere``; the MCP search tool added a fixed
sentence of its own (``_MCP_FIXES``) that named no session and no profile. The
stage now carries the same fixes spelled as MCP calls for that session, and
each does what it says.
"""

from __future__ import annotations

import json
import uuid

import pytest

from lasagna.api._search_orchestrator import _note_documents_elsewhere
from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings, get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager
from lasagna.utils import degraded

MINILM, SMALL = "all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"


@pytest.fixture(autouse=True)
def models_load(monkeypatch):
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


def _mgr(sid):
    return get_knowledge_config_manager(session_id=sid)


def _session_on(profile: str, documents_collection: str) -> str:
    sid = f"fix-{uuid.uuid4().hex[:8]}"
    _mgr(sid).set_active_profile(profile, session_id=sid, force=True)
    get_rag_database().save_document(
        {
            "document_id": f"a_{sid}",
            "session_id": sid,
            "filename": "a.md",
            "file_size": 10,
            "file_type": "md",
            "status": "ready",
            "chunks_count": 1,
        },
        "/nonexistent/a.md",
        collection_name=documents_collection,
    )
    return sid


def _mcp_note(sid: str) -> str:
    from lasagna.api._profile_guard import forget_documents_elsewhere
    from lasagna.mcp import server

    forget_documents_elsewhere(sid)
    with degraded.collecting():
        _note_documents_elsewhere(sid, get_rag_config().with_session(sid))
        (note,) = server._stage_notes()
    return note


MINILM_COLLECTION = collection_name_for_settings({"embedding_model": MINILM, "embedding_dim": 384})


def test_the_mcp_note_names_this_sessions_profile_and_both_fixes():
    sid = _session_on("standard", MINILM_COLLECTION)
    note = _mcp_note(sid)
    assert note.startswith("documents_elsewhere: 1 of 1 document")
    assert f"set_active_profile(profile_name='standard', session_id='{sid}', reindex=true)" in note
    assert f"set_active_profile(profile_name='speed', session_id='{sid}')" in note
    # The CLI's, in the reason, from the same place.
    assert f"lasagna configure --profile standard -s {sid} --reindex" in note

    from lasagna.mcp import server

    server.set_active_profile("speed", session_id=sid)
    assert get_rag_config().with_session(sid).collection_name == MINILM_COLLECTION


def test_with_an_embedding_override_the_mcp_switch_back_sets_the_documents_model():
    sid = _session_on("speed", MINILM_COLLECTION)
    _mgr(sid).patch_config({"embedding_model": SMALL, "embedding_dim": 384}, sid, force=True)
    note = _mcp_note(sid)
    assert f"set_active_profile(profile_name='speed', session_id='{sid}', reindex=true)" in note
    patch = {"embedding_model": MINILM, "embedding_dim": 384}
    assert (
        f"patch_rag_config(session_id='{sid}', config_patch={json.dumps(json.dumps(patch))})"
        in note
    )

    from lasagna.mcp import server

    answer = json.loads(server.patch_rag_config(json.dumps(patch), session_id=sid))
    assert "error" not in answer, answer
    assert get_rag_config().with_session(sid).collection_name == MINILM_COLLECTION
