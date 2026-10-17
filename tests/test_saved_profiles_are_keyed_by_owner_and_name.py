"""Saved profiles are one owner's each, and saving or deleting one leaves other sessions searchable.

The store keeps a profile per (owner, name), but the session settings looked
profiles up by name only. So, with auth on, tenant B saving a profile under a
name the CLI had saved (owner ``__anonymous__``) wrote a second row, and tenant
C, on the shared profile of that name, was handed B's settings (its
``agentic_prompts`` included) once the list was read again; and every tenant
could change or delete the profiles the CLI or a stdio MCP server had saved.
Re-saving or deleting a profile checked the caller's session only: the new
settings reached every other session on that profile, and a delete put them on
``default``, both leaving their documents in a collection they no longer
searched.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.core import model_downloads
from lasagna.core.config import collection_name_for_settings
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import KnowledgeConfigManager
from lasagna.core.session_config._data_models import ANONYMOUS

MINILM, SMALL = "all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"
MINILM_SETTINGS = {"embedding_model": MINILM, "embedding_dim": 384}


@pytest.fixture(autouse=True)
def models_load(monkeypatch):
    monkeypatch.setattr(model_downloads, "refusal", lambda *a, **k: None)


def _name(prefix: str = "team") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _sid() -> str:
    return f"own-{uuid.uuid4().hex[:8]}"


def _documents(sid: str, collection: str, n: int = 2) -> None:
    for i in range(n):
        get_rag_database().save_document(
            {
                "document_id": f"d{i}_{sid}",
                "session_id": sid,
                "filename": f"d{i}.md",
                "file_size": 10,
                "file_type": "md",
                "status": "ready",
                "chunks_count": 1,
            },
            f"/nonexistent/d{i}.md",
            collection_name=collection,
        )


def _rows(name: str) -> dict:
    return {
        r["user_id"]: r["config"]
        for r in get_rag_database().get_saved_profiles()
        if r["profile_name"] == name
    }


def test_a_tenant_saving_a_shared_name_never_reaches_another_tenant():
    name = _name()
    cli = KnowledgeConfigManager()
    cli.save_profile(name, {"rerank_top_k": 3}, session_id=_sid(), set_active=False)

    tenant_b = KnowledgeConfigManager()
    b = _sid()
    tenant_b.save_profile(
        name,
        {"rerank_top_k": 9, "agentic_prompts": {"plan": "B's private prompt"}},
        session_id=b,
        user_id="tenant-b",
    )
    assert _rows(name)[ANONYMOUS] == {"rerank_top_k": 3}  # the shared row is untouched
    assert _rows(name)["tenant-b"]["rerank_top_k"] == 9

    c = _sid()
    tenant_c = KnowledgeConfigManager()
    tenant_c.set_active_profile(name, session_id=c, user_id="tenant-c")
    for manager in (tenant_c, KnowledgeConfigManager()):  # this process, and a fresh one
        manager._refresh_saved_profiles(force=True)
        resolved = manager.get_resolved_config(c)
        assert resolved["rerank_top_k"] == 3
        assert "agentic_prompts" not in manager.get_chosen_config(c)
        shown = manager.get_config(c, user_id="tenant-c")["profiles"]
        assert all("B's private prompt" not in str(p) for p in shown)
    # B's own session keeps B's.
    assert KnowledgeConfigManager().get_resolved_config(b)["rerank_top_k"] == 9


def test_with_auth_on_a_tenant_cannot_change_or_delete_a_shared_profile():
    name = _name()
    KnowledgeConfigManager().save_profile(
        name, {"rerank_top_k": 3}, session_id=_sid(), set_active=False
    )
    tenant = KnowledgeConfigManager()
    with pytest.raises(ValueError, match="not yours"):
        tenant.delete_profile(name, session_id=_sid(), user_id="tenant-b")
    with pytest.raises(ValueError, match="not yours"):
        tenant.rename_profile(name, _name("renamed"), session_id=_sid(), user_id="tenant-b")
    assert _rows(name) == {ANONYMOUS: {"rerank_top_k": 3}}
    # Saving under that name makes the tenant's own, beside it.
    tenant.save_profile(name, {"rerank_top_k": 4}, session_id=_sid(), user_id="tenant-b")
    assert _rows(name) == {ANONYMOUS: {"rerank_top_k": 3}, "tenant-b": {"rerank_top_k": 4}}


def _two_sessions_on(name: str) -> tuple:
    """``name`` (MiniLM), session ``other`` on it with documents, and the caller's session."""
    manager = KnowledgeConfigManager()
    other = _sid()
    manager.save_profile(name, dict(MINILM_SETTINGS), session_id=other)
    _documents(other, collection_name_for_settings(manager.get_resolved_config(other)))
    return manager, other, _sid()


def test_a_resave_that_would_move_another_sessions_documents_is_refused():
    from lasagna.api._profile_guard import DocumentsLeftBehind

    name = _name()
    manager, other, mine = _two_sessions_on(name)
    small = {"embedding_model": SMALL, "embedding_dim": 384}
    with pytest.raises(DocumentsLeftBehind) as refused:
        manager.save_profile(name, small, session_id=mine, set_active=False)
    message = str(refused.value)
    assert other in message and "2 documents" in message and "Nothing was changed" in message
    assert "under a new name" in message and f"-s {other} --reindex" in message
    assert _rows(name)[ANONYMOUS]["embedding_model"] == MINILM
    assert KnowledgeConfigManager().get_resolved_config(other)["embedding_model"] == MINILM

    # The admin can force it; the other session then searches the new model.
    manager.save_profile(name, small, session_id=mine, set_active=False, force=True)
    assert KnowledgeConfigManager().get_resolved_config(other)["embedding_model"] == SMALL


def test_a_delete_that_would_strand_another_sessions_documents_is_refused():
    from lasagna.api._profile_guard import DocumentsLeftBehind

    name = _name()
    manager, other, mine = _two_sessions_on(name)
    with pytest.raises(DocumentsLeftBehind) as refused:
        manager.delete_profile(name, session_id=mine)
    assert other in str(refused.value) and "Nothing was changed" in str(refused.value)
    assert _rows(name) == {ANONYMOUS: MINILM_SETTINGS}  # still saved
    assert KnowledgeConfigManager().get_active_profile(other) == name

    manager.delete_profile(name, session_id=mine, force=True)
    assert _rows(name) == {}
