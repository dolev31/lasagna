"""A write that changes the embedding size is checked first, and a refusal changes nothing.

Every session shares one collection embedded at one size. The guard against
switching a session to another size ran on three surfaces, after the switch,
then switched back:

* The configurator's save and import, and an MCP patch of ``embedding_dim``,
  never reached it, and reached the same state with a 200.
* Switching back dropped the user's overrides, or kept the refused request's.
* A store it could not read (Milvus Lite locked by a running server) counted as
  "nothing to protect", so the switch went through.
"""

from __future__ import annotations

import copy

import pytest

import lasagna.api._profile_guard as guard
from lasagna.api._profile_guard import EmbeddingSizeConflict, EmbeddingSizeUnverified
from tests._isolated_config import isolated_manager, rest_client, stored_at


@pytest.fixture
def manager(tmp_path, monkeypatch):
    return isolated_manager(tmp_path, monkeypatch)


def _unreadable(monkeypatch):
    def stored(collection, vector_backend=None):
        raise ConnectionError("another process holds the lock on the store")

    monkeypatch.setattr(guard, "stored_embedding_dim", stored)


def _everything(mgr, sid):
    """All of a session's state that a refused write must leave alone."""
    from lasagna.providers import get_session_provider

    state = mgr.get_session_state(sid)
    return copy.deepcopy(
        {
            "active": state.active_profile,
            "overrides": state.session_overrides,
            "profiles": [p.to_dict() for p in state.profiles],
            "saved": [p.to_dict() for p in mgr._global_custom_profiles],
            "resolved": mgr.get_resolved_config(sid),
            "search": get_session_provider().get_resolved_config(sid),
        }
    )


def test_a_refused_profile_switch_changes_nothing(manager, monkeypatch):
    sid = "switch-refused"
    manager.set_active_profile("speed", session_id=sid)
    manager.patch_config({"enable_hybrid_search": True, "hybrid_alpha": 0.7}, session_id=sid)
    stored_at(monkeypatch, 384)
    before = _everything(manager, sid)

    with pytest.raises(EmbeddingSizeConflict, match="384") as refused:
        manager.set_active_profile("balanced", session_id=sid)
    assert "'speed' is still active" in str(refused.value)
    assert _everything(manager, sid) == before


def test_a_refused_switch_with_config_keeps_the_old_overrides(manager, monkeypatch):
    sid = "switch-config-refused"
    manager.set_active_profile("speed", session_id=sid)
    manager.patch_config({"rerank_top_k": 17}, session_id=sid)
    stored_at(monkeypatch, 384)
    before = _everything(manager, sid)

    with pytest.raises(EmbeddingSizeConflict):
        manager.set_active_profile(
            "balanced", config={"hybrid_alpha": 0.1, "rerank_top_k": 3}, session_id=sid
        )
    assert _everything(manager, sid) == before


def test_force_applies_the_switch(manager, monkeypatch):
    sid = "switch-forced"
    manager.set_active_profile("speed", session_id=sid)
    stored_at(monkeypatch, 384)
    manager.set_active_profile("balanced", session_id=sid, force=True)
    assert manager.get_active_profile(sid) == "balanced"


_WIDE = {"embedding_model": "BAAI/bge-base-en-v1.5", "embedding_dim": 768}

_WRITES = {
    "save as active": lambda m, sid: m.save_profile("wide", dict(_WIDE), session_id=sid),
    "save over the active profile": lambda m, sid: m.save_profile(
        "mine", {"embedding_dim": 768}, session_id=sid, set_active=False
    ),
    "import overrides": lambda m, sid: m.import_config(
        {"schema_version": 2, "profiles": [], "session_overrides": dict(_WIDE)}, session_id=sid
    ),
    "import an active profile": lambda m, sid: m.import_config(
        {
            "schema_version": 2,
            "profiles": [{"name": "wide", "config": {"embedding_dim": 768}}],
            "active_profile": "wide",
            "session_overrides": {},
        },
        session_id=sid,
    ),
    "patch": lambda m, sid: m.patch_config(dict(_WIDE), session_id=sid),
    "reset": lambda m, sid: m.reset_config("active", session_id=sid),
    "delete the active profile": lambda m, sid: m.delete_profile("mine", session_id=sid),
}


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_every_write_that_changes_the_embedding_size_is_checked(manager, monkeypatch, write):
    sid = f"write-{write.replace(' ', '-')}"
    # Nothing stored yet: put the session on 384, by an override (what the CLI
    # and SDK clients leave behind) or on a saved profile.
    if write in ("save over the active profile", "delete the active profile"):
        manager.save_profile("mine", {"embedding_dim": 384}, session_id=sid)
    else:
        manager.patch_config({"embedding_model": "all-MiniLM-L6-v2", "embedding_dim": 384}, sid)
    stored_at(monkeypatch, 384)
    before = _everything(manager, sid)

    with pytest.raises(EmbeddingSizeConflict):
        _WRITES[write](manager, sid)
    after = _everything(manager, sid)
    for key in ("active", "overrides", "resolved", "search"):
        assert after[key] == before[key], key


def test_an_unreadable_store_refuses_the_switch(manager, monkeypatch):
    """For a session with documents; one without has nothing in the store to protect
    (tests/test_configure_works_while_serve_runs.py)."""
    from lasagna.core.config import collection_name_for_settings
    from lasagna.core.database import get_rag_database

    sid = "store-locked"
    manager.set_active_profile("speed", session_id=sid)
    resolved, _ = manager.preview_active_profile("standard", sid)
    get_rag_database().save_document(
        {
            "document_id": "doc_store_locked",
            "session_id": sid,
            "filename": "a.md",
            "file_size": 10,
            "file_type": "md",
            "status": "ready",
            "chunks_count": 1,
        },
        "/nonexistent/a.md",
        # Already in standard's collection, so only the size check is left.
        collection_name=collection_name_for_settings(resolved),
    )
    _unreadable(monkeypatch)
    before = _everything(manager, sid)

    with pytest.raises(EmbeddingSizeUnverified, match="another process holds the lock"):
        manager.set_active_profile("standard", session_id=sid)
    assert _everything(manager, sid) == before


def test_an_unreadable_store_lets_an_empty_session_switch(manager, monkeypatch):
    sid = "store-locked-empty"
    manager.set_active_profile("speed", session_id=sid)
    _unreadable(monkeypatch)
    manager.set_active_profile("standard", session_id=sid)
    assert manager.get_active_profile(sid) == "standard"


def test_a_write_that_keeps_the_embedding_size_never_reads_the_store(manager, monkeypatch):
    sid = "same-size"
    manager.set_active_profile("speed", session_id=sid)
    _unreadable(monkeypatch)
    manager.patch_config({"rerank_top_k": 9}, session_id=sid)
    manager.set_active_profile("speed", config={"hybrid_alpha": 0.4}, session_id=sid)
    assert manager.get_resolved_config(sid)["hybrid_alpha"] == 0.4


def test_switching_to_the_stored_size_is_allowed(manager, monkeypatch):
    sid = "back-to-stored"
    manager.set_active_profile("speed", session_id=sid, force=True)
    stored_at(monkeypatch, 1024)
    manager.set_active_profile("balanced", session_id=sid)
    assert manager.get_active_profile(sid) == "balanced"


@pytest.mark.parametrize("route", ["save", "import"])
def test_the_configurator_save_and_import_are_a_409_at_another_embedding_size(
    manager, monkeypatch, route
):
    """The two routes that reached a 768-dim session over 384-dim documents with a 200."""
    sid = f"rest-guard-{route}"
    manager.set_active_profile("speed", session_id=sid)
    stored_at(monkeypatch, 384)
    client = rest_client()
    if route == "save":
        response = client.post(
            "/api/config/knowledge/profile",
            json={"profile_name": "wide", "config": _WIDE, "set_active": True},
            headers={"X-Session-ID": sid},
        )
    else:
        response = client.post(
            "/api/config/knowledge/import",
            json={"schema_version": 2, "profiles": [], "session_overrides": _WIDE},
            headers={"X-Session-ID": sid},
        )
    assert response.status_code == 409, response.text
    assert "384" in response.json()["detail"]
    assert "force" not in response.json()["detail"]  # REST has no force to offer
    assert manager.get_resolved_config(sid)["embedding_dim"] == 384


def test_an_unknown_profile_is_a_typed_error(manager):
    from lasagna.core.session_config import UnknownProfileError

    with pytest.raises(UnknownProfileError, match="Available: .*speed"):
        manager.set_active_profile("no_such_profile", session_id="typo")


def test_the_sdk_raises_profile_not_found_for_an_unknown_profile(manager):
    """It matched the text "not found"; the manager says "Unknown profile"."""
    import asyncio
    from types import SimpleNamespace

    from lasagna.sdk.exceptions import ProfileNotFoundError
    from lasagna.sdk.managers.config_manager import ConfigManager

    config_manager = ConfigManager.__new__(ConfigManager)
    config_manager._client = SimpleNamespace(session_id="sdk-typo")
    with pytest.raises(ProfileNotFoundError) as refused:
        asyncio.run(config_manager.set_profile("no_such_profile"))
    assert refused.value.profile_name == "no_such_profile"
