"""With auth on, a tenant's saved profiles are its own: counted, listed and used.

Reproduced with separate tokens: tenant B saved 48 profiles and tenant A could
then save none ("Maximum number of profiles (50) reached"), and every tenant's
``GET /api/config/knowledge`` and ``/export`` listed every other tenant's
profile names and full configs. Built-in profiles stay visible to everyone.
"""

from __future__ import annotations

import pytest

from lasagna.api.rest import session_token
from tests._tokens import minted_token
from tests._isolated_config import isolated_manager, rest_client


@pytest.fixture
def client(tmp_path, monkeypatch):
    isolated_manager(tmp_path, monkeypatch)
    return rest_client()


@pytest.fixture
def tenants(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "k" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    a = minted_token("session-a", "tenant-a")
    b = minted_token("session-b", "tenant-b")
    return {"X-Session-Token": a}, {"X-Session-Token": b}


def _save(client, name, headers, config=None):
    return client.post(
        "/api/config/knowledge/profile",
        json={"profile_name": name, "config": config or {}, "set_active": False},
        headers=headers,
    )


def _names(response):
    return {p["name"] for p in response.json()["profiles"]}


def test_a_tenant_does_not_see_or_use_another_tenants_profiles(client, tenants):
    as_a, as_b = tenants
    assert _save(client, "b_private", as_b, {"rerank_top_k": 7}).status_code == 200
    assert "b_private" in _names(client.get("/api/config/knowledge", headers=as_b))
    shown = _names(client.get("/api/config/knowledge", headers=as_a))
    assert "b_private" not in shown
    assert {"default", "speed", "standard"} <= shown
    assert "b_private" not in _names(client.get("/api/config/knowledge/export", headers=as_a))
    refused = client.put(
        "/api/config/knowledge/active", json={"active_profile": "b_private"}, headers=as_a
    )
    assert refused.status_code == 422
    assert "b_private" not in refused.json()["detail"].split("Available:")[-1]


def test_one_tenant_filling_its_profiles_does_not_stop_another_saving(client, tenants):
    from lasagna.core.session_config import MAX_GLOBAL_PROFILES

    as_a, as_b = tenants
    for i in range(MAX_GLOBAL_PROFILES):
        assert _save(client, f"b_{i}", as_b).status_code == 200
    assert _save(client, "b_one_more", as_b).status_code == 422
    assert _save(client, "a_first", as_a).status_code == 200


def test_an_oversized_profile_config_is_refused(client, tenants):
    as_a, _ = tenants
    response = _save(client, "huge", as_a, {"reranker_model": "x" * 20_000})
    assert response.status_code == 422
    assert "limit" in response.json()["detail"]


def test_a_profile_saved_from_another_session_can_be_activated(tmp_path, monkeypatch):
    """`list_profiles` showed it; `set_active_profile` said "Unknown profile"."""
    manager = isolated_manager(tmp_path, monkeypatch)
    manager.get_session_state("old-session")
    manager.save_profile("team", {"rerank_top_k": 4}, session_id="other", set_active=False)
    manager.set_active_profile("team", session_id="old-session")
    assert manager.get_resolved_config("old-session")["rerank_top_k"] == 4


def test_without_auth_a_saved_profile_can_be_saved_again_by_the_next_request(client):
    """Each unauthenticated request got a new anon id, so it owned nothing it saved."""
    for _ in range(2):
        assert _save(client, "mine", {"X-Session-ID": "rest-anon"}).status_code == 200


def test_over_mcp_a_tenant_does_not_see_another_tenants_profiles(client, tenants, monkeypatch):
    import json

    from lasagna.mcp import server, transport_guard

    monkeypatch.setattr(transport_guard, "_transport", "sse")
    (a,), (b,) = (t.values() for t in tenants)
    saved = server.save_rag_profile("b_mcp", "{}", set_active=False, session_token=b)
    assert "saved successfully" in saved
    names = lambda token: {p["name"] for p in json.loads(server.list_profiles(session_token=token))}  # noqa: E731
    assert "b_mcp" in names(b)
    assert "b_mcp" not in names(a)
    assert "speed" in names(a)
