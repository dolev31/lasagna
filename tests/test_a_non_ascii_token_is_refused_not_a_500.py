"""A token with a non-ASCII character is refused like any wrong token.

`secrets.compare_digest(str, str)` raises TypeError on non-ASCII input, so an
``X-Admin-Token`` holding Latin-1 bytes reached the client as a 500 on the GC
policy route and on session minting, instead of the 401/403 a wrong token gets.
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ADMIN = "correct-admin-token-0123456789abcdef0123"
_FOREIGN = "tokén-ü"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RAG_ADMIN_TOKEN", _ADMIN)
    monkeypatch.setenv("RAG_JWT_SECRET", "k" * 64)
    from lasagna.api.rest.routes import gc_admin, sessions

    app = FastAPI()
    app.include_router(gc_admin.router, prefix="/api/v1")
    app.include_router(sessions.router, prefix="/api/v1")
    return TestClient(app)


def test_the_gc_policy_route_answers_401(client):
    response = client.get(
        "/api/v1/admin/gc/policy", headers={"X-Admin-Token": _FOREIGN.encode("latin-1")}
    )
    assert response.status_code == 401


def test_minting_a_named_session_answers_403(client):
    response = client.post(
        "/api/v1/sessions",
        json={"session_id": "victim"},
        headers={"X-Admin-Token": _FOREIGN.encode("latin-1")},
    )
    assert response.status_code == 403


def test_the_mcp_admin_check_refuses(monkeypatch):
    from lasagna.mcp import transport_guard

    monkeypatch.setenv("RAG_ADMIN_TOKEN", _ADMIN)
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setattr(transport_guard, "_transport", "sse")
    with pytest.raises(transport_guard.MCPAuthError):
        transport_guard.guard_admin(_FOREIGN, tool="rebuild_collection")


def test_the_right_token_still_matches():
    from lasagna.api.rest.auth_mode import secret_matches

    assert secret_matches(_ADMIN, _ADMIN)
    assert not secret_matches(_FOREIGN, _ADMIN)
