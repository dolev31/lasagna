"""Taking a key out of ``RAG_API_KEYS`` ends the sessions it minted.

A session token names the key that minted it (``prn``, a digest of the key),
but decoding never looked at it: an operator who removed a leaked key from
``RAG_API_KEYS`` and restarted still saw its tokens accepted, on REST and on
MCP, until they expired seven days later.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lasagna.api.rest import session_token

KEPT = "kept-key-for-tests-0123456789abcdef0123"
REMOVED = "removed-key-for-tests-0123456789abcdef"


@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "a-jwt-secret-for-tests-0123456789abcdef")
    monkeypatch.setenv("RAG_API_KEYS", f"{KEPT},{REMOVED}")
    monkeypatch.setattr(session_token, "_fallback_secret", None)


@pytest.fixture
def client(auth_on):
    from lasagna.api.rest.app import create_app

    return TestClient(create_app(), base_url="http://localhost")


def _mint(client, key):
    response = client.post("/api/v1/sessions", headers={"X-API-Key": key}, json={})
    assert response.status_code == 201, response.text
    return response.json()["token"]


def _pipelines(client, token):
    return client.get("/api/v1/pipelines", headers={"X-Session-Token": token})


def test_rest_refuses_a_token_whose_key_was_removed(client, monkeypatch):
    removed, kept = _mint(client, REMOVED), _mint(client, KEPT)
    assert _pipelines(client, removed).status_code == 200

    monkeypatch.setenv("RAG_API_KEYS", KEPT)
    refused = _pipelines(client, removed)
    assert refused.status_code == 401
    assert "RAG_API_KEYS" in refused.json()["detail"]
    assert _pipelines(client, kept).status_code == 200


def test_mcp_refuses_a_token_whose_key_was_removed(client, monkeypatch):
    from lasagna.mcp import server as mcp_server
    from lasagna.mcp.transport_guard import MCPAuthError

    token = _mint(client, REMOVED)
    sid, _user = mcp_server._resolve_session(session_token=token, tool="search")

    monkeypatch.setenv("RAG_API_KEYS", KEPT)
    with pytest.raises(MCPAuthError):
        mcp_server._resolve_session(session_token=token, tool="search")
    assert sid.startswith("sess_")


def test_tokens_the_admin_minted_are_not_tied_to_a_key(auth_on, monkeypatch):
    token, _ = session_token.create_session_token("s", "u", principal="admin")
    monkeypatch.setenv("RAG_API_KEYS", KEPT)
    assert session_token.decode_session_claims(token) is not None


def test_with_auth_off_the_keys_are_not_consulted(monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    monkeypatch.delenv("RAG_API_KEYS", raising=False)
    from lasagna.api.rest.auth_mode import principal_for_api_key

    monkeypatch.setenv("RAG_API_KEYS", REMOVED)
    principal = principal_for_api_key(REMOVED)
    monkeypatch.delenv("RAG_API_KEYS")
    token, _ = session_token.create_session_token("s", "u", principal=principal)
    assert session_token.decode_session_claims(token) is not None
