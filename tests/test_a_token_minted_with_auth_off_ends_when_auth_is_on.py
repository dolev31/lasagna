"""A session token minted while auth was off stops working once auth is on.

With ``RAG_REQUIRE_AUTH`` off, ``POST /api/v1/sessions`` mints a token that
names no principal (no API key minted it). Switching auth on with the same
``RAG_JWT_SECRET`` left every such token valid for the rest of its lifetime,
seven days by default: anyone who had one kept the session without a key.
With auth on, a token must name its key, or ``admin`` for one minted with the
admin token; with only ``RAG_ADMIN_TOKEN`` set (no keys) the old token was
still accepted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lasagna.api.rest import auth_mode, session_token
from lasagna.api.rest.app import create_app

KEY = "k" * 40
ADMIN = "a" * 40


@pytest.fixture
def secret(monkeypatch):
    monkeypatch.setenv("RAG_JWT_SECRET", "s" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    monkeypatch.delenv(auth_mode.REQUIRE_AUTH_ENV, raising=False)


def _mint(**headers):
    response = TestClient(create_app()).post("/api/v1/sessions", headers=headers, json={})
    assert response.status_code == 201, response.text
    return response.json()["token"]


def _auth_on(monkeypatch, keys=KEY):
    monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")
    monkeypatch.setenv(auth_mode.API_KEYS_ENV, keys)
    monkeypatch.setenv("RAG_ADMIN_TOKEN", ADMIN)


def test_a_token_from_before_auth_is_refused(secret, monkeypatch):
    token = _mint()
    assert session_token.decode_session_claims(token) is not None  # auth still off
    _auth_on(monkeypatch)
    assert session_token.decode_session_claims(token) is None
    response = TestClient(create_app()).get("/api/v1/documents", headers={"X-Session-Token": token})
    assert response.status_code == 401


def test_a_token_minted_with_a_key_is_accepted(secret, monkeypatch):
    _auth_on(monkeypatch)
    token = _mint(**{"X-API-Key": KEY})
    claims = session_token.decode_session_claims(token)
    assert claims is not None and claims["prn"].startswith(auth_mode.KEY_PRINCIPAL_PREFIX)


def test_an_admin_token_is_accepted(secret, monkeypatch):
    _auth_on(monkeypatch)
    token = _mint(**{"X-Admin-Token": ADMIN})
    assert session_token.decode_session_claims(token)["prn"] == "admin"


def test_a_token_from_before_auth_is_refused_with_only_an_admin_token(secret, monkeypatch):
    token = _mint()
    monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")
    monkeypatch.delenv(auth_mode.API_KEYS_ENV, raising=False)
    monkeypatch.setenv("RAG_ADMIN_TOKEN", ADMIN)
    assert session_token.decode_session_claims(token) is None
    admin = _mint(**{"X-Admin-Token": ADMIN})
    assert session_token.decode_session_claims(admin)["prn"] == "admin"
