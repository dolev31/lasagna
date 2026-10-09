"""An unauthenticated caller must not be able to choose whose session it gets.

`POST /api/v1/sessions` took `session_id` and `user_id` from the request body
and minted a signed token for them, with no authentication and no router guard.
Sessions are the only isolation boundary in this system — `documents.py` and
`search.py` reference `get_user_id` zero times — so a caller who could name
another tenant's `session_id` could list, read and delete their documents. The
shipped k8s manifests run the API at `replicas: 2`, so this is the documented
deployment shape rather than a local-only path.

Harmless for the single-user local case the README describes, which is
presumably how it survived. Not harmless for anything shared.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402

from lasagna.api.rest.routes import sessions as sessions_route  # noqa: E402


@pytest.fixture
def client():
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(sessions_route.router, prefix="/api/v1")
    return TestClient(app)


class TestAnonymousCallersGetAFreshSession:
    """The normal path must stay frictionless — this is not an auth wall."""

    def test_an_empty_request_is_issued_a_session(self, client):
        r = client.post("/api/v1/sessions", json={})
        assert r.status_code == 201
        body = r.json()
        assert body["session_id"].startswith("sess_")
        assert body["user_id"].startswith("user_")
        assert body["token"]

    def test_no_body_at_all_also_works(self, client):
        assert client.post("/api/v1/sessions").status_code == 201

    def test_two_calls_get_different_sessions(self, client):
        a = client.post("/api/v1/sessions", json={}).json()["session_id"]
        b = client.post("/api/v1/sessions", json={}).json()["session_id"]
        assert a != b


class TestClaimingAnIdentityIsRefused:
    @pytest.fixture(autouse=True)
    def _no_admin_token(self, monkeypatch):
        monkeypatch.delenv("RAG_ADMIN_TOKEN", raising=False)

    def test_naming_a_session_id_is_forbidden(self, client):
        r = client.post("/api/v1/sessions", json={"session_id": "victim-tenant"})
        assert r.status_code == 403, (
            "an unauthenticated caller was issued a token for a session it named"
        )

    def test_naming_a_user_id_is_forbidden(self, client):
        assert client.post("/api/v1/sessions", json={"user_id": "someone-else"}).status_code == 403

    def test_the_refusal_explains_the_alternative(self, client):
        detail = client.post("/api/v1/sessions", json={"session_id": "x"}).json()["detail"]
        assert "X-Admin-Token" in detail
        assert "Omit both" in detail

    def test_it_refuses_rather_than_silently_substituting(self, client):
        """Returning a *different* session than asked for looks like success.

        The failure would then surface later, somewhere unrelated, as an empty
        search against a session the caller believes it is using.
        """
        r = client.post("/api/v1/sessions", json={"session_id": "victim-tenant"})
        assert r.status_code == 403
        assert "session_id" not in r.json()

    def test_a_wrong_admin_token_is_still_refused(self, client, monkeypatch):
        monkeypatch.setenv("RAG_ADMIN_TOKEN", "correct-token-0123456789abcdef0123456789")
        r = client.post(
            "/api/v1/sessions",
            json={"session_id": "victim-tenant"},
            headers={"X-Admin-Token": "wrong-token"},
        )
        assert r.status_code == 403


class TestAdminsMayStillProvisionSessions:
    def test_a_valid_admin_token_may_name_a_session(self, client, monkeypatch):
        monkeypatch.setenv("RAG_ADMIN_TOKEN", "correct-token-0123456789abcdef0123456789")
        r = client.post(
            "/api/v1/sessions",
            json={"session_id": "tenant-a", "user_id": "u1"},
            headers={"X-Admin-Token": "correct-token-0123456789abcdef0123456789"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["session_id"] == "tenant-a"
        assert body["user_id"] == "u1"


class TestTheGuardItself:
    """Fails closed, and does not leak the token by timing."""

    def test_no_configured_token_means_nobody_may_claim(self, monkeypatch):
        monkeypatch.delenv("RAG_ADMIN_TOKEN", raising=False)
        assert sessions_route._may_claim_an_identity("anything") is False

    def test_an_empty_configured_token_does_not_authorise(self, monkeypatch):
        monkeypatch.setenv("RAG_ADMIN_TOKEN", "")
        assert sessions_route._may_claim_an_identity("") is False

    def test_a_missing_header_does_not_authorise(self, monkeypatch):
        monkeypatch.setenv("RAG_ADMIN_TOKEN", "correct-token-0123456789abcdef0123456789")
        assert sessions_route._may_claim_an_identity(None) is False

    def test_comparison_is_constant_time(self):
        """A prefix-leaking comparison is not a token check."""
        import inspect

        from lasagna.api.rest import auth_mode

        src = inspect.getsource(sessions_route._may_claim_an_identity)
        assert "secret_matches" in src
        assert "compare_digest" in inspect.getsource(auth_mode.secret_matches)
