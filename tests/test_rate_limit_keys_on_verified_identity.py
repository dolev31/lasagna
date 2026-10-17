"""With auth on, a request without a valid token must not spend a session's budget.

The limiter keyed its buckets on the asserted X-Session-ID before
authentication ran, so an unauthenticated client could exhaust any chosen
session's bucket (and, via the shared "default" bucket, block every sign-up).
"""

from __future__ import annotations

from starlette.requests import Request

from lasagna.api.rest import session_token
from lasagna.api.rest.app import _rate_limit_key


def _request(headers, client=("203.0.113.7", 5000)):
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/api/v1/documents",
        "query_string": b"",
        "headers": [(k.lower().encode(), v.encode()) for k, v in headers.items()],
        "client": client,
    }
    return Request(scope)


def test_an_asserted_session_id_is_counted_against_the_caller_not_the_victim(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    assert _rate_limit_key(_request({"X-Session-ID": "victim"})) == "ip:203.0.113.7"


def test_a_verified_token_is_counted_against_its_key(monkeypatch):
    from lasagna.api.rest import auth_mode

    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "k" * 64)
    monkeypatch.setenv(auth_mode.API_KEYS_ENV, "a" * 40)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    principal = auth_mode.principal_for_api_key("a" * 40)
    token, _expires = session_token.create_session_token(
        "alice-session", "alice", principal=principal
    )
    assert _rate_limit_key(_request({"X-Session-Token": token})) == f"prn:{principal}"


def test_a_token_naming_no_key_is_counted_against_the_address(monkeypatch):
    """Minted while auth was off: with auth on it verifies nothing."""
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "k" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    token, _expires = session_token.create_session_token("alice-session", "alice")
    assert _rate_limit_key(_request({"X-Session-Token": token})) == "ip:203.0.113.7"


def test_without_auth_the_asserted_id_is_used(monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    assert _rate_limit_key(_request({"X-Session-ID": "dev"})) == "dev"
