"""A 413 or 429 sent to an allowed origin carries the CORS headers.

CORS was the innermost middleware, so the body limit and the rate limiter
answered before it ran: their 413 and 429 had no
``Access-Control-Allow-Origin``, the browser hid them from the page, and a
frontend on an allowed origin saw a bare network error instead of "too large"
or "retry later".
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

ORIGIN = "http://app.example"


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("RAG_CORS_ORIGINS", ORIGIN)
    from lasagna.api.rest.app import create_app

    return TestClient(create_app(), base_url="http://localhost")


def test_a_body_over_the_limit_is_refused_with_cors_headers(client):
    response = client.post(
        "/api/v1/search",
        content=b"x" * (2 << 20),
        headers={"Origin": ORIGIN, "Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert response.headers.get("access-control-allow-origin") == ORIGIN


def test_a_rate_limited_request_is_refused_with_cors_headers(client, monkeypatch):
    import lasagna.core.rate_limiter as rate_limiter

    limiter = SimpleNamespace(
        is_enabled=True,
        try_acquire=lambda key, path: (False, 1.0),
        record_request_latency=lambda ms: None,
    )
    monkeypatch.setattr(rate_limiter, "get_rate_limiter", lambda: limiter)
    response = client.get("/api/v1/documents", headers={"Origin": ORIGIN})
    assert response.status_code == 429
    assert response.headers.get("access-control-allow-origin") == ORIGIN


def test_another_origin_still_gets_none(client):
    response = client.post(
        "/api/v1/search",
        content=b"x" * (2 << 20),
        headers={"Origin": "https://attacker.example", "Content-Type": "application/json"},
    )
    assert response.status_code == 413
    assert "access-control-allow-origin" not in response.headers
