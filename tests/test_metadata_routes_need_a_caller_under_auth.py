"""With auth on, the instance's metadata answers only a verified caller.

``GET /api/v1/stats`` (embedding model, collection name, feature flags),
``GET /api/v1/pipelines`` (pipeline names and descriptions, which can be a
tenant's custom pipeline) and the configurator's schema and validate routes
answered with no credential on an instance advertised as locked down. The
liveness and readiness probes stay open for orchestrators.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lasagna.api.rest import session_token
from lasagna.api.rest.dependencies import get_backend

_ROUTES = [
    ("get", "/api/v1/stats", None),
    ("get", "/api/v1/pipelines", None),
    ("get", "/configurator/api/schema", None),
    ("post", "/configurator/api/validate", {"config": {}}),
]


@pytest.fixture
def app(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "k" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    from fastapi import FastAPI

    from lasagna.api.rest.routes import router
    from lasagna.configurator import mount_configurator

    app = FastAPI()
    app.include_router(router)
    mount_configurator(app)
    app.dependency_overrides[get_backend] = lambda: SimpleNamespace(get_stats=lambda: {})
    return TestClient(app)


@pytest.mark.parametrize("method, path, body", _ROUTES)
def test_an_unauthenticated_caller_is_refused(app, method, path, body):
    response = getattr(app, method)(path, **({"json": body} if body else {}))
    assert response.status_code == 401, response.text


@pytest.mark.parametrize("method, path, body", _ROUTES)
def test_a_verified_caller_is_answered(app, method, path, body):
    token, _ = session_token.create_session_token("s", "u")
    response = getattr(app, method)(
        path, headers={"X-Session-Token": token}, **({"json": body} if body else {})
    )
    assert response.status_code == 200, response.text


def test_the_probes_stay_open(app):
    assert app.get("/api/v1/health/live").status_code == 200
