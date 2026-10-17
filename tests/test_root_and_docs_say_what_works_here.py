"""`/` advertises only a configurator that exists, and `/docs` says what to do offline.

In a source checkout `/` listed `"configurator": "/configurator"`, which
answered 404 because the page is not built (the serve banner already said
so). `/docs` loads Swagger UI from a CDN, and offline it was a blank page.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def _app(monkeypatch, configurator_built: bool):
    import lasagna.configurator as configurator
    from lasagna.api.rest.app import create_app

    monkeypatch.setenv("RAG_CONFIGURATOR", "true")
    if not configurator_built:
        monkeypatch.setattr(configurator, "FRONTEND_DIST", configurator.CONFIGURATOR_DIR / "nope")
    return TestClient(create_app())


def test_root_does_not_advertise_a_page_that_is_not_built(monkeypatch):
    body = _app(monkeypatch, configurator_built=False).get("/").json()
    assert body["configurator"] is None
    assert "npm run build" in body["configurator_note"]


def test_docs_explains_itself_when_swagger_cannot_load(monkeypatch):
    page = _app(monkeypatch, configurator_built=False).get("/docs")
    assert page.status_code == 200
    assert "swagger-ui-bundle" in page.text
    assert "typeof SwaggerUIBundle === 'undefined'" in page.text
    assert "/openapi.json" in page.text and "endpoints.md" in page.text
