"""The REST server reports the package's version and the address it serves on.

``/api/v1/health`` and ``/`` reported ``"version": "1.0.0"`` for 0.1.0a1, and
the startup log said ``API: http://0.0.0.0:8000/api/v1/health`` for a server
bound to 127.0.0.1:8931.
"""

from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

import lasagna


def test_health_and_root_report_the_package_version(monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    from lasagna.api.rest.app import create_app
    from lasagna.api.rest.dependencies import get_backend

    app = create_app()
    app.dependency_overrides[get_backend] = lambda: SimpleNamespace()
    client = TestClient(app)
    assert client.get("/").json()["version"] == lasagna.__version__
    assert client.get("/api/v1/health").json()["version"] == lasagna.__version__
    assert app.version == lasagna.__version__


def test_serve_prints_the_address_it_binds(monkeypatch):
    from typer.testing import CliRunner

    from lasagna import cli

    ran = {}
    monkeypatch.setattr("uvicorn.run", lambda *a, **k: ran.update(k))
    result = CliRunner().invoke(
        cli.app, ["serve", "--port", "8931", "--no-open", "--host", "127.0.0.1"]
    )
    assert result.exit_code == 0, result.output
    assert "http://127.0.0.1:8931/api/v1/health" in result.output
    assert "0.0.0.0" not in result.output and "8000" not in result.output
    assert ran["port"] == 8931
