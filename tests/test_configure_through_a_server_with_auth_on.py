"""`lasagna configure` through a running server with auth on sends the admin's session token.

While `lasagna serve` holds the local store, `configure` switches the profile
through it. It sent no credential, so with ``RAG_REQUIRE_AUTH`` on the server
answered 401 and the CLI printed the server's sentence about session tokens.
With the server's admin token in ``RAG_ADMIN_TOKEN`` it now mints a session
token for the named session and switches with it; without one, a 401 names the
setting to set.
"""

from __future__ import annotations

import os
import uuid
from urllib.parse import urlparse

import pytest
from typer.testing import CliRunner

from lasagna import cli
from lasagna.core import local_server

ADMIN = "admin-token-for-tests-0123456789abcdef"
URL = "http://127.0.0.1:8765"


@pytest.fixture
def server(monkeypatch):
    """The real REST app with auth on, reached by httpx as a running server would be."""
    import httpx
    from fastapi.testclient import TestClient

    from lasagna.api.rest.app import create_app

    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "test-secret-not-a-real-one-0123456789abcdef")
    monkeypatch.setenv("RAG_ADMIN_TOKEN", ADMIN)
    client = TestClient(create_app())
    sent = []

    def call(method):
        def send(url, **kwargs):
            sent.append((method, urlparse(url).path, dict(kwargs.get("headers") or {})))
            kwargs.pop("timeout", None)
            # The server has its own environment: its admin token is ADMIN
            # whatever the CLI's says.
            cli_token = os.environ.get("RAG_ADMIN_TOKEN")
            os.environ["RAG_ADMIN_TOKEN"] = ADMIN
            try:
                return getattr(client, method)(urlparse(url).path, **kwargs)
            finally:
                if cli_token is None:
                    os.environ.pop("RAG_ADMIN_TOKEN", None)
                else:
                    os.environ["RAG_ADMIN_TOKEN"] = cli_token

        return send

    monkeypatch.setattr(httpx, "put", call("put"))
    monkeypatch.setattr(httpx, "post", call("post"))
    monkeypatch.setattr(local_server, "local_store_held", lambda: True)
    monkeypatch.setattr(local_server, "running_server", lambda: URL)
    return sent


def _configure(sid):
    result = CliRunner().invoke(cli.app, ["configure", "--profile", "speed", "-s", sid])
    return result, " ".join(result.output.split())


def test_the_switch_goes_with_a_session_token_the_admin_token_minted(server):
    sid = f"auth-{uuid.uuid4().hex[:8]}"
    result, out = _configure(sid)
    assert result.exit_code == 0, out
    assert "Applied profile: speed" in out
    (mint, switch) = server
    assert mint[:2] == ("post", "/api/v1/sessions") and mint[2] == {"X-Admin-Token": ADMIN}
    assert switch[:2] == ("put", "/api/config/knowledge/active")
    assert set(switch[2]) == {"X-Session-Token"}
    from lasagna.core.session_config import get_knowledge_config_manager

    assert get_knowledge_config_manager(session_id=sid).get_active_profile(sid) == "speed"


def test_without_the_admin_token_a_401_names_what_to_set(server, monkeypatch):
    monkeypatch.delenv("RAG_ADMIN_TOKEN")
    result, out = _configure(f"auth-{uuid.uuid4().hex[:8]}")
    assert result.exit_code == 1
    assert "requires authentication (RAG_REQUIRE_AUTH is on)" in out
    assert "set RAG_ADMIN_TOKEN to the server's value" in out


def test_a_wrong_admin_token_is_named(server, monkeypatch):
    monkeypatch.setenv("RAG_ADMIN_TOKEN", "a-different-admin-token-0123456789abcd")
    result, out = _configure(f"auth-{uuid.uuid4().hex[:8]}")
    assert result.exit_code == 1
    assert "refused the admin token in RAG_ADMIN_TOKEN" in out
