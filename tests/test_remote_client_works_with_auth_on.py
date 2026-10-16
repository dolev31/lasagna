"""``KnowledgeClient.from_url`` closes cleanly and works against a server with auth on.

``from_url`` built the client field by field with ``cls.__new__`` and missed
``_in_flight_documents``, so ``close()`` raised ``AttributeError``. And it sent
``X-Session-ID`` with the API key as a Bearer token, never an
``X-Session-Token``, so a server with ``RAG_REQUIRE_AUTH`` on (the Kubernetes
ConfigMap's setting) answered 401 to every call. The client now mints a
session token with the key when the server first answers 401, or sends one
it was given.
"""

from __future__ import annotations

import asyncio
import functools

import httpx
import pytest

from lasagna.api.rest import session_token
from lasagna.api.rest.app import create_app
from lasagna.api.rest.dependencies import get_backend
from lasagna.sdk import KnowledgeClient
from lasagna.sdk import remote_backend

KEY = "k" * 40


class _Backend:
    """Records which session each listing was for."""

    def __init__(self):
        self.sessions = []

    def get_documents(self, session_id):
        self.sessions.append(session_id)
        return []


@pytest.fixture
def server(monkeypatch):
    """The REST app behind every httpx client `from_url` makes; returns what it saw."""
    monkeypatch.setenv("RAG_JWT_SECRET", "s" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    backend = _Backend()
    paths = []
    app = create_app()
    app.dependency_overrides[get_backend] = lambda: backend

    async def recording(scope, receive, send):
        if scope["type"] == "http":
            paths.append((scope["method"], scope["path"]))
        await app(scope, receive, send)

    monkeypatch.setattr(
        remote_backend.httpx,
        "AsyncClient",
        functools.partial(httpx.AsyncClient, transport=httpx.ASGITransport(app=recording)),
    )
    return backend, paths


@pytest.fixture
def auth_on(monkeypatch):
    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_API_KEYS", KEY)


@pytest.fixture
def auth_off(monkeypatch):
    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)


def _list_twice_and_close(client):
    async def go():
        try:
            await client.list_documents()
            await client.list_documents()
        finally:
            await client.close()

    asyncio.run(go())


def test_close_on_a_remote_client_does_not_raise(server, auth_off):
    asyncio.run(KnowledgeClient.from_url("http://testserver").close())


def test_with_auth_off_the_named_session_is_used(server, auth_off):
    backend, paths = server
    client = KnowledgeClient.from_url("http://testserver", api_key=KEY, session_id="my-project")

    _list_twice_and_close(client)

    assert backend.sessions == ["my-project", "my-project"]
    assert ("POST", "/api/v1/sessions") not in paths


def test_with_auth_on_the_key_mints_one_session_token(server, auth_on):
    backend, paths = server
    client = KnowledgeClient.from_url("http://testserver", api_key=KEY)

    _list_twice_and_close(client)

    assert paths.count(("POST", "/api/v1/sessions")) == 1
    assert len(backend.sessions) == 2 and len(set(backend.sessions)) == 1
    assert backend.sessions[0].startswith("sess_")
    assert client.session_id == backend.sessions[0]


def test_with_auth_on_a_given_token_is_sent(server, auth_on):
    backend, paths = server
    token, _ = session_token.create_session_token("sess_given", "user_1", principal="ci")
    client = KnowledgeClient.from_url("http://testserver", session_token=token)

    _list_twice_and_close(client)

    assert backend.sessions == ["sess_given", "sess_given"]
    assert ("POST", "/api/v1/sessions") not in paths


def test_with_auth_on_and_no_key_the_refusal_reaches_the_caller(server, auth_on):
    client = KnowledgeClient.from_url("http://testserver")

    with pytest.raises(httpx.HTTPStatusError) as err:
        _list_twice_and_close(client)
    assert err.value.response.status_code == 401


def test_with_auth_on_a_named_session_is_not_quietly_replaced(server, auth_on):
    backend, _ = server
    client = KnowledgeClient.from_url("http://testserver", api_key=KEY, session_id="theirs")

    with pytest.raises(httpx.HTTPStatusError) as err:
        _list_twice_and_close(client)
    assert err.value.response.status_code == 403
    assert "X-Admin-Token" in err.value.response.text
    assert backend.sessions == []
