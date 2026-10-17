"""The MCP status tool needs a caller under auth, and shows the operator's details to the operator.

``get_system_status`` took no credential while every other tool needs a session
token on a networked transport with ``RAG_REQUIRE_AUTH`` on, and it answered
with the metadata database's path, the upload folder and the row count of the
collection every session on the model shares. Now it resolves its session like
the other tools, and the paths and the shared count need the admin token.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from lasagna.api.rest import auth_mode, session_token
from lasagna.mcp import server, transport_guard
from tests._tokens import minted_token

ADMIN = "a" * 40


class _Cfg(SimpleNamespace):
    def __getattr__(self, name):
        return 0


@pytest.fixture
def status(monkeypatch):
    import lasagna.core.database as database
    from lasagna.api.rest.routes import health

    store = SimpleNamespace(
        backend_name="milvus",
        has_collection=lambda name: True,
        get_collection_stats=lambda name: {"row_count": 1234},
    )
    backend = SimpleNamespace(
        embedding_manager=SimpleNamespace(
            model_name="m", dimension=8, is_loaded=True, get_cache_stats=dict
        ),
        get_vector_backend=lambda *a: store,
        get_session_config=lambda sid: _Cfg(
            collection_name="c", embedding_model="m", db_path="/srv/db/rag.db", upload_dir="/srv/up"
        ),
    )
    monkeypatch.setattr(health, "PING_TIMEOUT_S", 0.5)
    monkeypatch.setattr(
        database, "get_rag_database", lambda: SimpleNamespace(get_documents=lambda sid: [])
    )
    monkeypatch.setattr(server, "_get_backend", lambda: backend)
    monkeypatch.setattr(server, "_get_config", lambda: _Cfg(collection_name="c"))
    monkeypatch.delenv("RAG_SESSION_ID", raising=False)

    def call(**kwargs):
        return json.loads(server.get_system_status(**kwargs))

    return call


@pytest.fixture
def networked_with_auth(monkeypatch):
    monkeypatch.setattr(transport_guard, "get_transport", lambda: "sse")
    monkeypatch.setattr(transport_guard, "is_networked", lambda: True)
    monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "s" * 64)
    monkeypatch.setenv("RAG_ADMIN_TOKEN", ADMIN)
    monkeypatch.setattr(session_token, "_fallback_secret", None)


def test_without_a_token_it_is_refused(status, networked_with_auth):
    with pytest.raises(transport_guard.MCPAuthError):
        status(session_id="tenant-a")


def test_a_tenant_sees_no_paths_and_no_shared_count(status, networked_with_auth):
    body = status(session_token=minted_token("tenant-a"))
    text = json.dumps(body)
    assert body["session_id"] == "tenant-a"
    assert "/srv/db" not in text and "/srv/up" not in text
    assert "row_count" not in body["components"]["milvus"]


def test_the_operator_sees_them(status, networked_with_auth):
    body = status(session_id="tenant-a", admin_token=ADMIN)
    assert body["components"]["sqlite"]["db_path"] == "/srv/db/rag.db"
    assert body["storage"]["upload_dir"] == "/srv/up"
    assert body["components"]["milvus"]["row_count"] == 1234


def test_a_wrong_admin_token_is_refused(status, networked_with_auth):
    with pytest.raises(transport_guard.MCPAuthError):
        status(session_id="tenant-a", admin_token="b" * 40)


def test_with_auth_off_the_local_user_sees_everything(status, monkeypatch):
    monkeypatch.delenv(auth_mode.REQUIRE_AUTH_ENV, raising=False)
    body = status()
    assert body["session_id"] == "default"
    assert body["components"]["milvus"]["row_count"] == 1234
    assert body["storage"]["upload_dir"] == "/srv/up"
