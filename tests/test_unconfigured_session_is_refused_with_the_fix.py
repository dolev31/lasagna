"""A session whose model is not downloaded gets the fix, never a 500 or a download.

The README's curl with no session header, or on ``my-corpus`` before it was
configured, answered ``{"detail":"Internal server error"} [500]`` (offline) or
started a 2.3 GB BGE-M3 download inside the request, while ``/health/ready``
reported ``embedding_model up BAAI/bge-m3`` from the model's name alone. MCP
``get_system_status`` reported ``row_count 0`` for a session with 76 chunks,
and ``lasagna search`` with no ``-s`` printed "Search failed: Search failed:
Could not load any embedding model" with no word of which session to use.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from lasagna import cli
from lasagna.core import model_downloads as md


@pytest.fixture
def not_cached(monkeypatch):
    monkeypatch.setattr(md, "is_cached", lambda model: False)


def test_offline_and_not_cached_is_refused_with_the_reason(monkeypatch, not_cached):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    problem = md.refusal("BAAI/bge-m3")
    assert isinstance(problem, md.ModelNotDownloaded)
    assert "offline" in str(problem) and "BAAI/bge-m3" in str(problem)


def test_a_server_refuses_a_download_unless_the_setting_allows_it(monkeypatch, not_cached):
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    assert md.refusal("BAAI/bge-m3") is None  # the CLI and SDK download on first use
    with md.refuse_downloads_in_requests():
        assert md.refusal("BAAI/bge-m3") is not None
        monkeypatch.setattr(md, "_allowed", lambda: True)
        assert md.refusal("BAAI/bge-m3") is None


def test_the_embedding_manager_does_not_start_the_download(monkeypatch, not_cached):
    from lasagna.infrastructure.embedding.manager import EmbeddingManager

    manager = EmbeddingManager()
    calls = []
    monkeypatch.setattr(manager, "_load_model", lambda: calls.append("load"))
    with md.refuse_downloads_in_requests(), pytest.raises(md.ModelNotDownloaded):
        manager._ensure_loaded()
    assert calls == []
    # Not remembered as a load failure: outside a server, online, it loads as
    # before. (Offline, a model that is not cached is refused everywhere.)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    manager._ensure_loaded()
    assert calls == ["load"]


class _Backend:
    def __init__(self, error):
        self.error = error
        self.embedding_manager = SimpleNamespace(
            model_name="BAAI/bge-m3", dimension=1024, is_loaded=False
        )

    async def async_search(self, **kw):
        raise self.error

    def get_vector_backend(self):
        return SimpleNamespace(backend_name="fake")


def _client(backend):
    from lasagna.api.rest.app import create_app
    from lasagna.api.rest.dependencies import get_backend

    app = create_app()
    app.dependency_overrides[get_backend] = lambda: backend
    return TestClient(app)


def test_search_answers_503_naming_the_fix(monkeypatch):
    # A session with documents whose model is not downloaded. One with none
    # is answered 200 with a note before any model loads
    # (tests/test_an_empty_session_a_security_link_and_configure_flags_say_what_is_wrong.py).
    monkeypatch.setattr(md, "sessions_with_documents", lambda: {"default": 2, "my-corpus": 3})
    backend = _Backend(md.ModelNotDownloaded("BAAI/bge-m3", offline=True))
    with _client(backend) as client:
        r = client.post("/api/v1/search", json={"query": "q", "top_k": 5})
    assert r.status_code == 503
    detail = r.json()["detail"]
    assert "not downloaded" in detail
    assert "lasagna configure --profile speed -s default" in detail


def test_readiness_says_the_default_model_cannot_load(monkeypatch, not_cached):
    from lasagna.api.rest.routes import health

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(health, "_dependency_pings", lambda backend: {})
    with _client(_Backend(RuntimeError("unused"))) as client:
        r = client.get("/api/v1/health/ready")
    assert r.status_code == 503
    check = r.json()["checks"]["embedding_model"]
    assert check["status"] == "down" and "not downloaded" in check["detail"]


def test_mcp_status_reports_the_sessions_collection(monkeypatch):
    from lasagna.mcp import server

    rows = {"rag_documents_v2__all_minilm_l6_v2__384": 76}
    store = SimpleNamespace(
        has_collection=lambda name: name in rows,
        get_collection_stats=lambda name: {"row_count": rows[name]},
    )
    session = SimpleNamespace(
        collection_name="rag_documents_v2__all_minilm_l6_v2__384",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
    )

    class _Cfg(SimpleNamespace):
        def __getattr__(self, name):
            return 0

    backend = SimpleNamespace(
        get_session_config=lambda sid: _Cfg(**vars(session), db_path="", upload_dir=""),
        get_vector_backend=lambda: store,
        embedding_manager=SimpleNamespace(
            model_name="BAAI/bge-m3", dimension=1024, is_loaded=False, get_cache_stats=dict
        ),
    )
    monkeypatch.setattr(server, "_get_backend", lambda: backend)
    monkeypatch.setenv("RAG_SESSION_ID", "my-corpus")
    status = json.loads(server.get_system_status())
    assert status["session_id"] == "my-corpus"
    assert status["components"]["milvus"]["row_count"] == 76
    assert status["components"]["embedding_model"]["model_name"] == "all-MiniLM-L6-v2"


def test_cli_search_on_an_empty_session_names_the_ones_with_documents(monkeypatch):
    monkeypatch.setattr(md, "sessions_with_documents", lambda: {"my-corpus": 3})

    def _never(*a, **k):
        raise AssertionError("a model was loaded to search a session with no documents")

    monkeypatch.setattr(cli, "_get_client", _never)
    r = CliRunner().invoke(cli.app, ["search", "what did it show?"], terminal_width=200)
    out = " ".join(r.output.split())
    assert r.exit_code == 1
    assert "Session 'default' has no documents" in out
    assert "my-corpus (3)" in out and "-s my-corpus" in out


def test_cli_search_says_search_failed_once(monkeypatch):
    from lasagna.sdk.exceptions import SearchError

    monkeypatch.setattr(md, "sessions_with_documents", lambda: None)

    def boom(coro):
        coro.close()
        try:
            raise RuntimeError("Could not load any embedding model")
        except RuntimeError as exc:
            raise SearchError(f"Search failed: {exc}") from exc

    monkeypatch.setattr(cli, "_run", boom)
    monkeypatch.setattr(cli, "_get_client", lambda *a, **k: None)
    r = CliRunner().invoke(cli.app, ["search", "q"], terminal_width=200)
    assert r.output.count("Search failed") == 1, r.output
