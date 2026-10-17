"""Liveness, readiness, detailed status and MCP status describe one model.

`/api/v1/health` answered 200 with ``"status": "unhealthy", "failing": []``,
`/api/v1/health/ready` answered 503 with ``["embedding_model"]``, the MCP
``get_system_status`` tool said "healthy", and the deployment guide said both
"always with 200" and "503 while ... not loaded". Now: liveness checks
nothing; readiness answers 503 naming each failing component, with Redis down
counted as degraded (its users fall back); detailed status and MCP report the
same component states, always with 200.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from lasagna.core import model_downloads


class _Cfg(SimpleNamespace):
    def __getattr__(self, name):
        return 0


def _backend(model_loaded=True):
    store = SimpleNamespace(
        backend_name="milvus",
        has_collection=lambda name: True,
        get_collection_stats=lambda name: {"row_count": 3},
    )
    return SimpleNamespace(
        embedding_manager=SimpleNamespace(
            model_name="BAAI/bge-m3", dimension=1024, is_loaded=model_loaded, get_cache_stats=dict
        ),
        get_vector_backend=lambda: store,
    )


@pytest.fixture
def surfaces(monkeypatch):
    """The REST app and the MCP status tool over one stand-in backend."""
    import lasagna.core.database as database
    from lasagna.api.rest.app import create_app
    from lasagna.api.rest.dependencies import get_backend
    from lasagna.api.rest.routes import health
    from lasagna.mcp import server

    monkeypatch.setattr(health, "PING_TIMEOUT_S", 0.5)
    monkeypatch.setattr(
        database, "get_rag_database", lambda: SimpleNamespace(get_documents=lambda sid: [])
    )
    monkeypatch.delenv("RAG_SESSION_ID", raising=False)

    def make(backend):
        app = create_app()
        app.dependency_overrides[get_backend] = lambda: backend
        monkeypatch.setattr(server, "_get_backend", lambda: backend)
        monkeypatch.setattr(
            server, "_get_config", lambda: _Cfg(collection_name="c", db_path="", upload_dir="")
        )
        backend.get_session_config = lambda sid: _Cfg(
            collection_name="c", embedding_model="BAAI/bge-m3", db_path="", upload_dir=""
        )
        client = TestClient(app)

        def answers():
            health_ = client.get("/api/v1/health")
            ready = client.get("/api/v1/health/ready")
            status = json.loads(server.get_system_status())
            return health_, ready, status

        return client, answers

    return make


def test_a_model_that_cannot_load_fails_every_answer_alike(surfaces, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(model_downloads, "is_cached", lambda model: False)
    _client, answers = surfaces(_backend(model_loaded=False))
    health, ready, mcp = answers()
    assert health.status_code == 200 and ready.status_code == 503
    for body in (health.json(), ready.json(), mcp):
        assert body["status"] == "unhealthy"
        assert body["failing"] == ["embedding_model"] and body["degraded"] == []


def test_everything_up_is_healthy_everywhere(surfaces):
    _client, answers = surfaces(_backend(model_loaded=True))
    health, ready, mcp = answers()
    assert health.status_code == 200 and ready.status_code == 200
    for body in (health.json(), ready.json(), mcp):
        assert (body["status"], body["failing"], body["degraded"]) == ("healthy", [], [])


def test_redis_down_is_degraded_everywhere_and_still_ready(surfaces, monkeypatch):
    import lasagna.core.config as config_module

    monkeypatch.setattr(
        config_module,
        "rag_config",
        SimpleNamespace(
            collection_name="c",
            cache_backend="redis",
            redis_url="redis://127.0.0.1:1",  # nothing listens on port 1
            celery_enabled=False,
        ),
        raising=False,
    )
    _client, answers = surfaces(_backend(model_loaded=True))
    health, ready, mcp = answers()
    assert health.status_code == 200 and ready.status_code == 200
    for body in (health.json(), ready.json(), mcp):
        assert (body["status"], body["failing"], body["degraded"]) == ("degraded", [], ["redis"])


def test_liveness_checks_no_dependency(surfaces):
    def unreachable():
        raise AssertionError("liveness touched the backend")

    client, _answers = surfaces(_backend())
    client.app.dependency_overrides.clear()
    import lasagna.api.rest.dependencies as deps

    original = deps.get_rag_backend
    deps.get_rag_backend = unreachable
    try:
        r = client.get("/api/v1/health/live")
    finally:
        deps.get_rag_backend = original
    assert r.status_code == 200 and r.json() == {"status": "alive"}


def test_readiness_checks_the_model_the_served_session_uses(surfaces, monkeypatch):
    """`serve -s my-corpus` on MiniLM is ready although the settings file's BGE-M3 is absent."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(model_downloads, "is_cached", lambda model: model in ("all-MiniLM-L6-v2",))
    backend = _backend(model_loaded=False)
    client, _answers = surfaces(backend)
    from lasagna.api.rest.routes import health

    served = SimpleNamespace(model_name="all-MiniLM-L6-v2", dimension=384, is_loaded=False)
    monkeypatch.setattr(health, "_served_embedder", lambda b: served)
    r = client.get("/api/v1/health/ready")
    assert r.status_code == 200, r.json()
    assert "all-MiniLM-L6-v2" in r.json()["checks"]["embedding_model"]["detail"]


def test_the_deployment_guide_states_one_model():
    from tests._paths import PACKAGE_DIR

    guide = (PACKAGE_DIR / "deploy" / "README.md").read_text()
    assert "always with 200" in guide
    assert "503 while the backend has not loaded" not in guide
