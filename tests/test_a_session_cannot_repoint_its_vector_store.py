"""A session sets retrieval and generation settings, never where the server keeps data.

Any key a session wrote was stored and resolved. ``patch_rag_config`` (in the
MCP server's default tool set) with ``{"vector_backend": "qdrant",
"vector_backend_config": {"url": "http://169.254.169.254"}}`` came back
valid, and the session's next search made the server connect there
(``resolve_backend`` calls ``get_backend(name, **config)``). The same went for
another collection, a database or upload path, a Redis URL,
``reranker_trust_remote_code`` and keys that do not exist, through REST
(``/api/config/knowledge/profile``, ``/import``), MCP and the config manager.
"""

from __future__ import annotations

import json

import pytest

from lasagna.core.session_config._session_keys import SessionSettingRefused
from tests._isolated_config import isolated_manager, rest_client

#: One per class of setting a session must not set, with the key the error names.
REFUSED = {
    "vector store": ({"vector_backend": "qdrant"}, "vector_backend"),
    "vector store address": (
        {"vector_backend_config": {"url": "http://169.254.169.254:6333"}},
        "vector_backend_config",
    ),
    "another collection": ({"collection_name": "another_tenants_documents"}, "collection_name"),
    "a database path": ({"db_path": "/tmp/elsewhere.db"}, "db_path"),
    "an upload directory": ({"upload_dir": "/etc"}, "upload_dir"),
    "a Redis URL": ({"redis_url": "redis://attacker.example:6379"}, "redis_url"),
    "a credential": ({"database_dsn": "postgresql://u:p@db/x"}, "database_dsn"),
    "remote code": ({"reranker_trust_remote_code": True}, "reranker_trust_remote_code"),
    "server limits": ({"rate_limiting_rpm": 987654}, "rate_limiting_rpm"),
    "an unknown key": ({"rerank_topk": 3}, "rerank_topk"),
    "a model from a path": ({"reranker_model": "/srv/models/private"}, "reranker_model"),
    "a model from a URL": ({"hyde_model": "http://attacker.example/m"}, "hyde_model"),
}

ALLOWED = {"rerank_top_k": 7, "enable_hyde": False, "hybrid_alpha": 0.4}


@pytest.fixture
def manager(tmp_path, monkeypatch):
    return isolated_manager(tmp_path, monkeypatch)


def _search_sees(sid):
    from lasagna.providers import get_session_provider

    return get_session_provider().get_resolved_config(sid)


_WRITES = {
    "patch": lambda m, sid, cfg: m.patch_config(cfg, session_id=sid),
    "switch with config": lambda m, sid, cfg: m.set_active_profile(
        "mine", config=cfg, session_id=sid
    ),
    "save": lambda m, sid, cfg: m.save_profile("mine", cfg, session_id=sid),
    "import overrides": lambda m, sid, cfg: m.import_config(
        {"schema_version": 2, "profiles": [], "session_overrides": cfg}, session_id=sid
    ),
    "import a profile": lambda m, sid, cfg: m.import_config(
        {"schema_version": 2, "profiles": [{"name": "mine", "config": cfg}]}, session_id=sid
    ),
}


@pytest.mark.parametrize("write", sorted(_WRITES))
@pytest.mark.parametrize("case", sorted(REFUSED))
def test_every_write_refuses_every_kind_of_server_setting(manager, write, case):
    sid = "tenant"
    config, key = REFUSED[case]
    with pytest.raises(SessionSettingRefused, match=key):
        _WRITES[write](manager, sid, dict(config))
    assert key not in manager.get_session_state(sid).session_overrides
    assert not any(
        key in p.config for p in manager.get_session_state(sid).profiles if p.name == "mine"
    )
    assert key not in _search_sees(sid)


@pytest.mark.parametrize("write", sorted(_WRITES))
def test_retrieval_settings_still_apply(manager, write):
    sid = f"knobs-{write.replace(' ', '-')}"
    _WRITES[write](manager, sid, dict(ALLOWED))
    resolved = manager.get_resolved_config(sid)
    if write == "import a profile":
        manager.set_active_profile("mine", session_id=sid)
        resolved = manager.get_resolved_config(sid)
    assert {k: resolved[k] for k in ALLOWED} == ALLOWED


def test_the_error_names_the_key_and_where_server_settings_go(manager):
    with pytest.raises(SessionSettingRefused) as refused:
        manager.patch_config({"vector_backend_config": {"url": "http://x"}, "rerank_topk": 1})
    message = str(refused.value)
    assert "vector_backend_config" in message and "rerank_topk" in message
    assert "did you mean 'rerank_top_k'" in message
    assert "settings.lasagna.toml" in message


def test_an_exported_configuration_imports_again(manager):
    """An export carries every built-in profile's whole configuration: the
    server's own values are dropped, not refused."""
    sid = "round-trip"
    manager.patch_config(dict(ALLOWED), session_id=sid)
    exported = manager.export_config(sid)
    assert any("vector_backend" in p["config"] for p in exported["profiles"])
    manager.import_config(exported, session_id="round-trip-2")
    resolved = manager.get_resolved_config("round-trip-2")
    assert {k: resolved[k] for k in ALLOWED} == ALLOWED


def test_the_server_keeps_its_own_vector_store(manager):
    from lasagna.core.config import rag_config

    with pytest.raises(SessionSettingRefused):
        manager.patch_config(
            {"vector_backend": "qdrant", "vector_backend_config": {"url": "http://x"}},
            session_id="probe",
        )
    session = rag_config.with_session("probe")
    assert session.vector_backend == rag_config.vector_backend
    assert session.vector_backend_config == rag_config.vector_backend_config


# -- the surfaces ----------------------------------------------------------------


@pytest.mark.parametrize("case", ["vector store address", "a database path", "remote code"])
def test_mcp_patch_and_save_refuse_them(manager, case):
    from lasagna.mcp import server

    config, key = REFUSED[case]
    reply = json.loads(server.patch_rag_config("mcp-tenant", json.dumps(config)))
    assert key in reply["error"]
    reply = json.loads(server.save_rag_profile("mcp-tenant", "evil", json.dumps(config)))
    assert key in reply["error"]
    assert key not in _search_sees("mcp-tenant")
    ok = json.loads(server.patch_rag_config("mcp-tenant", json.dumps(ALLOWED)))
    assert "error" not in ok


@pytest.mark.parametrize("route", ["profile", "import", "patch"])
def test_rest_refuses_them_with_a_422_naming_the_key(manager, route):
    client = rest_client()
    headers = {"X-Session-ID": "rest-tenant"}
    bad = {"vector_backend": "qdrant"}
    if route == "profile":
        response = client.post(
            "/api/config/knowledge/profile",
            json={"profile_name": "evil", "config": bad, "set_active": True},
            headers=headers,
        )
    elif route == "import":
        response = client.post(
            "/api/config/knowledge/import",
            json={"schema_version": 2, "profiles": [], "session_overrides": bad},
            headers=headers,
        )
    else:
        response = client.patch("/api/v1/config", json=bad, headers=headers)
    assert response.status_code == 422, response.text
    assert "vector_backend" in response.json()["detail"]
    assert "vector_backend" not in _search_sees("rest-tenant")


def test_the_sdk_config_manager_saves_only_session_settings(manager):
    import asyncio

    from lasagna.sdk import KnowledgeClient, KnowledgeConfig

    client = KnowledgeClient(
        config=KnowledgeConfig(rerank_top_k=7, enable_hyde=False, log_level="DEBUG"),
        session_id="sdk-tenant",
    )
    asyncio.run(client.config_manager.save_profile("sdk-saved", set_active=False))
    saved = next(p for p in manager._global_custom_profiles if p.name == "sdk-saved").config
    assert saved["rerank_top_k"] == 7 and "log_level" not in saved
    with pytest.raises(Exception, match="vector_backend_config"):
        asyncio.run(
            client.config_manager.save_profile(
                "sdk-evil", config={"vector_backend_config": {"url": "http://x"}}
            )
        )


def test_a_pipeline_in_this_process_still_names_its_collection(manager):
    """The evaluation harness gives an embedding arm a collection of its own."""
    from lasagna.core.pipeline import KnowledgePipeline
    from lasagna.core.strict import strict

    pipeline = KnowledgePipeline(config={"collection_name": "rag_bench_arm_1", "rerank_top_k": 3})
    with strict():
        pipeline._apply_session_config("harness-session")
    assert manager.get_resolved_config("harness-session")["collection_name"] == "rag_bench_arm_1"
