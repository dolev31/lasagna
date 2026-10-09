"""A `KnowledgeConfigManager` and REST client that touch nothing shared.

Saved profiles otherwise persist in the suite's one rag.db and leak between
tests; the vector store is stubbed so the embedding-size check reads what a
test says is stored.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import lasagna.api._profile_guard as guard


def isolated_manager(tmp_path, monkeypatch, *, stored_dim=None):
    with patch(
        "lasagna.core.session_config.KnowledgeConfigManager._resolve_user_data_path",
        return_value=str(tmp_path),
    ):
        from lasagna.core.session_config import KnowledgeConfigManager

        mgr = KnowledgeConfigManager()
    mgr._global_custom_profiles = []
    mgr._global_profile_names = set()
    for name in ("_persist_global_profiles", "_persist_session_states", "_reload_if_changed"):
        monkeypatch.setattr(mgr, name, lambda *a, **k: None)
    monkeypatch.setattr(
        "lasagna.core.session_config.get_knowledge_config_manager", lambda *a, **k: mgr
    )
    monkeypatch.setattr(
        "lasagna.api.rest.routes.knowledge_config.get_knowledge_config_manager",
        lambda *a, **k: mgr,
    )
    stored_at(monkeypatch, stored_dim)
    return mgr


def stored_at(monkeypatch, dim):
    """Documents are stored at ``dim`` (None: nothing stored)."""
    monkeypatch.setattr(guard, "stored_embedding_dim", lambda collection, vector_backend=None: dim)


def rest_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from lasagna.api.rest.dependencies import get_backend
    from lasagna.api.rest.routes.config import router as config_router
    from lasagna.api.rest.routes.knowledge_config import router as knowledge_router
    from lasagna.core.config import rag_config

    app = FastAPI()
    app.include_router(config_router, prefix="/api/v1")
    app.include_router(knowledge_router)
    app.dependency_overrides[get_backend] = lambda: SimpleNamespace(
        get_session_config=rag_config.with_session
    )
    return TestClient(app)
