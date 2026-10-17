"""A store lock held past its wait is a 503 with the reason on the configuration routes.

``StoreLockTimeout`` has a 503 handler on the app, but every
``/api/config/knowledge`` route (and ``PATCH /api/v1/config``) caught it as any
other error first and answered 500 "Internal server error".
"""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lasagna.core.session_config import KnowledgeConfigManager
from lasagna.core.store_locks import StoreLockTimeout, lock_timeout_message

BUSY = lock_timeout_message("session:r1", 300)


@pytest.fixture
def client(monkeypatch):
    from lasagna.api.rest.app import _answer_switch_in_progress
    from lasagna.api.rest.dependencies import get_backend
    from lasagna.api.rest.routes.config import router as v1
    from lasagna.api.rest.routes.knowledge_config import router

    def busy(self, *args, **kwargs):
        raise StoreLockTimeout(BUSY)

    for name in ("patch_config", "save_profile", "delete_profile", "reset_config", "import_config"):
        monkeypatch.setattr(KnowledgeConfigManager, name, busy)
    monkeypatch.setattr(KnowledgeConfigManager, "set_active_profile", busy)
    app = FastAPI()
    _answer_switch_in_progress(app)
    app.include_router(router)
    app.include_router(v1, prefix="/api/v1")
    app.dependency_overrides[get_backend] = lambda: None
    return TestClient(app)


@pytest.mark.parametrize(
    "method, path, body",
    [
        ("put", "/api/config/knowledge/active", {"active_profile": "speed"}),
        ("post", "/api/config/knowledge/profile", {"profile_name": "mine", "config": {}}),
        ("post", "/api/config/knowledge/profile/delete", {"profile_name": "mine"}),
        ("post", "/api/config/knowledge/reset", {"scope": "active"}),
        ("post", "/api/config/knowledge/import", {"schema_version": 2, "profiles": []}),
        ("patch", "/api/v1/config", {"rerank_top_k": 4}),
    ],
)
def test_each_route_answers_503_with_the_reason(client, method, path, body):
    response = getattr(client, method)(path, json=body, headers={"X-Session-ID": "r1"})
    assert response.status_code == 503, response.text
    assert response.json()["detail"] == BUSY
    assert response.headers["Retry-After"] == "30"
