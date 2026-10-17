"""With auth on, quotas count per API key, and a key mints a bounded number of sessions.

Document, storage, chunk and search quotas were kept per session, and a session
cost nothing: one API key minted as many as it liked, each with a fresh
allowance, and could fill the uploads volume and the vector store. A session
minted with a key is now recorded with that key, in the metadata store every
process reads (API workers, the Celery worker, garbage collection), and its
usage counts against the key's quota summed over its sessions. A key mints at
most ``RAG_MAX_SESSIONS_PER_KEY`` sessions per token lifetime. With auth off
(loopback, one user) nothing changes.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lasagna.api.rest import auth_mode, session_token
from lasagna.core.database import RAGDatabase
from lasagna.core.quota import QuotaExceededError, QuotaManager, TenantQuota

KEY = "k" * 40
OTHER_KEY = "o" * 40
ADMIN = "a" * 40


@pytest.fixture
def db(tmp_path, monkeypatch):
    database = RAGDatabase(str(tmp_path / "rag.db"))
    monkeypatch.setattr("lasagna.core.database.get_rag_database", lambda *a: database)
    return database


@pytest.fixture
def auth(monkeypatch, db):
    monkeypatch.setenv("RAG_JWT_SECRET", "s" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")
    monkeypatch.setenv(auth_mode.API_KEYS_ENV, f"{KEY},{OTHER_KEY}")
    monkeypatch.setenv("RAG_ADMIN_TOKEN", ADMIN)


def _mint(client, **headers):
    return client.post("/api/v1/sessions", headers=headers, json={})


def _session(client, key=KEY):
    response = _mint(client, **{"X-API-Key": key})
    assert response.status_code == 201, response.text
    return response.json()["session_id"]


@pytest.fixture
def client(auth):
    from lasagna.api.rest.app import create_app

    return TestClient(create_app())


def test_one_key_shares_one_quota_across_its_sessions(client):
    qm = QuotaManager(enabled=True, default_quota=TenantQuota(max_documents=2))
    first, second, third = (_session(client) for _ in range(3))
    qm.check_and_reserve_upload(first)
    qm.check_and_reserve_upload(second)
    with pytest.raises(QuotaExceededError, match="API key"):
        qm.check_and_reserve_upload(third)
    assert qm.get_usage(third)["documents"] == 2


def test_another_key_has_its_own_quota(client):
    qm = QuotaManager(enabled=True, default_quota=TenantQuota(max_documents=1))
    qm.check_and_reserve_upload(_session(client))
    qm.check_and_reserve_upload(_session(client, OTHER_KEY))


def test_a_key_mints_a_bounded_number_of_sessions(client, monkeypatch):
    monkeypatch.setenv("RAG_MAX_SESSIONS_PER_KEY", "2")
    _session(client)
    _session(client)
    refused = _mint(client, **{"X-API-Key": KEY})
    assert refused.status_code == 429
    assert "RAG_MAX_SESSIONS_PER_KEY" in refused.json()["detail"]
    _session(client, OTHER_KEY)  # another key is not affected
    assert _mint(client, **{"X-Admin-Token": ADMIN}).status_code == 201  # nor the operator


def test_garbage_collection_rebuilds_the_keys_counters(client, db, monkeypatch):
    from lasagna.lifecycle.gc import GarbageCollector

    first, second = _session(client), _session(client)
    for sid, doc in ((first, "d1"), (second, "d2")):
        db.save_document(
            {
                "document_id": doc,
                "session_id": sid,
                "filename": f"{doc}.txt",
                "file_size": 10,
                "file_type": ".txt",
                "status": "ready",
                "chunks_count": 3,
            },
            file_path="",
        )
    qm = QuotaManager(enabled=True, default_quota=TenantQuota())
    import lasagna.lifecycle.gc as gc_mod

    monkeypatch.setattr(gc_mod, "get_quota_manager", lambda: qm)
    collector = GarbageCollector.__new__(GarbageCollector)
    collector._db = db
    collector._phase6_reconcile_quotas()
    usage = qm.get_usage(first)
    assert (usage["documents"], usage["storage_bytes"], usage["chunks"]) == (2, 20, 6)


def test_with_auth_off_each_session_has_its_own_quota(db, monkeypatch):
    monkeypatch.delenv(auth_mode.REQUIRE_AUTH_ENV, raising=False)
    db.bind_session_principal("s1", "key_abc")
    db.bind_session_principal("s2", "key_abc")
    qm = QuotaManager(enabled=True, default_quota=TenantQuota(max_documents=1))
    qm.check_and_reserve_upload("s1")
    qm.check_and_reserve_upload("s2")


@pytest.mark.skipif(not __import__("os").environ.get("DATABASE_DSN"), reason="needs DATABASE_DSN")
def test_on_a_live_postgres_the_session_limit_holds():
    import os
    import uuid

    from lasagna.core.database_pg import PostgresRAGDatabase

    db = PostgresRAGDatabase(os.environ["DATABASE_DSN"], pool_min=1, pool_max=2)
    principal = f"key_{uuid.uuid4().hex[:12]}"
    try:
        assert db.bind_session_principal("s-a-" + principal, principal, limit=2)
        assert db.bind_session_principal("s-b-" + principal, principal, limit=2)
        assert not db.bind_session_principal("s-c-" + principal, principal, limit=2)
        assert db.session_principal("s-a-" + principal) == principal
        assert db.session_principal("s-c-" + principal) is None
    finally:
        db.close()
