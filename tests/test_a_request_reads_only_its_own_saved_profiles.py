"""A request reads its own saved profiles, and one API key has one set of them.

Every request on PostgreSQL built a manager that loaded every tenant's saved
profiles, again each second, and the owner of a session's profiles was a random
``user_<id>`` per minted session, so the limit of saved profiles applied per
session: one API key could store 50 profiles in each of its sessions, with no
end. A manager now loads the caller's and the shared profiles, and with auth on
the sessions one API key mints share the key's owner. Anonymous is never handed
another owner's profile of a name.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from lasagna.api.rest import auth_mode, session_token
from lasagna.core.session_config import MAX_GLOBAL_PROFILES
from lasagna.core.session_config._data_models import ANONYMOUS
from tests._postgres_store import FakePostgres, manager_class_on

CONFIG = {"rerank_top_k": 3}


def _save(manager_class, owner, name, session="s"):
    manager_class(user_id=owner, session_id=session).save_profile(
        name, CONFIG, session_id=session, user_id=owner
    )


def test_a_manager_reads_the_callers_profiles_and_the_shared_ones_only(monkeypatch):
    store = FakePostgres()
    on_postgres = manager_class_on(monkeypatch, store)
    for owner in ("alice", "bob", ANONYMOUS):
        _save(on_postgres, owner, f"{owner}-profile", session=f"s-{owner}")
    store.profile_owners_read.clear()

    alice = on_postgres(user_id="alice", session_id="s-alice")
    alice._refresh_saved_profiles(force=True)
    names = {p.name for p in alice._global_custom_profiles}
    assert "bob-profile" not in names
    assert {"alice-profile", f"{ANONYMOUS}-profile"} <= names
    assert store.profile_owners_read
    assert all(o is not None and "bob" not in o for o in store.profile_owners_read)


def test_anonymous_is_not_handed_another_owners_profile(monkeypatch):
    store = FakePostgres()
    on_postgres = manager_class_on(monkeypatch, store)
    _save(on_postgres, "bob", "bobs")
    anonymous = on_postgres(user_id=ANONYMOUS, session_id="s2")
    assert anonymous._saved_for("bobs", ANONYMOUS) is None
    assert anonymous._saved_for("bobs", "bob") is not None


def test_the_limit_of_saved_profiles_is_counted_for_the_owner(monkeypatch):
    store = FakePostgres()
    on_postgres = manager_class_on(monkeypatch, store)
    for i in range(MAX_GLOBAL_PROFILES):
        _save(on_postgres, "key_1", f"p{i}", session=f"s{i}")
    with pytest.raises(ValueError, match="(?i)maximum"):
        _save(on_postgres, "key_1", "one-more", session="s-new")
    _save(on_postgres, "key_2", "mine", session="s-other")


KEYS = ("k" * 40, "o" * 40)


@pytest.fixture
def client(monkeypatch, tmp_path):
    from lasagna.core.database import RAGDatabase

    database = RAGDatabase(str(tmp_path / "rag.db"))
    monkeypatch.setattr("lasagna.core.database.get_rag_database", lambda *a: database)
    monkeypatch.setenv("RAG_JWT_SECRET", "s" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    monkeypatch.setenv(auth_mode.REQUIRE_AUTH_ENV, "true")
    monkeypatch.setenv(auth_mode.API_KEYS_ENV, ",".join(KEYS))
    from lasagna.api.rest.app import create_app

    return TestClient(create_app())


def _user_of(client, key):
    response = client.post("/api/v1/sessions", headers={"X-API-Key": key}, json={})
    assert response.status_code == 201, response.text
    return response.json()["user_id"]


def test_the_sessions_one_key_mints_share_one_owner_and_another_key_has_its_own(client):
    first, second = _user_of(client, KEYS[0]), _user_of(client, KEYS[0])
    other = _user_of(client, KEYS[1])
    assert first == second and first.startswith(auth_mode.KEY_PRINCIPAL_PREFIX)
    assert other != first
    assert KEYS[0] not in first  # a digest, never the key
