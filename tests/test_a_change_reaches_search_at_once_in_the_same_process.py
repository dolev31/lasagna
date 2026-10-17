"""A change saved through one manager reaches the others of the process at once.

With PostgreSQL every request gets a manager of its own, and the session
provider (which search reads) keeps one for the whole process. A manager
compares its copy of a session with the store at most once a second, so a REST
patch returned and the next search in the same process, within that second,
ran with the old settings. Changes made in this process are now seen by every
manager of it at once; another process's changes still take up to a second.
"""

from __future__ import annotations

import uuid

import pytest

from lasagna.core.session_config import KnowledgeConfigManager, _session_state


@pytest.fixture
def sid():
    return f"at-once-{uuid.uuid4().hex[:8]}"


@pytest.fixture(autouse=True)
def _default_recheck(monkeypatch):
    monkeypatch.setattr(_session_state, "_RECHECK_SECONDS", 1.0)


def test_a_patch_reaches_a_manager_that_read_the_session_just_before(sid):
    search_side, request_side = KnowledgeConfigManager(), KnowledgeConfigManager()
    before = search_side.get_resolved_config(sid)["rerank_top_k"]

    request_side.patch_config({"rerank_top_k": before + 12}, session_id=sid)

    assert search_side.get_resolved_config(sid)["rerank_top_k"] == before + 12


def test_a_profile_switch_and_a_reset_reach_it_too(sid):
    search_side, request_side = KnowledgeConfigManager(), KnowledgeConfigManager()
    assert search_side.get_active_profile(sid) == "default"

    request_side.set_active_profile("speed", session_id=sid, force=True)
    assert search_side.get_active_profile(sid) == "speed"

    request_side.patch_config({"rerank_top_k": 13}, session_id=sid)
    assert search_side.get_resolved_config(sid)["rerank_top_k"] == 13
    request_side.reset_config(session_id=sid)
    assert "rerank_top_k" not in search_side.get_session_state(sid).session_overrides


def test_search_reads_it_through_the_session_provider(sid, monkeypatch):
    from lasagna.core.config import rag_config
    from lasagna.providers import get_session_provider

    provider = get_session_provider()
    if not hasattr(provider, "_manager"):
        pytest.skip("the persistent session store is off in this run")
    monkeypatch.setattr(provider, "_manager", KnowledgeConfigManager())
    before = rag_config.with_session(sid).rerank_top_k

    KnowledgeConfigManager().patch_config({"rerank_top_k": before + 9}, session_id=sid)

    assert rag_config.with_session(sid).rerank_top_k == before + 9
