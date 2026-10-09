"""The default agentic graph searches under its own arm's settings.

The orchestrator hands an agentic pipeline the session config with the arm's
per-query overrides applied, but the graph's retrieve and final-search nodes
searched with ``_from_routing=True`` and nothing else, so each search re-read
the shared session store. Under a sweep, where concurrent arms write that
store, an agentic arm's hops could run with another arm's reranker or
embedding settings, and the difference would be attributed to the loop.
"""

from __future__ import annotations

from tests._loop_stubs import ArmConfig, hit, install_backend

ARM = {"reranker_model": "arm-reranker", "hybrid_alpha": 0.3}


def _state(**extra):
    return {
        "current_query": "who wrote hamlet",
        "original_question": "who wrote hamlet",
        "session_id": "arm-session",
        "top_k": 3,
        "session_config": ArmConfig(**ARM, use_final_search=True),
        **extra,
    }


async def test_a_retrieval_hop_searches_under_the_arms_settings(monkeypatch):
    from lasagna.agentic.nodes import retrieve

    monkeypatch.setattr(retrieve, "_embed_query_for_dedup", lambda q: None)
    backend = install_backend(monkeypatch, {"who wrote hamlet": [hit("c1", "Shakespeare.")]})
    out = await retrieve.retrieve_node(_state())

    (call,) = backend.calls
    assert call["_from_routing"] is True
    assert call["query_config_overrides"] == {**ARM, "use_final_search": True}
    assert [c.chunk_id for c in out["retrieved_chunks"]] == ["c1"]


async def test_the_final_search_runs_under_the_arms_settings(monkeypatch):
    from lasagna.agentic.nodes import final_search

    backend = install_backend(monkeypatch, {"who wrote hamlet": [hit("c1", "Shakespeare.")]})
    await final_search.final_search_node(_state())

    (call,) = backend.calls
    assert call["query_config_overrides"] == {**ARM, "use_final_search": True}


async def test_a_session_without_per_query_settings_searches_as_before(monkeypatch):
    from types import SimpleNamespace

    from lasagna.agentic.nodes import retrieve

    monkeypatch.setattr(retrieve, "_embed_query_for_dedup", lambda q: None)
    backend = install_backend(monkeypatch, {"who wrote hamlet": [hit("c1", "Shakespeare.")]})
    await retrieve.retrieve_node(_state(session_config=SimpleNamespace()))
    assert backend.calls[0]["query_config_overrides"] is None
