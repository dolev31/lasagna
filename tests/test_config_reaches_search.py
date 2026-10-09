"""Changing configuration must change what search does.

There were two disconnected configuration stores. `KnowledgeConfigManager`
persisted to its own SQLite-backed session state; `rag_config.with_session()` —
what the search orchestrator actually resolves through — read from
`SessionConfigProvider`. Nothing bridged them.

So `patch_config({"rerank_top_k": 17})` returned a success payload showing 17
while search kept using 5, and `set_active_profile("speed")` recorded the switch
without disabling reranking. Nine of the 31 MCP tools were inert with respect to
retrieval.

Silent success is the worst available failure shape. An error gets noticed; this
left an agent believing it had tuned retrieval, with every downstream number
quietly measuring the old configuration.

These tests assert against the *read* path, never the manager's own report —
the manager was always self-consistent, which is exactly why the split survived.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def manager_and_session(request):
    from lasagna.core.session_config import get_knowledge_config_manager

    session_id = f"cfg-reaches-{request.node.name}"
    return get_knowledge_config_manager(session_id=session_id), session_id


def _search_config(session_id):
    """Exactly what the search orchestrator resolves."""
    from lasagna.core.config import rag_config

    return rag_config.with_session(session_id)


class TestPatchReachesSearch:
    def test_a_scalar_override_applies(self, manager_and_session):
        manager, session_id = manager_and_session
        before = _search_config(session_id).rerank_top_k
        manager.patch_config({"rerank_top_k": before + 12}, session_id)
        assert _search_config(session_id).rerank_top_k == before + 12

    def test_a_boolean_toggle_applies(self, manager_and_session):
        manager, session_id = manager_and_session
        manager.patch_config({"enable_reranking": False}, session_id)
        assert _search_config(session_id).enable_reranking is False

    def test_successive_patches_accumulate(self, manager_and_session):
        manager, session_id = manager_and_session
        manager.patch_config({"rerank_top_k": 11}, session_id)
        manager.patch_config({"enable_mmr": False}, session_id)
        config = _search_config(session_id)
        assert config.rerank_top_k == 11, "the second patch dropped the first"
        assert config.enable_mmr is False


class TestProfileSwitchReachesSearch:
    def test_speed_actually_disables_reranking(self, manager_and_session):
        """The documented escape hatch from the multi-gigabyte first run."""
        manager, session_id = manager_and_session
        manager.set_active_profile("speed", session_id=session_id)
        assert _search_config(session_id).enable_reranking is False


class TestResetReachesSearch:
    def test_resetting_restores_the_default(self, manager_and_session):
        """A reset that leaves the override in force is worse than no reset."""
        manager, session_id = manager_and_session
        default = _search_config(session_id).rerank_top_k
        manager.patch_config({"rerank_top_k": default + 9}, session_id)
        assert _search_config(session_id).rerank_top_k == default + 9

        manager.reset_config(session_id=session_id)
        assert _search_config(session_id).rerank_top_k == default


class TestSessionsStayIsolated:
    def test_one_session_does_not_leak_into_another(self):
        """Publishing writes to a shared provider; scoping must survive that."""
        from lasagna.core.session_config import get_knowledge_config_manager

        manager = get_knowledge_config_manager(session_id="cfg-iso-a")
        baseline = _search_config("cfg-iso-b").rerank_top_k
        manager.patch_config({"rerank_top_k": baseline + 21}, "cfg-iso-a")

        assert _search_config("cfg-iso-a").rerank_top_k == baseline + 21
        assert _search_config("cfg-iso-b").rerank_top_k == baseline


class TestPublishingDoesNotRecurse:
    """The fix for the split had to not break the backend that was already fine.

    `PersistentSessionProvider` resolves *through* this manager, and its
    `set_config_overrides` calls `patch_config`. Publishing unconditionally from
    inside `patch_config` therefore recursed: measured at 49 nested calls and a
    RecursionError for a single user-level patch, each level persisting to
    SQLite. The provider advertises the delegation with `_delegates_to_manager`,
    which the REST config route already honoured.
    """

    def test_a_delegating_provider_is_not_written_back_to(self, manager_and_session, monkeypatch):
        manager, session_id = manager_and_session

        calls = []

        class _Delegating:
            _delegates_to_manager = True

            def set_config_overrides(self, session_id, overrides):
                calls.append(overrides)

        import lasagna.providers as providers

        monkeypatch.setattr(providers, "get_session_provider", lambda: _Delegating())
        manager.patch_config({"rerank_top_k": 8}, session_id)
        assert calls == [], "wrote back into the provider that reads from us"

    def test_a_non_delegating_provider_is_written_to(self, manager_and_session, monkeypatch):
        """The guard must not disable publishing for the default backend."""
        manager, session_id = manager_and_session

        calls = []

        class _Independent:
            def set_config_overrides(self, session_id, overrides):
                calls.append(overrides)

        import lasagna.providers as providers

        monkeypatch.setattr(providers, "get_session_provider", lambda: _Independent())
        manager.patch_config({"rerank_top_k": 8}, session_id)
        assert calls, "the default in-memory provider still needs the write-through"
        assert calls[-1].get("rerank_top_k") == 8

    def test_one_patch_is_one_patch(self, manager_and_session, monkeypatch):
        """Counts calls rather than catching RecursionError, so a merely
        expensive fan-out is caught too, not only an unbounded one."""
        manager, session_id = manager_and_session

        import lasagna.core.session_config._session_state as state_module

        original = state_module.SessionStateMixin.patch_config
        count = {"n": 0}

        def counting(self, patch, session_id=None):
            count["n"] += 1
            return original(self, patch, session_id)

        monkeypatch.setattr(state_module.SessionStateMixin, "patch_config", counting)
        manager.patch_config({"rerank_top_k": 6}, session_id)
        assert count["n"] == 1, f"one patch caused {count['n']} nested patch_config calls"


class TestPublishingIsNotFatal:
    def test_a_failed_publish_does_not_lose_the_write(self, manager_and_session, monkeypatch):
        """The config change must still be recorded even if publishing fails."""
        manager, session_id = manager_and_session

        import lasagna.providers as providers

        def boom():
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(providers, "get_session_provider", boom)
        result = manager.patch_config({"rerank_top_k": 33}, session_id)
        assert result["session_overrides"]["rerank_top_k"] == 33
