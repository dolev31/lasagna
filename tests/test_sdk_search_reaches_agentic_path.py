"""The SDK's agentic search must reach the agent, not quietly degrade.

`SearchManager._search_internal` is a coroutine on the caller's event loop. It
used to call the synchronous `backend.search()`, whose agentic branch refuses to
run inside a running loop -- and that refusal sat inside the branch's own broad
`except`, which served standard results instead. Every agentic search through
`KnowledgeClient`, the LangChain retriever and `KnowledgeEvaluator` was standard
retrieval under an agentic label, with one ERROR log line as the only trace.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lasagna.api._search_orchestrator import SearchOrchestrator
from lasagna.sdk.managers.search_manager import SearchManager


def _fake_client(backend):
    config = SimpleNamespace(
        enable_reranking=False,
        enable_hyde=False,
        enable_hybrid_search=False,
        hybrid_alpha=0.5,
        enable_citations=False,
    )
    client = SimpleNamespace(
        session_id="s1",
        config=config,
        focused_document_ids=None,
    )
    client._get_backend = AsyncMock(return_value=backend)
    return client


def test_sdk_search_awaits_the_async_backend_path():
    backend = MagicMock()
    backend.async_search = AsyncMock(return_value=[])
    backend.search = MagicMock(side_effect=AssertionError("sync search() called on the event loop"))

    manager = SearchManager(_fake_client(backend))
    asyncio.run(manager._search_internal("what changed?", routing_tier="comprehensive"))

    backend.search.assert_not_called()
    backend.async_search.assert_awaited_once()
    assert backend.async_search.await_args.kwargs["routing_tier"] == "comprehensive"


def test_agentic_search_inside_a_running_loop_raises_instead_of_falling_back():
    orchestrator = MagicMock(spec=SearchOrchestrator)
    orchestrator._get_session_config.return_value = SimpleNamespace(max_agent_iterations=1, rerank_top_k=5)

    async def call_from_inside_a_loop():
        return SearchOrchestrator._search_comprehensive(
            orchestrator,
            query="q",
            session_id="s1",
            top_k=5,
            min_score=0.0,
            document_ids=None,
            trace_id=None,
            routing_decision=None,
        )

    with patch("lasagna.api._search_orchestrator.get_quota_manager"):
        with pytest.raises(RuntimeError, match="async_search"):
            asyncio.run(call_from_inside_a_loop())

    # The old behavior: the violation was caught and standard search served.
    orchestrator.search.assert_not_called()
