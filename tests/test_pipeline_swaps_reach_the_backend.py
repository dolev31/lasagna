"""Swapping a retriever, agentic pipeline or graph node must reach search.

KnowledgePipeline passes these as _retriever_override, _agentic_pipeline_override
and _graph_node_overrides, but RAGBackend.search did not accept them, so three
of the seven advertised swap points raised TypeError on the first query.
"""

from __future__ import annotations

from unittest.mock import create_autospec

import pytest

from lasagna.api.rag_backend import RAGBackend


@pytest.mark.parametrize(
    "name", ["_retriever_override", "_agentic_pipeline_override", "_graph_node_overrides"]
)
def test_the_backend_accepts_and_forwards_each_override(name):
    backend = object.__new__(RAGBackend)
    backend._search_orchestrator = create_autospec(
        __import__(
            "lasagna.api._search_orchestrator", fromlist=["SearchOrchestrator"]
        ).SearchOrchestrator,
        instance=True,
    )
    backend._search_orchestrator.search.return_value = []
    value = {"retrieve": object()} if name == "_graph_node_overrides" else object()
    RAGBackend.search(backend, "q", session_id="s", **{name: value})
    assert backend._search_orchestrator.search.call_args.kwargs[name] is value
