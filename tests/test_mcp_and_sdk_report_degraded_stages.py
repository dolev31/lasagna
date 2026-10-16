"""MCP and the SDK report which search stages fell back, as the REST API does.

A REST search names its fallbacks (hybrid running dense-only, HyDE without an
LLM) in ``degraded_stages``. The MCP ``search`` tool and the SDK's
``SearchResults`` dropped them, so an agent or a script saw a well-formed
result that was not the configured pipeline's.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lasagna.utils import degraded


def _chunk(i=0):
    return SimpleNamespace(
        chunk_id=f"c{i}",
        content="tide tables",
        document_id="d1",
        chunk_index=i,
        page_number=None,
        metadata={},
    )


@pytest.fixture
def mcp(monkeypatch):
    from lasagna.mcp import server

    def search_with_citations(**kw):
        if fallback:
            degraded.record("hybrid", "BM25 index empty; dense results only")
        return [SimpleNamespace(chunk=_chunk(), score=0.9)], [], "<knowledge_base>tides"

    fallback = []
    backend = SimpleNamespace(search_with_citations=search_with_citations)
    config = SimpleNamespace(use_agentic_rag=False, rerank_top_k=5)
    monkeypatch.setattr(server, "_resolve_session", lambda sid, token=None, **kw: ("s1", "u1"))
    monkeypatch.setattr(server, "_prepare_search", lambda *a, **k: (backend, config, None))
    for name in ("_store_citations", "_log_citation_diagnostics", "_validate_page_mapping"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    return server, fallback


def _metadata(reply: str) -> dict:
    return json.loads(reply.split("<search_metadata>")[1].split("</search_metadata>")[0])


def test_the_mcp_search_names_a_stage_that_fell_back(mcp):
    server, fallback = mcp
    fallback.append(True)

    reply = asyncio.run(server.search("tides", session_id="s1"))

    assert _metadata(reply)["degraded_stages"] == ["hybrid"]


def test_a_search_that_ran_as_configured_names_none(mcp):
    server, _ = mcp

    reply = asyncio.run(server.search("tides", session_id="s1"))

    assert "degraded_stages" not in _metadata(reply)


def _client(backend):
    async def get_backend():
        return backend

    return SimpleNamespace(
        _get_backend=get_backend,
        session_id="s1",
        focused_document_ids=None,
        config=SimpleNamespace(
            enable_reranking=True,
            enable_hyde=True,
            enable_hybrid_search=True,
            hybrid_alpha=0.5,
            enable_citations=False,
        ),
    )


def test_the_sdk_search_results_name_a_stage_that_fell_back():
    from lasagna.sdk.managers.search_manager import SearchManager

    async def async_search(**kw):
        def in_a_thread():
            degraded.record("hyde", "no hypothetical document; searched the query")
            raw_doc = SimpleNamespace(document_id="d1", filename="tides.txt")
            return [SimpleNamespace(chunk=_chunk(), score=0.8, document=raw_doc)]

        return await asyncio.to_thread(in_a_thread)

    manager = SearchManager(_client(SimpleNamespace(async_search=async_search)))

    results = asyncio.run(manager._search_internal("tides"))

    assert results.degraded_stages == ["hyde"]
    assert len(results) == 1


def test_a_remote_search_passes_on_the_servers_stages():
    from lasagna.sdk.remote_backend import RemoteRAGBackend

    body = {"results": [], "degraded_stages": ["hybrid"]}
    response = SimpleNamespace(raise_for_status=lambda: None, json=lambda: body)

    async def post(url, json=None):
        return response

    backend = RemoteRAGBackend.__new__(RemoteRAGBackend)
    backend._client = SimpleNamespace(post=post, is_closed=True)
    backend._api = lambda path: path
    backend._parse_json = lambda resp: resp.json()

    async def run():
        with degraded.collecting() as stages:
            await backend._async_search("tides", "s1")
        return stages

    assert asyncio.run(run()) == ["hybrid"]
