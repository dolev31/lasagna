"""The MCP `search` tool says why it found nothing when a stage knows why.

Over stdio, a session whose documents sit in another embedding model's
collection got "No relevant information found ... Try rephrasing your search",
while the reason (`documents_elsewhere`, with the commands that fix it) went to
the server's stderr, which an agent never reads. The reply now carries each
fallen-back stage, its reason and the MCP call that fixes it, in the text and
in `<search_metadata>`; results that came back short carry the same as notes.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lasagna.utils import degraded

ELSEWHERE = (
    "1 of 1 document in session 's1' is in all-MiniLM-L6-v2's collection, and this "
    "search uses BAAI/bge-small-en-v1.5's, so it cannot find it."
)


def _chunk():
    return SimpleNamespace(
        chunk_id="c0", content="tides", document_id="d1", chunk_index=0, page_number=None
    )


@pytest.fixture
def server(monkeypatch):
    from lasagna.mcp import server

    monkeypatch.setattr(server, "_resolve_session", lambda sid, token=None, **kw: ("s1", "u1"))
    for name in ("_store_citations", "_log_citation_diagnostics", "_validate_page_mapping"):
        monkeypatch.setattr(server, name, lambda *a, **k: None)
    return server


def _use(monkeypatch, server, backend):
    config = SimpleNamespace(use_agentic_rag=False, rerank_top_k=5, default_top_k=5)
    monkeypatch.setattr(server, "_prepare_search", lambda *a, **k: (backend, config, None))


def _metadata(reply: str) -> dict:
    return json.loads(reply.split("<search_metadata>")[1].split("</search_metadata>")[0])


def _missed_everything(**kw):
    degraded.record("documents_elsewhere", ELSEWHERE)
    return [], [], ""


def test_an_empty_search_names_the_reason_and_the_fix(server, monkeypatch):
    _use(monkeypatch, server, SimpleNamespace(search_with_citations=_missed_everything))

    reply = asyncio.run(server.search("tides", session_id="s1"))

    text = reply.split("<search_metadata>")[0]
    assert "Try rephrasing" not in text
    assert ELSEWHERE in text
    assert "set_active_profile with reindex=true" in text
    meta = _metadata(reply)
    assert meta["degraded_stages"] == ["documents_elsewhere"]
    assert ELSEWHERE in meta["warnings"][0]


def test_the_older_backend_path_says_it_too(server, monkeypatch):
    async def async_search(**kw):
        degraded.record("documents_elsewhere", ELSEWHERE)
        return []

    _use(monkeypatch, server, SimpleNamespace(async_search=async_search))

    reply = asyncio.run(server.search("tides", session_id="s1"))

    assert ELSEWHERE in reply and "Try rephrasing" not in reply


def test_results_that_came_back_short_carry_the_reason_as_a_note(server, monkeypatch):
    def some(**kw):
        degraded.record("documents_elsewhere", ELSEWHERE)
        return [SimpleNamespace(chunk=_chunk(), score=0.9)], [], "<knowledge_base>tides"

    _use(monkeypatch, server, SimpleNamespace(search_with_citations=some))

    reply = asyncio.run(server.search("tides", session_id="s1"))

    assert f"Note: documents_elsewhere: {ELSEWHERE}" in reply.split("<search_metadata>")[0]
    assert ELSEWHERE in _metadata(reply)["warnings"][0]


def test_a_search_that_found_nothing_for_no_known_reason_keeps_the_old_advice(server, monkeypatch):
    _use(monkeypatch, server, SimpleNamespace(search_with_citations=lambda **kw: ([], [], "")))

    reply = asyncio.run(server.search("tides", session_id="s1"))

    assert reply == server._NO_RESULTS_MSG


def test_reasons_are_collected_per_search():
    with degraded.collecting():
        degraded.record("hybrid", "BM25 index empty; dense results only")
        assert degraded.reasons() == {"hybrid": "BM25 index empty; dense results only"}
    with degraded.collecting():
        assert degraded.reasons() == {}
    assert degraded.reasons() == {}
