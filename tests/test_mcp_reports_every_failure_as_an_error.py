"""Every MCP tool failure reaches the client as an error, not as content.

Reproduced over stdio, eleven tools answered a failure with isError=False, so
an agent read it as retrieved knowledge or as success: a document or profile
that does not exist, a cancelled clear_session, focus on ids that do not exist
(after which every search in the session quietly returned nothing), an upload
whose processing failed, an empty query, a negative top_k or offset, and a
document-context lookup that raised. REST answers each of these with a 4xx.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

pytest.importorskip("mcp")

from mcp.shared.memory import create_connected_server_and_client_session  # noqa: E402

from lasagna.mcp import server as mcp_server  # noqa: E402

_DOC = SimpleNamespace(
    document_id="d1",
    session_id="s1",
    filename="empty.txt",
    file_type="txt",
    file_size=0,
    upload_date=datetime.now(timezone.utc),
    chunks_count=0,
    processing_time=0.0,
    text_length=0,
    error_message="the file has no text",
    status=SimpleNamespace(value="error"),
)


class _Backend:
    def get_documents(self, session_id):
        return [_DOC]

    def get_document_chunks(self, **kwargs):
        return []

    def delete_document(self, document_id, session_id):
        return False

    def upload_document(self, path, session_id, filename=None):
        return _DOC

    def get_document_by_id(self, document_id, session_id):
        return _DOC


@pytest.fixture
def call(monkeypatch, tmp_path):
    monkeypatch.setattr(mcp_server, "_get_backend", lambda: _Backend())
    # A networked transport reads server-side files only inside this root.
    monkeypatch.setenv("RAG_MCP_UPLOAD_ROOT", str(tmp_path))

    def go(tool, arguments):
        async def run():
            # "all": delete_document, delete_rag_profile and clear_session are
            # not in the default safe set, and "Unknown tool" is not the error
            # under test.
            srv = mcp_server.create_mcp_server(host="127.0.0.1", port=8999, tools="all")
            async with create_connected_server_and_client_session(srv._mcp_server) as client:
                return await client.call_tool(tool, arguments)

        return asyncio.run(run())

    return go


@pytest.mark.parametrize(
    "tool, arguments",
    [
        ("get_document_info", {"document_id": "nope", "session_id": "s1"}),
        ("get_document_chunks", {"document_id": "nope", "session_id": "s1"}),
        ("get_document_chunks", {"document_id": "d1", "session_id": "s1", "offset": -1}),
        ("get_document_chunks", {"document_id": "d1", "session_id": "s1", "limit": 0}),
        ("delete_document", {"document_id": "nope", "session_id": "s1"}),
        ("delete_rag_profile", {"session_id": "s1", "profile_name": "never_saved"}),
        ("reindex_document", {"document_id": "nope", "session_id": "s1"}),
        ("clear_session", {"session_id": "s1"}),
        ("set_focused_documents", {"session_id": "s1", "document_ids": "d1,ghost"}),
        ("search", {"query": "  ", "session_id": "s1"}),
        ("search", {"query": "tides", "session_id": "s1", "top_k": -3}),
    ],
)
def test_a_failure_is_an_mcp_error(call, tool, arguments):
    result = call(tool, arguments)
    assert result.isError is True, result.content[0].text


def test_an_upload_whose_processing_failed_is_an_error_with_the_reason(call, tmp_path):
    source = tmp_path / "empty.txt"
    source.write_text("")
    result = call("upload_document", {"file_path": str(source), "session_id": "s1"})
    assert result.isError is True
    assert "the file has no text" in result.content[0].text


def test_a_document_context_lookup_that_raises_is_an_error(call, monkeypatch):
    async def broken(*args, **kwargs):
        raise RuntimeError("store unavailable")

    monkeypatch.setattr("lasagna.agentic.document_awareness.get_document_awareness_context", broken)
    result = call("get_document_context", {"session_id": "s1"})
    assert result.isError is True
    assert "store unavailable" in result.content[0].text


def test_focus_on_unknown_ids_changes_nothing(call):
    from lasagna.core.session_config import get_knowledge_config_manager

    manager = get_knowledge_config_manager()
    manager.set_focused_document_ids(["d1"], "s1")
    call("set_focused_documents", {"session_id": "s1", "document_ids": "ghost"})
    assert manager.get_focused_document_ids("s1") == ["d1"]
    manager.set_focused_document_ids([], "s1")


def test_a_document_with_no_chunks_is_an_answer_not_an_error(call):
    result = call("get_document_chunks", {"document_id": "d1", "session_id": "s1"})
    assert result.isError is False
    assert '"total_chunks": 0' in result.content[0].text
