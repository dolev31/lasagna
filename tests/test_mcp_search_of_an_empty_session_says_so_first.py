"""MCP `search` on a session with no documents says so before anything about models.

Session `nope` got "The embedding model BAAI/bge-m3 is not downloaded..." and
only then that the session has no documents. The CLI and REST already check
the session first; the MCP tool now does too, before any model loads.
"""

from __future__ import annotations

import asyncio

import pytest

from lasagna.core import model_downloads
from lasagna.mcp import server


@pytest.fixture
def stubs(monkeypatch):
    monkeypatch.setattr(server, "_resolve_session", lambda sid, token=None, **kw: (sid, "u1"))

    def loads_a_model(*a, **k):
        pytest.fail("a model was loaded for a session with no documents")

    monkeypatch.setattr(server, "_get_backend", loads_a_model)


def test_an_unknown_session_is_reported_empty_before_a_model_loads(stubs, monkeypatch):
    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {"my-corpus": 3})

    reply = asyncio.run(server.search("tides", session_id="nope"))

    assert reply.startswith("No results. Session 'nope' has no documents")
    assert "my-corpus (3)" in reply
    assert "session_id argument" in reply
    assert "not downloaded" not in reply


def test_no_session_with_documents_says_how_to_add_one(stubs, monkeypatch):
    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {})

    reply = asyncio.run(server.search("tides", session_id="nope"))

    assert "Session 'nope' has no documents" in reply and "upload_document" in reply
