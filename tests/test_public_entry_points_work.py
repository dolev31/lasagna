"""Public entry points that crashed on first call, each now exercised once.

A basic type-check run found every one of these: the SDK called backend and
manager methods that do not exist or with arguments they do not take, so the
call failed the first time a user tried it. The mocks here are spec'd against
the real classes, so calling a method that does not exist fails the test.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lasagna.agentic.citation_manager import CitationManager as AgenticCitationManager
from lasagna.api.rag_backend import RAGBackend
from lasagna.sdk.exceptions import CitationNotFoundError
from lasagna.sdk.managers.citation_manager import CitationManager
from lasagna.sdk.managers.document_manager import DocumentManager


def _client(backend, session_id="s1"):
    client = SimpleNamespace(session_id=session_id, config=SimpleNamespace(enable_citations=True))
    client._get_backend = AsyncMock(return_value=backend)
    return client


def test_documents_get_uses_the_backend_method_that_exists():
    backend = MagicMock(spec=RAGBackend)
    backend.get_document_by_id.return_value = SimpleNamespace(
        document_id="d1",
        filename="a.pdf",
        session_id="s1",
        status="ready",
        file_path="/x/a.pdf",
        file_type="pdf",
        file_size=10,
        chunks_count=3,
        upload_date=datetime.now(timezone.utc),
    )
    doc = asyncio.run(DocumentManager(_client(backend)).get("d1"))
    assert doc is not None and doc.document_id == "d1" and doc.chunk_count == 3
    backend.get_document_by_id.assert_called_once_with("d1", "s1")


def _citation_row(session_id="s1"):
    return {
        "citation_id": "c1",
        "session_id": session_id,
        "chunk_id": "k1",
        "document_id": "d1",
        "document_name": "a.pdf",
        "page_number": 2,
        "content_preview": "text",
        "score": 0.9,
        "hop_index": 0,
    }


def test_citations_get_and_verify_work_and_stay_in_their_session():
    store = MagicMock(spec=AgenticCitationManager)
    backend = MagicMock(spec=RAGBackend)
    backend.get_document_by_id.return_value = object()
    with patch("lasagna.agentic.citation_manager.get_citation_manager", return_value=store):
        store.get_citation.return_value = _citation_row("s1")
        mine = CitationManager(_client(backend, "s1"))
        citation = asyncio.run(mine.get("c1"))
        assert citation.citation_id == "c1" and citation.filename == "a.pdf"
        assert asyncio.run(mine.verify("c1")) is True

        backend.get_document_by_id.return_value = None
        assert asyncio.run(mine.verify("c1")) is False  # cited document is gone

        other = CitationManager(_client(backend, "s2"))
        with pytest.raises(CitationNotFoundError):
            asyncio.run(other.get("c1"))
        assert asyncio.run(other.verify("c1")) is False


def test_create_agentic_rag_binds_the_llm_not_the_backend():
    from lasagna import standalone

    llm = MagicMock(name="llm")
    with (
        patch.object(standalone, "create_rag_backend") as make_backend,
        patch(
            "lasagna.providers.get_llm_provider", return_value=SimpleNamespace(get_llm=lambda: llm)
        ),
        patch("lasagna.agentic.agentic_rag.AgenticRAG") as agentic_cls,
    ):
        standalone.create_agentic_rag(
            llm_provider="ollama", rag_settings={"max_agent_iterations": 2}
        )
    make_backend.assert_called_once()
    assert agentic_cls.call_args.kwargs["llm"] is llm
    assert agentic_cls.call_args.kwargs["max_iterations"] == 2


def test_sdk_mcp_server_starts_on_the_requested_host_and_port():
    pytest.importorskip("mcp")
    from mcp.server.fastmcp import FastMCP

    from lasagna.sdk.integrations.mcp import create_mcp_server

    with patch.object(FastMCP, "run_streamable_http_async", new=AsyncMock()) as run:
        server = create_mcp_server(MagicMock())
        asyncio.run(server.start(port=8123, host="127.0.0.1"))
    run.assert_awaited_once()
    assert (server._server.settings.host, server._server.settings.port) == ("127.0.0.1", 8123)


def test_gc_audit_log_reads_the_database_instead_of_returning_nothing():
    rows = [{"run_id": "r1", "phase": "scan"}]
    db = MagicMock()
    db.get_gc_audit_log.return_value = rows
    backend = object.__new__(RAGBackend)
    with patch("lasagna.core.database.get_rag_database", return_value=db):
        assert backend.get_gc_audit_log(limit=5) == rows
    db.get_gc_audit_log.assert_called_once_with(limit=5)
