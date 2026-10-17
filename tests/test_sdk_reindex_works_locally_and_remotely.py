"""``client.documents.reindex()`` reindexes, on a local client and a remote one.

It called ``backend.reindex_document``, which neither ``RAGBackend`` nor
``RemoteRAGBackend`` had, so every call failed, and it returned an
``UploadResult`` with status FAILED instead of raising, so a script that did
not read the status carried on as if the document had been reindexed. The SDK
user guide documents it.
"""

from __future__ import annotations

import asyncio
import functools
from types import SimpleNamespace

import httpx
import pytest

from lasagna.api.rest.app import create_app
from lasagna.api.rest.dependencies import get_backend
from lasagna.sdk import KnowledgeClient, remote_backend
from lasagna.sdk.exceptions import DocumentError, DocumentNotFoundError
from lasagna.sdk.managers.document_manager import DocumentManager
from tests._document_harness import MODEL_B, SESSION, Harness

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 2


@pytest.fixture
def h(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    harness.document_id = harness.upload(harness.write(tmp_path, "paper.txt", TEXT)).document_id
    harness.old_collection = harness.row(harness.document_id)["collection_name"]
    return harness


@pytest.fixture(params=["local", "remote"])
def reindex(request, h, monkeypatch):
    """``client.documents.reindex`` on a local client, or on one talking to the REST app."""
    if request.param == "local":

        async def backend():
            return h.backend

        manager = DocumentManager(SimpleNamespace(_get_backend=backend, session_id=SESSION))
        return lambda document_id: asyncio.run(manager.reindex(document_id))

    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    app = create_app()
    app.dependency_overrides[get_backend] = lambda: h.backend
    monkeypatch.setattr(
        remote_backend.httpx,
        "AsyncClient",
        functools.partial(httpx.AsyncClient, transport=httpx.ASGITransport(app=app)),
    )

    def run(document_id):
        async def go():
            client = KnowledgeClient.from_url("http://testserver", session_id=SESSION)
            try:
                return await client.documents.reindex(document_id)
            finally:
                await client.close()

        return asyncio.run(go())

    return run


def test_reindex_moves_the_document_to_the_new_model(h, reindex):
    h.settings = dict(MODEL_B)
    result = reindex(h.document_id)
    assert result.document_id == h.document_id
    assert result.chunk_count == h.row(h.document_id)["chunks_count"] > 0
    assert h.row(h.document_id)["collection_name"] == h.config().collection_name
    assert h.chunks(h.document_id, h.old_collection) == []
    assert h.searchable(h.document_id)


def test_a_failed_reindex_raises_and_leaves_the_document(h, reindex):
    h.settings = dict(MODEL_B)
    h.collections.not_downloaded.add(MODEL_B["embedding_model"])
    with pytest.raises(DocumentError) as raised:
        reindex(h.document_id)
    assert not isinstance(raised.value, DocumentNotFoundError)
    assert MODEL_B["embedding_model"] in str(raised.value)
    assert "unchanged" in str(raised.value)
    assert h.searchable(h.document_id)
    assert h.row(h.document_id)["collection_name"] == h.old_collection


def test_reindexing_an_unknown_document_raises_not_found(h, reindex):
    with pytest.raises(DocumentNotFoundError):
        reindex("doc_never_uploaded")
