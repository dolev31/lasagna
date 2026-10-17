"""A reindex that fails leaves the document as it was, searchable.

Every reindex path deleted the document before it could store the new copy:
REST ``/documents/{id}/reindex`` and the bulk ``/api/knowledge/reindex`` called
``delete_document`` (row and vectors) and then ``upload_document``, and the MCP
tool deleted the chunks and then processed the file again. Anything that failed
in between lost the document: a model that was not downloaded after a model
switch, an embedding size that could not be read, a quota, a file type or size
the settings no longer accept, an extraction or insert error. The bulk route
reported "partial" and kept going, and the files stayed on disk with nothing
pointing to them. The MCP tool left the row "ready" with no vectors.
"""

from __future__ import annotations

import asyncio

import pytest
from fastapi import HTTPException

from lasagna.api.rest.routes import documents as documents_route
from lasagna.api.rest.routes import reindex as reindex_route
from lasagna.core.quota import QuotaManager, TenantQuota
from lasagna.mcp import server
from tests._document_harness import MODEL_B, SESSION, Harness

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 3


@pytest.fixture
def harness(tmp_path, monkeypatch):
    def make(**kwargs):
        h = Harness(tmp_path, monkeypatch, **kwargs)
        monkeypatch.setattr(server, "_get_backend", lambda: h.backend)
        doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))
        assert h.searchable(doc.document_id)
        h.document_id = doc.document_id
        h.original = h.texts(doc.document_id)
        return h

    return make


def _rest(h):
    try:
        asyncio.run(
            documents_route.reindex_document(h.document_id, session_id=SESSION, backend=h.backend)
        )
    except HTTPException as exc:
        return exc
    return None


def _bulk(h):
    return reindex_route._reindex_sync(h.backend, SESSION)


def _mcp(h):
    return server.reindex_document(h.document_id, session_id=SESSION)


ROUTES = {"rest": _rest, "bulk": _bulk, "mcp": _mcp}


def _model_not_downloaded(h):
    h.settings = dict(MODEL_B)
    h.collections.not_downloaded.add(MODEL_B["embedding_model"])


def _size_unknown(h):
    h.settings = {"embedding_model": "someone/an-unknown-model"}


def _chunk_quota(h):
    h.quota.set_tenant_quota(SESSION, TenantQuota(max_chunks=len(h.original) + 1))
    h.chunk_words = 2


def _type_refused(h):
    h.settings = {**h.settings, "supported_extensions": [".pdf"]}


def _size_refused(h):
    h.settings = {**h.settings, "max_file_size": 10}


def _extraction_fails(h):
    h.fail_chunking = RuntimeError("the extractor crashed")


def _insert_fails(h):
    h.chunk_words = 2
    h.vectors.fail_insert = ConnectionError("the vector store went away")


FAILURES = {
    "model_not_downloaded": _model_not_downloaded,
    "embedding_size_unknown": _size_unknown,
    "chunk_quota": _chunk_quota,
    "type_refused": _type_refused,
    "size_refused": _size_refused,
    "extraction_fails": _extraction_fails,
    "insert_fails": _insert_fails,
}


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("failure", FAILURES)
def test_a_failed_reindex_leaves_the_document_searchable(harness, route, failure):
    h = harness(quota=QuotaManager(enabled=True, default_quota=TenantQuota()))
    files_before = sorted(p.name for p in h.uploads.rglob("*") if p.is_file())
    FAILURES[failure](h)

    outcome = ROUTES[route](h)

    if route == "rest":
        assert isinstance(outcome, HTTPException) and outcome.status_code >= 400
    elif route == "bulk":
        assert outcome["status"] == "partial" and outcome["documents_reindexed"] == 0
        assert h.document_id in outcome["errors"][0]
    else:
        assert outcome.startswith("Failed to reindex document")
    assert h.searchable(h.document_id), (h.row(h.document_id), h.chunks(h.document_id))
    assert h.texts(h.document_id) == h.original
    assert sorted(p.name for p in h.uploads.rglob("*") if p.is_file()) == files_before


@pytest.mark.parametrize("route", ROUTES)
def test_a_reindex_over_the_hourly_upload_limit_is_not_an_upload(harness, route):
    """It used to delete the document, then be refused as the hour's second upload."""
    h = harness(quota=QuotaManager(enabled=True, default_quota=TenantQuota(max_uploads_per_hour=1)))
    h.chunk_words = 2
    ROUTES[route](h)
    assert h.searchable(h.document_id)
    assert h.texts(h.document_id) != h.original


@pytest.mark.parametrize("route", ROUTES)
def test_a_reindex_after_a_model_switch_moves_the_document(harness, route):
    h = harness()
    old_collection = h.row(h.document_id)["collection_name"]
    h.settings = dict(MODEL_B)

    ROUTES[route](h)

    row = h.row(h.document_id)
    assert row["collection_name"] == h.config().collection_name != old_collection
    assert h.searchable(h.document_id)
    assert h.chunks(h.document_id, old_collection) == []


@pytest.mark.parametrize("route", ROUTES)
def test_a_reindex_in_the_same_collection_replaces_every_chunk(harness, route):
    h = harness()
    h.chunk_words = 2
    ROUTES[route](h)
    assert h.searchable(h.document_id)
    chunks = h.chunks(h.document_id)
    assert len({c["chunk_id"] for c in chunks}) == len(chunks) > len(h.original)


def test_the_quota_counts_the_reindexed_document_once(harness):
    h = harness(quota=QuotaManager(enabled=True, default_quota=TenantQuota()))
    before = h.quota.get_usage(SESSION)
    h.chunk_words = 2
    h.pipeline.reindex_document(h.document_id, SESSION)
    after = h.quota.get_usage(SESSION)
    assert after["documents"] == before["documents"] == 1
    assert after["storage_bytes"] == before["storage_bytes"]
    assert after["chunks"] == h.row(h.document_id)["chunks_count"] == len(h.chunks(h.document_id))


def test_a_document_still_processing_is_not_reindexed(harness):
    h = harness()
    h.db.update_document_status(h.document_id, "processing")
    from lasagna.api._document_pipeline import DocumentBusy

    with pytest.raises(DocumentBusy, match="still being processed"):
        h.pipeline.reindex_document(h.document_id, SESSION)
    assert h.texts(h.document_id) == h.original
