"""Uploading a file again whose earlier upload failed retries that document.

Each offline rerun of the README quickstart added another failed copy of
paper.pdf: `lasagna status` said "Documents: 3" after three runs, while the
README says a rerun uploads nothing new. The duplicate check skipped failed
documents, which is how a failure is retried, and every retry was a new
document. A file whose only copies in the session have failed now replaces
the newest of them under the same id, whichever model it was tried with, on
every surface (SDK, CLI, REST and MCP all upload through the same pipeline).
"""

from __future__ import annotations

import asyncio
import io

import pytest
from fastapi import UploadFile

from lasagna.core.quota import QuotaManager, TenantQuota
from tests._document_harness import MODEL_B, SESSION, Harness

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 3


@pytest.fixture
def h(tmp_path, monkeypatch):
    return Harness(
        tmp_path, monkeypatch, quota=QuotaManager(enabled=True, default_quota=TenantQuota())
    )


def _ids(h):
    return [r["document_id"] for r in h.db.get_documents(SESSION)]


def _upload_failing(h, path):
    with pytest.raises(RuntimeError, match="the extractor crashed"):
        h.upload(path)


def test_three_failed_uploads_of_one_file_are_one_document(h, tmp_path):
    path = h.write(tmp_path, "paper.txt", TEXT)
    h.fail_chunking = RuntimeError("the extractor crashed")
    for _ in range(3):
        _upload_failing(h, path)
    (failed,) = _ids(h)
    assert h.row(failed)["status"] == "error"

    h.fail_chunking = None
    doc = h.upload(path)

    assert doc.document_id == failed and doc.duplicate_of is None
    assert _ids(h) == [failed]
    assert h.searchable(failed)
    assert h.quota.get_usage(SESSION)["documents"] == 1


def test_a_rest_upload_retries_the_failed_copy(h, tmp_path):
    from lasagna.api.rest.routes import documents as documents_route

    path = h.write(tmp_path, "paper.txt", TEXT)
    h.fail_chunking = RuntimeError("the extractor crashed")
    _upload_failing(h, path)
    (failed,) = _ids(h)
    h.fail_chunking = None

    response = asyncio.run(
        documents_route.upload_document(
            file=UploadFile(io.BytesIO(TEXT.encode()), filename="paper.txt"),
            document_id=None,
            allow_duplicates=False,
            session_id=SESSION,
            backend=h.backend,
        )
    )
    assert response.document_id == failed and response.status == "ready"
    assert _ids(h) == [failed]


def test_other_content_is_a_new_document(h, tmp_path):
    h.fail_chunking = RuntimeError("the extractor crashed")
    _upload_failing(h, h.write(tmp_path, "paper.txt", TEXT))
    h.fail_chunking = None
    doc = h.upload(h.write(tmp_path, "other.txt", TEXT + " and more"))
    assert len(_ids(h)) == 2 and doc.document_id in _ids(h)


def test_a_failed_copy_from_another_model_is_replaced_by_the_retry(h, tmp_path):
    """The retry after a profile switch replaces the failed copy.

    The README quickstart fails offline with the default profile (its model is
    not cached), and `lasagna configure --profile speed` makes the rerun
    succeed. The failed copy was in the first model's collection, so it was
    not found, and `lasagna status` listed it with 0 chunks next to the ready
    copy. A failed document stored no vectors: the model does not matter.
    """
    path = h.write(tmp_path, "paper.txt", TEXT)
    h.fail_chunking = RuntimeError("the extractor crashed")
    _upload_failing(h, path)
    (failed,) = _ids(h)
    h.fail_chunking = None
    h.settings = dict(MODEL_B)

    doc = h.upload(path)
    assert doc.document_id == failed
    assert _ids(h) == [failed]
    assert h.searchable(failed)


def test_a_ready_copy_from_another_model_is_still_not_a_duplicate(h, tmp_path):
    """Only failed copies are found across models: a ready one is not searchable with the new model."""
    path = h.write(tmp_path, "paper.txt", TEXT)
    first = h.upload(path)
    h.settings = dict(MODEL_B)

    doc = h.upload(path)
    assert doc.document_id != first.document_id
    assert sorted(_ids(h)) == sorted([first.document_id, doc.document_id])


def test_allow_duplicates_still_makes_a_new_document(h, tmp_path):
    path = h.write(tmp_path, "paper.txt", TEXT)
    h.fail_chunking = RuntimeError("the extractor crashed")
    _upload_failing(h, path)
    h.fail_chunking = None
    h.upload(path, allow_duplicates=True)
    assert len(_ids(h)) == 2
