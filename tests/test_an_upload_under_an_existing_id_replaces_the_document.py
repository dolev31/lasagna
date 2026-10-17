"""Uploading under the id of a document the session holds replaces it.

`api/docs/endpoints.md` promised that re-uploading with a `document_id` the
session owns "replaces that document instead of creating a duplicate", and
nothing removed the old copy: its chunks, its file and its quota stayed. On
Milvus, whose `insert` keeps two rows of one id, every chunk was in the store
twice and search returned the old text beside the new; on Qdrant and pgvector,
which upsert, the old chunks past the new count stayed searchable. The quota
counted two documents.
"""

from __future__ import annotations

import pytest

from lasagna.core.quota import QuotaManager, TenantQuota
from tests._document_harness import SESSION, Harness, MemoryVectors

OLD = "the first draft says the reranker lowered ndcg on every benchmark we ran " * 2
NEW = "the revision says recall rose"


@pytest.fixture(params=["milvus-insert", "upsert"])
def harness(request, tmp_path, monkeypatch):
    # Milvus Lite itself when LASAGNA_MILVUS_DB_PATH is set, a store that keeps
    # duplicate ids otherwise; and one that replaces a row of the same id.
    vectors = MemoryVectors(upsert=True) if request.param == "upsert" else None
    h = Harness(
        tmp_path,
        monkeypatch,
        vectors,
        quota=QuotaManager(enabled=True, default_quota=TenantQuota()),
    )
    first = h.upload(h.write(tmp_path, "draft.txt", OLD))
    h.document_id = first.document_id
    h.old_file = h.row(first.document_id)["file_path"]
    h.tmp = tmp_path
    return h


def test_the_new_copy_is_the_only_copy(harness):
    h = harness
    again = h.upload(h.write(h.tmp, "revision.txt", NEW), document_id=h.document_id)

    assert again.document_id == h.document_id
    assert h.searchable(h.document_id)
    stored = h.chunks(h.document_id)
    assert " ".join(c["content"] for c in stored).split() == NEW.split()
    assert (
        len({c["chunk_id"] for c in stored}) == len(stored) == h.row(h.document_id)["chunks_count"]
    )
    row = h.row(h.document_id)
    assert row["filename"] == "revision.txt"
    assert open(row["file_path"]).read() == NEW


def test_the_replaced_file_is_removed(harness):
    h = harness
    import os

    h.upload(h.write(h.tmp, "revision.txt", NEW), document_id=h.document_id)
    assert not os.path.exists(h.old_file)
    files = [p for p in h.uploads.rglob("*") if p.is_file()]
    assert len(files) == 1 and files[0].read_text() == NEW


def test_the_quota_counts_one_document(harness):
    h = harness
    h.upload(h.write(h.tmp, "revision.txt", NEW), document_id=h.document_id)
    usage = h.quota.get_usage(SESSION)
    assert usage["documents"] == 1
    assert usage["storage_bytes"] == len(NEW.encode())
    assert usage["chunks"] == h.row(h.document_id)["chunks_count"]


def test_a_replacement_that_fails_keeps_the_old_copy(harness):
    h = harness
    before = h.texts(h.document_id)
    h.fail_chunking = RuntimeError("the extractor crashed")
    with pytest.raises(RuntimeError, match="extractor crashed"):
        h.upload(h.write(h.tmp, "revision.txt", NEW), document_id=h.document_id)
    assert h.searchable(h.document_id)
    assert h.texts(h.document_id) == before
    assert open(h.row(h.document_id)["file_path"]).read() == OLD
    assert [p.read_text() for p in h.uploads.rglob("*") if p.is_file()] == [OLD]
    assert h.quota.get_usage(SESSION)["documents"] == 1


def test_an_id_from_another_session_is_refused(harness):
    h = harness
    with pytest.raises(ValueError, match="another session"):
        h.pipeline.upload_document(
            h.write(h.tmp, "revision.txt", NEW), "someone-else", document_id=h.document_id
        )
    assert h.row(h.document_id)["session_id"] == SESSION
    assert h.searchable(h.document_id)


def test_rest_answers_409_for_a_document_still_processing(harness):
    """A replacement of a document mid-processing would race it; it was a 500."""
    import asyncio
    import io

    from fastapi import HTTPException, UploadFile

    from lasagna.api.rest.routes import documents

    h = harness
    h.db.update_document_status(h.document_id, "processing")
    upload = UploadFile(file=io.BytesIO(NEW.encode()), filename="revision.txt")
    with pytest.raises(HTTPException) as refused:
        asyncio.run(
            documents.upload_document(
                file=upload, document_id=h.document_id, session_id=SESSION, backend=h.backend
            )
        )
    assert refused.value.status_code == 409
    assert "still being processed" in refused.value.detail
