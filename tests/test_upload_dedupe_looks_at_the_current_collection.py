"""An upload is a duplicate only of a document the session can search now.

Upload deduplication matched any ready or processing row of the session with
the same bytes. After a model switch, uploading the file again, which
docs/configuration.md tells users to do, came back as ``duplicate_of`` the old
model's copy, and the new model's collection stayed empty. A document stuck
"processing" counted too, so the upload that would retry it was refused until
GC failed it, an hour or more later. Two identical uploads at the same time
both passed the check and both indexed, and the batch route had no
``allow_duplicates`` at all.
"""

from __future__ import annotations

import asyncio
import io
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import UploadFile

from lasagna.api.rest.routes import documents as documents_route
from tests._document_harness import MODEL_B, SESSION, Harness

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall"


@pytest.fixture
def h(tmp_path, monkeypatch):
    harness = Harness(tmp_path, monkeypatch)
    harness.path = harness.write(tmp_path, "paper.txt", TEXT)
    return harness


def test_the_same_file_in_the_same_collection_is_a_duplicate(h):
    first = h.upload(h.path)
    again = h.upload(h.path)
    assert again.duplicate_of == first.document_id


def test_after_a_model_switch_the_same_file_is_indexed_for_the_new_model(h):
    first = h.upload(h.path)
    h.settings = dict(MODEL_B)
    again = h.upload(h.path)
    assert again.duplicate_of is None and again.document_id != first.document_id
    new_collection = h.config().collection_name
    assert h.row(again.document_id)["collection_name"] == new_collection
    assert h.searchable(again.document_id)
    # The old model's copy is untouched.
    assert h.searchable(first.document_id)


def _mark(h, document_id, status, age):
    h.db.update_document_fields(
        document_id,
        {"status": status, "upload_date": (datetime.now(timezone.utc) - age).isoformat()},
    )


def test_a_document_stuck_processing_does_not_block_its_retry(h):
    stuck = h.upload(h.path)
    _mark(h, stuck.document_id, "processing", timedelta(hours=2))
    retry = h.upload(h.path)
    assert retry.duplicate_of is None and h.searchable(retry.document_id)


def test_a_document_still_being_processed_is_a_duplicate(h):
    busy = h.upload(h.path)
    _mark(h, busy.document_id, "processing", timedelta(seconds=5))
    assert h.upload(h.path).duplicate_of == busy.document_id


def test_two_identical_uploads_at_once_index_one_copy(h, monkeypatch):
    """Each paused after its check until the other had passed it too."""
    both_checked = threading.Barrier(2, timeout=2)
    store = h.pipeline._store_file

    def store_after_both_checked(*args):
        try:
            both_checked.wait()
        except threading.BrokenBarrierError:
            pass
        return store(*args)

    monkeypatch.setattr(h.pipeline, "_store_file", store_after_both_checked)
    results = []
    threads = [threading.Thread(target=lambda: results.append(h.upload(h.path))) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)

    indexed = [r for r in results if r.duplicate_of is None]
    assert len(results) == 2 and len(indexed) == 1
    assert [r.duplicate_of for r in results if r.duplicate_of] == [indexed[0].document_id]
    assert len(h.db.get_documents(SESSION)) == 1


def test_the_batch_route_takes_allow_duplicates(h):
    first = h.upload(h.path)

    def batch(**kwargs):
        upload = UploadFile(file=io.BytesIO(TEXT.encode()), filename="paper.txt")
        return asyncio.run(
            documents_route.upload_documents_batch(
                files=[upload], session_id=SESSION, backend=h.backend, **kwargs
            )
        ).results[0]

    assert batch(allow_duplicates=False).duplicate_of == first.document_id
    forced = batch(allow_duplicates=True)
    assert forced.duplicate_of is None and forced.document_id != first.document_id


def test_identical_files_in_one_batch_index_one_copy(h):
    uploads = [UploadFile(file=io.BytesIO(TEXT.encode()), filename="paper.txt") for _ in range(2)]
    results = asyncio.run(
        documents_route.upload_documents_batch(
            files=uploads, session_id=SESSION, backend=h.backend, allow_duplicates=False
        )
    ).results
    assert sorted(r.duplicate_of is None for r in results) == [False, True]
