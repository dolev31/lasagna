"""Orphan repair deletes only vectors that no document record owns.

Detection compared the vector store with the ``ready`` documents only, so
every document still uploading, processing or indexing, every one in
``error`` and every archived (``soft_deleted``) one showed up as a
vector-only orphan, and repair (the weekly Celery scan with ``dry_run=False``,
``POST /admin/gc/repair``) deleted its vectors: an upload in progress lost
the chunks already written, and restoring an archived document brought back
a record with nothing to search.

A vector is an orphan only when no record, in any status, owns it: the
owner was hard-deleted (the only permanent deletion) or never existed.
"""

from __future__ import annotations

import pytest

from lasagna.core.database import RAGDatabase
from lasagna.core.schema import DocumentStatus
from lasagna.lifecycle.orphan_detector import OrphanDetector

COLLECTION = "rag_documents_v2__test__8"


class _VectorStore:
    """Chunks per document id; what the detector lists and repair deletes."""

    def __init__(self, doc_ids):
        self.doc_ids = set(doc_ids)

    def list_document_ids(self, collection_name, session_id=None, limit=None):
        return sorted(self.doc_ids)

    def delete(self, collection_name, filter_expr):
        self.doc_ids.discard(filter_expr.value)


@pytest.fixture
def db(tmp_path):
    database = RAGDatabase(str(tmp_path / "metadata.db"))
    for status in DocumentStatus:
        database.save_document(
            {
                "document_id": f"doc_{status.value}",
                "session_id": "s1",
                "filename": f"{status.value}.txt",
                "file_size": 10,
                "file_type": "txt",
                "status": status.value,
            },
            "/uploads/x",
            COLLECTION,
        )
    # Archiving sets deleted_at as well as the status.
    database.soft_delete_document(f"doc_{DocumentStatus.SOFT_DELETED.value}")
    return database


def _store_with_every_document(extra=()):
    return _VectorStore([f"doc_{s.value}" for s in DocumentStatus] + list(extra))


@pytest.mark.parametrize("status", list(DocumentStatus), ids=lambda s: s.value)
def test_a_document_in_any_status_keeps_its_vectors(db, status):
    store = _store_with_every_document()
    detector = OrphanDetector(db=db, backend=store)

    report = detector.detect(COLLECTION)
    detector.repair(report, COLLECTION, dry_run=False)

    assert f"doc_{status.value}" not in {
        o.document_id for o in report.orphans if o.kind == "vector_only"
    }
    assert f"doc_{status.value}" in store.doc_ids


def test_vectors_with_no_record_are_the_only_ones_repaired(db):
    store = _store_with_every_document(extra=["doc_never_recorded"])
    detector = OrphanDetector(db=db, backend=store)

    report = detector.detect(COLLECTION)
    repair = detector.repair(report, COLLECTION, dry_run=False)

    assert [o.document_id for o in report.orphans if o.kind == "vector_only"] == [
        "doc_never_recorded"
    ]
    assert repair.repaired_vector_only == 1
    assert store.doc_ids == {f"doc_{s.value}" for s in DocumentStatus}


def test_a_hard_deleted_document_leaves_orphans_that_repair_removes(db):
    store = _store_with_every_document()
    db.delete_document("doc_ready")
    detector = OrphanDetector(db=db, backend=store)

    report = detector.detect(COLLECTION)
    detector.repair(report, COLLECTION, dry_run=False)

    assert "doc_ready" not in store.doc_ids
    assert store.doc_ids == {f"doc_{s.value}" for s in DocumentStatus} - {"doc_ready"}


def test_a_report_older_than_the_record_deletes_nothing(db):
    store = _store_with_every_document(extra=["doc_reuploaded"])
    detector = OrphanDetector(db=db, backend=store)
    report = detector.detect(COLLECTION)
    assert any(o.document_id == "doc_reuploaded" for o in report.orphans)

    # The same document id is uploaded again before the repair runs.
    db.save_document(
        {
            "document_id": "doc_reuploaded",
            "session_id": "s1",
            "filename": "again.txt",
            "file_size": 10,
            "file_type": "txt",
            "status": "processing",
        },
        "/uploads/y",
        COLLECTION,
    )
    repair = detector.repair(report, COLLECTION, dry_run=False)

    assert "doc_reuploaded" in store.doc_ids
    assert repair.repaired_vector_only == 0


def test_records_that_cannot_be_read_are_not_called_orphans(db, monkeypatch):
    store = _store_with_every_document(extra=["doc_never_recorded"])

    def unreadable(*args, **kwargs):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "get_documents_batch", unreadable)
    report = OrphanDetector(db=db, backend=store).detect(COLLECTION)

    assert report.vector_only_count == 0
