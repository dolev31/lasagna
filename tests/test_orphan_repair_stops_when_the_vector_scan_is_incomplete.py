"""Orphan repair changes nothing when the vector store could not be read in full.

Detection turned any error from the vector store scan into an empty set of
vectors, so every ready document looked like a document with no vectors (a
db-only orphan), and ``POST /admin/gc/repair?dry_run=false`` (or the weekly
Celery scan) archived every one of them. A scan that stopped at
``vector_scan_limit`` (50,000 documents) did the same to every document past
the limit. An incomplete scan is reported as incomplete, with the reason, and
repair refuses to act on it.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from lasagna.core.database import RAGDatabase
from lasagna.lifecycle._results import ConsistencyReport, OrphanRecord
from lasagna.lifecycle.orphan_detector import OrphanDetector

COLLECTION = "rag_documents_v2__test__8"
DOC_IDS = [f"doc_{i}" for i in range(5)]


class _UnreachableStore:
    def list_document_ids(self, collection_name, session_id=None, limit=None):
        raise ConnectionError("milvus:19530 refused the connection")

    def delete(self, collection_name, filter_expr):  # pragma: no cover - must not run
        raise AssertionError("repair deleted vectors after a failed scan")


class _TruncatingStore:
    """Returns at most ``limit`` ids, like a backend that stops at the scan limit."""

    def __init__(self, doc_ids):
        self.doc_ids = list(doc_ids)
        self.deleted = []

    def list_document_ids(self, collection_name, session_id=None, limit=None):
        return self.doc_ids[:limit] if limit else list(self.doc_ids)

    def delete(self, collection_name, filter_expr):
        self.deleted.append(filter_expr.value)


@pytest.fixture
def db(tmp_path):
    database = RAGDatabase(str(tmp_path / "metadata.db"))
    for doc_id in DOC_IDS:
        database.save_document(
            {
                "document_id": doc_id,
                "session_id": "s1",
                "filename": f"{doc_id}.txt",
                "file_size": 10,
                "file_type": "txt",
                "status": "ready",
            },
            "/uploads/x",
            COLLECTION,
        )
    return database


def _ready(db):
    return set(db.list_all_active_document_ids(None, only_ready=True))


def test_a_failed_scan_is_incomplete_and_repair_archives_nothing(db):
    detector = OrphanDetector(db=db, backend=_UnreachableStore())

    report = detector.detect(COLLECTION)
    repair = detector.repair(report, COLLECTION, dry_run=False)

    assert report.incomplete
    assert "refused the connection" in report.incomplete_reason
    assert report.orphans == []
    assert not report.is_consistent
    assert repair.aborted and repair.repaired_db_only == 0
    assert "vector store" in repair.abort_reason
    assert _ready(db) == set(DOC_IDS)


def test_a_scan_that_hit_the_limit_is_incomplete_and_repair_archives_nothing(db):
    store = _TruncatingStore(DOC_IDS)
    detector = OrphanDetector(db=db, backend=store)

    report = detector.detect(COLLECTION, vector_scan_limit=2)
    repair = detector.repair(report, COLLECTION, dry_run=False)

    assert report.scan_limited and report.incomplete
    assert "vector_scan_limit" in report.incomplete_reason
    assert report.orphans == []
    assert repair.aborted and repair.repaired_db_only == 0
    assert store.deleted == []
    assert _ready(db) == set(DOC_IDS)


def test_repair_refuses_a_truncated_report_that_lists_orphans(db):
    """A report built elsewhere (or by an older version) is checked too."""
    store = _TruncatingStore(DOC_IDS[:2] + ["doc_leaked"])
    report = ConsistencyReport(
        scanned_at=datetime.now(timezone.utc),
        session_id=None,
        db_count=5,
        vector_count=3,
        orphans=[
            OrphanRecord(document_id="doc_4", in_db=True, in_vector=False),
            OrphanRecord(document_id="doc_leaked", in_db=False, in_vector=True),
        ],
        scan_limited=True,
        vector_scan_limit=3,
    )

    repair = OrphanDetector(db=db, backend=store).repair(report, COLLECTION, dry_run=False)

    assert repair.aborted
    assert repair.to_dict()["aborted"] is True and repair.to_dict()["abort_reason"]
    assert store.deleted == [] and _ready(db) == set(DOC_IDS)


def test_a_complete_scan_still_finds_and_repairs_orphans(db):
    store = _TruncatingStore(DOC_IDS[:4] + ["doc_leaked"])
    detector = OrphanDetector(db=db, backend=store)

    report = detector.detect(COLLECTION)
    repair = detector.repair(report, COLLECTION, dry_run=False)

    assert not report.incomplete and report.incomplete_reason is None
    assert {(o.document_id, o.kind) for o in report.orphans} == {
        ("doc_4", "db_only"),
        ("doc_leaked", "vector_only"),
    }
    assert not repair.aborted
    assert (repair.repaired_db_only, repair.repaired_vector_only) == (1, 1)
    assert store.deleted == ["doc_leaked"]
    assert _ready(db) == set(DOC_IDS[:4])
