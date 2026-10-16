"""The orphan scan covers every embedding model's collection, and repair archives only real orphans.

Each embedding model has its own collection, recorded in ``collection_meta``,
but the scan read only the default model's: leaked vectors in any other
collection were never found. When the database could not say which collection
each document was written to, the scan assumed the default one, so every
document of another model looked like one with no vectors, and repair archived
it. Repair also archived a db-only orphan from a report made before the
document's vectors were written.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lasagna.core.database import RAGDatabase
from lasagna.lifecycle.manager import DocumentLifecycleManager
from lasagna.lifecycle.orphan_detector import OrphanDetector
from lasagna.lifecycle.policy import LifecyclePolicy

DEFAULT = "rag_documents_v2__small__384"
OTHER = "rag_documents_v2__m3__1024"


class _VectorStore:
    """document ids per collection; what the detector lists and repair deletes."""

    backend_name = "milvus"

    def __init__(self, contents):
        self.contents = {name: set(ids) for name, ids in contents.items()}

    def has_collection(self, name):
        return name in self.contents

    def list_document_ids(self, collection_name, session_id=None, limit=None):
        return sorted(self.contents[collection_name])

    def delete(self, collection_name, filter_expr):
        self.contents[collection_name].discard(filter_expr.value)


@pytest.fixture
def db(tmp_path):
    store = RAGDatabase(str(tmp_path / "metadata.db"))
    for name, model, dim in ((DEFAULT, "small", 384), (OTHER, "m3", 1024)):
        store.stamp_collection(name, model, dim)
    for doc_id, collection in (("doc_small", DEFAULT), ("doc_m3", OTHER)):
        store.save_document(
            {
                "document_id": doc_id,
                "session_id": "s1",
                "filename": f"{doc_id}.txt",
                "file_size": 10,
                "file_type": "txt",
                "status": "ready",
            },
            "/uploads/x",
            collection,
        )
    return store


def _manager(db, store):
    return DocumentLifecycleManager(
        db=db,
        pipeline=SimpleNamespace(),
        backend=store,
        policy=LifecyclePolicy(),
        collection_name=DEFAULT,
    )


def test_leaked_vectors_in_another_models_collection_are_found_and_deleted(db):
    store = _VectorStore({DEFAULT: {"doc_small"}, OTHER: {"doc_m3", "doc_leaked"}})
    manager = _manager(db, store)

    report = manager.detect_orphans()
    assert [(o.document_id, o.kind, o.collection_name) for o in report.orphans] == [
        ("doc_leaked", "vector_only", OTHER)
    ]

    manager.repair_orphans(report, dry_run=False)
    assert store.contents[OTHER] == {"doc_m3"}
    assert store.contents[DEFAULT] == {"doc_small"}


def test_a_document_missing_from_another_models_collection_is_found(db):
    store = _VectorStore({DEFAULT: {"doc_small"}, OTHER: set()})
    report = _manager(db, store).detect_orphans()
    assert [(o.document_id, o.kind, o.collection_name) for o in report.orphans] == [
        ("doc_m3", "db_only", OTHER)
    ]


def test_unreadable_collections_make_the_scan_incomplete_not_wider(db, monkeypatch):
    store = _VectorStore({DEFAULT: {"doc_small"}, OTHER: {"doc_m3"}})
    detector = OrphanDetector(db=db, backend=store)

    def unreadable(ids, include_soft_deleted=False):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(db, "get_documents_batch", unreadable)
    report = detector.detect(DEFAULT)

    assert report.incomplete
    assert "which collection" in report.incomplete_reason
    assert report.orphans == []
    detector.repair(report, DEFAULT, dry_run=False)
    assert db.get_document("doc_m3")["status"] == "ready"


def test_repair_keeps_a_document_whose_vectors_arrived_after_the_scan(db):
    store = _VectorStore({DEFAULT: set(), OTHER: {"doc_m3"}})
    manager = _manager(db, store)
    report = manager.detect_orphans()
    assert [o.document_id for o in report.orphans if o.kind == "db_only"] == ["doc_small"]

    store.contents[DEFAULT].add("doc_small")  # its indexing finished meanwhile
    repair = manager.repair_orphans(report, dry_run=False)

    assert repair.repaired_db_only == 0
    assert db.get_document("doc_small")["status"] == "ready"


def test_repair_still_archives_a_document_that_still_has_no_vectors(db):
    store = _VectorStore({DEFAULT: set(), OTHER: {"doc_m3"}})
    manager = _manager(db, store)

    repair = manager.repair_orphans(manager.detect_orphans(), dry_run=False)

    assert repair.repaired_db_only == 1
    assert db.get_document("doc_small")["status"] == "soft_deleted"
