"""Tests for OrphanDetector — cross-store consistency checking.

Covers:
- db_only orphans (DB record, no vectors)
- vec_only orphans (vectors, no DB record)
- Dry run: detect returns report, repair makes no changes
- Repair: actual deletion of orphans
"""

from __future__ import annotations

import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from lasagna.lifecycle.orphan_detector import OrphanDetector
from lasagna.lifecycle._results import (
    ConsistencyReport,
    OrphanRecord,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(
    db_only_ids: list[str] = (),
    vec_only_ids: list[str] = (),
) -> ConsistencyReport:
    """Build a ConsistencyReport with the given orphan sets."""
    orphans = [OrphanRecord(document_id=d, in_db=True, in_vector=False) for d in db_only_ids] + [
        OrphanRecord(document_id=d, in_db=False, in_vector=True) for d in vec_only_ids
    ]
    return ConsistencyReport(
        scanned_at=datetime.now(timezone.utc),
        session_id=None,
        db_count=len(db_only_ids),
        vector_count=len(vec_only_ids),
        orphans=orphans,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_db():
    db = MagicMock()
    db.list_all_active_document_ids.return_value = []
    return db


@pytest.fixture()
def mock_backend():
    backend = MagicMock()
    backend.list_document_ids.return_value = []
    return backend


@pytest.fixture()
def detector(mock_db, mock_backend):
    return OrphanDetector(db=mock_db, backend=mock_backend)


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


class TestDetect:
    def test_no_orphans_when_perfectly_consistent(self, detector, mock_db, mock_backend):
        """detect() returns empty report when DB and vector store are in sync."""
        ids = ["doc_1", "doc_2", "doc_3"]
        mock_db.list_all_active_document_ids.return_value = ids
        mock_backend.list_document_ids.return_value = ids

        report = detector.detect(collection_name="test_col")

        assert report.db_only_count == 0
        assert report.vector_only_count == 0

    def test_db_only_orphan_detected(self, detector, mock_db, mock_backend):
        """detect() flags documents in DB but missing from vector store."""
        mock_db.list_all_active_document_ids.return_value = ["doc_db_only", "doc_both"]
        mock_backend.list_document_ids.return_value = ["doc_both"]

        report = detector.detect(collection_name="test_col")

        db_only = [o for o in report.orphans if o.kind == "db_only"]
        assert len(db_only) == 1
        assert db_only[0].document_id == "doc_db_only"

    def test_vec_only_orphan_detected(self, detector, mock_db, mock_backend):
        """detect() flags documents in vector store but missing from DB."""
        mock_db.list_all_active_document_ids.return_value = ["doc_both"]
        mock_backend.list_document_ids.return_value = ["doc_both", "doc_vec_only"]

        report = detector.detect(collection_name="test_col")

        vec_only = [o for o in report.orphans if o.kind == "vector_only"]
        assert len(vec_only) == 1
        assert vec_only[0].document_id == "doc_vec_only"

    def test_both_orphan_types_detected(self, detector, mock_db, mock_backend):
        """detect() correctly identifies both db_only and vec_only orphans."""
        mock_db.list_all_active_document_ids.return_value = ["doc_db", "doc_both"]
        mock_backend.list_document_ids.return_value = ["doc_vec", "doc_both"]

        report = detector.detect(collection_name="test_col")

        db_only_ids = {o.document_id for o in report.orphans if o.kind == "db_only"}
        vec_only_ids = {o.document_id for o in report.orphans if o.kind == "vector_only"}
        assert db_only_ids == {"doc_db"}
        assert vec_only_ids == {"doc_vec"}

    def test_detect_scoped_to_session(self, detector, mock_db, mock_backend):
        """detect() passes session_id filter to both DB and vector backend."""
        detector.detect(collection_name="test_col", session_id="sess_123")
        mock_db.list_all_active_document_ids.assert_called_once()
        # session_id is passed as positional arg to list_all_active_document_ids
        assert mock_db.list_all_active_document_ids.call_args[0][0] == "sess_123"
        mock_backend.list_document_ids.assert_called_once_with(
            "test_col", session_id="sess_123", limit=50_000
        )

    def test_detect_is_read_only(self, detector, mock_db, mock_backend):
        """detect() makes no write calls to DB or backend."""
        mock_db.list_all_active_document_ids.return_value = ["doc_db"]
        mock_backend.list_document_ids.return_value = ["doc_vec"]

        detector.detect(collection_name="test_col")

        # No writes
        mock_db.delete_document.assert_not_called()
        mock_db.soft_delete_document.assert_not_called()
        mock_backend.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Repair — dry run
# ---------------------------------------------------------------------------


class TestRepairDryRun:
    def test_dry_run_makes_no_changes(self, detector, mock_db, mock_backend):
        """repair(dry_run=True) returns counts without deleting anything."""
        report = _make_report(db_only_ids=["doc_db"], vec_only_ids=["doc_vec"])
        mock_db.list_all_active_document_ids.return_value = ["doc_db"]  # still ready

        repair = detector.repair(report, collection_name="test_col", dry_run=True)

        assert repair.dry_run is True
        assert repair.repaired_db_only == 1  # dry_run counts as "would repair"
        assert repair.repaired_vector_only == 1
        mock_db.delete_document.assert_not_called()
        mock_db.soft_delete_document.assert_not_called()
        mock_backend.delete.assert_not_called()


# ---------------------------------------------------------------------------
# Repair — apply
# ---------------------------------------------------------------------------


class TestRepairApply:
    def test_repair_soft_deletes_db_only_orphans(self, detector, mock_db, mock_backend):
        """repair(dry_run=False) soft-deletes DB-only orphan records."""
        report = _make_report(db_only_ids=["doc_db_only"])
        mock_db.list_all_active_document_ids.return_value = ["doc_db_only"]  # still ready

        repair = detector.repair(report, collection_name="test_col", dry_run=False)

        mock_db.soft_delete_document.assert_called_once_with("doc_db_only")
        assert repair.repaired_db_only == 1

    def test_repair_deletes_vec_only_orphans(self, detector, mock_db, mock_backend):
        """repair(dry_run=False) deletes vector-only orphan chunks from backend."""
        report = _make_report(vec_only_ids=["doc_vec_only"])

        repair = detector.repair(report, collection_name="test_col", dry_run=False)

        mock_backend.delete.assert_called_once()
        delete_call_args = mock_backend.delete.call_args
        assert delete_call_args[0][0] == "test_col"
        assert repair.repaired_vector_only == 1

    def test_repair_handles_partial_errors(self, detector, mock_db, mock_backend):
        """repair() counts errors when individual deletions fail."""
        report = _make_report(db_only_ids=["doc_1", "doc_2"])
        mock_db.list_all_active_document_ids.return_value = ["doc_1", "doc_2"]  # still ready
        # First succeeds, second fails
        mock_db.soft_delete_document.side_effect = [True, RuntimeError("db error")]

        repair = detector.repair(report, collection_name="test_col", dry_run=False)

        assert repair.repaired_db_only == 1
        assert repair.errors == 1

    def test_empty_report_repairs_nothing(self, detector, mock_db, mock_backend):
        """repair() on an empty ConsistencyReport makes no changes."""
        report = _make_report()

        repair = detector.repair(report, collection_name="test_col", dry_run=False)

        assert repair.repaired_db_only == 0
        assert repair.repaired_vector_only == 0
        assert repair.errors == 0
        mock_db.soft_delete_document.assert_not_called()
        mock_backend.delete.assert_not_called()
