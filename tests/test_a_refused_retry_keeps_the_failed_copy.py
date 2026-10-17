"""A retry of a failed upload that is refused leaves the failed copy where it was.

Uploading a file again whose earlier upload failed retries that document under
its id. The failed copy (row, file, chunks) was deleted first and the upload
checked after, so a retry the quota or the file type refused lost both: the
session had neither the failed copy, to look at or retry, nor the new one. The
failed copy now goes only once the retry is admitted.
"""

from __future__ import annotations

import os

import pytest

from lasagna.core.quota import QuotaExceededError, QuotaManager, TenantQuota
from tests._document_harness import SESSION, Harness

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 3


@pytest.fixture
def h(tmp_path, monkeypatch):
    return Harness(
        tmp_path, monkeypatch, quota=QuotaManager(enabled=True, default_quota=TenantQuota())
    )


def _failed_copy(h, tmp_path):
    path = h.write(tmp_path, "paper.txt", TEXT)
    h.fail_chunking = RuntimeError("the extractor crashed")
    with pytest.raises(RuntimeError, match="the extractor crashed"):
        h.upload(path)
    h.fail_chunking = None
    (row,) = h.db.get_documents(SESSION)
    assert row["status"] == "error"
    return path, row


def test_a_retry_the_quota_refuses_keeps_the_failed_copy(h, tmp_path):
    path, failed = _failed_copy(h, tmp_path)
    h.quota.set_tenant_quota(SESSION, TenantQuota(max_storage_bytes=1))

    with pytest.raises(QuotaExceededError):
        h.upload(path)

    row = h.row(failed["document_id"])
    assert row is not None and row["status"] == "error"
    assert os.path.exists(row["file_path"])


def test_a_retry_the_file_type_refuses_keeps_the_failed_copy(h, tmp_path, monkeypatch):
    path, failed = _failed_copy(h, tmp_path)
    h.settings["supported_extensions"] = [".pdf"]  # the server stopped accepting text

    with pytest.raises(ValueError, match="Unsupported file type"):
        h.upload(path)

    assert h.row(failed["document_id"])["status"] == "error"


def test_an_admitted_retry_still_takes_the_failed_copys_place(h, tmp_path):
    path, failed = _failed_copy(h, tmp_path)
    doc = h.upload(path)
    assert doc.document_id == failed["document_id"]
    assert [r["document_id"] for r in h.db.get_documents(SESSION)] == [failed["document_id"]]
    assert h.searchable(doc.document_id)
