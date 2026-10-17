"""The batch upload and the bulk reindex report the reason the single routes give.

``POST /api/v1/documents/batch`` returned each failed file's raw exception
text, and ``POST /api/knowledge/reindex`` each failed document's: internal
paths and library messages went to the client, while the single upload and
reindex routes answer with a plain reason (and "the server log has the cause"
for an internal error).
"""

from __future__ import annotations

import asyncio
import io
from types import SimpleNamespace

from fastapi import UploadFile

from lasagna.api.rest.routes import documents as documents_route
from lasagna.api.rest.routes import reindex as reindex_route
from lasagna.core.model_downloads import ModelNotDownloaded

SECRET = "/srv/lasagna/internal/state.db"


def _batch(error):
    def upload_document(**kwargs):
        raise error

    backend = SimpleNamespace(upload_document=upload_document)
    files = [UploadFile(io.BytesIO(b"text"), filename="a.txt")]
    response = asyncio.run(
        documents_route.upload_documents_batch(
            files=files, allow_duplicates=False, session_id="s", backend=backend
        )
    )
    (item,) = response.results
    return item.error


def test_an_internal_error_in_a_batch_names_no_internals(monkeypatch):
    monkeypatch.setattr(documents_route, "_refuse_when_the_embedder_cannot_load", lambda b, s: None)
    error = _batch(RuntimeError(f"sqlite3.OperationalError at {SECRET}"))
    assert SECRET not in error
    assert "internal error" in error and "server log" in error


def test_a_refused_file_keeps_its_reason(monkeypatch):
    monkeypatch.setattr(documents_route, "_refuse_when_the_embedder_cannot_load", lambda b, s: None)
    assert _batch(ValueError("Unsupported file type: .xyz")) == "Unsupported file type: .xyz"


def test_a_model_that_is_not_downloaded_says_how_to_get_it(monkeypatch):
    monkeypatch.setattr(documents_route, "_refuse_when_the_embedder_cannot_load", lambda b, s: None)
    monkeypatch.setattr(
        "lasagna.core.model_downloads.fix_for_session", lambda sid, model: "Run: hf download m."
    )
    error = _batch(ModelNotDownloaded("m", offline=True))
    assert "not downloaded" in error and "hf download m" in error


def test_the_bulk_reindex_names_no_internals():
    def reindex_document(document_id, session_id):
        raise RuntimeError(f"pymilvus said no at {SECRET}")

    backend = SimpleNamespace(
        get_documents=lambda sid: [SimpleNamespace(document_id="doc_1")],
        reindex_document=reindex_document,
    )
    result = reindex_route._reindex_sync(backend, "s")
    (entry,) = result["errors"]
    assert entry.startswith("doc_1: ") and SECRET not in entry
    assert "the document is unchanged" in entry
