"""An upload under a new ``document_id`` never overwrites a row someone else saved meanwhile.

The upload checked that the id was free, then saved its row with an upsert
(SQLite ``INSERT OR REPLACE``; PostgreSQL ``ON CONFLICT DO UPDATE``, which also
rewrote ``session_id``). Two uploads under one new id both found it free: the
second's row replaced the first's, and from another session it took the
document over. The check and the save now hold the document's lock, and the
save is a plain insert that refuses a taken id.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from lasagna.core.database import DocumentIdTaken, RAGDatabase
from tests._document_harness import SESSION, Harness

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 3


def _row(document_id, session_id):
    return {
        "document_id": document_id,
        "session_id": session_id,
        "filename": "a.txt",
        "file_size": 1,
        "file_type": ".txt",
        "status": "processing",
    }


def test_sqlite_keeps_the_first_row(tmp_path):
    db = RAGDatabase(str(tmp_path / "rag.db"))
    db.save_document(_row("doc_x", "alice"), file_path="/a", replace=False)
    with pytest.raises(DocumentIdTaken, match="doc_x"):
        db.save_document(_row("doc_x", "mallory"), file_path="/m", replace=False)
    assert db.get_document("doc_x")["session_id"] == "alice"
    # The evaluation harness registers its corpus again: that still replaces.
    db.save_document(_row("doc_x", "alice"), file_path="/b")
    assert db.get_document("doc_x")["file_path"] == "/b"


class _Cursor:
    def __init__(self, conn):
        self.conn = conn
        self.rowcount = -1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params):
        self.conn.statements.append(sql)
        taken = params[0] in self.conn.rows
        if "DO NOTHING" in sql and taken:
            self.rowcount = 0
            return
        self.conn.rows[params[0]] = params[1]
        self.rowcount = 1


class _Connection:
    def __init__(self):
        self.rows, self.statements = {}, []

    def cursor(self):
        return _Cursor(self)

    def commit(self):
        pass


def test_postgres_keeps_the_first_row():
    from contextlib import contextmanager

    from lasagna.core.database_pg import PostgresRAGDatabase

    conn = _Connection()
    db = PostgresRAGDatabase.__new__(PostgresRAGDatabase)

    @contextmanager
    def _conn():
        yield conn

    db._conn = _conn
    db.save_document(_row("doc_x", "alice"), file_path="/a", replace=False)
    with pytest.raises(DocumentIdTaken):
        db.save_document(_row("doc_x", "mallory"), file_path="/m", replace=False)
    assert conn.rows["doc_x"] == "alice"
    assert "DO UPDATE SET" not in conn.statements[-1]


def _slow_store(h, monkeypatch, delay=0.5):
    from lasagna.api._document_pipeline import DocumentPipeline

    store = DocumentPipeline._store_file

    def slow(self, *a, **k):
        time.sleep(delay)
        return store(self, *a, **k)

    monkeypatch.setattr(DocumentPipeline, "_store_file", slow)


def _race(h, path, sessions):
    outcomes = SimpleNamespace(errors={}, docs={})

    def upload(session_id):
        try:
            outcomes.docs[session_id] = h.pipeline.upload_document(
                path, session_id, "paper.txt", document_id="doc_shared"
            )
        except Exception as exc:
            outcomes.errors[session_id] = exc

    threads = [threading.Thread(target=upload, args=(s,)) for s in sessions]
    for t in threads:
        t.start()
        time.sleep(0.05)
    for t in threads:
        t.join(30)
    return outcomes


def test_another_session_cannot_take_over_a_new_id(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    _slow_store(h, monkeypatch)
    path = h.write(tmp_path, "paper.txt", TEXT)

    outcomes = _race(h, path, [SESSION, "mallory"])

    assert h.row("doc_shared")["session_id"] == SESSION
    assert isinstance(outcomes.errors.get("mallory"), ValueError)
    assert h.searchable("doc_shared")


def test_two_uploads_under_one_new_id_index_it_once(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    _slow_store(h, monkeypatch)
    path = h.write(tmp_path, "paper.txt", TEXT)

    _race(h, path, [SESSION, SESSION])

    assert h.searchable("doc_shared")  # its collection holds exactly the chunks its row counts


def test_rest_answers_409_for_an_id_taken_meanwhile(tmp_path, monkeypatch):
    import asyncio
    import io

    from fastapi import HTTPException, UploadFile

    from lasagna.api.rest.routes import documents as documents_route

    h = Harness(tmp_path, monkeypatch)

    def taken(*a, **k):
        raise DocumentIdTaken("doc_new")

    monkeypatch.setattr(h.pipeline, "upload_document", taken)
    h.backend.upload_document = taken
    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            documents_route.upload_document(
                file=UploadFile(io.BytesIO(b"x"), filename="a.txt"),
                document_id="doc_new",
                allow_duplicates=False,
                session_id=SESSION,
                backend=h.backend,
            )
        )
    assert caught.value.status_code == 409


@pytest.mark.skipif(not __import__("os").environ.get("DATABASE_DSN"), reason="needs DATABASE_DSN")
def test_on_a_live_postgres_a_taken_id_is_refused():
    import os
    import uuid

    from lasagna.core.database_pg import PostgresRAGDatabase

    db = PostgresRAGDatabase(os.environ["DATABASE_DSN"], pool_min=1, pool_max=2)
    doc_id = f"doc_{uuid.uuid4().hex[:12]}"
    try:
        db.save_document(_row(doc_id, "alice"), file_path="/a", replace=False)
        with pytest.raises(DocumentIdTaken):
            db.save_document(_row(doc_id, "mallory"), file_path="/m", replace=False)
        assert db.get_document(doc_id)["session_id"] == "alice"
    finally:
        db.delete_document(doc_id)
        db.close()
