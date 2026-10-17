"""Uploads, reindexes and deletes of one document wait for each other across processes.

The duplicate check of an upload and the lock a reindex and a delete take were
``threading`` locks, while the image runs four gunicorn workers, Kubernetes two
to ten API pods, and the MCP server and the Celery worker are processes of
their own. Two identical uploads on two workers both passed the duplicate check
and were both indexed. A delete on one pod during a reindex on another removed
the row; the reindex then stored its chunks, updated no row, left the chunks in
the vector store and failed the request with a 500.

The processes here share a SQLite metadata store and an upload folder, as the
workers of one host do. On PostgreSQL the same lock is an advisory lock.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from lasagna.api._document_pipeline import DocumentRemoved
from tests._document_harness import SESSION, Harness
from tests._paths import PACKAGE_DIR

ROOT = Path(__file__).resolve().parents[1]
TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 3

CHILD = r"""
import json, sys, time
from pathlib import Path

import pytest

from lasagna.api._document_pipeline import DocumentPipeline
from tests._document_harness import SESSION, Harness

role, base, document_id = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
h = Harness(base, pytest.MonkeyPatch())


def wait_for(name):
    while not (base / name).exists():
        time.sleep(0.01)


if role == "upload":
    store = DocumentPipeline._store_file

    def slow_store(self, *a, **k):
        # Between the duplicate check and the row: where two uploads raced.
        time.sleep(0.6)
        return store(self, *a, **k)

    DocumentPipeline._store_file = slow_store
    wait_for("go")
    doc = h.upload(str(base / "paper.txt"))
    print(json.dumps({"document_id": doc.document_id, "duplicate_of": doc.duplicate_of}))
elif role == "reindex":
    chunk = h._chunk

    def slow_chunk(*a, **k):
        (base / "reindexing").touch()
        time.sleep(1.0)
        return chunk(*a, **k)

    h._chunk = slow_chunk
    import lasagna.api._document_pipeline as pipeline_mod

    pipeline_mod.chunk_document = slow_chunk
    try:
        doc = h.pipeline.reindex_document(document_id, SESSION)
        outcome = {"result": None if doc is None else doc.document_id}
    except Exception as exc:
        outcome = {"error": type(exc).__name__}
    outcome["finished"] = time.time()
    print(json.dumps(outcome))
elif role == "delete":
    wait_for("reindexing")
    started = time.time()
    deleted = h.pipeline.delete_document(document_id, SESSION)
    print(json.dumps({"deleted": deleted, "started": started, "finished": time.time()}))
"""


def _start(tmp_path, role, document_id=""):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=os.pathsep.join([str(ROOT), str(PACKAGE_DIR.parent)]),
        HF_HUB_OFFLINE="1",
    )
    return subprocess.Popen(
        [sys.executable, "-c", CHILD, role, str(tmp_path), document_id],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _result(proc):
    out, err = proc.communicate(timeout=120)
    assert proc.returncode == 0, err[-3000:]
    return json.loads(out.strip().splitlines()[-1])


def test_two_processes_uploading_the_same_file_keep_one_copy(tmp_path):
    (tmp_path / "paper.txt").write_text(TEXT)
    procs = [_start(tmp_path, "upload") for _ in range(2)]
    time.sleep(3.0)  # both imported and waiting
    (tmp_path / "go").touch()
    first, second = sorted((_result(p) for p in procs), key=lambda r: bool(r["duplicate_of"]))

    from lasagna.core.database import RAGDatabase

    rows = RAGDatabase(str(tmp_path / "rag.db")).get_documents(SESSION)
    assert [r["document_id"] for r in rows] == [first["document_id"]]
    assert first["duplicate_of"] is None
    assert second["duplicate_of"] == first["document_id"]


def test_a_delete_in_another_process_waits_for_the_reindex(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))

    reindex = _start(tmp_path, "reindex", doc.document_id)
    delete = _start(tmp_path, "delete", doc.document_id)
    reindexed, deleted = _result(reindex), _result(delete)

    assert reindexed == {"result": doc.document_id, "finished": reindexed["finished"]}
    assert deleted["deleted"] is True
    assert deleted["finished"] >= reindexed["finished"]
    assert h.row(doc.document_id) is None


def test_a_store_lock_holds_against_another_process(tmp_path):
    """The lock itself: a second process waits until the first lets go."""
    script = (
        "import sys, time; from lasagna.core.database import RAGDatabase\n"
        "db = RAGDatabase(sys.argv[1])\n"
        "with db.exclusive('document:doc_1'):\n"
        "    print(time.time(), flush=True); time.sleep(1.0); print(time.time(), flush=True)\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent)}
    db_path = str(tmp_path / "rag.db")
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, db_path], env=env, stdout=subprocess.PIPE, text=True
        )
        for _ in range(2)
    ]
    spans = sorted(tuple(map(float, p.communicate(timeout=60)[0].split())) for p in procs)
    assert spans[1][0] >= spans[0][1], spans


def test_a_thread_may_take_a_lock_it_holds(tmp_path):
    from lasagna.core.database import RAGDatabase

    db = RAGDatabase(str(tmp_path / "rag.db"))
    with db.exclusive("document:a"), db.exclusive("document:a"):
        pass
    with db.exclusive("document:a"):  # released, not left held by the nested exit
        pass


@pytest.fixture
def harness(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))
    h.document_id = doc.document_id
    h.monkeypatch = monkeypatch
    return h


def _delete_row_while_chunking(h):
    """A delete that does not wait for the lock (one made by older code, or by hand)."""
    chunk = h._chunk

    def chunk_then_delete(*a, **k):
        h.db.delete_document(h.document_id)
        return chunk(*a, **k)

    import lasagna.api._document_pipeline as pipeline_mod

    h.monkeypatch.setattr(pipeline_mod, "chunk_document", chunk_then_delete)


def test_a_reindex_whose_row_vanished_removes_its_chunks_and_says_so(harness):
    h = harness
    h.chunk_words = 2
    _delete_row_while_chunking(h)

    with pytest.raises(DocumentRemoved, match="was deleted while it was being processed"):
        h.pipeline.reindex_document(h.document_id, SESSION)

    assert h.chunks(h.document_id) == []
    files = [p for p in h.uploads.rglob("*") if p.is_file()]
    assert all("__r" not in str(p) for p in files), files


def test_rest_answers_404_with_the_reason(harness):
    import asyncio

    from fastapi import HTTPException

    from lasagna.api.rest.routes import documents as documents_route

    _delete_row_while_chunking(harness)
    with pytest.raises(HTTPException) as caught:
        asyncio.run(
            documents_route.reindex_document(
                harness.document_id, session_id=SESSION, backend=harness.backend
            )
        )
    assert caught.value.status_code == 404
    assert "deleted while it was being" in caught.value.detail


def test_an_upload_whose_row_vanished_leaves_no_chunks(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    path = h.write(tmp_path, "paper.txt", TEXT)
    seen = SimpleNamespace(document_id=None)
    chunk = h._chunk

    def chunk_then_delete(text, document_id, *a, **k):
        seen.document_id = document_id
        h.db.delete_document(document_id)
        return chunk(text, document_id, *a, **k)

    import lasagna.api._document_pipeline as pipeline_mod

    monkeypatch.setattr(pipeline_mod, "chunk_document", chunk_then_delete)
    with pytest.raises(DocumentRemoved):
        h.upload(path)
    assert h.chunks(seen.document_id) == []


class _FakeCursor:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.log.append((sql, params))

    def fetchone(self):
        return (True,)


class _FakeConnection:
    def __init__(self, log, autocommit):
        self.log, self.autocommit, self.closed = log, autocommit, False

    def cursor(self):
        return _FakeCursor(self.log)

    def close(self):
        self.closed = True


def test_on_postgres_the_lock_is_an_advisory_lock_on_its_own_connection(monkeypatch):
    import threading

    import lasagna.core.database_pg as database_pg

    log, connections = [], []

    def connect(dsn, autocommit=False):
        connections.append(_FakeConnection(log, autocommit))
        return connections[-1]

    monkeypatch.setattr(database_pg, "_psycopg", SimpleNamespace(connect=connect))
    db = database_pg.PostgresRAGDatabase.__new__(database_pg.PostgresRAGDatabase)
    db._dsn, db._driver, db._lock = "postgresql://x", "psycopg3", threading.Lock()

    with db.exclusive("document:doc_1"), db.exclusive("document:doc_1"):
        assert len(connections) == 1 and connections[0].autocommit
    from lasagna.core.store_locks import lock_key

    key = lock_key("document:doc_1")
    # Waited for a bounded time (RAG_STORE_LOCK_TIMEOUT, 300 s by default).
    assert log == [
        ("SET lock_timeout = 300000", ()),
        ("SELECT pg_advisory_lock(%s)", (key,)),
        ("SET lock_timeout = 0", ()),
        ("SELECT pg_advisory_unlock(%s)", (key,)),
    ]
    assert connections[0].closed


@pytest.mark.skipif(not os.environ.get("DATABASE_DSN"), reason="needs DATABASE_DSN (PostgreSQL)")
def test_on_a_live_postgres_a_second_holder_waits():
    import threading

    from lasagna.core.database_pg import PostgresRAGDatabase

    spans = []

    def hold():
        db = PostgresRAGDatabase(os.environ["DATABASE_DSN"], pool_min=1, pool_max=2)
        try:
            with db.exclusive(f"document:race-{os.getpid()}"):
                start = time.time()
                time.sleep(0.5)
                spans.append((start, time.time()))
        finally:
            db.close()

    threads = [threading.Thread(target=hold) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(60)
    spans.sort()
    assert len(spans) == 2 and spans[1][0] >= spans[0][1], spans


def test_processes_starting_on_a_new_database_all_open_it(tmp_path):
    """Four gunicorn workers start at once; the second one's switch to WAL failed."""
    rounds = 12
    script = (
        "import sys, time, pathlib\n"
        "base = pathlib.Path(sys.argv[1])\n"
        "from lasagna.core.database import RAGDatabase\n"
        f"for i in range({rounds}):\n"
        "    while not (base / f'go{i}').exists(): time.sleep(0.002)\n"
        "    RAGDatabase(str(base / f'rag{i}.db'))\n"
    )
    env = {**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent)}
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", script, str(tmp_path)],
            env=env,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(8)
    ]
    time.sleep(3.0)
    for i in range(rounds):
        (tmp_path / f"go{i}").touch()
        time.sleep(0.3)
    errors = [p.communicate(timeout=60)[1] for p in procs]
    assert [p.returncode for p in procs] == [0] * 8, "\n".join(e[-500:] for e in errors if e)
