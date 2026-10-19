"""Two experiment processes on one store never delete each other's corpus.

Nothing held across processes from the harness's corpus check to its ingest. On
a vector store two processes share (a Milvus or Qdrant server), the second saw
the first's ingest in progress as a "stale partial corpus" and deleted those
vectors while the first went on evaluating against them, and no result said
so. Now the check and the ingest hold the corpus's store lock, a run that waits
for it says so, and a task whose corpus changed under it fails instead of
recording numbers.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

# The live-run fixtures of the ranked-order test: a stub retriever over the smoke corpus.
from tests.test_retrieval_metrics_score_the_ranked_order import (  # noqa: F401
    _run,
    live,
    orchestrator,
)

REPO = Path(__file__).resolve().parents[1]

#: One experiment process: the smoke corpus into a vector store shared through
#: a SQLite file, ingested slowly, telling ``mark`` once a few documents are in
#: and then waiting for the other process to be about to check the store
#: (``ready``), so that its check meets this ingest half done.
_RUN = textwrap.dedent(
    """
    import asyncio, json, os, sqlite3, sys, time
    from types import SimpleNamespace

    from loguru import logger

    import lasagna.api.rag_backend as rag_backend
    from lasagna.knowledge_evaluation_sdk import _fast_ingest
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    logger.enable("lasagna")  # as the CLI does

    vectors, role, mark, results, ready = sys.argv[1:6]

    def db():
        conn = sqlite3.connect(vectors, timeout=30)
        conn.execute("CREATE TABLE IF NOT EXISTS chunks (session TEXT, doc TEXT)")
        conn.execute("CREATE TABLE IF NOT EXISTS events (role TEXT, what TEXT)")
        return conn

    class Shared:
        def has_collection(self, name):
            return True

        def session_filter(self, collection, session):
            return session

        def query(self, **kwargs):
            raise NotImplementedError

        def list_document_ids(self, collection_name, session_id, limit):
            with db() as conn:
                rows = conn.execute(
                    "SELECT DISTINCT doc FROM chunks WHERE session = ?", (session_id,)
                ).fetchall()
            return [r[0] for r in rows][:limit]

        def delete(self, collection_name, filter_expr):
            with db() as conn:
                conn.execute("INSERT INTO events VALUES (?, 'delete')", (role,))
                conn.execute("DELETE FROM chunks WHERE session = ?", (filter_expr,))

    store = Shared()
    backend = SimpleNamespace(
        _collection_manager=SimpleNamespace(resolve_backend=lambda s: store),
        hybrid_searcher=SimpleNamespace(clear_session=lambda s: None),
    )
    rag_backend.get_rag_backend = lambda *a, **k: backend

    async def slow_ingest(corpus, corpus_to_valid, ready_ids, session_id, **kwargs):
        todo = sorted(set(corpus_to_valid.values()) - set(ready_ids))
        for i, doc in enumerate(todo):
            with db() as conn:
                conn.execute("INSERT INTO chunks VALUES (?, ?)", (session_id, doc))
            if i == 2:
                open(mark, "w").close()
                deadline = time.monotonic() + 120
                while not os.path.exists(ready) and time.monotonic() < deadline:
                    time.sleep(0.05)
            time.sleep(0.15)
        return len(todo), set(todo), set()

    _fast_ingest.fast_ingest_corpus = slow_ingest
    exp._results_root = lambda: results
    if role == "second":
        open(ready, "w").close()
    info = asyncio.run(exp._ensure_corpus_ingested("smoke", []))
    print("RESULT " + json.dumps({"session": info["session_id"], "role": role}))
    """
)


def _env(tmp_path: Path) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        PYTHONPATH=str(REPO / "src"),
        HOME=str(tmp_path / "home"),
        RAG_DB_DIR=str(tmp_path / "dbs"),  # one metadata store, as on one host
        HF_HUB_OFFLINE="1",
    )
    (tmp_path / "home").mkdir(exist_ok=True)
    return env


def _start(tmp_path: Path, role: str, mark: Path) -> subprocess.Popen:
    args = [
        str(tmp_path / "vectors.db"),
        role,
        str(mark),
        str(tmp_path / "results"),
        str(tmp_path / "second-is-ready"),
    ]
    return subprocess.Popen(
        [sys.executable, "-c", _RUN, *args],
        env=_env(tmp_path),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def test_the_second_process_waits_for_the_first_and_reuses_its_corpus(tmp_path):
    first_mark, second_mark = tmp_path / "first", tmp_path / "second"
    first = _start(tmp_path, "first", first_mark)
    deadline = time.monotonic() + 120
    while not first_mark.exists():
        assert first.poll() is None, first.communicate()[1][-3000:]
        assert time.monotonic() < deadline, "the first process never started ingesting"
        time.sleep(0.05)
    second = _start(tmp_path, "second", second_mark)
    out1, err1 = first.communicate(timeout=300)
    out2, err2 = second.communicate(timeout=300)
    assert first.returncode == 0, err1[-3000:]
    assert second.returncode == 0, err2[-3000:]

    with sqlite3.connect(tmp_path / "vectors.db") as conn:
        deletes = conn.execute("SELECT role FROM events WHERE what = 'delete'").fetchall()
        sessions = conn.execute("SELECT session, COUNT(*) FROM chunks GROUP BY session").fetchall()
    assert deletes == [], "a process deleted the corpus the other was ingesting"
    assert not second_mark.exists(), "the second process ingested again"
    results = [
        json.loads(line[7:]) for line in (out1 + out2).splitlines() if line.startswith("RESULT ")
    ]
    assert len({r["session"] for r in results}) == 1
    assert len(sessions) == 1  # one copy of the corpus, of every document once
    assert "Waiting for another process" in err2, err2[-3000:]


def test_a_corpus_that_changed_under_a_run_is_noticed():
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    present = {"d1": 2, "d2": 3}

    class Store:
        def query(self, collection_name, filter_expr, output_fields, limit):
            return [{"document_id": d} for d, n in present.items() for _ in range(n)]

    store = Store()
    census = exp._corpus_census(store, "c", "s", ["d1", "d2"])
    slot = {"session_id": "s", "corpus_census": census, "_census_store": lambda: store}
    assert exp._corpus_changed(slot) is None

    present["d2"] = 6  # ingested again on top: every chunk twice
    changed = exp._corpus_changed(slot)
    assert changed and "changed while this run used it" in changed
    present.pop("d2")  # or deleted by another process
    assert "1 of 2" in exp._corpus_changed(slot)


@pytest.mark.usefixtures("live")
def test_a_task_whose_corpus_changed_records_no_numbers(tmp_path, monkeypatch):
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    changed = "The corpus of session stub-session changed while this run used it: ..."
    monkeypatch.setattr(exp, "_corpus_changed", lambda corpus_info: changed)
    result = _run(tmp_path, "speed")
    assert result.error == changed
    assert not result.metrics


class _SharedStore:
    """A vector store two backends in one process use, as two processes share a server."""

    def __init__(self):
        self.rows = []

    def query(self, collection_name, filter_expr, output_fields, limit):
        return [{"document_id": d} for d in self.rows]


def _register(db, session, documents, stamp):
    for doc in documents:
        db.save_document(
            {
                "document_id": doc,
                "session_id": session,
                "filename": f"{doc}.txt",
                "file_size": 0,
                "file_type": "txt",
                "status": "ready",
                "upload_date": stamp,
            },
            file_path="",
            collection_name="c",
        )


def test_a_corpus_deleted_and_ingested_again_under_a_run_is_noticed(tmp_path, monkeypatch):
    """Same documents, same chunk counts, but another process ingested them again.

    The end-of-task census compared counts only, so a corpus that another
    process deleted and ingested again looked unchanged once it was back, and
    the searches that ran while it was missing or half there were recorded.
    Every ingest stamps its documents with a new time; the census compares it.
    """
    import lasagna.core.database as database
    from lasagna.core.database import RAGDatabase
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    db = RAGDatabase(str(tmp_path / "rag.db"))
    monkeypatch.setattr(database, "get_rag_database", lambda: db)
    store = _SharedStore()
    store.rows = ["d1", "d1", "d2", "d2", "d2"]
    _register(db, "s", ["d1", "d2"], "2026-10-19T10:00:00+00:00")
    census = exp._corpus_census(store, "c", "s", ["d1", "d2"])
    slot = {"session_id": "s", "corpus_census": census, "_census_store": lambda: store}
    assert exp._corpus_changed(slot) is None

    # The other process deletes the corpus, ingests it again, and finishes.
    store.rows = []
    store.rows = ["d1", "d1", "d2", "d2", "d2"]
    _register(db, "s", ["d1", "d2"], "2026-10-19T10:30:00+00:00")
    changed = exp._corpus_changed(slot)
    assert changed and "ingested again" in changed
    assert "numbers are not recorded" in changed and "run this one again" in changed


def test_a_count_that_fails_once_is_asked_again(monkeypatch):
    """One failed count (a dropped connection) no longer discards the task's numbers."""
    from lasagna.knowledge_evaluation_sdk import experiment as exp

    monkeypatch.setattr(exp, "_RECOUNT_PAUSE", 0.0)
    store = _SharedStore()
    store.rows = ["d1", "d2"]
    census = exp._corpus_census(store, "c", "s", ["d1", "d2"])
    calls = {"n": 0}
    answer = store.query

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ConnectionError("reset by peer")
        return answer(*a, **k)

    store.query = flaky
    slot = {"session_id": "s", "corpus_census": census, "_census_store": lambda: store}
    assert exp._corpus_changed(slot) is None
    assert calls["n"] == 2

    def broken(*a, **k):
        raise ConnectionError("reset by peer")

    store.query = broken
    unknown = exp._corpus_changed(slot)
    assert unknown and "could not be counted" in unknown
