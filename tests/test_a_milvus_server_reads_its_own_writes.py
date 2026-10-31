"""On a Milvus server, a filter read moments after a write sees that write.

A Milvus server reads with the collection's consistency level, Bounded by
default, so a query right after an insert misses the newest rows. On the
stack's Milvus 2.4 each count after an insert read one row short. The
benchmark's corpus census is taken right after ingestion: it counted 0 of 24
documents, the end-of-task check counted 24, and every cold `lasagna benchmark
run` in the Compose stack was refused as "changed while this run used it".
Milvus Lite has no staleness window, so it never showed on a laptop.

Filter reads (``query``, ``iter_all``, ``query_raw``) now ask a server for
Strong consistency. The fake server below behaves as the real one did: without
Strong, it hides the rows written since its last read.
"""

from __future__ import annotations

import pytest

from lasagna.knowledge_evaluation_sdk import experiment
from lasagna.vector_backends.milvus_backend import MilvusBackend


class _BoundedServer:
    """A Milvus server whose default reads lag behind its writes."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.visible = 0  # rows a Bounded read can see
        self.calls = []

    def _read(self, kwargs):
        self.calls.append(kwargs)
        if kwargs.get("consistency_level") == "Strong":
            return list(self.rows)
        seen, self.visible = self.rows[: self.visible], len(self.rows)
        return seen

    def query(self, **kwargs):
        rows = self._read(kwargs)
        return rows[: kwargs.get("limit", len(rows))]

    def query_iterator(self, **kwargs):
        rows = self._read(kwargs)

        class _It:
            def __init__(self):
                self.batches = [rows] if rows else []

            def next(self):
                return self.batches.pop(0) if self.batches else []

            def close(self):
                pass

        return _It()

    def load_collection(self, *a, **kw):
        pass

    def get_load_state(self, *a, **kw):
        return {"state": "Loaded"}


ROWS = [{"document_id": f"doc_{i}", "chunk_id": f"c{i}"} for i in range(24)]


@pytest.fixture
def server(monkeypatch):
    backend = MilvusBackend(uri="http://milvus:19530")
    backend._client = _BoundedServer(ROWS)
    monkeypatch.setattr(backend, "_ensure_loaded", lambda client, name: None)
    return backend


def test_a_query_right_after_ingest_sees_every_row(server):
    assert len(server.query("rag_documents", limit=100)) == 24
    assert server._client.calls[-1]["consistency_level"] == "Strong"


def test_iter_all_and_query_raw_read_strong_too(server):
    assert sum(1 for _ in server.iter_all("rag_documents")) == 24
    assert server._client.calls[-1]["consistency_level"] == "Strong"
    assert len(server.query_raw("rag_documents", 'chunk_id != ""', limit=100)) == 24
    assert server._client.calls[-1]["consistency_level"] == "Strong"


def test_the_census_after_ingest_matches_the_end_of_task_count(server):
    ids = [r["document_id"] for r in ROWS]
    at_publish = experiment._count_present(server, "rag_documents", "s1", ids)
    at_end = experiment._count_present(server, "rag_documents", "s1", ids)
    assert at_publish == at_end == (24, 24)


def test_milvus_lite_gets_no_extra_argument(tmp_path, monkeypatch):
    backend = MilvusBackend(db_path=str(tmp_path / "rag.db"))
    backend._client = _BoundedServer(ROWS)
    monkeypatch.setattr(backend, "_ensure_loaded", lambda client, name: None)
    backend.query("rag_documents", limit=100)
    assert "consistency_level" not in backend._client.calls[-1]
