"""On a Milvus server, the reads that compare counts see the writes before them.

A Milvus server answers with the collection's consistency level, Bounded by
default, so a query moments after an insert misses the newest rows. On the
stack's Milvus 2.4 each count after an insert read one row short. The
benchmark's corpus census, taken right after ingestion, counted 0 of 24
documents; the end-of-task check counted 24, and every cold
``lasagna benchmark run`` in the Compose stack was refused as "changed while
this run used it". Milvus Lite has no staleness window, so a laptop never
showed it.

Strong costs about 200 ms per query on that server (Bounded: under 1 ms), so
it is asked for only inside ``read_fresh()``, which the census and the
pre-flight use; the search paths keep Bounded. pymilvus's query iterator
ignores ``consistency_level``, so ``iter_all`` on a server first runs one
Strong query, after which the scan sees every earlier write.

The fake server below behaves as the measured one: default reads see only the
rows that existed when the server last caught up, a Strong query catches it
up, and the iterator reads at the default level whatever it is asked.
"""

from __future__ import annotations

import threading

import pytest

from lasagna.knowledge_evaluation_sdk import experiment
from lasagna.vector_backends.milvus_backend import MilvusBackend

ROWS = [{"document_id": f"doc_{i}", "chunk_id": f"c{i}"} for i in range(24)]


class _BoundedServer:
    """A Milvus server whose default reads lag behind its writes."""

    def __init__(self, rows):
        self.rows = list(rows)
        self.applied = 0  # rows a default read can see
        self.calls = []

    def _visible(self, kwargs):
        if kwargs.get("consistency_level") == "Strong":
            self.applied = len(self.rows)
        return self.rows[: self.applied]

    def query(self, **kwargs):
        self.calls.append(("query", kwargs))
        rows = self._visible(kwargs)
        if kwargs.get("output_fields") == ["count(*)"]:
            return [{"count(*)": len(rows)}]
        return rows[: kwargs.get("limit", len(rows))]

    def query_iterator(self, **kwargs):
        self.calls.append(("query_iterator", kwargs))
        rows = self.rows[: self.applied]  # pymilvus ignores consistency_level here

        class _It:
            def __init__(self):
                self.batches = [rows] if rows else []

            def next(self):
                return self.batches.pop(0) if self.batches else []

            def close(self):
                pass

        return _It()


def _backend(**kw) -> MilvusBackend:
    backend = MilvusBackend(**kw)
    backend._client = _BoundedServer(ROWS)
    backend._ensure_loaded = lambda client, name: None
    return backend


@pytest.fixture
def server():
    return _backend(uri="http://milvus:19530")


def test_a_search_path_query_keeps_the_fast_default(server):
    server.query("rag_documents", limit=100)
    assert "consistency_level" not in server._client.calls[-1][1]


def test_inside_read_fresh_a_query_sees_every_row(server):
    with server.read_fresh():
        assert len(server.query("rag_documents", limit=100)) == 24
        assert server._client.calls[-1][1]["consistency_level"] == "Strong"
        assert len(server.query_raw("rag_documents", 'chunk_id != ""', limit=100)) == 24
        assert server._client.calls[-1][1]["consistency_level"] == "Strong"
    server.query("rag_documents", limit=100)
    assert "consistency_level" not in server._client.calls[-1][1]


def test_read_fresh_nests_and_stays_on_its_own_thread(server):
    seen = {}

    def other_thread():
        server.query("rag_documents", limit=1)
        seen["kwargs"] = server._client.calls[-1][1]

    with server.read_fresh():
        with server.read_fresh():
            pass
        server.query("rag_documents", limit=1)
        assert server._client.calls[-1][1]["consistency_level"] == "Strong"
        worker = threading.Thread(target=other_thread)
        worker.start()
        worker.join()
    assert "consistency_level" not in seen["kwargs"]


def test_a_full_scan_catches_the_server_up_first(server):
    rows = list(server.iter_all("rag_documents", partition_key="t_s1"))
    assert len(rows) == 24
    (probe_name, probe), (scan_name, scan) = server._client.calls[-2:]
    assert (probe_name, scan_name) == ("query", "query_iterator")
    assert probe["consistency_level"] == "Strong"
    assert probe["filter"] == scan["filter"]
    assert probe["partition_names"] == scan["partition_names"] == ["t_s1"]


def test_without_the_catch_up_the_scan_comes_up_short():
    """The iterator alone reads stale: the property the catch-up exists for."""
    client = _BoundedServer(ROWS)
    it = client.query_iterator(collection_name="c", filter="", consistency_level="Strong")
    assert it.next() == []


def test_the_census_after_ingest_matches_the_end_of_task_count(server):
    ids = [r["document_id"] for r in ROWS]
    at_publish = experiment._count_present(server, "rag_documents", "s1", ids)
    at_end = experiment._count_present(server, "rag_documents", "s1", ids)
    assert at_publish == at_end == (24, 24)


def test_the_preflight_sees_documents_just_written(server):
    ids = [r["document_id"] for r in ROWS]
    assert experiment._present_document_ids(server, "rag_documents", "s1", ids) == set(ids)


def test_milvus_lite_gets_no_extra_read(tmp_path):
    lite = _backend(db_path=str(tmp_path / "rag.db"))
    lite._client.applied = len(ROWS)  # Lite has no staleness window
    with lite.read_fresh():
        lite.query("rag_documents", limit=100)
    assert "consistency_level" not in lite._client.calls[-1][1]
    list(lite.iter_all("rag_documents"))
    assert [name for name, _ in lite._client.calls[-1:]] == ["query_iterator"]


def test_a_store_without_read_fresh_still_counts():
    class _Plain:
        def query(self, **kwargs):
            return [{"document_id": "d1"}]

    assert experiment._count_present(_Plain(), "c", "s", ["d1"]) == (1, 1)


# The Compose stack runs a Milvus server behind a PooledVectorBackend (the
# registry pools whenever pool_size > 1). The pool kept the base no-op
# read_fresh, so the census read stale there although every test above
# passed: a cold run in the stack was refused again. These go through it.


def _pool(size: int):
    from lasagna.vector_backends.pool import PooledVectorBackend

    def factory():
        backend = _backend(uri="http://milvus:19530")
        backend.connect = lambda: None
        return backend

    pool = PooledVectorBackend(factory=factory, pool_size=size)
    pool.connect()
    return pool


def test_the_census_through_a_pool_sees_every_row():
    pool = _pool(2)
    ids = [r["document_id"] for r in ROWS]
    assert experiment._count_present(pool, "rag_documents", "s1", ids) == (24, 24)
    assert experiment._present_document_ids(pool, "rag_documents", "s1", ids) == set(ids)


def test_read_fresh_through_a_pool_reaches_every_instance():
    pool = _pool(3)
    with pool.read_fresh():
        for _ in range(6):  # the pool rotates its instances
            pool.query("rag_documents", limit=1)
    sent = [kw for member in pool._members for name, kw in member._client.calls if name == "query"]
    assert len(sent) == 6 and all(kw.get("consistency_level") == "Strong" for kw in sent)
    pool.query("rag_documents", limit=1)
    last = [m._client.calls[-1][1] for m in pool._members if m._client.calls]
    assert any("consistency_level" not in kw for kw in last)


def test_a_full_scan_through_a_pool_catches_up_and_frees_its_instance():
    pool = _pool(1)
    rows = pool.iter_all("rag_documents")
    first = next(rows)
    # The only instance is free again while the caller consumes the rows.
    assert pool.query("rag_documents", limit=1) is not None
    assert len([first, *rows]) == 24
    names = [name for name, _ in pool._members[0]._client.calls]
    assert names[:2] == ["query", "query_iterator"]


def test_a_failed_catch_up_raises_instead_of_scanning_stale(server):
    def refuse(**kwargs):
        if kwargs.get("output_fields") == ["count(*)"]:
            raise TimeoutError("server busy")
        return []

    server._client.query = refuse
    with pytest.raises(TimeoutError):
        list(server.iter_all("rag_documents"))


def test_the_paginated_fallback_reads_fresh(server):
    def no_iterator(**kwargs):
        raise AttributeError("no query_iterator in this pymilvus")

    server._client.query_iterator = no_iterator
    assert len(list(server.iter_all("rag_documents"))) == 24
    pages = [kw for name, kw in server._client.calls if name == "query"][1:]
    assert pages and all(kw.get("consistency_level") == "Strong" for kw in pages)


def test_a_pool_says_whether_it_holds_milvus_lite(tmp_path):
    from lasagna.vector_backends.pool import PooledVectorBackend

    def lite():
        backend = _backend(db_path=str(tmp_path / "rag.db"))
        backend.connect = lambda: None
        return backend

    pool = PooledVectorBackend(factory=lite, pool_size=2)
    assert pool.is_lite is False  # nothing connected yet
    pool.connect()
    assert pool.is_lite is True
    assert _pool(2).is_lite is False
