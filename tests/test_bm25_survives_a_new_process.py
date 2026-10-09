"""Hybrid search must stay hybrid in a process that did not upload.

The BM25 index lives in memory and was filled only at upload, so the next CLI
call or a server ran dense-only with a DEBUG line: the same query on the same
store returned different results depending on which process asked.
"""

from __future__ import annotations

import pytest

from lasagna.retrieval.search import hybrid
from lasagna.retrieval.search.hybrid import HybridSearcher

pytestmark = pytest.mark.skipif(
    not hybrid.BM25_FALLBACK_AVAILABLE, reason="rank_bm25 not installed"
)

ROWS = [
    {"chunk_id": f"c{i}", "content": text, "document_id": f"d{i}", "metadata": {}}
    for i, text in enumerate(
        ["the quick brown fox", "lazy dogs sleep all day", "foxes and hounds", "tea at noon"]
    )
]


class _Store:
    backend_name = "milvus"

    def __init__(self, rows):
        self.rows, self.scans = rows, 0

    def iter_all(self, collection_name, filter_expr=None, output_fields=None, **kw):
        self.scans += 1
        return iter([dict(r) for r in self.rows])


def _searcher():
    s = HybridSearcher(enable_periodic_cleanup=False)
    s.is_native_mode_available = lambda: False
    s._backend_supports_native_hybrid = lambda sid=None: False
    return s


def test_a_fresh_process_fills_the_index_from_the_store():
    s, store = _searcher(), _Store(ROWS)
    assert s.hydrate_session_index("sess", store, "chunks", "filter") == 4
    index = s.get_session_index("sess")
    assert index._is_built
    assert [d["chunk_id"] for d in index.documents] == ["c0", "c1", "c2", "c3"]
    hits = index.search("noon", top_k=2)  # in one document, stemmed or not
    assert hits, "BM25 must answer once filled"


def test_it_runs_once_and_adds_only_what_is_missing():
    s, store = _searcher(), _Store(ROWS)
    # This process uploaded c0 itself; the rest came from an earlier process.
    s.index_chunks([dict(ROWS[0])], "sess")
    assert s.hydrate_session_index("sess", store, "chunks", "filter") == 3
    assert s.hydrate_session_index("sess", store, "chunks", "filter") == 0
    assert store.scans == 1
    assert sorted(d["chunk_id"] for d in s.get_session_index("sess").documents) == [
        "c0",
        "c1",
        "c2",
        "c3",
    ]


def test_an_evicted_session_is_filled_again():
    s, store = _searcher(), _Store(ROWS)
    s.hydrate_session_index("sess", store, "chunks", "filter")
    s.clear_session("sess")
    assert s.hydrate_session_index("sess", store, "chunks", "filter") == 4


def test_falling_back_to_dense_only_is_a_warning_and_a_flag():
    from loguru import logger

    s = _searcher()
    seen = []
    sink = logger.add(lambda m: seen.append(m.record["level"].name), level="WARNING")
    try:
        _, metrics = s._fallback_hybrid_search("q", [], "empty", top_k=3, alpha=0.5)
    finally:
        logger.remove(sink)
    assert metrics["hybrid_mode"] == "dense_only_fallback"
    assert "WARNING" in seen


class _FlakyStore(_Store):
    failed = False

    def iter_all(self, *a, **kw):
        if not self.failed:
            self.failed = True
            self.scans += 1
            raise TimeoutError("store busy")
        return super().iter_all(*a, **kw)


def test_a_scan_that_failed_is_tried_again():
    # The collection was marked before the scan ran, so one timeout left BM25
    # empty for the life of the process and hybrid search silently dense-only.
    s, store = _searcher(), _FlakyStore(ROWS)
    assert s.hydrate_session_index("sess", store, "chunks", "filter") == 0
    assert s.hydrate_session_index("sess", store, "chunks", "filter") == 4
    assert store.scans == 2


def test_a_concurrent_first_search_waits_for_the_scan():
    import threading
    import time

    class _SlowStore(_Store):
        def iter_all(self, *a, **kw):
            time.sleep(0.2)
            return super().iter_all(*a, **kw)

    s, store = _searcher(), _SlowStore(ROWS)
    first = threading.Thread(target=s.hydrate_session_index, args=("sess", store, "chunks", "f"))
    first.start()
    time.sleep(0.05)  # the first call holds the scan
    s.hydrate_session_index("sess", store, "chunks", "f")
    # By the time the second call returns, the index is filled.
    assert len(s.get_session_index("sess").documents) == 4
    first.join()
    assert store.scans == 1
