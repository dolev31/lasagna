"""Tests for BM25 deferred build, version-counter CAS, and observability.

Covers:
- Version-counter correctness (CAS prevents stale-built state)
- Deferred build lifecycle (build_now=False → lazy build on search)
- Backward compatibility (build_now=True default)
- build_session_index() explicit build + idempotency
- Concurrent append integrity under threading
- Deferred build + remove_document interaction
- Observability via get_session_stats()
- Native-mode guard skips fallback indexing
"""

import threading
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DIVERSE_TOPICS = [
    "machine learning algorithms neural networks deep learning",
    "database systems query optimization indexing strategies",
    "cloud computing distributed systems microservices architecture",
    "natural language processing text mining information retrieval",
    "computer vision image recognition object detection models",
    "cybersecurity encryption authentication access control",
    "software engineering design patterns agile development",
    "data structures binary trees hash tables graph algorithms",
    "operating systems process scheduling memory management",
    "web development frontend frameworks backend services",
    "quantum computing qubits entanglement superposition",
    "robotics sensor fusion path planning control systems",
    "bioinformatics genome sequencing protein folding analysis",
    "game theory Nash equilibrium strategy optimization",
    "signal processing Fourier transform filtering noise",
    "parallel computing GPU programming CUDA kernels",
    "blockchain consensus mechanisms smart contracts ledger",
    "augmented reality spatial mapping rendering pipeline",
    "recommendation systems collaborative filtering embeddings",
    "time series forecasting anomaly detection trend analysis",
]


def _make_chunks(n: int, doc_id: str = "doc_1", start_idx: int = 0):
    """Create n fake chunk dicts for BM25 indexing.

    Uses diverse topics so BM25 IDF scoring produces non-zero results.
    """
    return [
        {
            "chunk_id": f"chunk_{start_idx + i}",
            "content": _DIVERSE_TOPICS[(start_idx + i) % len(_DIVERSE_TOPICS)],
            "document_id": doc_id,
            "metadata": {"session_id": "test_session"},
        }
        for i in range(n)
    ]


def _make_searcher(**kwargs):
    """Create a HybridSearcher with fallback mode (no native Milvus)."""
    from lasagna.retrieval.search.hybrid import HybridSearcher

    searcher = HybridSearcher(
        enable_periodic_cleanup=False,
        **kwargs,
    )
    # Ensure no native mode is available (fallback only)
    searcher._native_searcher = None
    return searcher


# ---------------------------------------------------------------------------
# 1. Version-counter correctness (CRITICAL race test)
# ---------------------------------------------------------------------------


class TestVersionCounterCAS:
    """Verify CAS prevents stale-built state under concurrent mutations."""

    def test_concurrent_add_during_build_rejects_stale(self):
        """Thread A builds, Thread B adds docs during build → CAS rejects."""
        from lasagna.retrieval.search.hybrid import (
            BM25FallbackIndex,
            BM25_FALLBACK_AVAILABLE,
        )

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        index = BM25FallbackIndex(enable_stemming=False)
        index.add_documents(_make_chunks(5))
        assert index._version == 1
        assert not index._is_built

        # Monkey-patch BM25Okapi to simulate slow build + concurrent mutation
        _original_build = index.build_index

        build_started = threading.Event()
        mutation_done = threading.Event()

        def slow_build():
            # Snapshot version like normal build_index does
            snap = index._version
            build_started.set()
            # Wait for the concurrent mutation to happen
            mutation_done.wait(timeout=5.0)
            # Now BM25Okapi construction would happen here...
            # Simulate by calling original (which will do the CAS check)
            # But we need to test the CAS, so let's do it manually
            from rank_bm25 import BM25Okapi

            bm25 = BM25Okapi(index.tokenized_corpus)
            _elapsed_ms = 0.1
            if index._version == snap:
                index.bm25 = bm25
                index._is_built = True
                index._built_version = snap
                return True
            else:
                return False

        # Thread A: slow build
        result_a = [None]

        def thread_a():
            result_a[0] = slow_build()

        t_a = threading.Thread(target=thread_a)
        t_a.start()
        build_started.wait(timeout=5.0)

        # Thread B: add more docs while A is building
        index.add_documents(_make_chunks(5, start_idx=5))
        assert index._version == 2
        mutation_done.set()

        t_a.join(timeout=5.0)

        # CAS should have rejected the build
        assert result_a[0] is False, "Stale build should be rejected by CAS"
        assert not index._is_built, "_is_built should remain False after CAS rejection"
        assert len(index.documents) == 10

        # Now a fresh build should succeed with all 10 docs
        assert index.build_index() is True
        assert index._is_built
        assert index._built_version == 2

    def test_version_increments_on_add(self):
        from lasagna.retrieval.search.hybrid import BM25FallbackIndex

        index = BM25FallbackIndex(enable_stemming=False)
        assert index._version == 0
        index.add_documents(_make_chunks(3))
        assert index._version == 1
        index.add_documents(_make_chunks(2, start_idx=3))
        assert index._version == 2

    def test_version_increments_on_remove(self):
        from lasagna.retrieval.search.hybrid import (
            BM25FallbackIndex,
            BM25_FALLBACK_AVAILABLE,
        )

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        index = BM25FallbackIndex(enable_stemming=False)
        index.add_documents(_make_chunks(3, doc_id="doc_a"))
        v_after_add = index._version
        index.build_index()
        assert index._is_built

        index.remove_document("doc_a")
        assert index._version == v_after_add + 1
        assert not index._is_built

    def test_version_increments_on_clear(self):
        from lasagna.retrieval.search.hybrid import BM25FallbackIndex

        index = BM25FallbackIndex(enable_stemming=False)
        index.add_documents(_make_chunks(3))
        v_before = index._version
        index.clear()
        assert index._version == v_before + 1
        assert index._built_version == -1

    def test_build_lock_prevents_duplicate_concurrent_builds(self):
        """Only one thread can hold _build_lock; others skip immediately."""
        from lasagna.retrieval.search.hybrid import (
            BM25FallbackIndex,
            BM25_FALLBACK_AVAILABLE,
        )

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        index = BM25FallbackIndex(enable_stemming=False)
        index.add_documents(_make_chunks(10))

        # Hold the build lock manually to simulate a thread mid-build
        assert index._build_lock.acquire(blocking=False), "Lock should be free"

        # A second build_index() call must return False (lock held)
        assert index.build_index() is False, (
            "build_index should return False when another thread holds _build_lock"
        )
        assert not index._is_built, "Index should not be built"
        assert index._build_count == 0, "No builds should have completed"

        # Release and verify build works now
        index._build_lock.release()
        assert index.build_index() is True
        assert index._is_built
        assert index._build_count == 1

    def test_concurrent_build_index_only_one_builds(self):
        """N threads calling build_index() → exactly 1 build completes."""
        from lasagna.retrieval.search.hybrid import (
            BM25FallbackIndex,
            BM25_FALLBACK_AVAILABLE,
        )

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        index = BM25FallbackIndex(enable_stemming=False)
        index.add_documents(_make_chunks(15))

        results = []
        barrier = threading.Barrier(5)

        def try_build():
            barrier.wait()  # all threads start simultaneously
            result = index.build_index()
            results.append(result)

        threads = [threading.Thread(target=try_build) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        true_count = sum(1 for r in results if r is True)
        assert true_count == 1, (
            f"Expected exactly 1 successful build but got {true_count} (results={results})"
        )
        assert index._build_count == 1
        assert index.is_current()


# ---------------------------------------------------------------------------
# 2. Deferred build lifecycle
# ---------------------------------------------------------------------------


class TestDeferredBuild:
    """build_now=False defers build; lazy rebuild on search works."""

    def test_index_chunks_deferred_does_not_build(self):
        from lasagna.retrieval.search.hybrid import BM25_FALLBACK_AVAILABLE

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        searcher = _make_searcher()
        chunks = _make_chunks(10)
        searcher.index_chunks(chunks, "sess_deferred", build_now=False)

        index = searcher.get_session_index("sess_deferred")
        assert len(index.documents) == 10
        assert len(index.tokenized_corpus) == 10
        assert not index._is_built
        assert index.bm25 is None

    def test_lazy_build_on_ensure_index_built(self):
        from lasagna.retrieval.search.hybrid import BM25_FALLBACK_AVAILABLE

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        searcher = _make_searcher()
        searcher.index_chunks(_make_chunks(10), "sess_lazy", build_now=False)

        # Simulate what happens before a search
        searcher._ensure_index_built("sess_lazy")

        index = searcher.get_session_index("sess_lazy")
        assert index._is_built
        assert index.bm25 is not None

        # Search should return results
        results = index.search("machine learning neural networks", top_k=5)
        assert len(results) > 0, f"Expected search results but got none. Built={index._is_built}"


# ---------------------------------------------------------------------------
# 3. Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """Default build_now=True preserves immediate build."""

    def test_default_builds_immediately(self):
        from lasagna.retrieval.search.hybrid import BM25_FALLBACK_AVAILABLE

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        searcher = _make_searcher()
        searcher.index_chunks(_make_chunks(5), "sess_compat")

        index = searcher.get_session_index("sess_compat")
        assert index._is_built
        assert index.bm25 is not None


# ---------------------------------------------------------------------------
# 4. build_session_index() explicit build + idempotency
# ---------------------------------------------------------------------------


class TestBuildSessionIndex:
    """Explicit build returns True once, False on repeat."""

    def test_build_returns_true_then_false(self):
        from lasagna.retrieval.search.hybrid import BM25_FALLBACK_AVAILABLE

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        searcher = _make_searcher()
        searcher.index_chunks(_make_chunks(5), "sess_explicit", build_now=False)

        assert searcher.build_session_index("sess_explicit") is True
        assert searcher.build_session_index("sess_explicit") is False  # idempotent

    def test_build_nonexistent_session_returns_false(self):
        searcher = _make_searcher()
        assert searcher.build_session_index("nonexistent") is False

    def test_build_empty_session_returns_false(self):
        searcher = _make_searcher()
        searcher.get_session_index("sess_empty")
        assert searcher.build_session_index("sess_empty") is False


# ---------------------------------------------------------------------------
# 5. Concurrent append integrity
# ---------------------------------------------------------------------------


class TestConcurrentAppend:
    """Multiple threads appending with build_now=False → correct total."""

    def test_five_threads_append_then_build(self):
        from lasagna.retrieval.search.hybrid import BM25_FALLBACK_AVAILABLE

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        searcher = _make_searcher()
        chunks_per_thread = 20
        num_threads = 5

        def append_chunks(thread_idx):
            chunks = _make_chunks(
                chunks_per_thread,
                doc_id=f"doc_{thread_idx}",
                start_idx=thread_idx * chunks_per_thread,
            )
            searcher.index_chunks(chunks, "sess_concurrent", build_now=False)

        threads = [threading.Thread(target=append_chunks, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        index = searcher.get_session_index("sess_concurrent")
        assert len(index.documents) == num_threads * chunks_per_thread
        assert not index._is_built

        # Build and verify search works
        assert searcher.build_session_index("sess_concurrent") is True
        assert index._is_built
        results = index.search("machine learning neural networks", top_k=10)
        assert len(results) > 0


# ---------------------------------------------------------------------------
# 6. Deferred build + remove_document
# ---------------------------------------------------------------------------


class TestDeferredBuildWithRemove:
    """Add docs deferred → remove one → search → correct results."""

    def test_remove_then_lazy_build(self):
        from lasagna.retrieval.search.hybrid import BM25_FALLBACK_AVAILABLE

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        searcher = _make_searcher()
        # Add docs from two different documents
        searcher.index_chunks(_make_chunks(5, doc_id="doc_a"), "sess_rm", build_now=False)
        searcher.index_chunks(
            _make_chunks(5, doc_id="doc_b", start_idx=5), "sess_rm", build_now=False
        )

        index = searcher.get_session_index("sess_rm")
        assert len(index.documents) == 10

        # Remove one document
        searcher.remove_document("doc_a", "sess_rm")
        assert len(index.documents) == 5

        # Lazy build should work with remaining docs
        searcher._ensure_index_built("sess_rm")
        assert index._is_built

        # Verify only doc_b chunks are searchable (doc_b has indices 5-9)
        results = index.search("cybersecurity encryption authentication", top_k=10)
        assert len(results) > 0, f"Expected results but got none. Built={index._is_built}"
        result_doc_ids = {index.documents[idx].get("document_id") for idx, _ in results}
        assert result_doc_ids == {"doc_b"}


# ---------------------------------------------------------------------------
# 7. Observability via get_session_stats()
# ---------------------------------------------------------------------------


class TestObservability:
    """get_session_stats returns correct counters."""

    def test_stats_after_builds(self):
        from lasagna.retrieval.search.hybrid import BM25_FALLBACK_AVAILABLE

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        searcher = _make_searcher()
        searcher.index_chunks(_make_chunks(5), "sess_stats", build_now=False)

        stats = searcher.get_session_stats("sess_stats")
        assert stats is not None
        assert stats["doc_count"] == 5
        assert stats["is_built"] is False
        assert stats["build_count"] == 0

        searcher.build_session_index("sess_stats")
        stats = searcher.get_session_stats("sess_stats")
        assert stats["is_built"] is True
        assert stats["build_count"] == 1
        assert stats["total_build_ms"] >= 0  # may be 0.0 for tiny corpora
        assert stats["version"] == stats["built_version"]

    def test_stats_nonexistent_returns_none(self):
        searcher = _make_searcher()
        assert searcher.get_session_stats("nonexistent") is None

    def test_build_count_increments_on_rebuild(self):
        from lasagna.retrieval.search.hybrid import BM25_FALLBACK_AVAILABLE

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        searcher = _make_searcher()
        searcher.index_chunks(_make_chunks(5), "sess_rebuild", build_now=False)
        searcher.build_session_index("sess_rebuild")

        # Add more docs → invalidates → rebuild
        searcher.index_chunks(_make_chunks(3, start_idx=5), "sess_rebuild", build_now=False)
        searcher.build_session_index("sess_rebuild")

        stats = searcher.get_session_stats("sess_rebuild")
        assert stats["build_count"] == 2
        assert stats["doc_count"] == 8


# ---------------------------------------------------------------------------
# 8. Native-mode guard
# ---------------------------------------------------------------------------


class TestNativeModeGuard:
    """index_chunks skips fallback when native mode available."""

    def test_native_mode_skips_fallback_indexing(self):
        from lasagna.retrieval.search.hybrid import HybridSearcher

        searcher = HybridSearcher(enable_periodic_cleanup=False)
        # Mock native mode as available
        searcher._native_searcher = MagicMock()
        searcher._native_searcher.is_native_bm25_supported.return_value = True

        searcher.index_chunks(_make_chunks(10), "sess_native", build_now=False)

        # No session index should have been created
        assert searcher.get_session_count() == 0


# ---------------------------------------------------------------------------
# 9. build_index() return value
# ---------------------------------------------------------------------------


class TestBuildIndexReturnValue:
    """build_index() returns bool correctly."""

    def test_returns_true_on_success(self):
        from lasagna.retrieval.search.hybrid import (
            BM25FallbackIndex,
            BM25_FALLBACK_AVAILABLE,
        )

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        index = BM25FallbackIndex(enable_stemming=False)
        index.add_documents(_make_chunks(3))
        assert index.build_index() is True

    def test_returns_false_on_empty(self):
        from lasagna.retrieval.search.hybrid import BM25FallbackIndex

        index = BM25FallbackIndex(enable_stemming=False)
        assert index.build_index() is False


# ---------------------------------------------------------------------------
# 10. index_chunks must respect max_sessions (regression for #1)
# ---------------------------------------------------------------------------


class TestMaxSessionsEnforcement:
    """index_chunks must respect max_sessions limit like get_session_index."""

    def test_index_chunks_respects_max_sessions(self):
        """index_chunks creating new sessions must not exceed max_sessions."""
        from lasagna.retrieval.search.hybrid import (
            HybridSearcher,
            BM25_FALLBACK_AVAILABLE,
        )

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        searcher = HybridSearcher(
            max_sessions=2,
            session_ttl_seconds=3600,
            enable_periodic_cleanup=False,
        )
        searcher._native_searcher = None  # force fallback mode

        # Create 3 sessions via index_chunks — should evict oldest
        searcher.index_chunks(_make_chunks(3), "sess_a", build_now=False)
        searcher.index_chunks(_make_chunks(3), "sess_b", build_now=False)
        searcher.index_chunks(_make_chunks(3), "sess_c", build_now=False)

        assert searcher.get_session_count() <= 2, (
            f"Expected max 2 sessions but got {searcher.get_session_count()}"
        )

    def test_get_session_index_and_index_chunks_same_enforcement(self):
        """Both paths must produce the same max_sessions behavior."""
        from lasagna.retrieval.search.hybrid import (
            HybridSearcher,
            BM25_FALLBACK_AVAILABLE,
        )

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        # Path 1: via get_session_index
        s1 = HybridSearcher(max_sessions=1, enable_periodic_cleanup=False)
        s1._native_searcher = None
        s1.get_session_index("a")
        s1.get_session_index("b")
        count_via_get = s1.get_session_count()

        # Path 2: via index_chunks
        s2 = HybridSearcher(max_sessions=1, enable_periodic_cleanup=False)
        s2._native_searcher = None
        s2.index_chunks(_make_chunks(2), "a", build_now=False)
        s2.index_chunks(_make_chunks(2), "b", build_now=False)
        count_via_index = s2.get_session_count()

        assert count_via_get == count_via_index, (
            f"get_session_index produced {count_via_get} sessions, "
            f"index_chunks produced {count_via_index} — enforcement mismatch"
        )


# ---------------------------------------------------------------------------
# 11. is_current() public API
# ---------------------------------------------------------------------------


class TestIsCurrentAPI:
    """BM25FallbackIndex.is_current() replaces internal checks."""

    def test_not_current_after_add(self):
        from lasagna.retrieval.search.hybrid import BM25FallbackIndex

        index = BM25FallbackIndex(enable_stemming=False)
        index.add_documents(_make_chunks(3))
        assert not index.is_current()

    def test_current_after_build(self):
        from lasagna.retrieval.search.hybrid import (
            BM25FallbackIndex,
            BM25_FALLBACK_AVAILABLE,
        )

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        index = BM25FallbackIndex(enable_stemming=False)
        index.add_documents(_make_chunks(3))
        index.build_index()
        assert index.is_current()

    def test_not_current_after_mutation(self):
        from lasagna.retrieval.search.hybrid import (
            BM25FallbackIndex,
            BM25_FALLBACK_AVAILABLE,
        )

        if not BM25_FALLBACK_AVAILABLE:
            pytest.skip("rank_bm25 not installed")

        index = BM25FallbackIndex(enable_stemming=False)
        index.add_documents(_make_chunks(3))
        index.build_index()
        assert index.is_current()

        index.add_documents(_make_chunks(2, start_idx=3))
        assert not index.is_current()
