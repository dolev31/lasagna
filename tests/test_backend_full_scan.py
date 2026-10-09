"""Scanning a whole collection must not stop partway through.

`query()` takes limit/offset, and Milvus has no native offset — it asks for
`limit + offset` rows and slices locally. So paginating a full scan is quadratic
*and* fails outright once `limit + offset` passes 16384. A few thousand
documents chunked normally is already past that.

The failure mode is what makes this worth pinning. A BM25 index built from the
first 16k chunks of a larger corpus still returns results, so the experiment
reports a plausible retrieval number computed against a corpus that was never
fully indexed. Nothing downstream can tell the difference.

`iter_all` is deliberately concrete on the base class rather than abstract:
vector backends are a published plugin point, and adding an abstract method
would break every third-party backend on upgrade.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from lasagna.vector_backends.base import VectorStoreAdapter
from lasagna.vector_backends.filters import FilterExpression


class _PagedBackend(VectorStoreAdapter):
    """A backend that emulates offset by over-fetching, as Milvus does.

    Raises past the cap rather than truncating, which is what the real server
    does — truncating silently would hide the bug this test exists to catch.
    """

    CAP = 16_384

    def __init__(self, n_rows: int, cap: Optional[int] = None):
        self.rows = [{"chunk_id": f"c{i}", "document_id": f"d{i // 3}"} for i in range(n_rows)]
        self.cap = self.CAP if cap is None else cap
        self.calls = 0

    backend_name = "paged"

    def query(
        self,
        collection_name: str,
        filter_expr: Optional[FilterExpression] = None,
        output_fields: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0,
        partition_key: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        self.calls += 1
        if limit + offset > self.cap:
            raise RuntimeError(f"limit+offset {limit + offset} exceeds cap {self.cap}")
        return self.rows[offset : offset + limit]


# `iter_all` and `list_document_ids` are concrete and depend only on `query()`,
# which is implemented above. Connecting, indexing and searching are irrelevant
# here, so the ABC's remaining requirements are waived rather than stubbed.
_PagedBackend.__abstractmethods__ = frozenset()


class TestFullScanIsExhaustive:
    def test_reads_every_row_below_the_cap(self):
        backend = _PagedBackend(2_500)
        assert len(list(backend.iter_all("c"))) == 2_500

    def test_reads_every_row_past_the_cap(self):
        """The regression: 20k rows is beyond limit+offset, and must still scan."""
        backend = _PagedBackend(20_000, cap=999_999)
        rows = list(backend.iter_all("c"))
        assert len(rows) == 20_000
        assert rows[-1]["chunk_id"] == "c19999"

    def test_an_empty_collection_yields_nothing(self):
        assert list(_PagedBackend(0).iter_all("c")) == []

    def test_a_partial_final_page_ends_the_scan(self):
        backend = _PagedBackend(1_500)
        assert len(list(backend.iter_all("c", page_size=1000))) == 1_500
        assert backend.calls == 2, "should stop on the short page, not query again"

    def test_it_streams_rather_than_materialising(self):
        """A full corpus scan must not require the whole corpus in memory."""
        backend = _PagedBackend(5_000)
        it = backend.iter_all("c")
        next(it)
        assert backend.calls == 1, "consuming one row should not have read every page"


class TestTheDefaultStillFailsPastTheCap:
    """Proves the cap is real, so the Milvus override is not cargo-cult.

    If this ever stops raising, the base-class path is safe and the native
    cursor override could be reconsidered.
    """

    def test_paginating_past_the_cap_raises(self):
        backend = _PagedBackend(20_000)
        with pytest.raises(RuntimeError, match="exceeds cap"):
            list(backend.iter_all("c"))


class TestListDocumentIdsUsesTheScan:
    def test_deduplicates_across_pages(self):
        """3 chunks per document, so dedup must span page boundaries."""
        backend = _PagedBackend(3_000)
        assert len(backend.list_document_ids("c")) == 1_000

    def test_respects_limit(self):
        assert len(_PagedBackend(3_000).list_document_ids("c", limit=10)) == 10

    def test_offset_skips_documents_not_rows(self):
        """Row-offset paging over distinct values re-returns documents.

        The previous implementation passed `offset` straight to `query()` as a
        row offset while deduplicating locally, so page 2 had no memory of page
        1 and repeated documents it had already returned.
        """
        backend = _PagedBackend(3_000)
        first = backend.list_document_ids("c", limit=5)
        second = backend.list_document_ids("c", limit=5, offset=5)
        assert not set(first) & set(second), f"overlap between pages: {first} vs {second}"
        assert second == [f"d{i}" for i in range(5, 10)]

    def test_reads_documents_past_the_cap(self):
        backend = _PagedBackend(60_000, cap=999_999)
        assert len(backend.list_document_ids("c", limit=100_000)) == 20_000


class TestPluginCompatibility:
    def test_iter_all_is_not_abstract(self):
        """An abstract method here would break every installed third-party backend."""
        assert "iter_all" not in VectorStoreAdapter.__abstractmethods__

    def test_milvus_overrides_it(self):
        from lasagna.vector_backends.milvus_backend import MilvusBackend

        assert MilvusBackend.iter_all is not VectorStoreAdapter.iter_all, (
            "Milvus must use its native cursor; the base implementation cannot "
            "read past the limit+offset cap"
        )


class TestAnUnsupportedFeatureRefusesRatherThanSubstitutes:
    """Qdrant answered a hybrid query with dense-only results and a warning.

    For an evaluation framework that is the worst available option: a
    hybrid-vs-dense comparison on Qdrant compared dense against dense, produced
    a difference of about zero, and looked exactly like a finding. The backend
    already advertised `supports_hybrid_search=False`, so the behaviour also
    contradicted its own capability declaration.
    """

    def test_qdrant_declares_no_hybrid_support(self):
        from lasagna.vector_backends.qdrant_backend import QdrantBackend

        backend = QdrantBackend.__new__(QdrantBackend)
        backend._capabilities = None
        assert backend.capabilities.supports_hybrid_search is False

    def test_asking_for_hybrid_raises_instead_of_returning_dense(self):
        from lasagna.vector_backends.qdrant_backend import QdrantBackend

        backend = QdrantBackend.__new__(QdrantBackend)
        with pytest.raises(NotImplementedError, match="hybrid"):
            backend.hybrid_search(
                collection_name="c",
                query_embedding=[0.1, 0.2],
                query_text="q",
            )

    def test_the_error_says_what_to_do_instead(self):
        from lasagna.vector_backends.qdrant_backend import QdrantBackend

        backend = QdrantBackend.__new__(QdrantBackend)
        with pytest.raises(NotImplementedError) as exc:
            backend.hybrid_search(collection_name="c", query_embedding=[0.1], query_text="q")
        message = str(exc.value)
        assert "milvus" in message or "pgvector" in message, "no working alternative named"
        assert "enable_hybrid_search=False" in message, "no way out named"
