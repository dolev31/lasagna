"""Tests for the fast corpus ingestion pipeline (_fast_ingest.py).

Covers:
- Stage A: chunking produces correct chunk dicts
- Stage B: cross-doc embedding batching
- Stage C: insert schema parity with standard path
- Pipeline orchestration: error isolation, backpressure, timeout
- Resume model: exact ID-set comparison, allowlist persistence
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_corpus(n: int = 10) -> Dict[str, str]:
    """Create a test corpus of n documents with diverse content."""
    topics = [
        "Machine learning algorithms use neural networks for deep learning tasks.",
        "Database systems optimize queries using indexing strategies and B-trees.",
        "Cloud computing enables distributed systems with microservices architecture.",
        "Natural language processing extracts meaning from unstructured text.",
        "Computer vision models detect objects and recognize faces in images.",
        "Cybersecurity protocols use encryption for authentication and access control.",
        "Software engineering principles guide agile development and code reviews.",
        "Data structures like hash tables enable efficient key-value lookups.",
        "Operating systems manage memory allocation and process scheduling.",
        "Web development frameworks streamline frontend and backend integration.",
    ]
    return {f"doc_{i}": f"Document {i}: {topics[i % len(topics)]} " * 5 for i in range(n)}


def _make_corpus_to_valid(corpus: Dict[str, str]) -> Dict[str, str]:
    """Create corpus_id → valid_id mapping."""
    import hashlib

    return {cid: f"doc_{hashlib.md5(cid.encode()).hexdigest()[:16]}" for cid in corpus}


class MockEmbeddingManager:
    """Mock embedding manager that returns fixed-dim vectors."""

    def __init__(self, dim: int = 384):
        self.dim = dim
        self.model_name = "mock-model"
        self.dimension = dim
        self._encode_lock = threading.Lock()
        self._call_count = 0

    def encode_documents_batch(self, texts, show_progress=True):
        with self._encode_lock:
            self._call_count += 1
            return [np.random.randn(self.dim).astype(np.float32) for _ in texts]

    def encode_documents_late_chunking(self, full_text, chunk_texts):
        return self.encode_documents_batch(chunk_texts, show_progress=False)

    def encode_document(self, text):
        return np.random.randn(self.dim).astype(np.float32)

    def encode_sparse(self, texts):
        return [{"token_1": 0.5, "token_2": 0.3} for _ in texts]

    def encode_query(self, text, use_cache=True):
        return self.encode_document(text)

    def similarity(self, a, b):
        # What semantic chunking asks of the manager between sentences.
        return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0))


class MockVectorBackend:
    """Mock vector backend that stores inserts in memory."""

    def __init__(self):
        self.inserted: List[Dict[str, Any]] = []
        self.backend_name = "milvus"
        self._deleted = False

    def insert(self, collection_name, data, partition_key=None):
        if isinstance(data, dict):
            data = [data]
        self.inserted.extend(data)
        return [f"id_{i}" for i in range(len(data))]

    def list_document_ids(self, collection_name, session_id=None, limit=10000, offset=0):
        doc_ids = set()
        for item in self.inserted:
            doc_ids.add(item.get("document_id", ""))
        return sorted(doc_ids)

    def delete(self, collection_name, filter_expr=None):
        self.inserted = []
        self._deleted = True

    @property
    def capabilities(self):
        mock = MagicMock()
        mock.supports_hybrid_search = False
        mock.supports_native_bm25 = False
        return mock


class MockCollectionManager:
    """Mock collection manager."""

    def __init__(self, backend, embedding_manager):
        self._backend = backend
        self.embedding_manager = embedding_manager

    def embedder_for(self, config):
        return self.embedding_manager

    def resolve_backend(self, session_id):
        return self._backend


class MockBackend:
    """Mock RAGBackend with all required components."""

    def __init__(self):
        self._vector_backend = MockVectorBackend()
        self._embedding_manager = MockEmbeddingManager()
        self._collection_manager = MockCollectionManager(
            self._vector_backend, self._embedding_manager
        )
        from lasagna.retrieval.search.hybrid import HybridSearcher

        self.hybrid_searcher = HybridSearcher(enable_periodic_cleanup=False)
        self.hybrid_searcher._native_searcher = None


class MockConfig:
    """Mock RAGConfig for testing."""

    def __init__(self, **kwargs):
        defaults = {
            "chunking_strategy": "fixed",
            "chunk_size": 200,
            "chunk_overlap": 0,
            "enable_hierarchical": False,
            "enable_contextual_retrieval": False,
            "enable_graph_retrieval": False,
            "enable_raptor": False,
            "enable_late_chunking": False,
            "enable_hybrid_search": True,
            "max_content_length": 4000,
            "collection_name": "test_collection",
            "min_chunk_size": 50,
            "max_chunk_size": 500,
            "window_size": 3,
            "sentence_overlap": 1,
            "semantic_threshold": 0.75,
            "preserve_tables": False,
            "preserve_headers": False,
            "extract_metadata": True,
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
            "auto_merge_threshold": 0.5,
        }
        defaults.update(kwargs)
        for k, v in defaults.items():
            setattr(self, k, v)


# ---------------------------------------------------------------------------
# Test Stage A: chunking
# ---------------------------------------------------------------------------


class TestChunkOnDoc:
    """_chunk_one_doc produces correct chunk dicts."""

    def test_basic_chunking(self):
        from lasagna.knowledge_evaluation_sdk._fast_ingest import (
            _chunk_one_doc,
        )

        config = MockConfig()
        doc_id, chunks, parents = _chunk_one_doc(
            corpus_id="test_doc",
            text="Hello world. This is a test document with enough text to chunk properly. " * 20,
            valid_id="doc_abc123",
            session_id="test_session",
            filename="doc_abc123.txt",
            config=config,
        )
        assert doc_id == "doc_abc123"
        assert len(chunks) > 0
        assert parents is None  # hierarchical disabled

        # Verify chunk dict structure
        chunk = chunks[0]
        assert "content" in chunk
        assert "document_id" in chunk
        assert chunk["document_id"] == "doc_abc123"
        assert "session_id" in chunk
        assert "metadata" in chunk
        assert "chunk_index" in chunk

    def test_chunk_index_sequential(self):
        from lasagna.knowledge_evaluation_sdk._fast_ingest import (
            _chunk_one_doc,
        )

        config = MockConfig()
        _, chunks, _ = _chunk_one_doc(
            corpus_id="doc_1",
            text="Sentence one. Sentence two. Sentence three. " * 30,
            valid_id="doc_test",
            session_id="sess",
            filename="doc_test.txt",
            config=config,
        )
        indices = [c["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


# ---------------------------------------------------------------------------
# Test Stage C: insert schema parity
# ---------------------------------------------------------------------------


class TestInsertSchemaParity:
    """Insert payload matches _document_pipeline.py:570-579 schema."""

    def test_insert_doc_fields(self):
        from lasagna.knowledge_evaluation_sdk._fast_ingest import (
            _flush_insert,
        )

        backend = MockVectorBackend()
        chunks = [
            {
                "chunk_id": "doc1_chunk_0",
                "content": "Test content",
                "document_id": "doc1",
                "session_id": "sess",
                "metadata": {"session_id": "sess", "document_id": "doc1"},
                "embedding": np.array([1.0, 2.0, 3.0]),
            }
        ]
        from lasagna.retrieval.search.hybrid import HybridSearcher

        hs = HybridSearcher(enable_periodic_cleanup=False)
        hs._native_searcher = None

        n = _flush_insert(
            chunks,
            backend,
            "test_coll",
            hs,
            "sess",
            True,
            None,
        )
        assert n == 1
        assert len(backend.inserted) == 1

        doc = backend.inserted[0]
        # Required fields per _document_pipeline.py:570-579
        assert "chunk_id" in doc
        assert "document_id" in doc
        assert "session_id" in doc
        assert "content" in doc
        assert "embedding" in doc
        assert "metadata" in doc
        # embedding must be list, not numpy
        assert isinstance(doc["embedding"], list)
        # session_id in both top-level and metadata
        assert doc["metadata"]["session_id"] == "sess"


# ---------------------------------------------------------------------------
# Test Pipeline: error isolation
# ---------------------------------------------------------------------------


class TestPipelineErrorIsolation:
    """Corrupt docs don't crash the pipeline."""

    @pytest.mark.asyncio
    async def test_empty_text_handled(self):
        from lasagna.knowledge_evaluation_sdk._fast_ingest import (
            fast_ingest_corpus,
        )

        corpus = {"good_doc": "Valid text content. " * 30, "empty_doc": ""}
        corpus_to_valid = {"good_doc": "v_good", "empty_doc": "v_empty"}

        backend = MockBackend()
        config = MockConfig()

        total, ingested, errors = await fast_ingest_corpus(
            corpus=corpus,
            corpus_to_valid=corpus_to_valid,
            ready_ids=set(),
            session_id="test_sess",
            config=config,
            backend=backend,
        )
        # `assert total >= 0` and `len(errors) >= 0` were both true for every
        # possible run -- a count cannot be negative -- so this asserted that
        # the pipeline returned, and nothing about what it returned.
        assert ingested == {"v_good"}, f"the valid document was not ingested; errors={errors}"
        # `total` counts chunks, not documents. It equalled the document count
        # only while chunking ignored the config's chunk_size (200 here) and cut
        # the 600-character document at the global 800, into one chunk.
        assert total == 3, f"600 characters at chunk_size 200 should be 3 chunks, got {total}"


# ---------------------------------------------------------------------------
# Test Pipeline: backpressure
# ---------------------------------------------------------------------------


class TestBackpressure:
    """Tiny queues don't cause deadlock."""

    @pytest.mark.asyncio
    async def test_small_queue_no_deadlock(self):
        from lasagna.knowledge_evaluation_sdk._fast_ingest import (
            fast_ingest_corpus,
        )

        # Use enough docs to exceed typical queue sizes and trigger backpressure
        corpus = _make_corpus(30)
        corpus_to_valid = _make_corpus_to_valid(corpus)
        backend = MockBackend()
        config = MockConfig()

        # Force very small batch sizes, so the queues actually fill and the
        # backpressure path this test is named for gets exercised.
        #
        # This patches the `rag_config` singleton itself rather than a name in
        # `_fast_ingest`. An earlier version patched
        # `_fast_ingest._global_rag_config`, a module-level alias that the
        # ingest function never reads — it re-imports `rag_config` locally —
        # so the batch sizes below had no effect and the test ran against the
        # defaults, exercising no backpressure at all while passing.
        # These are read-only properties, so they are patched on the class.
        from lasagna.core.config import rag_config

        cfg_cls = type(rag_config)
        with (
            patch.object(
                cfg_cls, "ingest_extract_workers", new_callable=PropertyMock, return_value=2
            ),
            patch.object(
                cfg_cls, "ingest_embed_batch_size", new_callable=PropertyMock, return_value=4
            ),
            patch.object(
                cfg_cls, "ingest_insert_batch_size", new_callable=PropertyMock, return_value=8
            ),
        ):
            assert rag_config.ingest_embed_batch_size == 4, (
                "the patch must reach the config the ingest function reads, or this "
                "test exercises no backpressure while still passing"
            )
            total, ingested, errors = await fast_ingest_corpus(
                corpus=corpus,
                corpus_to_valid=corpus_to_valid,
                ready_ids=set(),
                session_id="bp_test",
                config=config,
                backend=backend,
            )

        assert total > 0
        assert len(ingested) > 0


# ---------------------------------------------------------------------------
# Test: config properties
# ---------------------------------------------------------------------------


class TestIngestConfig:
    """Config keys exist and have correct defaults."""

    def test_default_values(self):
        from lasagna.core.config import rag_config

        assert isinstance(rag_config.ingest_extract_workers, int)
        assert rag_config.ingest_extract_workers >= 1
        assert isinstance(rag_config.ingest_embed_batch_size, int)
        assert rag_config.ingest_embed_batch_size > 0
        assert isinstance(rag_config.ingest_insert_batch_size, int)
        assert rag_config.ingest_insert_batch_size > 0

    def test_env_override(self):
        import os
        from unittest.mock import patch as _patch

        with _patch.dict(os.environ, {"RAG_INGEST_EMBED_BATCH_SIZE": "128"}):
            from lasagna.core.config import RAGConfig

            cfg = RAGConfig()
            assert cfg.ingest_embed_batch_size == 128


# ---------------------------------------------------------------------------
# Test: chunks cut to max_content_length
# ---------------------------------------------------------------------------


class TestContentCut:
    """A chunk cut before embedding is reported, not cut silently.

    A whole-document arm (chunk_size 100000) had its longer documents cut at
    4,000 characters, embedded and indexed that way, and logged nothing.
    """

    @staticmethod
    def _run(lengths, max_content_length):
        import queue as _queue

        from loguru import logger

        from lasagna.knowledge_evaluation_sdk._fast_ingest import _embed_stage

        chunk_q: _queue.Queue = _queue.Queue()
        insert_q: _queue.Queue = _queue.Queue()
        for i, n in enumerate(lengths):
            chunk = {"content": "x" * n, "document_id": f"d{i}", "chunk_index": 0}
            chunk_q.put((f"d{i}", [chunk], None))
        warnings: List[str] = []
        sink = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
        try:
            _embed_stage(
                chunk_q,
                insert_q,
                MockEmbeddingManager(),
                embed_batch_size=8,
                total_docs=len(lengths),
                enable_late_chunking=False,
                enable_sparse=False,
                max_content_length=max_content_length,
            )
        finally:
            logger.remove(sink)
        stored = []
        while (batch := insert_q.get()) is not None:
            stored.extend(batch)
        return warnings, stored

    def test_a_cut_chunk_is_reported(self):
        warnings, stored = self._run([5000, 100], max_content_length=4000)
        assert [len(c["content"]) for c in stored] == [4000, 100]
        assert any("1 chunks were longer than max_content_length=4000" in w for w in warnings)

    def test_nothing_is_reported_when_nothing_is_cut(self):
        warnings, _ = self._run([5000, 100], max_content_length=100_000)
        assert not any("max_content_length" in w for w in warnings)


# ---------------------------------------------------------------------------
# Test: worker count rules
# ---------------------------------------------------------------------------


class TestWorkerCountRules:
    """LLM features force workers=1."""

    @pytest.mark.asyncio
    async def test_semantic_forces_single_worker(self):
        """When chunking_strategy=semantic, workers should be 1."""
        from lasagna.knowledge_evaluation_sdk._fast_ingest import (
            fast_ingest_corpus,
        )

        corpus = _make_corpus(5)
        corpus_to_valid = _make_corpus_to_valid(corpus)
        backend = MockBackend()
        config = MockConfig(chunking_strategy="semantic")

        # The function should not crash; it will use workers=1
        total, ingested, errors = await fast_ingest_corpus(
            corpus=corpus,
            corpus_to_valid=corpus_to_valid,
            ready_ids=set(),
            session_id="semantic_test",
            config=config,
            backend=backend,
        )
        assert isinstance(total, int) and not errors, "semantic chunking ran on every document"


# ---------------------------------------------------------------------------
# Test: tenant partitions
# ---------------------------------------------------------------------------


class _RecordingVectorBackend(MockVectorBackend):
    """Records the partition each insert was routed to."""

    def __init__(self):
        super().__init__()
        self.partitions: List[Any] = []

    def insert(self, collection_name, data, partition_key=None):
        self.partitions.append(partition_key)
        return super().insert(collection_name, data, partition_key=partition_key)


class _PartitionedTenants:
    """A tenant manager in partition mode: every session has its own partition."""

    def __init__(self, fail_to_create: bool = False):
        self.ensured: List[str] = []
        self._fail = fail_to_create

    def resolve_partition_key(self, session_id):
        return f"tenant_{session_id}"

    def ensure_partition(self, backend, collection_name, partition_key):
        if self._fail:
            raise RuntimeError("partition quota exceeded")
        self.ensured.append(partition_key)


class TestTenantPartitions:
    """Benchmark ingestion writes where search reads.

    Search reads only the session's partition when tenant isolation is
    "partition". The fast path used to import the tenant manager from a module
    that does not define it, swallow the ImportError, and insert into the
    default partition: retrieval then returned nothing, with no error.
    """

    @staticmethod
    def _backend_recording_partitions():
        backend = MockBackend()
        recording = _RecordingVectorBackend()
        backend._vector_backend = recording
        backend._collection_manager = MockCollectionManager(recording, backend._embedding_manager)
        return backend, recording

    @pytest.mark.asyncio
    async def test_benchmark_chunks_land_in_the_session_partition(self, monkeypatch):
        from lasagna.knowledge_evaluation_sdk._fast_ingest import fast_ingest_corpus

        tenants = _PartitionedTenants()
        monkeypatch.setattr("lasagna.core.tenant.get_tenant_manager", lambda: tenants)
        backend, recording = self._backend_recording_partitions()
        corpus = _make_corpus(3)

        total, ingested, errors = await fast_ingest_corpus(
            corpus=corpus,
            corpus_to_valid=_make_corpus_to_valid(corpus),
            ready_ids=set(),
            session_id="s1",
            config=MockConfig(),
            backend=backend,
        )

        assert total > 0 and not errors
        assert tenants.ensured == ["tenant_s1"]
        assert recording.partitions and set(recording.partitions) == {"tenant_s1"}

    @pytest.mark.asyncio
    async def test_a_partition_that_cannot_be_created_fails_the_ingest(self, monkeypatch):
        from lasagna.knowledge_evaluation_sdk._fast_ingest import fast_ingest_corpus

        monkeypatch.setattr(
            "lasagna.core.tenant.get_tenant_manager",
            lambda: _PartitionedTenants(fail_to_create=True),
        )
        backend, recording = self._backend_recording_partitions()
        corpus = _make_corpus(2)

        with pytest.raises(ValueError, match="partition"):
            await fast_ingest_corpus(
                corpus=corpus,
                corpus_to_valid=_make_corpus_to_valid(corpus),
                ready_ids=set(),
                session_id="s1",
                config=MockConfig(),
                backend=backend,
            )
        assert recording.inserted == []


class TestAFailureSharedByEveryDocumentIsReportedOnce:
    """A model that cannot load failed every document alike, and ``benchmark run``
    printed "Chunking failed for doc_..." once per document, 24 identical lines."""

    @pytest.mark.asyncio
    async def test_one_line_and_a_count(self):
        from loguru import logger

        from lasagna.knowledge_evaluation_sdk._fast_ingest import fast_ingest_corpus

        class _Broken:
            def chunk(self, text, metadata=None):
                raise RuntimeError("Cannot load embedding model 'BAAI/bge-m3'")

        corpus = _make_corpus(6)
        lines: List[str] = []
        sink = logger.add(lambda m: lines.append(m.record["message"]), level="WARNING")
        try:
            with pytest.raises(RuntimeError, match="Chunking failed for 6 of 6"):
                await fast_ingest_corpus(
                    corpus=corpus,
                    corpus_to_valid=_make_corpus_to_valid(corpus),
                    ready_ids=set(),
                    session_id="broken",
                    config=MockConfig(),
                    backend=MockBackend(),
                    chunker=_Broken(),
                )
        finally:
            logger.remove(sink)
        failed = [line for line in lines if line.startswith("Chunking failed")]
        assert len(failed) == 2, failed
        assert "bge-m3" in failed[0]
        assert failed[1] == "Chunking failed the same way for 5 more document(s)"
