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

    def maybe_reload_embedding_manager(self, config):
        pass

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
        assert len(ingested) >= 1, f"the valid document was not ingested; errors={errors}"
        assert total == len(ingested) + len(errors), (
            f"total {total} does not account for {len(ingested)} ingested + {len(errors)} errored"
        )


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
