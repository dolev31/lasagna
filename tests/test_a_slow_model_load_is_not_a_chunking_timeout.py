"""A document's chunking limit does not include loading the embedding model.

Benchmark ingest gave each document 120 seconds to chunk, counted from when the
first result was waited for. With a semantic or structure-aware chunker the
first document is the one that loads the embedding model, so on a busy machine
(two experiments at once on four CPUs) or a first run that downloads it, that
document alone failed with "Chunking failed for doc_...: " and no reason, and
the task failed for a corpus with one document missing.
"""

from __future__ import annotations

import time

import pytest
from loguru import logger

from tests.test_fast_ingest import (
    MockBackend,
    MockConfig,
    MockEmbeddingManager,
    _make_corpus,
    _make_corpus_to_valid,
)

LIMIT = 0.5


class _SlowToLoad(MockEmbeddingManager):
    """Loads its model, taking longer than the chunking limit, on first use."""

    def __init__(self):
        super().__init__()
        self._loaded = False

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def model(self):
        if not self._loaded:
            time.sleep(LIMIT * 3)
            self._loaded = True
        return object()

    def encode_documents_batch(self, texts, show_progress=True):
        self.model
        return super().encode_documents_batch(texts, show_progress=show_progress)


async def _ingest(monkeypatch, backend, config, chunker=None, corpus=None):
    from lasagna.knowledge_evaluation_sdk import _fast_ingest

    monkeypatch.setattr(_fast_ingest, "_CHUNK_TIMEOUT_SECONDS", LIMIT)
    corpus = corpus or _make_corpus(3)
    return await _fast_ingest.fast_ingest_corpus(
        corpus=corpus,
        corpus_to_valid=_make_corpus_to_valid(corpus),
        ready_ids=set(),
        session_id="slow_load",
        config=config,
        backend=backend,
        chunker=chunker,
    )


@pytest.mark.asyncio
async def test_the_first_document_is_not_charged_for_the_model_load(monkeypatch):
    backend = MockBackend()
    backend._collection_manager.embedding_manager = _SlowToLoad()

    total, ingested, errors = await _ingest(
        monkeypatch, backend, MockConfig(chunking_strategy="semantic")
    )

    assert not errors, f"documents failed to chunk: {errors}"
    assert len(ingested) == 3


@pytest.mark.asyncio
async def test_a_chunking_timeout_names_the_limit(monkeypatch):
    class _Stuck:
        def chunk(self, text, metadata=None):
            time.sleep(LIMIT * 2)
            return []

    lines = []
    sink = logger.add(lambda m: lines.append(m.record["message"]), level="WARNING")
    try:
        with pytest.raises(RuntimeError, match="Chunking failed for 1 of 1"):
            await _ingest(
                monkeypatch,
                MockBackend(),
                MockConfig(),
                chunker=_Stuck(),
                corpus=_make_corpus(1),
            )
    finally:
        logger.remove(sink)

    failed = [line for line in lines if line.startswith("Chunking failed for")]
    assert len(failed) == 1, failed
    assert "TimeoutError" in failed[0], failed[0]
    assert "longer than" in failed[0] and "busy" in failed[0], failed[0]
    assert not failed[0].rstrip().endswith(":"), failed[0]
