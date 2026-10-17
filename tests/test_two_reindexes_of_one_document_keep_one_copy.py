"""Two reindexes of one document at once leave one copy of it.

A reindex read the document's row before it took the document's lock. The
second of two concurrent reindexes waited for the lock holding a row the first
had made stale: after a model switch it named the old collection, where the
first had already retired every chunk, so the second retired nothing and
stored its own copy beside the first one. Search then returned each passage
twice.
"""

from __future__ import annotations

import threading

from tests._document_harness import MODEL_B, SESSION, Harness

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 3


def test_two_reindexes_after_a_model_switch_leave_one_copy(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    doc = h.upload(h.write(tmp_path, "paper.txt", TEXT))
    h.settings = dict(MODEL_B)

    second_read_its_row = threading.Event()
    read = h.db.get_document

    def get_document(document_id):
        row = read(document_id)
        if threading.current_thread().name == "second":
            second_read_its_row.set()
        return row

    monkeypatch.setattr(h.db, "get_document", get_document)
    chunk = h._chunk
    import lasagna.api._document_pipeline as pipeline_mod

    first_is_chunking = threading.Event()

    def chunk_when_the_second_is_waiting(*a, **k):
        if threading.current_thread().name == "first":
            first_is_chunking.set()
            # Fixed, the second waits for the lock before it reads anything.
            second_read_its_row.wait(timeout=1.0)
        return chunk(*a, **k)

    monkeypatch.setattr(pipeline_mod, "chunk_document", chunk_when_the_second_is_waiting)

    errors = []

    def reindex():
        try:
            h.pipeline.reindex_document(doc.document_id, SESSION)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    first = threading.Thread(target=reindex, name="first")
    second = threading.Thread(target=reindex, name="second")
    first.start()
    assert first_is_chunking.wait(timeout=30)
    second.start()
    first.join(30)
    second.join(30)

    assert not errors, errors
    row = h.row(doc.document_id)
    assert row["collection_name"] == h.config().collection_name
    stored = h.chunks(doc.document_id, row["collection_name"])
    assert len(stored) == row["chunks_count"], (len(stored), row["chunks_count"])
    assert h.searchable(doc.document_id)
    assert h.chunks(doc.document_id) == stored
