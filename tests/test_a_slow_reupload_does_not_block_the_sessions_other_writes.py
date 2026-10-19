"""A re-upload under an existing document_id leaves the session free while it runs.

The upload held the session's lock from its settings read to the end of the
replacement. Any other write to the session (a settings change, another
upload) waited for the whole replacement, then failed with a lock timeout
after RAG_STORE_LOCK_TIMEOUT seconds. Now the replacement holds only the
document's lock.
"""

from __future__ import annotations

import threading

from tests._document_harness import SESSION, Harness

TEXT = "the reranker lowered ndcg in the ablation and the dense retriever kept recall " * 3


def test_another_upload_to_the_session_completes_during_a_slow_replacement(tmp_path, monkeypatch):
    import lasagna.api._document_pipeline as pipeline_mod

    monkeypatch.setenv("RAG_STORE_LOCK_TIMEOUT", "1")
    h = Harness(tmp_path, monkeypatch)
    doc = h.upload(h.write(tmp_path, "paper.txt", TEXT), document_id="paper")
    other = h.write(tmp_path, "other.txt", "a different file with its own words " * 3)
    again = h.write(tmp_path, "paper.txt", TEXT + " and a new last sentence")

    chunk = h._chunk
    replacing = threading.Event()
    release = threading.Event()

    def slow_for_the_replacement(*a, **k):
        if threading.current_thread().name == "replacement":
            replacing.set()
            release.wait(timeout=30)
        return chunk(*a, **k)

    monkeypatch.setattr(pipeline_mod, "chunk_document", slow_for_the_replacement)
    errors = []

    def replace():
        try:
            h.upload(again, document_id=doc.document_id)
        except Exception as exc:  # pragma: no cover - reported below
            errors.append(exc)

    thread = threading.Thread(target=replace, name="replacement")
    thread.start()
    try:
        assert replacing.wait(timeout=30)
        # Held, this waited the lock's 1 second and raised StoreLockTimeout.
        second = h.upload(other)
        assert second.document_id != doc.document_id
        from lasagna.core.session_config._session_state import session_lock

        with session_lock(SESSION):
            pass
    finally:
        release.set()
        thread.join(30)

    assert not errors, errors
    assert h.searchable(doc.document_id)
    assert h.searchable(second.document_id)
