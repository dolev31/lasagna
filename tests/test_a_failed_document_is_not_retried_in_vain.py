"""A document that cannot be indexed fails once, and a fixable mismatch is fixed.

Reproduced with the server's default model at 1024 dimensions and a session on
a 384-dimension profile: the session's first upload hit the empty collection
created at startup, Milvus Lite 3 answered "vector field 'embedding' expected
dim 1024, got 384", and the insert fallback that rebuilds an *empty*
collection did not recognise that wording. The document went to error and the
worker retried it twice, 30 s apart, failing the same way each time; a
`lasagna upload` of an empty file likewise waited out both retries before
reporting the failure.
"""

from __future__ import annotations

import pytest

from lasagna.api._document_pipeline import is_dim_mismatch_error
from lasagna.infrastructure.tasks import document_tasks


@pytest.mark.parametrize(
    "message",
    [
        "<MilvusException: (code=6, message=vector field 'embedding' expected dim 1024, got 384)>",
        "the length of float data should divide the dim(1024)",
        "the dimension of the vector does not match the length 384",
    ],
)
def test_every_milvus_wording_of_a_size_mismatch_is_recognised(message):
    assert is_dim_mismatch_error(RuntimeError(message))


@pytest.fixture
def attempts(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(document_tasks.time, "sleep", sleeps.append)

    def run(error):
        def impl(*args, **kwargs):
            calls.append(kwargs["retry_ctx"].retries)
            raise error

        monkeypatch.setattr(document_tasks, "_process_document_impl", impl)
        semaphore = document_tasks.threading.Semaphore(0)
        document_tasks._thread_pool_wrapper("d", "s", "/f", "", 0, semaphore)
        assert semaphore.acquire(blocking=False), "the worker slot was not released"
        return calls, sleeps

    return run


@pytest.mark.parametrize(
    "error",
    [
        ValueError("No text could be extracted from document"),
        ValueError("EMBEDDING DIMENSION MISMATCH during insert"),
        TypeError("chunker returned a str"),
        RuntimeError("vector field 'embedding' expected dim 1024, got 384"),
    ],
)
def test_a_permanent_failure_is_not_retried(attempts, error):
    calls, sleeps = attempts(error)
    assert calls == [0]
    assert sleeps == []


def test_a_transient_failure_is_still_retried(attempts):
    calls, sleeps = attempts(ConnectionError("vector store unreachable"))
    assert calls == [0, 1, 2]
    assert sleeps == [30, 30]


def test_the_celery_path_does_not_retry_a_permanent_failure(monkeypatch):
    retried = []
    ctx = document_tasks.TaskRetryContext(retries=0, max_retries=2, retry_fn=retried.append)

    class _Backend:
        def _get_document_metadata(self, document_id):
            raise ValueError("No text could be extracted from document")

    monkeypatch.setattr("lasagna.api.rag_backend.get_rag_backend", lambda: _Backend())
    with pytest.raises(ValueError):
        document_tasks._process_document_impl("d", "s", "/f", retry_ctx=ctx)
    assert retried == []
