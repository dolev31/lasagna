"""The first upload to a fresh store logs no pymilvus RPC error.

Under `lasagna serve` and `lasagna-mcp`, the backend starts on the process's
default model, so the first upload on a session with another model reached the
collection identity check before that model's collection existed. Reading the
collection's embedding size described it, and pymilvus logged
"[ERROR][_log_rpc_error]: RPC error: [describe_collection] ... does not exist"
with a 20-line traceback, for an expected "not created yet". The size of a
collection that does not exist is now known without asking: there is none.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

pytest.importorskip("milvus_lite")

from lasagna.api._collection_manager import CollectionManager  # noqa: E402
from lasagna.vector_backends.milvus_backend import MilvusBackend  # noqa: E402


class _Records(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def pymilvus_logs():
    handler = _Records()
    target = logging.getLogger("pymilvus")
    target.addHandler(handler)
    yield handler.records
    target.removeHandler(handler)


@pytest.fixture
def backend(tmp_path):
    store = MilvusBackend(db_path=str(tmp_path / "lite.db"))
    store.connect()
    yield store
    store.disconnect()


def test_reading_the_size_of_a_missing_collection_logs_nothing(backend, pymilvus_logs):
    manager = SimpleNamespace(_default_backend=backend)
    assert CollectionManager.get_collection_embedding_dim(manager, "never_created__384") is None
    errors = [r.getMessage() for r in pymilvus_logs if r.levelno >= logging.WARNING]
    assert not errors, errors


def test_the_identity_check_on_a_new_collection_logs_nothing(backend, pymilvus_logs):
    manager = CollectionManager.__new__(CollectionManager)
    manager._default_backend = backend
    config = SimpleNamespace(
        collection_name="rag_documents_v2__all_minilm_l6_v2__384",
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        embedding_truncate_dim=None,
    )
    manager.check_collection_identity(config)
    errors = [r.getMessage() for r in pymilvus_logs if r.levelno >= logging.WARNING]
    assert not errors, errors
