"""Loading a persisted collection must end, with an error that says what to do.

After an interrupted experiment, later runs on the same store sat at 0% CPU
forever, waiting on a load with no deadline. A deadline that runs out now
raises with the store's path and the way out.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pymilvus")

from lasagna.vector_backends.milvus_backend import MilvusBackend  # noqa: E402


class _Client:
    def __init__(self, load_error=None, state_error=None):
        self.load_error, self.state_error, self.timeouts = load_error, state_error, []

    def get_load_state(self, name, timeout=None):
        self.timeouts.append(("state", timeout))
        if self.state_error:
            raise self.state_error
        return {"state": "<LoadState: NotLoad>"}

    def load_collection(self, name, timeout=None):
        self.timeouts.append(("load", timeout))
        if self.load_error:
            raise self.load_error


def test_every_call_has_a_deadline():
    backend = MilvusBackend(db_path="/tmp/store.db", load_timeout=42)
    client = _Client()
    backend._ensure_loaded(client, "chunks")
    assert client.timeouts == [("state", 30.0), ("load", 42.0)]


def test_a_deadline_that_runs_out_names_the_store():
    backend = MilvusBackend(db_path="/tmp/store.db", load_timeout=5)
    client = _Client(load_error=RuntimeError("<_InactiveRpcError: DEADLINE_EXCEEDED>"))
    with pytest.raises(TimeoutError, match=r"/tmp/store\.db.*lasagna doctor"):
        backend._ensure_loaded(client, "chunks")


def test_other_errors_stay_best_effort():
    backend = MilvusBackend(db_path="/tmp/store.db")
    backend._ensure_loaded(_Client(state_error=ValueError("no such collection")), "chunks")
