"""A local Milvus Lite store keeps its connection open while idle.

pymilvus pings every 10 seconds, idle or not. The Milvus Lite server allows one
ping per five minutes without data, so about 20 seconds after the last request
it closed the connection with GOAWAY "too_many_pings". `lasagna serve` then
printed a gRPC error line on every idle period, and a request that arrived as
the connection closed failed. The suite saw it as an intermittent failure.

The connection failure that followed was then reported as "in use by another
process (a running `lasagna serve` ...)": the lock probe found the lock held,
by this process's own Milvus Lite server.
"""

from __future__ import annotations

import os

import pytest

from lasagna.vector_backends import milvus_backend
from lasagna.vector_backends.milvus_backend import DataDirLockedError, MilvusBackend


class _RecordingClient:
    calls: list = []

    def __init__(self, *args, **kwargs):
        type(self).calls.append((args, kwargs))


@pytest.fixture
def recording_client(monkeypatch):
    _RecordingClient.calls = []
    monkeypatch.setattr(milvus_backend, "MilvusClient", _RecordingClient)
    return _RecordingClient


def test_a_lite_store_does_not_ping_while_idle(tmp_path, recording_client):
    MilvusBackend(db_path=str(tmp_path / "store.db")).connect()
    ((_, kwargs),) = recording_client.calls
    options = kwargs["grpc_options"]
    assert options["grpc.keepalive_permit_without_calls"] is False
    # At or above the server's minimum ping interval (five minutes).
    assert options["grpc.keepalive_time_ms"] >= 300_000


def test_a_server_deployment_keeps_pymilvus_keepalive(recording_client):
    MilvusBackend(uri="http://milvus:19530").connect()
    ((_, kwargs),) = recording_client.calls
    assert "grpc_options" not in kwargs


def _failing_client(*args, **kwargs):
    raise RuntimeError("UNAVAILABLE: Too many pings")


def test_our_own_server_holding_the_lock_is_not_another_process(tmp_path, monkeypatch):
    db = str(tmp_path / "store.db")
    monkeypatch.setattr(milvus_backend, "MilvusClient", _failing_client)
    monkeypatch.setattr(milvus_backend, "_lite_store_locked", lambda path: True)

    from milvus_lite.server_manager import server_manager_instance

    monkeypatch.setitem(server_manager_instance._servers, os.path.abspath(db), (None, None, 0))
    with pytest.raises(Exception) as caught:
        MilvusBackend(db_path=db).connect()
    assert not isinstance(caught.value, DataDirLockedError)
    assert "another process" not in str(caught.value)


def test_a_lock_held_elsewhere_is_still_reported_as_held(tmp_path, monkeypatch):
    monkeypatch.setattr(milvus_backend, "MilvusClient", _failing_client)
    monkeypatch.setattr(milvus_backend, "_lite_store_locked", lambda path: True)
    with pytest.raises(DataDirLockedError):
        MilvusBackend(db_path=str(tmp_path / "other.db")).connect()
