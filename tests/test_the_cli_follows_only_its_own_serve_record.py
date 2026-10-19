"""The CLI goes through a ``lasagna serve`` only on a record that serve itself wrote.

``serve.json`` tells ``status``, ``search`` and ``upload`` where to send
documents and queries. Any URL in it was followed, so a record naming
``http://collector.example`` sent them there. A pid the command could not
signal (another user's process, after pid reuse) counted as a live server, and
so did pid 0, which ``os.kill`` reads as "this process group". ``serve`` wrote
the record with the default mode, and a record another user wrote, or could
write, was trusted all the same.
"""

from __future__ import annotations

import json
import os
import stat

import pytest

from lasagna.core import local_server


@pytest.fixture
def record(tmp_path, monkeypatch, live_pid):
    monkeypatch.setenv("LASAGNA_DATA_DIR", str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    path = tmp_path / "data" / "serve.json"

    def write(url="http://127.0.0.1:8765", pid=None, mode=0o600):
        path.write_text(json.dumps({"pid": live_pid if pid is None else pid, "url": url}))
        os.chmod(path, mode)
        return path

    return write


def test_a_loopback_record_from_a_live_server_is_followed(record):
    record()
    assert local_server.running_server() == "http://127.0.0.1:8765"
    record(url="http://[::1]:8765")
    assert local_server.running_server() == "http://[::1]:8765"


@pytest.mark.parametrize(
    "url",
    [
        "http://collector.example:8000",
        "https://127.0.0.1:8000",
        "http://127.0.0.1",
        "http://127.0.0.1:notaport",
        "http://127.0.0.1:99999",
        "http://user@127.0.0.1:8000",
        "file:///etc/passwd",
        42,
    ],
)
def test_a_url_that_is_not_loopback_http_with_a_port_is_ignored(record, url):
    record(url=url)
    assert local_server.running_server() is None


@pytest.mark.parametrize("pid", [0, -1, True, "123"])
def test_a_pid_that_is_not_a_process_id_is_ignored(record, pid):
    record(pid=pid)
    assert local_server.running_server() is None


def test_a_process_this_user_cannot_signal_is_not_its_server(record, monkeypatch):
    def kill(pid, signal):
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(local_server.os, "kill", kill)
    record()
    assert local_server.running_server() is None


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX ownership")
def test_a_record_another_user_owns_is_ignored(record, monkeypatch):
    record()
    real = os.getuid()
    monkeypatch.setattr(local_server.os, "getuid", lambda: real + 1)
    assert local_server.running_server() is None


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX modes")
def test_a_record_others_can_write_is_ignored(record):
    record(mode=0o666)
    assert local_server.running_server() is None


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX links")
def test_a_symlinked_record_is_ignored(record, tmp_path, live_pid):
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"pid": live_pid, "url": "http://127.0.0.1:8765"}))
    path = record()
    path.unlink()
    path.symlink_to(target)
    assert local_server.running_server() is None


@pytest.mark.skipif(not hasattr(os, "getuid"), reason="POSIX modes")
def test_serve_writes_its_record_readable_only_by_its_user(record):
    path = record(mode=0o644)
    local_server.announce("http://127.0.0.1:8765")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert json.loads(path.read_text())["pid"] == os.getpid()
