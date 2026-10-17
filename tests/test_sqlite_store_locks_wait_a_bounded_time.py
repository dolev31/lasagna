"""SQLite store locks wait a bounded time, and unrelated locks never share a file.

The "schema" lock that every starting process takes shared one of 64 lock files
with a 64th of all document locks, so a process started while a reindex held
such a document waited for the reindex to end; a long profile switch held a
session's file, which other sessions shared. And a lock was waited for for ever,
where PostgreSQL's gave up after ``RAG_STORE_LOCK_TIMEOUT`` with a plain error.
Each kind of lock now has files of its own, and a wait past the timeout fails
with the PostgreSQL message.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from lasagna.core import store_locks
from lasagna.core.store_locks import STRIPES, StoreLockTimeout, lock_key

REPO = Path(__file__).resolve().parents[1]

pytestmark = pytest.mark.skipif(sys.platform == "win32", reason="flock is POSIX")

#: Holds one lock of a database's for the life of the process, then says so.
_HOLDER = textwrap.dedent(
    """
    import sys, time
    from lasagna.core.database import RAGDatabase
    db = RAGDatabase(sys.argv[1])
    with db.exclusive(sys.argv[2]):
        open(sys.argv[3], "w").close()
        time.sleep(120)
    """
)


def _env(tmp_path: Path, **extra) -> dict:
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(PYTHONPATH=str(REPO / "src"), HOME=str(tmp_path), **extra)
    return env


def _hold(tmp_path: Path, db_path: str, name: str) -> subprocess.Popen:
    held = tmp_path / f"held-{abs(lock_key(name))}"
    holder = subprocess.Popen(
        [sys.executable, "-c", _HOLDER, db_path, name, str(held)], env=_env(tmp_path)
    )
    deadline = time.monotonic() + 60
    while not held.exists():
        assert holder.poll() is None, "the holder exited"
        assert time.monotonic() < deadline, "the holder never took the lock"
        time.sleep(0.05)
    return holder


def _document_sharing_the_schema_stripe() -> str:
    target = lock_key("schema") % STRIPES
    return next(
        f"document:doc_{i}"
        for i in range(100_000)
        if lock_key(f"document:doc_{i}") % STRIPES == target
    )


def test_a_process_starts_while_a_reindex_holds_a_document(tmp_path):
    from lasagna.core.database import RAGDatabase

    db_path = str(tmp_path / "rag.db")
    RAGDatabase(db_path)
    holder = _hold(tmp_path, db_path, _document_sharing_the_schema_stripe())
    try:
        start = subprocess.run(
            [
                sys.executable,
                "-c",
                f"from lasagna.core.database import RAGDatabase; RAGDatabase({db_path!r})",
            ],
            env=_env(tmp_path),
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert start.returncode == 0, start.stderr[-2000:]
    finally:
        holder.kill()
        holder.wait()


def test_a_lock_held_by_another_process_fails_after_the_wait(tmp_path, monkeypatch):
    from lasagna.core.database import RAGDatabase

    db_path = str(tmp_path / "rag.db")
    db = RAGDatabase(db_path)
    holder = _hold(tmp_path, db_path, "session:r1")
    monkeypatch.setenv("RAG_STORE_LOCK_TIMEOUT", "1")
    try:
        result = {}

        def wait():
            started = time.monotonic()
            try:
                with db.exclusive("session:r1"):
                    result["took"] = "the lock"
            except StoreLockTimeout as exc:
                result["error"] = str(exc)
            result["waited"] = time.monotonic() - started

        waiter = threading.Thread(target=wait, daemon=True)
        waiter.start()
        waiter.join(30)
        assert not waiter.is_alive(), "waited for ever"
        assert "error" in result, result
        assert 0.9 <= result["waited"] < 10
        assert "held session 'r1' for more than 1 seconds" in result["error"]
        assert (
            "nothing was changed" in result["error"] and "RAG_STORE_LOCK_TIMEOUT" in result["error"]
        )
    finally:
        holder.kill()
        holder.wait()
    with db.exclusive("session:r1"):  # free again once its holder is gone
        pass


def test_a_thread_of_this_process_waits_a_bounded_time_too(tmp_path, monkeypatch):
    directory = str(tmp_path / "locks")
    monkeypatch.setenv("RAG_STORE_LOCK_TIMEOUT", "1")
    taken, release = threading.Event(), threading.Event()

    def hold():
        with store_locks.file_lock(directory, "document:doc_1"):
            taken.set()
            release.wait(30)

    holder = threading.Thread(target=hold, daemon=True)
    holder.start()
    assert taken.wait(10)
    try:
        with pytest.raises(StoreLockTimeout):
            with store_locks.file_lock(directory, "document:doc_1"):
                pass
    finally:
        release.set()
        holder.join(10)


def test_each_kind_of_lock_has_files_of_its_own(tmp_path):
    directory = str(tmp_path / "locks")
    for name in ("schema", "session:a", "document:a", "corpus:s:c", "content:s:c:d"):
        with store_locks.file_lock(directory, name):
            pass
    kinds = sorted(p.name for p in Path(directory).iterdir() if p.is_dir())
    assert kinds == ["content", "corpus", "document", "schema", "session"]
    assert not list(Path(directory).glob("*.lock"))  # none shared between kinds
