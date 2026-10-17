"""Locks that hold across processes sharing one metadata store.

The REST image runs four gunicorn workers, Kubernetes runs two to ten API
pods, and the MCP server and the Celery worker are processes of their own. A
``threading`` lock serializes the threads of one of them only: two identical
uploads reaching two workers both passed the duplicate check and were both
indexed, and a delete on one pod during a reindex on another left the
reindexed chunks in the vector store with no row.

The lock lives where the processes meet. On PostgreSQL it is a session-level
advisory lock (``pg_advisory_lock``) on a connection of its own, so a process
that dies releases it with its connection. On SQLite, which only processes of
one host share, it is an ``flock`` on a lock file beside the database, one of
:data:`STRIPES` files chosen by the lock's name. Windows has no ``flock``;
there it serializes the threads of one process only. Either waits at most
``RAG_STORE_LOCK_TIMEOUT`` seconds, then raises ``StoreLockTimeout``.

A thread may take a lock it already holds (a delete inside a replace).

A lock's name says its kind (``schema``, ``session:<id>``, ``document:<id>``,
``corpus:<session>:<collection>``, ...), and on SQLite each kind has stripe
files of its own. Kinds shared one set of files: the ``schema`` lock every
starting process takes waited behind a reindex holding a document of its
stripe. A session's lock is taken before any document's, never after, so a
stripe two kinds name cannot leave two threads each waiting for the other.
"""

from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
from contextlib import contextmanager
from typing import Dict, Iterator, List, Optional

#: Lock files per SQLite database. Two names in one stripe wait for each other,
#: which is correct, only slower.
STRIPES = 64

#: How long a PostgreSQL store lock is waited for by default, in seconds
#: (``RAG_STORE_LOCK_TIMEOUT``).
DEFAULT_LOCK_TIMEOUT = 300.0


class StoreLockTimeout(TimeoutError):
    """Another process held a store lock past ``RAG_STORE_LOCK_TIMEOUT``; nothing was changed."""


def lock_timeout() -> float:
    """``RAG_STORE_LOCK_TIMEOUT`` (seconds, 1 to 86400), else 300."""
    raw = os.environ.get("RAG_STORE_LOCK_TIMEOUT", "").strip()
    if not raw:
        return DEFAULT_LOCK_TIMEOUT
    try:
        value = float(raw)
    except ValueError:
        value = -1.0
    if not 1.0 <= value <= 86400.0:
        raise ValueError(
            f"RAG_STORE_LOCK_TIMEOUT={raw!r} is not a number of seconds from 1 to 86400. "
            f"Set one, or unset it for the default ({DEFAULT_LOCK_TIMEOUT:g})."
        )
    return value


def lock_timeout_message(name: str, waited: float) -> str:
    """Why an operation stopped waiting for the lock ``name``."""
    kind, _, what = name.partition(":")
    held = f"{kind} {what!r}" if what else repr(name)
    return (
        f"Another process held {held} for more than {waited:g} seconds, so nothing was "
        f"changed. Retry when it is done (a profile switch holds a session, and a reindex "
        f"a document, for as long as they run); RAG_STORE_LOCK_TIMEOUT sets the wait."
    )


def lock_key(name: str) -> int:
    """A stable signed 64-bit key for ``name`` (Python's ``hash`` differs per process)."""
    return int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:8], "big", signed=True)


class _Stripe:
    """One lock file: a re-entrant lock for this process's threads, and its ``flock``."""

    def __init__(self, path: str):
        self.path = path
        self.local = threading.RLock()
        self.depth = 0  # only read and written by the thread holding ``local``
        self.fd = -1


_stripes: Dict[str, List[_Stripe]] = {}
_stripes_lock = threading.Lock()


def _stripes_for(directory: str) -> List[_Stripe]:
    with _stripes_lock:
        stripes = _stripes.get(directory)
        if stripes is None:
            stripes = [_Stripe(os.path.join(directory, f"{i:02d}.lock")) for i in range(STRIPES)]
            _stripes[directory] = stripes
        return stripes


def lock_kind(name: str) -> str:
    """``session`` for ``session:r1``; a name without a colon is its own kind (``schema``)."""
    return name.partition(":")[0] or "lock"


@contextmanager
def file_lock(directory: str, name: str, timeout: Optional[float] = None) -> Iterator[None]:
    """Hold ``name`` against every thread and process that locks it in ``directory``.

    Waited for at most ``timeout`` seconds (default ``RAG_STORE_LOCK_TIMEOUT``;
    0 tries once), then ``StoreLockTimeout``: it waited for ever.
    """
    wait = lock_timeout() if timeout is None else max(0.0, float(timeout))
    deadline = time.monotonic() + wait
    directory = os.path.join(directory, lock_kind(name))
    stripe = _stripes_for(directory)[lock_key(name) % STRIPES]
    taken = (
        stripe.local.acquire(blocking=False) if wait <= 0 else stripe.local.acquire(timeout=wait)
    )
    if not taken:
        raise StoreLockTimeout(lock_timeout_message(name, wait))
    try:
        if stripe.depth == 0 and sys.platform != "win32":
            stripe.fd = _flock(directory, stripe.path, name, wait, deadline)
        stripe.depth += 1
        try:
            yield
        finally:
            stripe.depth -= 1
            if stripe.depth == 0 and stripe.fd >= 0:
                fd, stripe.fd = stripe.fd, -1
                try:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
    finally:
        stripe.local.release()


def _flock(directory: str, path: str, name: str, wait: float, deadline: float) -> int:
    """An open ``path`` holding its ``flock``, polled until ``deadline``."""
    import fcntl

    os.makedirs(directory, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    pause = 0.005
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return fd
            except BlockingIOError:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise StoreLockTimeout(lock_timeout_message(name, wait)) from None
                time.sleep(min(pause, left))
                pause = min(pause * 2, 0.1)
    except BaseException:
        os.close(fd)
        raise


class HeldLocks(threading.local):
    """The advisory locks this thread holds: key -> how many times it took it."""

    def __init__(self):
        self.held: Dict[int, int] = {}


_local_locks = [threading.RLock() for _ in range(STRIPES)]


def store_lock(name: str, timeout: Optional[float] = None):
    """The metadata store's lock ``name``, held across every process that shares the store.

    The store's own ``exclusive``; a stand-in store without one (in a test)
    gets a lock that serializes this process's threads. ``timeout``: see
    ``file_lock`` (0 tries once).
    """
    from lasagna.core.database import get_rag_database

    exclusive = getattr(get_rag_database(), "exclusive", None)
    if callable(exclusive):
        return exclusive(name) if timeout is None else exclusive(name, timeout=timeout)
    return _local_lock(name, timeout)


@contextmanager
def _local_lock(name: str, timeout: Optional[float]) -> Iterator[None]:
    lock = _local_locks[lock_key(name) % STRIPES]
    wait = lock_timeout() if timeout is None else max(0.0, float(timeout))
    if not (lock.acquire(blocking=False) if wait <= 0 else lock.acquire(timeout=wait)):
        raise StoreLockTimeout(lock_timeout_message(name, wait))
    try:
        yield
    finally:
        lock.release()
