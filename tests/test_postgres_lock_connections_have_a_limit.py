"""A process holds at most half its pool's size of lock connections outside the pool.

Locks beyond half the pool each opened a connection of their own, with no
limit, so a process holding many locks (a bulk delete, many reindexes) opened
as many connections as locks and could exhaust ``max_connections``. Past the
limit a lock waits ``RAG_STORE_LOCK_TIMEOUT`` for one of them, then fails with a
plain error.
"""

from __future__ import annotations

import threading
import time
from contextlib import ExitStack

import pytest

from lasagna.core.store_locks import StoreLockTimeout
from tests.test_postgres_store_locks_wait_a_bounded_time import _Pool, _db


def test_past_the_limit_a_lock_waits_then_fails_plainly(monkeypatch):
    monkeypatch.setenv("RAG_STORE_LOCK_TIMEOUT", "1")
    pool = _Pool([])
    db, _, own = _db(monkeypatch, pool=pool, pool_max=4)
    with ExitStack() as held:
        for name in ("a", "b", "c", "d"):  # two pooled, two of their own
            held.enter_context(db.exclusive(f"document:{name}"))
        assert len(own) == 2
        started = time.monotonic()
        with pytest.raises(StoreLockTimeout) as refused:
            with db.exclusive("document:e"):
                pass
        assert 0.9 <= time.monotonic() - started < 10
        assert len(own) == 2  # no third connection was opened
    message = str(refused.value)
    assert "held 2 store locks on connections of their own" in message
    assert "nothing was changed" in message and "RAG_DATABASE_POOL_MAX" in message


def test_a_lock_past_the_limit_gets_a_connection_once_one_is_released(monkeypatch):
    monkeypatch.setenv("RAG_STORE_LOCK_TIMEOUT", "30")
    db, _, own = _db(monkeypatch, pool=_Pool([]), pool_max=4)
    release = threading.Event()

    def hold(name):
        with db.exclusive(name):
            release.wait(30)

    holders = [threading.Thread(target=hold, args=(f"document:{n}",)) for n in "abcd"]
    for t in holders:
        t.start()
    while len(own) < 2:
        time.sleep(0.01)
    took = []

    def wait():
        with db.exclusive("document:e"):
            took.append(1)

    waiter = threading.Thread(target=wait)
    waiter.start()
    time.sleep(0.2)
    assert not took
    release.set()
    waiter.join(10)
    for t in holders:
        t.join(10)
    assert took
