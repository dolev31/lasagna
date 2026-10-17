"""PostgreSQL store locks wait a bounded time, share the pool, and say when they were lost.

Each lock opened a connection of its own, outside the pool, whatever the load;
it waited for ever behind another process (a reindex holds its document's lock
while it runs, a profile switch its session's); a connection that dropped while
holding one released it without a word, so another process could run inside
it; and the session settings kept a second connection pool per process. Now a
lock is waited for ``RAG_STORE_LOCK_TIMEOUT`` seconds and then fails with a
plain error, takes a pooled connection while locks hold at most half the pool
(a connection of its own beyond that), warns when its connection dropped, and
the process has one pool.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

import lasagna.core.database_pg as database_pg
from lasagna.core import store_locks
from lasagna.core.store_locks import lock_key


class _Cursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=()):
        self.conn.log.append((self.conn.name, sql, params))
        if "pg_advisory_lock" in sql and self.conn.busy:
            raise _LockNotAvailable()

    def fetchone(self):
        return (self.conn.still_held,)


class _LockNotAvailable(Exception):
    sqlstate = "55P03"


class _Conn:
    def __init__(self, log, name, *, busy=False, still_held=True):
        self.log, self.name, self.busy, self.still_held = log, name, busy, still_held
        self.autocommit, self.closed = False, False

    def cursor(self):
        return _Cursor(self)

    def close(self):
        self.closed = True


class _Pool:
    def __init__(self, log):
        self.log, self.out = log, []

    @contextmanager
    def connection(self, timeout=None):
        conn = _Conn(self.log, f"pooled{len(self.out)}")
        self.out.append(conn)
        yield conn


def _db(monkeypatch, *, pool=None, pool_max=4, **conn_kwargs):
    log, own = [], []

    def connect(dsn, autocommit=False):
        own.append(_Conn(log, f"own{len(own)}", **conn_kwargs))
        own[-1].autocommit = autocommit
        return own[-1]

    monkeypatch.setattr(database_pg, "_psycopg", SimpleNamespace(connect=connect))
    db = database_pg.PostgresRAGDatabase.__new__(database_pg.PostgresRAGDatabase)
    db._dsn, db._driver, db._lock = "postgresql://x", "psycopg3", threading.Lock()
    db._pool, db._pool_max = pool, pool_max
    return db, log, own


def test_a_lock_held_elsewhere_fails_after_the_wait_with_a_plain_error(monkeypatch):
    monkeypatch.setenv("RAG_STORE_LOCK_TIMEOUT", "1.5")
    db, log, _ = _db(monkeypatch, busy=True)
    with pytest.raises(store_locks.StoreLockTimeout) as waited:
        with db.exclusive("document:doc_1"):
            pass
    assert log[0][1] == "SET lock_timeout = 1500"
    message = str(waited.value)
    assert "held document 'doc_1' for more than 1.5 seconds" in message
    assert "nothing was changed" in message and "RAG_STORE_LOCK_TIMEOUT" in message


def test_a_bad_wait_setting_names_the_variable(monkeypatch):
    monkeypatch.setenv("RAG_STORE_LOCK_TIMEOUT", "forever")
    db, _, _ = _db(monkeypatch)
    with pytest.raises(ValueError, match="RAG_STORE_LOCK_TIMEOUT='forever'"):
        with db.exclusive("document:doc_1"):
            pass


def test_locks_hold_pooled_connections_up_to_half_the_pool(monkeypatch):
    log = []
    pool = _Pool(log)
    db, own_log, own = _db(monkeypatch, pool=pool, pool_max=4)
    with db.exclusive("session:a"), db.exclusive("document:b"), db.exclusive("document:c"):
        assert [c.name for c in pool.out] == ["pooled0", "pooled1"]
        assert [c.name for c in own] == ["own0"]  # past half of four
        assert all(c.autocommit for c in pool.out + own)
    # Each lock was released on the connection that took it, and the pooled
    # ones went back as the pool hands them out.
    for name, key in (("pooled0", "session:a"), ("pooled1", "document:b"), ("own0", "document:c")):
        assert (name, "SELECT pg_advisory_unlock(%s)", (lock_key(key),)) in log + own_log
    assert not any(c.autocommit for c in pool.out) and own[0].closed


def test_a_lock_whose_connection_dropped_is_reported(monkeypatch):
    from loguru import logger

    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        db, _, _ = _db(monkeypatch, still_held=False)
        with db.exclusive("session:r1"):
            pass
    finally:
        logger.remove(sink)
    assert any("'session:r1' was no longer held" in m for m in seen), seen


def test_the_session_settings_use_the_metadata_stores_pool(monkeypatch):
    import lasagna.core.database as database
    from lasagna.core.session_config import _persistence_db

    shared, _, _ = _db(monkeypatch)
    monkeypatch.setattr(database, "get_rag_database", lambda: shared)
    assert _persistence_db._get_pg_db() is shared


def test_rest_answers_503_with_the_reason():
    from lasagna.api.rest.routes.documents import reindex_failure, upload_failure

    exc = store_locks.StoreLockTimeout(
        "Another process held document 'd' for more than 300 seconds"
    )
    for answer in (upload_failure(exc, None, "s"), reindex_failure(exc, "d", "s")):
        assert answer.status_code == 503 and "held document 'd'" in answer.detail


@pytest.mark.skipif(not os.environ.get("DATABASE_DSN"), reason="needs DATABASE_DSN (PostgreSQL)")
def test_on_a_live_postgres_a_waiter_gives_up_after_the_timeout(monkeypatch):
    from lasagna.core.database_pg import PostgresRAGDatabase

    monkeypatch.setenv("RAG_STORE_LOCK_TIMEOUT", "1")
    name = f"document:bounded-{os.getpid()}"
    holder = PostgresRAGDatabase(os.environ["DATABASE_DSN"], pool_min=1, pool_max=2)
    waiter = PostgresRAGDatabase(os.environ["DATABASE_DSN"], pool_min=1, pool_max=2)
    holding, done = threading.Event(), threading.Event()

    def hold():
        with holder.exclusive(name):
            holding.set()
            done.wait(30)

    thread = threading.Thread(target=hold)
    thread.start()
    try:
        assert holding.wait(30)
        start = time.time()
        with pytest.raises(store_locks.StoreLockTimeout):
            with waiter.exclusive(name):
                pass
        assert 0.9 <= time.time() - start < 10
    finally:
        done.set()
        thread.join(30)
        holder.close()
        waiter.close()
