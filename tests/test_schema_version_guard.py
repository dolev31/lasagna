"""A metadata store written by a newer Lasagna is refused, naming both versions.

Migrations only ran upward, so an older install opened a newer store and read
and wrote rows whose columns it did not know: after a downgrade, or with two
versions sharing one Postgres. PostgreSQL never recorded a version at all, and
its startup migrations raced when several replicas or workers started together.
"""

from __future__ import annotations

import sqlite3

import pytest

from lasagna.core import database, database_pg


def _recorded_version(path) -> int:
    with sqlite3.connect(path) as conn:
        row = conn.execute("SELECT value FROM schema_meta WHERE key = 'version'").fetchone()
    return int(row[0])


class TestSqlite:
    def test_a_new_store_records_the_current_version(self, tmp_path):
        path = tmp_path / "rag.db"
        database.RAGDatabase(str(path))
        assert _recorded_version(path) == database.STORE_VERSION

    def test_a_store_at_the_current_version_opens(self, tmp_path):
        path = tmp_path / "rag.db"
        database.RAGDatabase(str(path))
        database.RAGDatabase(str(path))  # reopened: no error, nothing changed
        assert _recorded_version(path) == database.STORE_VERSION

    def test_a_newer_store_is_refused_naming_both_versions(self, tmp_path):
        path = tmp_path / "rag.db"
        newer = database.STORE_VERSION + 3
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE schema_meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.execute("INSERT INTO schema_meta VALUES ('version', ?)", (str(newer),))
        with pytest.raises(database.StoreVersionError) as refused:
            database.RAGDatabase(str(path))
        message = str(refused.value)
        assert f"store version {newer}" in message
        assert f"up to {database.STORE_VERSION}" in message
        assert "pip install -U lasagna-rag" in message
        assert _recorded_version(path) == newer, "the newer store was rewritten"


class _Cursor:
    def __init__(self, recorded):
        self.recorded = recorded
        self.statements = []

    def execute(self, sql, params=None):
        self.statements.append((" ".join(sql.split()), params))

    def fetchone(self):
        return None if self.recorded is None else (str(self.recorded),)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Connection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True


def _postgres(recorded):
    from contextlib import contextmanager

    cursor = _Cursor(recorded)
    conn = _Connection(cursor)
    db = database_pg.PostgresRAGDatabase.__new__(database_pg.PostgresRAGDatabase)
    db._dsn = "postgresql://rag:secret@db.internal:5432/rag"

    @contextmanager
    def _conn():
        yield conn

    db._conn = _conn
    return db, cursor, conn


class TestPostgres:
    def test_schema_creation_runs_under_the_advisory_lock(self):
        db, cursor, conn = _postgres(recorded=None)
        db._create_schema()
        first_sql, params = cursor.statements[0]
        assert first_sql == "SELECT pg_advisory_xact_lock(%s)"
        assert params == (database_pg._SCHEMA_LOCK_KEY,)
        assert conn.committed

    def test_it_records_the_store_version_before_committing(self):
        db, cursor, conn = _postgres(recorded=None)
        db._create_schema()
        writes = [p for sql, p in cursor.statements if sql.startswith("INSERT INTO schema_meta")]
        assert writes == [(str(database.STORE_VERSION),)]
        assert "ON CONFLICT (key) DO UPDATE" in next(
            sql for sql, _ in cursor.statements if sql.startswith("INSERT INTO schema_meta")
        )

    def test_a_newer_store_is_refused_before_any_change(self):
        newer = database.STORE_VERSION + 1
        db, cursor, conn = _postgres(recorded=newer)
        with pytest.raises(database.StoreVersionError) as refused:
            db._create_schema()
        assert f"store version {newer}" in str(refused.value)
        assert f"up to {database.STORE_VERSION}" in str(refused.value)
        assert "secret" not in str(refused.value)
        assert not conn.committed
        assert not any(
            sql.startswith(("CREATE TABLE IF NOT EXISTS rag_", "ALTER"))
            for sql, _ in cursor.statements
        )

    def test_stats_report_the_real_version(self):
        assert database_pg._store_version() == database.STORE_VERSION
