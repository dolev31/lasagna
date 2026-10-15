"""Lasagna keeps its data in ~/.lasagna, and an earlier ~/.rag_knowledge moves there.

The default data directory was ``~/.rag_knowledge``, a name from before the
project was called Lasagna, written as a literal in five places. It is now one
function, ``core.paths.data_dir()``. A user upgrading must not lose sight of
their documents: the old directory is renamed when it is the only one, and
never merged with or deleted when both exist.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from loguru import logger

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    for var in (
        "LASAGNA_DATA_DIR",
        "RAG_DATA_DIR",
        "LASAGNA_DB_DIR",
        "RAG_DB_DIR",
        "LASAGNA_UPLOAD_DIR",
        "RAG_UPLOAD_DIR",
    ):
        monkeypatch.delenv(var, raising=False)
    from lasagna.core import paths

    monkeypatch.setattr(paths, "_checked", set())
    return tmp_path


@pytest.fixture
def logs():
    seen = []
    sink = logger.add(lambda m: seen.append((m.record["level"].name, m.record["message"])))
    yield seen
    logger.remove(sink)


def test_the_default_is_dot_lasagna(home):
    from lasagna.core.paths import data_dir, db_dir, upload_dir

    assert data_dir() == home / ".lasagna"
    assert db_dir() == home / ".lasagna" / "dbs"
    assert upload_dir() == home / ".lasagna" / "uploads"


def test_an_old_directory_alone_is_moved_once(home, logs):
    from lasagna.core.paths import data_dir

    (home / ".rag_knowledge" / "dbs").mkdir(parents=True)
    (home / ".rag_knowledge" / "dbs" / "rag.db").write_text("documents")

    assert data_dir() == home / ".lasagna"
    assert (home / ".lasagna" / "dbs" / "rag.db").read_text() == "documents"
    assert not (home / ".rag_knowledge").exists()
    moved = [m for level, m in logs if "Moved Lasagna's data" in m]
    assert len(moved) == 1 and [lv for lv, m in logs if m == moved[0]] == ["INFO"]
    data_dir()
    assert len([m for _, m in logs if "Moved Lasagna's data" in m]) == 1


def test_two_existing_directories_are_never_merged_or_deleted(home, logs):
    from lasagna.core.paths import data_dir

    (home / ".rag_knowledge" / "dbs").mkdir(parents=True)
    (home / ".rag_knowledge" / "dbs" / "rag.db").write_text("old documents")
    (home / ".lasagna" / "research").mkdir(parents=True)

    assert data_dir() == home / ".lasagna"
    assert (home / ".rag_knowledge" / "dbs" / "rag.db").read_text() == "old documents"
    assert not (home / ".lasagna" / "dbs").exists()
    warned = [m for level, m in logs if level == "WARNING" and ".rag_knowledge" in m]
    assert warned and "mv " in warned[0] and "LASAGNA_DATA_DIR=" in warned[0]


def test_the_data_directory_can_be_named(home, monkeypatch):
    from lasagna.core.paths import data_dir, db_dir

    monkeypatch.setenv("RAG_DATA_DIR", str(home / "rag-named"))
    assert data_dir() == home / "rag-named"
    monkeypatch.setenv("LASAGNA_DATA_DIR", str(home / "lasagna-named"))
    assert data_dir() == home / "lasagna-named"
    monkeypatch.setenv("RAG_DB_DIR", str(home / "only-dbs"))
    assert db_dir() == home / "only-dbs"
    monkeypatch.setenv("LASAGNA_DB_DIR", str(home / "lasagna-dbs"))
    assert db_dir() == home / "lasagna-dbs"


def test_a_fresh_process_stores_in_dot_lasagna_and_moves_an_old_store(tmp_path):
    (tmp_path / ".rag_knowledge" / "dbs").mkdir(parents=True)
    (tmp_path / ".rag_knowledge" / "dbs" / "marker").write_text("x")
    env = {
        **{k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))},
        "HOME": str(tmp_path),
        "PYTHONPATH": str(REPO / "src"),
    }
    code = (
        "from lasagna.providers import get_config_provider\n"
        "p = get_config_provider()\n"
        "print(p.db_dir)\n"
        "print(p.upload_dir)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120
    )
    assert out.returncode == 0, out.stderr
    db, uploads = out.stdout.split()
    assert Path(db) == tmp_path / ".lasagna" / "dbs"
    assert Path(uploads) == tmp_path / ".lasagna" / "uploads"
    assert (tmp_path / ".lasagna" / "dbs" / "marker").exists()


def test_no_default_path_names_the_old_directory():
    """Every default goes through core/paths.py, so the name changes in one place."""
    allowed = {
        REPO / "src" / "lasagna" / "core" / "paths.py",
        # Two --help strings; listed for the coordinator (cli lane) to change.
        REPO / "src" / "lasagna" / "mcp" / "server.py",
    }
    offenders = [
        str(path.relative_to(REPO))
        for path in (REPO / "src" / "lasagna").rglob("*.py")
        if path not in allowed and ".rag_knowledge" in path.read_text()
    ]
    assert not offenders, offenders
