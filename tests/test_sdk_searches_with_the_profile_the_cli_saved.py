"""A session configured with ``lasagna configure`` is searched with that profile from Python.

The SDK kept sessions' settings in memory unless ``RAG_SESSION_BACKEND=persistent``
was set, while the CLI and the MCP server kept them in the metadata database.
After ``lasagna configure --profile speed -s cfg`` and an upload,
``KnowledgeClient(session_id="cfg").search_documents(...)`` in a new process
searched with the default model and failed with "BAAI/bge-m3 is not
downloaded". Rerunning the README Quickstart without ``config=`` said the
session had no profile while ``lasagna status`` showed all-MiniLM-L6-v2.
Every surface now reads the one store by default; ``RAG_SESSION_BACKEND=memory``
keeps a process's sessions in memory.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests._paths import PACKAGE_DIR

RESOLVE = """
from lasagna.core.config import get_rag_config
from lasagna.providers import get_session_provider

config = get_rag_config().with_session("cfg")
print(type(get_session_provider()).__name__)
print(config.embedding_model)
print(config.collection_name)
"""

SEARCH = """
import asyncio
from lasagna.sdk import KnowledgeClient

async def main():
    client = KnowledgeClient(session_id="cfg")
    try:
        found = await client.search_documents("what did the ablation show?")
        print("results:", len(found.results))
        print(found.results[0].chunk.content)
    finally:
        await client.close()

asyncio.run(main())
"""


def _env(tmp_path, **extra):
    # A new user's process: none of this suite's settings, offline, so a search
    # with the default 2.3 GB model fails instead of downloading it.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent), HF_HUB_OFFLINE="1")
    env.update(extra)
    return env


def _run(tmp_path, *argv, **extra):
    return subprocess.run(
        [sys.executable, *argv],
        cwd=tmp_path,
        env=_env(tmp_path, **extra),
        capture_output=True,
        text=True,
        timeout=600,
    )


def _configure_speed(tmp_path):
    done = _run(tmp_path, "-m", "lasagna", "configure", "--profile", "speed", "-s", "cfg")
    assert done.returncode == 0, done.stdout + done.stderr[-3000:]


def test_a_new_python_process_reads_the_profile_the_cli_saved(tmp_path):
    _configure_speed(tmp_path)
    (tmp_path / "resolve.py").write_text(RESOLVE)
    run = _run(tmp_path, "resolve.py")
    assert run.returncode == 0, run.stderr[-3000:]
    provider, model, collection = run.stdout.split()
    assert provider == "PersistentSessionProvider"
    assert model == "all-MiniLM-L6-v2"
    assert collection.endswith("__384")


def test_memory_keeps_the_sessions_of_one_process(tmp_path):
    """The in-memory store stays available, for tests and notebooks."""
    _configure_speed(tmp_path)
    (tmp_path / "resolve.py").write_text(RESOLVE)
    run = _run(tmp_path, "resolve.py", RAG_SESSION_BACKEND="memory")
    assert run.returncode == 0, run.stderr[-3000:]
    provider, model, _ = run.stdout.split()
    assert provider == "InMemorySessionProvider"
    assert model != "all-MiniLM-L6-v2"


def test_an_unknown_session_backend_is_named(monkeypatch):
    from loguru import logger

    from lasagna.providers import session_backend

    seen = []
    sink = logger.add(lambda m: seen.append(str(m)), level="WARNING")
    try:
        monkeypatch.setenv("RAG_SESSION_BACKEND", "redis")
        assert session_backend() == "persistent"
    finally:
        logger.remove(sink)
    assert any("RAG_SESSION_BACKEND='redis' is not a session backend" in m for m in seen)


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    return isinstance(hit, str)


@pytest.mark.skipif(not _minilm_cached(), reason="all-MiniLM-L6-v2 is not cached")
def test_an_sdk_search_finds_what_the_cli_uploaded(tmp_path):
    (tmp_path / "paper.md").write_text(
        "# Ablation\n\nThe ablation showed that removing the reranker lowered NDCG@10.\n"
    )
    _configure_speed(tmp_path)
    upload = _run(tmp_path, "-m", "lasagna", "upload", "paper.md", "-s", "cfg")
    assert upload.returncode == 0, upload.stdout + upload.stderr[-3000:]
    (tmp_path / "search.py").write_text(SEARCH)
    found = _run(tmp_path, "search.py")
    assert found.returncode == 0, found.stdout + found.stderr[-3000:]
    assert "results: 1" in found.stdout
    assert "The ablation showed" in found.stdout
