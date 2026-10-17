"""A reindex into another model is the session's model, and a search that cannot see documents says so.

``KnowledgeClient(session_id="s3", config=config_for_speed().with_updates(
embedding_model="BAAI/bge-small-en-v1.5")).documents.reindex(id)`` succeeded and
the SDK found the documents; ``lasagna search -s s3`` then printed "0 results"
and ``status`` still named all-MiniLM-L6-v2. The same after an MCP
``patch_rag_config`` with ``force=true`` and no reindex. Now the reindex records
the model as the session's, and a search whose session has documents in a
collection other than the one it searches reports the ``documents_elsewhere``
stage (``degraded_stages`` over REST and MCP), with a warning in the SDK and
the CLI, instead of an empty list with no word of why.
"""

from __future__ import annotations

import asyncio
import uuid
import warnings
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from lasagna.core.config import collection_name_for_settings, get_rag_config
from lasagna.core.database import get_rag_database
from lasagna.core.session_config import get_knowledge_config_manager
from lasagna.utils import degraded

SMALL = {"embedding_model": "BAAI/bge-small-en-v1.5", "embedding_dim": 384}


def _speed_session_with_a_document() -> str:
    sid = f"elsewhere-{uuid.uuid4().hex[:8]}"
    mgr = get_knowledge_config_manager(session_id=sid)
    mgr.set_active_profile("speed", session_id=sid)
    get_rag_database().save_document(
        {
            "document_id": f"doc_{sid}",
            "session_id": sid,
            "filename": "a.md",
            "file_size": 10,
            "file_type": "md",
            "status": "ready",
            "chunks_count": 1,
        },
        "/nonexistent/a.md",
        collection_name=collection_name_for_settings(mgr.get_resolved_config(sid)),
    )
    return sid


def _moved_to_small(sid: str) -> None:
    """The session searches bge-small now; its document stays in MiniLM's collection."""
    get_knowledge_config_manager(session_id=sid).patch_config(SMALL, session_id=sid, force=True)


def test_a_search_reports_documents_in_another_collection():
    from lasagna.api._profile_guard import forget_documents_elsewhere
    from lasagna.api._search_orchestrator import _note_documents_elsewhere

    sid = _speed_session_with_a_document()
    config = get_rag_config().with_session(sid)
    with degraded.collecting() as stages:
        _note_documents_elsewhere(sid, config)
    assert stages == []

    _moved_to_small(sid)
    forget_documents_elsewhere(sid)
    with degraded.collecting() as stages:
        _note_documents_elsewhere(sid, get_rag_config().with_session(sid))
    assert stages == ["documents_elsewhere"]

    from lasagna.api._profile_guard import documents_elsewhere

    config = get_rag_config().with_session(sid)
    message = documents_elsewhere(sid, config.collection_name, config.embedding_model)
    assert "1 of 1 document in session" in message
    assert "all-MiniLM-L6-v2's collection" in message
    assert "BAAI/bge-small-en-v1.5's" in message
    assert f"lasagna configure --profile speed -s {sid}" in message


class _Backend:
    """Records the stage, as the orchestrator does, and finds nothing."""

    async def async_search(self, **kwargs):
        degraded.record("documents_elsewhere")
        return []


def test_the_sdk_warns(monkeypatch):
    from lasagna.api._profile_guard import DocumentsElsewhereWarning
    from lasagna.sdk import KnowledgeClient

    sid = _speed_session_with_a_document()
    _moved_to_small(sid)
    client = KnowledgeClient(session_id=sid)

    async def backend():
        return _Backend()

    monkeypatch.setattr(client, "_get_backend", backend)
    with warnings.catch_warnings(record=True) as warned:
        warnings.simplefilter("always")
        found = asyncio.run(client.search_documents("helium"))
    assert found.results == [] and found.degraded_stages == ["documents_elsewhere"]
    messages = [str(w.message) for w in warned if issubclass(w.category, DocumentsElsewhereWarning)]
    assert messages and "all-MiniLM-L6-v2's collection" in messages[0]


def test_the_cli_warns_after_the_results(monkeypatch):
    from lasagna import cli
    from lasagna.api._profile_guard import DocumentsElsewhereWarning

    sid = _speed_session_with_a_document()
    _moved_to_small(sid)

    async def search_documents(query, top_k=5):
        warnings.warn("stage", DocumentsElsewhereWarning)
        return []

    async def close():
        pass

    client = SimpleNamespace(search_documents=search_documents, close=close)
    monkeypatch.setattr(cli, "_serving_url", lambda: None)
    monkeypatch.setattr(cli, "_client_for", lambda session_id, url: client)
    result = CliRunner().invoke(cli.app, ["search", "helium", "-s", sid])
    out = " ".join(result.output.split())
    assert result.exit_code == 0, out
    assert "0 results" in out
    assert out.index("0 results") < out.index("Warning:")
    assert "all-MiniLM-L6-v2's collection" in out


def test_a_reindex_with_another_model_records_it_for_the_session(monkeypatch):
    """With RAG_SESSION_BACKEND=memory the client's model stayed in its process."""
    import lasagna.providers as providers
    from lasagna.sdk import KnowledgeClient

    sid = _speed_session_with_a_document()
    provider = providers.InMemorySessionProvider()
    provider.set_config_overrides(sid, SMALL)
    providers.set_session_provider(provider)
    client = KnowledgeClient(session_id=sid)
    client._record_reindexed_model()
    chosen = get_knowledge_config_manager(session_id=sid).get_chosen_config(sid)
    assert chosen["embedding_model"] == "BAAI/bge-small-en-v1.5"
    assert chosen["embedding_dim"] == 384


@pytest.mark.parametrize("missing", [True, False])
def test_the_doctor_and_the_search_agree(missing):
    from lasagna.api._profile_guard import documents_elsewhere, forget_documents_elsewhere
    from lasagna.core.diagnostics import _check_session_documents

    sid = _speed_session_with_a_document()
    if missing:
        _moved_to_small(sid)
    forget_documents_elsewhere(sid)
    config = get_rag_config().with_session(sid)
    message = documents_elsewhere(sid, config.collection_name, config.embedding_model)
    check = _check_session_documents(sid)
    assert (message is not None) == missing
    assert (check.status == "error") == missing


REINDEX = """
import asyncio
from lasagna.sdk import KnowledgeClient, config_for_speed

async def main():
    config = config_for_speed().with_updates(embedding_model="BAAI/bge-small-en-v1.5")
    client = KnowledgeClient(session_id="s3", config=config)
    try:
        for doc in await client.documents.list():
            await client.documents.reindex(doc.document_id)
    finally:
        await client.close()

asyncio.run(main())
"""


def _cached(*names) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return all(isinstance(try_to_load_from_cache(n, "config.json"), str) for n in names)


@pytest.mark.skipif(
    not _cached("sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"),
    reason="all-MiniLM-L6-v2 and bge-small-en-v1.5 are not cached",
)
@pytest.mark.parametrize("backend", ["persistent", "memory"])
def test_the_cli_finds_what_an_sdk_reindex_moved(tmp_path, backend):
    import os
    import subprocess
    import sys

    from tests._paths import PACKAGE_DIR

    (tmp_path / "a.md").write_text("# Alpha\n\nAlpha particles are helium nuclei.\n")
    (tmp_path / "reindex.py").write_text(REINDEX)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent), HF_HUB_OFFLINE="1")

    def run(*argv, **extra):
        return subprocess.run(
            [sys.executable, *argv],
            cwd=tmp_path,
            env={**env, **extra},
            capture_output=True,
            text=True,
            timeout=600,
        )

    for argv in (("configure", "--profile", "speed"), ("upload", "a.md")):
        done = run("-m", "lasagna", *argv, "-s", "s3")
        assert done.returncode == 0, done.stdout + done.stderr[-2000:]
    moved = run("reindex.py", RAG_SESSION_BACKEND=backend)
    assert moved.returncode == 0, moved.stderr[-3000:]
    status = run("-m", "lasagna", "status", "-s", "s3")
    assert "bge-small-en-v1.5" in status.stdout, status.stdout
    found = run("-m", "lasagna", "search", "helium", "-s", "s3")
    assert "1 result" in found.stdout and "Warning" not in found.stdout, found.stdout
