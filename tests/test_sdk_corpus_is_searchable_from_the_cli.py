"""A corpus built from Python is searched with its own model from the CLI, REST and MCP.

``KnowledgeClient(config=config_for_speed())`` kept its settings in its own
process: the in-memory session store and the process-wide embedder. After an
SDK upload, ``lasagna status -s <session>`` showed BGE-M3, and ``lasagna
search`` loaded (or, offline, failed to load) a model that had not built the
corpus. The client now saves its settings as the session's on the session's
first write, as ``lasagna configure`` does, unless the session has its own.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from lasagna.sdk import KnowledgeClient, config_for_speed
from tests._paths import PACKAGE_DIR


class _Manager:
    def __init__(self, chosen):
        self.chosen, self.patches = chosen, []

    def get_chosen_config(self, session_id):
        return self.chosen

    def patch_config(self, patch, session_id=None, trusted=False, **kwargs):
        # The client writes its own process's settings, so unchecked.
        assert trusted
        self.patches.append((session_id, patch))


def _remember(monkeypatch, chosen):
    import lasagna.core.session_config as sc

    manager = _Manager(chosen)
    monkeypatch.setattr(sc, "get_knowledge_config_manager", lambda **kw: manager)
    client = KnowledgeClient(config=config_for_speed(), session_id="sdk-corpus")
    client._remember_session_config()
    client._remember_session_config()  # once per session
    return manager


def test_the_first_write_saves_the_clients_model_for_the_session(monkeypatch):
    manager = _remember(monkeypatch, chosen={})
    assert len(manager.patches) == 1
    session, patch = manager.patches[0]
    assert session == "sdk-corpus"
    assert patch["embedding_model"] == "all-MiniLM-L6-v2"
    assert patch["enable_reranking"] is False


def test_a_configured_session_keeps_its_settings(monkeypatch):
    manager = _remember(monkeypatch, chosen={"embedding_model": "BAAI/bge-small-en-v1.5"})
    assert manager.patches == []


def test_the_cli_reads_the_saved_settings(monkeypatch):
    """`status` built the CLI's client from the profile alone."""
    from lasagna import cli
    import lasagna.core.session_config as sc

    saved = {"embedding_model": "all-MiniLM-L6-v2", "embedding_dim": 384}
    # The default profile with the client's saved overrides: what it chose.
    mgr = SimpleNamespace(get_chosen_config=lambda session_id=None: dict(saved))
    monkeypatch.setattr(sc, "get_knowledge_config_manager", lambda **kw: mgr)
    config = cli._load_profile_config("sdk-corpus")
    assert (config.embedding_model, config.embedding_dim) == ("all-MiniLM-L6-v2", 384)


UPLOAD = """
import asyncio
from lasagna.sdk import KnowledgeClient, config_for_speed

async def main():
    client = KnowledgeClient(session_id="sdk-corpus", config=config_for_speed())
    doc = await client.upload("paper.md")
    await client.documents.wait_for_processing(doc.document_id)
    await client.close()

asyncio.run(main())
"""


def _minilm_cached() -> bool:
    from huggingface_hub import try_to_load_from_cache

    hit = try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json")
    return isinstance(hit, str)


@pytest.mark.skipif(not _minilm_cached(), reason="all-MiniLM-L6-v2 is not cached")
def test_an_sdk_upload_is_found_by_lasagna_search(tmp_path):
    (tmp_path / "paper.md").write_text(
        "# Ablation\n\nThe ablation showed that removing the reranker lowered NDCG@10.\n"
    )
    (tmp_path / "upload.py").write_text(UPLOAD)
    # A new user's process: none of this suite's settings, offline, so a
    # search with the default 2.3 GB model fails instead of downloading it.
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent), HF_HUB_OFFLINE="1")

    def run(*argv):
        return subprocess.run(
            [sys.executable, *argv],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
        )

    upload = run("upload.py")
    assert upload.returncode == 0, upload.stderr[-3000:]
    status = run("-m", "lasagna", "status", "-s", "sdk-corpus")
    assert "all-MiniLM-L6-v2" in status.stdout, status.stdout + status.stderr[-2000:]
    found = run("-m", "lasagna", "search", "what did the ablation show?", "-s", "sdk-corpus")
    assert found.returncode == 0, found.stdout + found.stderr[-2000:]
    assert "The ablation showed" in found.stdout and "1 result" in found.stdout
