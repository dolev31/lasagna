"""The CLI applies settings in the documented order: session, environment, file, default.

``RAG_ENABLE_RERANKING=false lasagna status`` still listed reranking, and
``RAG_EMBEDDING_MODEL=BAAI/bge-small-en-v1.5 lasagna upload`` embedded with
BGE-M3, while docs/configuration.md says the environment outranks the settings
file. The CLI handed the client the default profile's stored config, a
snapshot of the settings file, as explicit fields, so every one of them beat
the environment (and was saved over the session's own settings).
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from tests._paths import PACKAGE_DIR


@pytest.fixture
def lasagna(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(
        HOME=str(tmp_path / "home"),
        PYTHONPATH=str(PACKAGE_DIR.parent),
        HF_HUB_OFFLINE="1",
        COLUMNS="200",
    )

    def run(*argv, **extra):
        done = subprocess.run(
            [sys.executable, "-m", "lasagna", *argv],
            cwd=tmp_path,
            env={**env, **extra},
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert done.returncode == 0, done.stdout + done.stderr[-3000:]
        return done.stdout

    return run


def _shown(status: str) -> dict:
    lines = [line.strip("│ ").strip() for line in status.splitlines()]
    return dict(line.split(": ", 1) for line in lines if line.startswith(("Embedding", "Features")))


def test_the_settings_file_then_the_environment_then_the_session(lasagna, tmp_path):
    shown = _shown(lasagna("status"))
    assert shown["Embedding"] == "BAAI/bge-m3" and "reranking" in shown["Features"]

    (tmp_path / "mine.toml").write_text("[rag]\nenable_reranking = false\nenable_mmr = false\n")
    shown = _shown(lasagna("status", RAG_SETTINGS_FILE=str(tmp_path / "mine.toml")))
    assert "reranking" not in shown["Features"] and "MMR" not in shown["Features"]

    # The environment over the settings file.
    shown = _shown(
        lasagna(
            "status",
            RAG_SETTINGS_FILE=str(tmp_path / "mine.toml"),
            RAG_ENABLE_MMR="true",
            RAG_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
        )
    )
    assert "MMR" in shown["Features"] and shown["Embedding"] == "BAAI/bge-small-en-v1.5"
    shown = _shown(lasagna("status", RAG_ENABLE_RERANKING="false"))
    assert "reranking" not in shown["Features"]

    # The session over the environment.
    lasagna("configure", "--profile", "speed", "-s", "sp")
    shown = _shown(
        lasagna(
            "status",
            "-s",
            "sp",
            RAG_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5",
            RAG_ENABLE_RERANKING="true",
        )
    )
    assert shown == {"Embedding": "all-MiniLM-L6-v2", "Features": "none"}

    # Reading the default session saved nothing over it.
    shown = _shown(lasagna("status", RAG_ENABLE_RERANKING="false"))
    assert "reranking" not in shown["Features"]


def _cached(model: str) -> bool:
    from huggingface_hub import try_to_load_from_cache

    return isinstance(try_to_load_from_cache(model, "config.json"), str)


@pytest.mark.skipif(not _cached("BAAI/bge-small-en-v1.5"), reason="bge-small is not cached")
def test_upload_and_search_embed_with_the_model_the_environment_names(lasagna, tmp_path):
    (tmp_path / "paper.md").write_text(
        "# Ablation\n\nThe ablation showed that removing the reranker lowered NDCG@10.\n"
    )
    env = dict(RAG_EMBEDDING_MODEL="BAAI/bge-small-en-v1.5", RAG_ENABLE_RERANKING="false")
    assert "Uploaded successfully" in lasagna("upload", "paper.md", "-s", "e2", **env)
    found = lasagna("search", "what did the ablation show?", "-s", "e2", **env)
    assert "The ablation showed" in found and "1 result" in found
