"""`status` checks the vector backend, its error names where the bad value is, and REST
uploads are refused while the embedding model cannot load.

- `LASAGNA_VECTOR_BACKEND=milvuss lasagna status` printed "No documents
  uploaded yet." and exited 0; `doctor` and the other commands refused it.
- The refusal said "set RAG_VECTOR_BACKEND to one of: ..." even when the typo
  was in the settings file, or in `LASAGNA_VECTOR_BACKEND`, which wins over it.
- A REST upload answered 201 while `/health/ready` reported the embedding model
  down; the document then failed in the worker. It is refused with 503 and the
  reason readiness gives.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from tests._paths import PACKAGE_DIR


@pytest.fixture
def run(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("RAG_", "LASAGNA_"))}
    env.update(HOME=str(tmp_path / "home"), PYTHONPATH=str(PACKAGE_DIR.parent), COLUMNS="200")

    def _run(*args: str, **extra: str):
        done = subprocess.run(
            [sys.executable, "-m", "lasagna", *args],
            cwd=tmp_path,
            env={**env, **extra},
            capture_output=True,
            text=True,
            timeout=300,
        )
        return done.returncode, " ".join((done.stdout + done.stderr).split())

    return _run


def test_status_refuses_an_unknown_backend(run):
    code, out = run("status", LASAGNA_VECTOR_BACKEND="milvuss")
    assert code == 1, out
    assert "unknown backend 'milvuss' (did you mean 'milvus'?)" in out
    assert "set LASAGNA_VECTOR_BACKEND to one of" in out
    assert "No documents uploaded yet" not in out


def test_a_typo_in_the_settings_file_names_the_file(run, tmp_path):
    settings = tmp_path / "settings.lasagna.toml"
    settings.write_text('[rag]\nvector_backend = "milvuss"\n')
    code, out = run("status", RAG_SETTINGS_FILE=str(settings))
    assert code == 1, out
    assert "set vector_backend in" in out and "settings.lasagna.toml to one of" in out
    assert "RAG_VECTOR_BACKEND" not in out


MODEL = "nobody/not-downloaded-anywhere"


class _Backend:
    """The backend a session on a model that is not downloaded sees."""

    def __init__(self):
        self.embedding_manager = SimpleNamespace(model_name=MODEL, dimension=384, is_loaded=False)
        self.uploads = []

    def get_session_config(self, session_id):
        return SimpleNamespace(embedding_model=MODEL, embedding_dim=384)

    def upload_document(self, **kwargs):
        self.uploads.append(kwargs)
        raise AssertionError("the upload was accepted")


@pytest.fixture
def offline_client(monkeypatch):
    from lasagna.api.rest.app import create_app
    from lasagna.api.rest.dependencies import get_backend

    monkeypatch.delenv("RAG_REQUIRE_AUTH", raising=False)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    backend = _Backend()
    app = create_app()
    app.dependency_overrides[get_backend] = lambda: backend
    with TestClient(app, base_url="http://localhost") as client:
        yield client, backend


def test_an_upload_is_refused_with_the_reason_readiness_gives(offline_client):
    client, backend = offline_client
    ready = client.get("/api/v1/health/ready")
    reason = ready.json()["checks"]["embedding_model"]["detail"]
    assert ready.status_code == 503 and MODEL in reason

    single = client.post(
        "/api/v1/documents",
        headers={"X-Session-ID": "default"},
        files={"file": ("a.txt", b"some text", "text/plain")},
    )
    assert single.status_code == 503, single.text
    assert single.json()["detail"] == reason

    batch = client.post(
        "/api/v1/documents/batch",
        headers={"X-Session-ID": "default"},
        files=[("files", ("a.txt", b"some text", "text/plain"))],
    )
    assert batch.status_code == 503, batch.text
    assert batch.json()["detail"] == reason
    assert backend.uploads == []
