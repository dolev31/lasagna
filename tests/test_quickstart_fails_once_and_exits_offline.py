"""Offline, on the default profile, the quickstart fails once, says why, and exits.

With BGE-M3 not downloaded and HF_HUB_OFFLINE set, ``wait_for_processing``
returned the failed document without a word, the README's code went on to
search and failed with a traceback ending in "Could not load any embedding
model", and the interpreter then hung about 58 s on exit: the worker thread
retried the missing model twice, 30 s apart, and Python joins those threads
before it exits. ``lasagna upload`` took 67 s and printed three ERROR lines
after its own error.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest

from lasagna.core import model_downloads
from lasagna.infrastructure.tasks import document_tasks
from tests._paths import PACKAGE_DIR

# The README's quickstart, verbatim, except that it records when it raised.
QUICKSTART = """
import asyncio, sys, time
from lasagna.sdk import KnowledgeClient

async def main():
    client = KnowledgeClient(session_id="quickstart")   # same session on every run
    doc = await client.upload("paper.md")
    await client.documents.wait_for_processing(doc.document_id)  # upload returns once queued

    results = await client.search_documents("What did the ablation show?")
    for r in results.results:
        print(f"[{r.score:.2f}] {r.chunk.content[:120]}")

    await client.close()

try:
    asyncio.run(main())
except Exception:
    open("raised_at", "w").write(repr(time.time()))
    raise
"""


def _new_user_env(tmp_path) -> dict:
    """A first-time user's process: offline, and an empty model cache."""
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("RAG_", "LASAGNA_", "HF_", "TRANSFORMERS_"))
    }
    env.update(
        HOME=str(tmp_path / "home"),
        HF_HOME=str(tmp_path / "hf"),
        HF_HUB_OFFLINE="1",
        PYTHONPATH=str(PACKAGE_DIR.parent),
    )
    return env


def _paper(tmp_path) -> None:
    (tmp_path / "paper.md").write_text(
        "# Ablation\n\nRemoving the reranker lowered NDCG@10 by 0.04 on SciFact.\n"
    )


def test_the_quickstart_raises_one_error_with_the_fix_and_exits_at_once(tmp_path):
    _paper(tmp_path)
    (tmp_path / "quickstart.py").write_text(QUICKSTART)
    run = subprocess.run(
        [sys.executable, "quickstart.py"],
        cwd=tmp_path,
        env=_new_user_env(tmp_path),
        capture_output=True,
        text=True,
        timeout=300,
    )
    exited_at = time.time()
    assert run.returncode == 1, run.stderr[-3000:]
    last = run.stderr.strip().splitlines()[-1]
    assert last.startswith("lasagna.sdk.exceptions.DocumentProcessingError: paper.md"), last
    assert "BAAI/bge-m3 is not downloaded" in last
    assert "lasagna configure --profile speed -s quickstart" in last
    assert "config_for_speed()" in last
    # It stopped at the wait, not at the search after it.
    assert "search_documents" not in run.stderr
    assert "Could not load any embedding model" not in run.stderr
    raised_at = float((tmp_path / "raised_at").read_text())
    assert exited_at - raised_at < 5, f"exited {exited_at - raised_at:.0f} s after the error"


def test_lasagna_upload_prints_one_error_with_the_fix(tmp_path):
    _paper(tmp_path)
    started = time.monotonic()
    run = subprocess.run(
        [sys.executable, "-m", "lasagna", "upload", "paper.md"],
        cwd=tmp_path,
        env={**_new_user_env(tmp_path), "COLUMNS": "400"},
        capture_output=True,
        text=True,
        timeout=300,
    )
    elapsed = time.monotonic() - started
    out = run.stdout + run.stderr
    assert run.returncode == 1, out[-3000:]
    errors = [line for line in out.splitlines() if line.lower().startswith("error")]
    assert len(errors) == 1, errors
    assert "BAAI/bge-m3 is not downloaded" in errors[0]
    assert "lasagna configure --profile speed -s default" in errors[0]
    assert "[Worker]" not in out and "Failed to load" not in out, out[-3000:]
    assert elapsed < 60, f"took {elapsed:.0f} s"


class _Backend:
    def __init__(self, status: str, error_message=None):
        self.row = SimpleNamespace(
            document_id="d1",
            filename="paper.pdf",
            session_id="s",
            status=status,
            chunks_count=0,
            file_size=10,
            upload_date=None,
            error_message=error_message,
        )

    def get_document_by_id(self, document_id, session_id):
        return self.row


def _manager(backend):
    from lasagna.sdk.managers.document_manager import DocumentManager

    async def get_backend():
        return backend

    client = SimpleNamespace(_get_backend=get_backend, session_id="s", _in_flight_documents=set())
    return DocumentManager(client)


def test_waiting_on_a_failed_document_raises_with_its_reason():
    from lasagna.sdk import DocumentProcessingError

    manager = _manager(_Backend("error", "The embedding model X is not downloaded."))
    with pytest.raises(DocumentProcessingError) as info:
        asyncio.run(manager.wait_for_processing("d1", poll_interval=0))
    assert info.value.document_id == "d1"
    assert info.value.reason == "The embedding model X is not downloaded."
    assert str(info.value) == (
        "paper.pdf (d1) could not be indexed: The embedding model X is not downloaded."
    )


def test_raise_on_failure_false_returns_the_failed_document():
    from lasagna.sdk.types import DocumentStatus

    manager = _manager(_Backend("error", "bad file"))
    doc = asyncio.run(manager.wait_for_processing("d1", poll_interval=0, raise_on_failure=False))
    assert doc.status == DocumentStatus.FAILED and doc.error_message == "bad file"


class _AuthenticationError(Exception):
    """Named like the OpenAI and Anthropic SDKs' error for a rejected key."""


class _HTTPError(Exception):
    status_code = 401


@pytest.mark.parametrize(
    "error",
    [
        model_downloads.ModelNotDownloaded("BAAI/bge-m3", offline=True),
        FileNotFoundError("/uploads/doc.pdf"),
        PermissionError("/uploads/doc.pdf"),
        ImportError("docling is not installed"),
        _AuthenticationError("invalid api key"),
        _HTTPError("401 Unauthorized"),
    ],
    ids=["model-not-downloaded", "missing-file", "unreadable-file", "import", "auth", "http-401"],
)
def test_a_permanent_failure_is_not_retried(monkeypatch, error):
    calls, waits = [], []
    monkeypatch.setattr(document_tasks, "_wait_before_retry", lambda d: waits.append(d) or True)

    def impl(*args, **kwargs):
        calls.append(kwargs["retry_ctx"].retries)
        raise error

    monkeypatch.setattr(document_tasks, "_process_document_impl", impl)
    document_tasks._thread_pool_wrapper("d", "s", "/f", "", 0, threading.Semaphore(0))
    assert calls == [0] and waits == []


def test_a_missing_model_is_recorded_with_the_fix(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    reason = document_tasks._failure_reason(
        model_downloads.ModelNotDownloaded("BAAI/bge-m3", offline=True), "quickstart"
    )
    assert "lasagna configure --profile speed -s quickstart" in reason
    assert "hf download BAAI/bge-m3" in reason and "unset HF_HUB_OFFLINE" in reason


def test_a_document_is_not_marked_failed_while_a_retry_is_pending(monkeypatch):
    """A client polling between attempts saw "failed", which the retry then reversed."""
    statuses, attempts = [], []

    class _Backend:
        def _get_document_metadata(self, document_id):
            attempts.append(document_id)
            raise ConnectionError("vector store unreachable")

    db = SimpleNamespace(update_document_status=lambda **kw: statuses.append((len(attempts), kw)))
    monkeypatch.setattr("lasagna.api.rag_backend.get_rag_backend", lambda: _Backend())
    monkeypatch.setattr("lasagna.core.database.get_rag_database", lambda: db)
    monkeypatch.setattr(document_tasks, "_wait_before_retry", lambda delay: True)
    document_tasks._thread_pool_wrapper("d", "s", "/f", "", 0, threading.Semaphore(0))
    assert len(attempts) == 3
    assert [n for n, _ in statuses] == [3], "marked failed before the last attempt"


RETRY_AT_EXIT = """
import threading, time
from lasagna.infrastructure.tasks import document_tasks

def impl(*args, **kwargs):
    raise ConnectionError("vector store unreachable")

document_tasks._process_document_impl = impl
document_tasks._fail_at_exit = lambda *a: open("failed_at_exit", "w").write("yes")
document_tasks._get_doc_executor().submit(
    document_tasks._thread_pool_wrapper, "d", "s", "/f", "", 0, threading.Semaphore(0)
)
time.sleep(0.5)  # the worker is now waiting 30 s to retry
"""


def test_a_pending_retry_does_not_hold_the_interpreter_at_exit(tmp_path):
    (tmp_path / "script.py").write_text(RETRY_AT_EXIT)
    started = time.monotonic()
    run = subprocess.run(
        [sys.executable, "script.py"],
        cwd=tmp_path,
        env={**os.environ, "HOME": str(tmp_path), "PYTHONPATH": str(PACKAGE_DIR.parent)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    elapsed = time.monotonic() - started
    assert run.returncode == 0, run.stderr[-2000:]
    assert elapsed < 20, f"exit waited {elapsed:.0f} s for the retry"
    assert (tmp_path / "failed_at_exit").read_text() == "yes"
