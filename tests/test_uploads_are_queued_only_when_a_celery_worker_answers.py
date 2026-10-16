"""Uploads are queued for Celery only when a worker is there to take them.

``docker-compose.yml`` sets ``RAG_CELERY_ENABLED=true`` whatever the
profiles, and with Redis reachable ``process_document_task.delay()``
succeeds whether or not a worker consumes the queue. So the stack started
without its ``celery`` profile accepted every upload and left it in
"processing" for good. The lifecycle scheduler made the same mistake:
``inspect().ping()`` returns None, not an error, when no worker answers, so
it left GC to a Celery Beat that was not running.

Now ``make`` derives the setting from ``COMPOSE_PROFILES`` and ``make
validate`` refuses ``true`` without the profile, and the api queues an
upload only when a worker answers on the broker, otherwise processing it
in its own threads with one log line saying why.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from loguru import logger

from lasagna.infrastructure.tasks import document_tasks

DEPLOY_DIR = Path(__file__).resolve().parents[1] / "src" / "lasagna" / "deploy"


def _never_called():
    raise AssertionError("the broker must not be probed")


def test_with_celery_off_the_broker_is_not_probed():
    assert document_tasks.choose_dispatch(False, True, _never_called) == ("local", None)


def test_with_a_worker_answering_uploads_are_queued():
    assert document_tasks.choose_dispatch(True, True, lambda: True) == ("celery", None)


def test_without_a_worker_uploads_are_processed_here_and_the_reason_names_the_fix():
    route, reason = document_tasks.choose_dispatch(
        True, True, lambda: False, broker="redis://redis:6379/0"
    )
    assert route == "local"
    assert "no Celery worker answered on redis://redis:6379/0" in reason
    assert "--profile celery" in reason and "RAG_CELERY_ENABLED=false" in reason


def test_without_celery_installed_uploads_are_processed_here():
    route, reason = document_tasks.choose_dispatch(True, False, _never_called)
    assert route == "local"
    assert "celery is not installed" in reason


@pytest.fixture
def broker_with_no_worker(monkeypatch):
    """A reachable broker on which no worker answers, as celery reports it."""
    app = MagicMock()
    app.control.ping.return_value = []
    app.control.inspect.return_value.ping.return_value = None
    fake = types.ModuleType("lasagna.infrastructure.tasks.celery_app")
    fake.celery_app = app
    monkeypatch.setitem(sys.modules, "lasagna.infrastructure.tasks.celery_app", fake)
    monkeypatch.setattr(document_tasks, "_worker_probe", (float("-inf"), False), raising=False)
    monkeypatch.setattr(document_tasks, "_last_fallback_reason", None, raising=False)
    return app


def test_an_upload_with_no_worker_is_not_queued(broker_with_no_worker):
    task = MagicMock()
    config = MagicMock(celery_enabled=True, async_processing=True, max_concurrent_uploads=5)
    lines = []
    sink = logger.add(lambda m: lines.append(str(m)), level="WARNING")
    try:
        with (
            patch.object(document_tasks, "process_document_task", task),
            patch.object(document_tasks, "_thread_pool_wrapper"),
            patch("lasagna.core.config.rag_config", config),
        ):
            assert document_tasks.dispatch_processing("doc1", "s1", "/data/uploads/a.txt")
            assert document_tasks.dispatch_processing("doc2", "s1", "/data/uploads/b.txt")
    finally:
        logger.remove(sink)

    task.delay.assert_not_called()
    said = [line for line in lines if "no Celery worker answered" in line]
    assert len(said) == 1, lines


def test_the_worker_probe_is_reused_for_a_while(monkeypatch):
    calls = []
    monkeypatch.setattr(document_tasks, "_worker_probe", (float("-inf"), False), raising=False)
    monkeypatch.setattr(
        document_tasks, "_ping_celery_workers", lambda timeout=1.0: calls.append(1) or True
    )
    ttl = document_tasks.WORKER_PROBE_TTL_S

    assert document_tasks.celery_workers_answer(now=1000.0)
    assert document_tasks.celery_workers_answer(now=1000.0 + ttl - 1)
    assert len(calls) == 1
    assert document_tasks.celery_workers_answer(now=1000.0 + ttl + 1)
    assert len(calls) == 2


def test_gc_runs_in_the_api_when_no_worker_answers(broker_with_no_worker):
    from lasagna.lifecycle.scheduler import LifecycleScheduler

    scheduler = LifecycleScheduler(gc=MagicMock(), policy=MagicMock())
    with patch.object(LifecycleScheduler, "_start_apscheduler") as start_embedded:
        scheduler.start()
    start_embedded.assert_called_once()


# -- make: the setting follows the profile ------------------------------------

needs_make = pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")


def _make(tmp_path, *args):
    return subprocess.run(
        ["make", "-s", "-C", str(DEPLOY_DIR), *args],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(tmp_path)},
    )


@needs_make
@pytest.mark.parametrize(
    "profiles, expected", [("celery,mcp", "true"), ("mcp", "false"), ("", "false")]
)
def test_make_derives_celery_from_the_profiles(tmp_path, profiles, expected):
    show = tmp_path / "show.mk"
    show.write_text("show:\n\t@printenv RAG_CELERY_ENABLED\n")
    result = _make(
        tmp_path, "-f", "Makefile", "-f", str(show), "show", f"COMPOSE_PROFILES={profiles}"
    )
    assert result.stdout.strip() == expected, result.stdout + result.stderr


@needs_make
@pytest.mark.parametrize("value", ["true", "TRUE", "1"])
def test_validate_refuses_celery_without_its_profile(tmp_path, value):
    result = _make(
        tmp_path,
        "validate",
        "VECTOR_BACKEND=qdrant",
        "RAG_JWT_SECRET=x",
        "POSTGRES_PASSWORD=y",
        "COMPOSE_PROFILES=mcp",
        f"RAG_CELERY_ENABLED={value}",
    )
    assert result.returncode != 0
    assert "COMPOSE_PROFILES=mcp starts none" in result.stdout + result.stderr


@needs_make
def test_validate_accepts_celery_with_its_profile(tmp_path):
    result = _make(
        tmp_path,
        "validate",
        "VECTOR_BACKEND=qdrant",
        "RAG_JWT_SECRET=x",
        "POSTGRES_PASSWORD=y",
        "COMPOSE_PROFILES=celery,mcp",
        "RAG_CELERY_ENABLED=true",
    )
    assert result.returncode == 0, result.stdout + result.stderr
