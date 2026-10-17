"""Celery application configuration for background task processing.

Creates a Celery app configured via RAG settings.  When ``celery_enabled``
is False (default), the module can still be imported but tasks will execute
synchronously via ``.apply()`` instead of being dispatched to workers.

Usage::

    # Start a worker:
    celery -A lasagna.infrastructure.tasks.celery_app worker \\
        --loglevel=info --concurrency=4

    # Start the beat scheduler (for periodic GC tasks):
    celery -A lasagna.infrastructure.tasks.celery_app beat \\
        --loglevel=info

    # Dispatch a task (from application code):
    from lasagna.infrastructure.tasks.document_tasks import process_document_task
    process_document_task.delay(document_id, session_id, file_path)
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from typing import Optional

from loguru import logger

from . import TASK_SOFT_TIME_LIMIT_S, TASK_TIME_LIMIT_S

try:
    from celery import Celery
except ImportError:
    Celery = None  # type: ignore[assignment,misc]


def _build_beat_schedule() -> dict:
    """Build the Celery Beat schedule from LifecyclePolicy.

    Uses interval schedules (timedelta) so that gc_cycle_hours and
    orphan_scan_hours from the policy / TOML / env are honoured — unlike
    hardcoded crontab expressions.
    """
    try:
        from lasagna.lifecycle.policy import LifecyclePolicy

        policy = LifecyclePolicy.load()
        gc_interval = timedelta(hours=max(1, policy.gc_cycle_hours))
        orphan_interval = timedelta(hours=max(1, policy.orphan_scan_hours))
    except Exception:
        gc_interval = timedelta(hours=24)
        orphan_interval = timedelta(hours=168)

    return {
        # Full GC cycle — interval driven by LifecyclePolicy.gc_cycle_hours
        "gc-full-cycle": {
            "task": "knowledge.gc.run_full_cycle",
            "schedule": gc_interval,
            "options": {"expires": 3600},  # Discard if not started within 1 h
        },
        # Orphan scan — interval driven by LifecyclePolicy.orphan_scan_hours
        "gc-orphan-scan": {
            "task": "knowledge.gc.orphan_scan",
            "schedule": orphan_interval,
            "kwargs": {"dry_run": True},
            "options": {"expires": 3600},
        },
    }


#: The module that defines the tasks (``rag.process_document`` and the GC tasks).
TASK_MODULE = "lasagna.infrastructure.tasks.document_tasks"


def refuse_model_downloads_in_worker(**_signal) -> None:
    """A worker processes uploads as the API does, so it refuses downloads the same way.

    It downloaded any model it lacked inside a task, whatever
    ``allow_model_download_in_requests`` (``RAG_ALLOW_MODEL_DOWNLOAD_IN_REQUESTS``)
    said; a document now fails with the model's name and the fix, as an
    upload to the API does.
    """
    from lasagna.core.model_downloads import refuse_downloads_for_this_process

    refuse_downloads_for_this_process()


#: The process serving this worker's metrics (``worker_metrics``), if any.
_metrics_server: Optional["subprocess.Popen"] = None


def serve_worker_metrics(**_signal) -> None:
    """Serve the worker's Prometheus metrics on ``RAG_WORKER_METRICS_PORT``.

    Ingests are counted where documents are processed, and with the celery
    profile that is the worker, which served no ``/metrics``: the api's
    ``lasagna_ingests_total`` counted only what the api processed itself.
    Every pool process writes to ``PROMETHEUS_MULTIPROC_DIR`` (compose and
    Kubernetes set it), and a process of its own serves them
    (:mod:`lasagna.infrastructure.tasks.worker_metrics`).
    """
    global _metrics_server
    port = os.environ.get("RAG_WORKER_METRICS_PORT", "").strip()
    if not port or _metrics_server is not None:
        return
    from lasagna.core.config import rag_config

    shared = os.environ.get("PROMETHEUS_MULTIPROC_DIR", "").strip()
    problem = None
    if not rag_config.enable_prometheus:
        problem = "RAG_ENABLE_PROMETHEUS is not on. Set RAG_ENABLE_PROMETHEUS=true"
    elif not port.isdigit():
        problem = f"RAG_WORKER_METRICS_PORT={port!r} is not a port number"
    elif not shared:
        problem = (
            "PROMETHEUS_MULTIPROC_DIR is not set, so the pool processes' counts cannot be "
            "collected. Set it to a directory the worker can write (the image has "
            "/tmp/prometheus_multiproc)"
        )
    if problem:
        logger.warning(f"The worker serves no metrics: {problem}.")
        return
    # This container's own directory: a restarted worker would add its counts
    # to the last run's.
    for stale in Path(shared).glob("*.db"):
        stale.unlink(missing_ok=True)
    _metrics_server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lasagna.infrastructure.tasks.worker_metrics",
            port,
            str(os.getpid()),
        ]
    )
    logger.info(f"Worker metrics: http://0.0.0.0:{port}/metrics")


def stop_worker_metrics(**_signal) -> None:
    """Stop the metrics process when the worker shuts down."""
    global _metrics_server
    if _metrics_server is not None:
        _metrics_server.terminate()
        _metrics_server = None


def _connect_worker_signals() -> None:
    """Run the worker-only setup in every worker process."""
    try:
        from celery.signals import worker_init, worker_process_init, worker_shutdown
    except ImportError:
        return
    # The main process, which forks the pool, and each child (a pool that
    # spawns does not inherit the main process's state).
    worker_init.connect(refuse_model_downloads_in_worker, weak=False)
    worker_process_init.connect(refuse_model_downloads_in_worker, weak=False)
    worker_init.connect(serve_worker_metrics, weak=False)
    worker_shutdown.connect(stop_worker_metrics, weak=False)


def create_celery_app() -> "Celery":
    """Create and configure the Celery application.

    Returns a Celery instance configured from RAG settings.
    Raises ImportError if celery is not installed.
    """
    if Celery is None:
        raise ImportError(
            "Celery is required for background tasks. Install with: pip install celery>=5.3.0"
        )

    from lasagna.core.config import rag_config

    broker_url = getattr(rag_config, "celery_broker_url", "redis://localhost:6379/0")
    result_backend = getattr(rag_config, "celery_result_backend", "redis://localhost:6379/1")

    app = Celery(
        "rag_tasks",
        broker=broker_url,
        backend=result_backend,
        # Imported by the worker at startup, so it knows the tasks. The
        # autodiscover_tasks call that stood here imports `<package>.tasks`,
        # which does not exist: the worker registered nothing and discarded
        # every upload and GC run as "unregistered task", while its
        # `inspect ping` healthcheck still passed.
        include=[TASK_MODULE],
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_soft_time_limit=TASK_SOFT_TIME_LIMIT_S,  # 9 min soft limit
        task_time_limit=TASK_TIME_LIMIT_S,  # 10 min hard limit
        # A worker that dies as a whole leaves its message unacknowledged, and
        # the broker delivers it again; the task skips a document that is no
        # longer processing. A worker *child* that dies (out of memory, a
        # crash in a parser, the hard limit) would die again on the same file,
        # so its task fails instead of being requeued without end, and GC
        # phase 0 marks the document failed.
        task_acks_late=True,
        task_reject_on_worker_lost=False,
        worker_prefetch_multiplier=1,  # Fair dispatch
        # Beat schedule is policy-driven — see _build_beat_schedule()
        beat_schedule=_build_beat_schedule(),
    )

    # The tasks are ``shared_task``s, which bind to Celery's *current* app, and
    # that is per thread. The REST routes dispatch from a threadpool thread
    # where this app was never made current, so ``.delay()`` used Celery's
    # unconfigured default app (AMQP on localhost), failed with "Connection
    # refused", and every upload fell back to the api's own threads with the
    # worker idle. The default app is shared by all threads.
    app.set_default()
    _connect_worker_signals()

    from lasagna.core.provenance import redact_url

    logger.info(f"Celery app created: broker={redact_url(broker_url)}")
    return app


# Module-level app for ``celery -A ... worker`` and ``celery -A ... beat``
# Only created if Celery is available
celery_app = create_celery_app() if Celery is not None else None
