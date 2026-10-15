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

from datetime import timedelta

from loguru import logger

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
    )

    app.conf.update(
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        timezone="UTC",
        enable_utc=True,
        task_soft_time_limit=540,  # 9 min soft limit
        task_time_limit=600,  # 10 min hard limit
        task_acks_late=True,  # Requeue on worker crash
        worker_prefetch_multiplier=1,  # Fair dispatch
        # Beat schedule is policy-driven — see _build_beat_schedule()
        beat_schedule=_build_beat_schedule(),
    )

    # Auto-discover tasks in this package
    app.autodiscover_tasks(["lasagna.infrastructure.tasks"])

    from lasagna.core.provenance import redact_url

    logger.info(f"Celery app created: broker={redact_url(broker_url)}")
    return app


# Module-level app for ``celery -A ... worker`` and ``celery -A ... beat``
# Only created if Celery is available
celery_app = create_celery_app() if Celery is not None else None
