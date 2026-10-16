"""Lifecycle scheduler — runs GC on a timer.

Two scheduler backends are supported:

Celery Beat (production)
    If a Celery broker is reachable and tasks have been registered, Celery Beat
    handles scheduling externally.  ``LifecycleScheduler`` simply exposes a
    ``trigger_gc()`` helper that enqueues the Celery task directly.

APScheduler (single-node / dev)
    When no Celery broker is reachable, ``LifecycleScheduler`` starts an
    embedded APScheduler ``BackgroundScheduler``.  Because multiple worker
    processes might all start APScheduler simultaneously, every scheduled job
    first tries to acquire the DB advisory lock via ``GarbageCollector`` —
    only the process that acquires the lock actually runs the cycle.

Usage::

    scheduler = LifecycleScheduler(gc=gc, policy=policy)
    scheduler.start()    # call once at application startup
    scheduler.stop()     # call at shutdown (APScheduler only)
    scheduler.trigger_gc()  # manually trigger a cycle
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Optional

from loguru import logger

if TYPE_CHECKING:
    from .gc import GarbageCollector
    from .policy import LifecyclePolicy


def _celery_workers_running() -> bool:
    """Whether a Celery worker answers on the broker (Beat runs beside it).

    A reachable broker with no worker used to count: ``inspect().ping()``
    returns None rather than raising, so the compose stack without its
    ``celery`` profile started no embedded scheduler and GC never ran.
    """
    from ..infrastructure.tasks.document_tasks import _ping_celery_workers

    return _ping_celery_workers(timeout=2.0)


class LifecycleScheduler:
    """Manages periodic GC scheduling.

    Automatically selects Celery Beat (production) or APScheduler
    (single-node / dev) based on broker availability.
    """

    def __init__(
        self,
        gc: "GarbageCollector",
        policy: "LifecyclePolicy",
    ) -> None:
        self._gc = gc
        self._policy = policy
        self._apscheduler: Optional[object] = None
        self._started = False
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the scheduler.  Idempotent (safe to call multiple times)."""
        with self._lock:
            if self._started:
                return

            if _celery_workers_running():
                logger.info(
                    "LifecycleScheduler: a Celery worker answered; GC scheduled via Celery Beat."
                )
                # Nothing to do here: celery_app.py beat_schedule already registers the tasks.
            else:
                logger.info("LifecycleScheduler: no Celery worker; starting embedded APScheduler.")
                self._start_apscheduler()

            self._started = True

    def stop(self) -> None:
        """Stop the embedded APScheduler (no-op for Celery mode)."""
        with self._lock:
            if self._apscheduler is not None:
                try:
                    self._apscheduler.shutdown(wait=False)
                except Exception:
                    pass
                self._apscheduler = None
            self._started = False

    def update_policy(self, new_policy: "LifecyclePolicy") -> None:
        """Update the active policy and reschedule APScheduler jobs if cadence changed.

        For Celery Beat deployments, cadence changes require a Beat process restart
        (Celery Beat reads the schedule only at startup).  In APScheduler mode,
        the interval jobs are rescheduled immediately via ``reschedule_job()``.
        """
        with self._lock:
            old_gc = self._policy.gc_cycle_hours
            old_orphan = self._policy.orphan_scan_hours
            self._policy = new_policy

            if self._apscheduler is not None:
                # Reschedule only if cadence values actually changed
                if new_policy.gc_cycle_hours != old_gc:
                    gc_hours = max(1, new_policy.gc_cycle_hours)
                    try:
                        self._apscheduler.reschedule_job(
                            "lifecycle_gc_cycle",
                            trigger="interval",
                            hours=gc_hours,
                        )
                        logger.info(f"LifecycleScheduler: GC interval rescheduled to {gc_hours}h")
                    except Exception as exc:
                        logger.warning(f"LifecycleScheduler: failed to reschedule GC job: {exc}")

                if new_policy.orphan_scan_hours != old_orphan:
                    orphan_hours = max(1, new_policy.orphan_scan_hours)
                    try:
                        self._apscheduler.reschedule_job(
                            "lifecycle_orphan_scan",
                            trigger="interval",
                            hours=orphan_hours,
                        )
                        logger.info(
                            f"LifecycleScheduler: orphan scan interval rescheduled to {orphan_hours}h"
                        )
                    except Exception as exc:
                        logger.warning(
                            f"LifecycleScheduler: failed to reschedule orphan scan job: {exc}"
                        )
            elif _celery_workers_running():
                if (
                    new_policy.gc_cycle_hours != old_gc
                    or new_policy.orphan_scan_hours != old_orphan
                ):
                    logger.warning(
                        "LifecycleScheduler: cadence fields changed, but Celery Beat reads the "
                        "schedule only at startup.  Restart the Beat process to apply new intervals."
                    )

    def trigger_gc(self) -> None:
        """Manually trigger a GC cycle.

        In Celery mode, enqueues a task.
        In APScheduler mode (or if Celery unavailable at call time), runs inline.
        """
        try:
            from ..infrastructure.tasks.document_tasks import run_gc_cycle_task

            if not _celery_workers_running():
                raise RuntimeError("no Celery worker answered")
            run_gc_cycle_task.delay()
            logger.info("LifecycleScheduler: GC task enqueued via Celery.")
        except Exception:
            logger.info("LifecycleScheduler: running GC inline (Celery not available).")
            self._run_gc()

    # =========================================================================
    # Private helpers
    # =========================================================================

    def _start_apscheduler(self) -> None:
        try:
            from apscheduler.schedulers.background import BackgroundScheduler
        except ImportError:
            # Optional: GC still runs when asked (POST /api/v1/admin/gc/run). A
            # warning on every `lasagna serve` start was noise for a core
            # install, which does not include the scheduler.
            logger.info(
                "Garbage collection runs when asked (POST /api/v1/admin/gc/run). To run "
                "it on a schedule, install the scheduler: pip install 'lasagna-rag[lifecycle]'."
            )
            return

        scheduler = BackgroundScheduler()
        gc_hours = max(1, self._policy.gc_cycle_hours)
        orphan_hours = max(1, self._policy.orphan_scan_hours)

        scheduler.add_job(
            func=self._run_gc,
            trigger="interval",
            hours=gc_hours,
            id="lifecycle_gc_cycle",
            max_instances=1,
            coalesce=True,
        )
        scheduler.add_job(
            func=self._run_orphan_scan,
            trigger="interval",
            hours=orphan_hours,
            id="lifecycle_orphan_scan",
            max_instances=1,
            coalesce=True,
        )

        scheduler.start()
        self._apscheduler = scheduler
        logger.info(f"APScheduler started: GC every {gc_hours}h, orphan scan every {orphan_hours}h")

    def _run_gc(self) -> None:
        """Run a GC cycle (acquires lock — safe to call from multiple processes)."""
        try:
            report = self._gc.run_full_cycle()
            if not report.aborted:
                logger.info(
                    f"Scheduled GC complete: cleaned={report.total_docs_cleaned}, "
                    f"freed={report.total_bytes_freed / 1024:.1f} KB"
                )
        except Exception as exc:
            logger.exception(f"Scheduled GC failed: {exc}")

    def _run_orphan_scan(self) -> None:
        """Run orphan detection (dry-run, logs results).

        Acquires the same DB advisory lock used by the GC cycle to prevent
        duplicate scans across multiple APScheduler workers/processes.
        """
        import uuid

        owner_id = f"orphan-{uuid.uuid4()}"
        lock_acquired = False
        try:
            if not self._gc._db.acquire_gc_lock(owner_id, ttl_seconds=3600):
                logger.info("Scheduled orphan scan skipped: advisory lock held by another process.")
                return
            lock_acquired = True
            from ..api.rag_backend import get_rag_backend

            backend = get_rag_backend()
            if backend is None:
                return
            report = backend.detect_orphans()
            if report.incomplete:
                logger.warning(f"Orphan scan incomplete: {report.incomplete_reason}")
            elif not report.is_consistent:
                logger.warning(
                    f"Orphan scan: {report.db_only_count} db-only, "
                    f"{report.vector_only_count} vector-only orphans detected. "
                    "Run POST /admin/gc/repair to fix."
                )
        except Exception as exc:
            logger.exception(f"Scheduled orphan scan failed: {exc}")
        finally:
            if lock_acquired:
                try:
                    self._gc._db.release_gc_lock(owner_id)
                except Exception:
                    pass
