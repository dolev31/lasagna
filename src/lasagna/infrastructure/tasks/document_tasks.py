"""Background document processing tasks via Celery or ThreadPoolExecutor.

Three-tier dispatch for document processing:

1. **Celery** — When ``celery_enabled = true`` and a broker is reachable,
   tasks are enqueued to Celery workers for fully distributed processing.
2. **Thread pool** — When Celery is unavailable and ``async_processing = true``
   (default), a bounded ``ThreadPoolExecutor`` processes documents in
   background threads.  The pool size and backpressure limit are controlled
   by ``max_concurrent_uploads`` (default 5).
3. **Synchronous** — When both async paths are unavailable or at capacity,
   processing happens inline in the caller's thread.

Usage::

    from lasagna.infrastructure.tasks.document_tasks import (
        dispatch_processing,
    )

    # Returns True if dispatched (Celery or thread pool), False = sync
    dispatched = dispatch_processing(document_id, session_id, file_path)
"""

from __future__ import annotations

import atexit
import concurrent.futures
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

from loguru import logger


# ---------------------------------------------------------------------------
# Retry context — decouples task logic from Celery infrastructure
# ---------------------------------------------------------------------------


@dataclass
class TaskRetryContext:
    """Encapsulates retry state so _process_document_impl can be tested
    without Celery.

    Attributes:
        retries: Number of retries already attempted.
        max_retries: Maximum number of retries allowed.
        retry_fn: Callable that raises an exception to trigger a retry.
            When ``None``, retries are disabled (e.g., sync fallback).
    """

    retries: int = 0
    max_retries: int = 2
    retry_fn: Optional[Callable] = None


# ---------------------------------------------------------------------------
# Core processing logic (Celery-independent, testable)
# ---------------------------------------------------------------------------


class _RetryRequested(Exception):
    """Sentinel raised by _process_document_impl when a retry is needed."""


def _is_permanent(exc: BaseException) -> bool:
    """Whether processing the same file again must fail the same way.

    The pipeline raises ValueError and TypeError for what retrying cannot fix:
    an unsupported or empty file, no extractable text, a size mismatch with a
    collection that holds documents. Retrying those (twice, 30 s apart) held a
    `lasagna upload` for over a minute before it reported the failure.
    """
    from lasagna.api._document_pipeline import is_dim_mismatch_error

    return isinstance(exc, (ValueError, TypeError)) or is_dim_mismatch_error(exc)


def _process_document_impl(
    document_id: str,
    session_id: str,
    file_path: str,
    reservation_id: str = "",
    reserved_file_size: int = 0,
    *,
    retry_ctx: Optional[TaskRetryContext] = None,
) -> dict:
    """Core document processing logic shared by Celery task and tests.

    Loads the RAG backend, retrieves the document metadata, and runs the
    full processing pipeline (text extraction → chunking → embedding →
    vector indexing).

    When ``reservation_id`` is provided (from the pre-dispatch
    ``check_and_reserve_upload()`` call), the function finalizes the
    reservation on success or releases it on terminal failure — keeping
    quota accounting atomic across the sync and async paths.

    Args:
        document_id: Unique document identifier.
        session_id: Owning session.
        file_path: Path to the stored file on disk.
        reservation_id: Quota reservation token (empty = legacy path).
        reserved_file_size: File size that was reserved (for rollback).
        retry_ctx: Retry state. When ``None``, no retries are attempted.

    Returns:
        dict with ``document_id``, ``status``, and ``chunks_count``.

    Raises:
        _RetryRequested: When a transient error occurs and retries remain.
        Exception: On terminal failure (retries exhausted or not configured).
    """
    if retry_ctx is None:
        retry_ctx = TaskRetryContext()

    logger.info(f"[Worker] Processing document {document_id} for session {session_id[:8]}...")

    try:
        from lasagna.api.rag_backend import get_rag_backend
        from lasagna.core.schema import DocumentStatus

        backend = get_rag_backend()

        # Retrieve the persisted document record
        doc = backend._get_document_metadata(document_id)
        if doc is None:
            raise ValueError(f"Document {document_id} not found in database")

        # Get session-scoped config
        config = backend.get_session_config(session_id)

        # Run the full processing pipeline
        backend._document_pipeline.process_document(doc, file_path, config=config)

        # Refresh from DB — must exist after processing
        updated = backend._get_document_metadata(document_id)
        if updated is None:
            raise ValueError(f"Document {document_id} missing from DB after processing")

        chunks_count = updated.chunks_count
        status = updated.status.value

        # Finalize quota reservation (or fall back to record_upload for
        # tasks dispatched without a reservation_id).
        from lasagna.core.quota import get_quota_manager

        quota_mgr = get_quota_manager()
        if reservation_id:
            quota_mgr.finalize_upload_reservation(session_id, chunks_count)
        else:
            quota_mgr.record_upload(
                session_id=session_id,
                file_size_bytes=doc.file_size,
                chunks=chunks_count,
            )

        # Invalidate search cache so new document is immediately searchable
        try:
            from lasagna.infrastructure.cache import get_search_cache

            search_config = backend.get_session_config(session_id)
            if search_config.enable_search_cache:
                search_cache = get_search_cache(search_config)
                invalidated = search_cache.invalidate_session(session_id)
                if invalidated > 0:
                    logger.debug(
                        f"[Worker] Search cache invalidated: {invalidated} entries for session {session_id[:8]}..."
                    )
        except Exception as cache_exc:
            logger.warning(f"[Worker] Cache invalidation failed: {cache_exc}")

        logger.info(f"[Worker] Document {document_id} processed: {chunks_count} chunks")
        return {
            "document_id": document_id,
            "status": status,
            "chunks_count": chunks_count,
        }

    except Exception as exc:
        logger.error(f"[Worker] Failed to process document {document_id}: {exc}")

        # Retry transient errors before marking as ERROR
        if (
            retry_ctx.retries < retry_ctx.max_retries
            and retry_ctx.retry_fn is not None
            and not _is_permanent(exc)
        ):
            logger.info(
                f"[Worker] Retrying document {document_id} (attempt {retry_ctx.retries + 1}/{retry_ctx.max_retries})"
            )
            # Do NOT release reservation on retry — it will be finalized
            # or released when the retry completes.
            raise retry_ctx.retry_fn(exc)

        # All retries exhausted — mark ERROR in database and release reservation
        from lasagna.utils.monitoring import METRICS

        METRICS.counter("lasagna_ingests", 1, tags={"outcome": "error"})
        try:
            from lasagna.core.database import get_rag_database
            from lasagna.core.schema import DocumentStatus

            rag_db = get_rag_database()
            rag_db.update_document_status(
                document_id=document_id,
                status=DocumentStatus.ERROR.value,
                error_message=str(exc),
            )
        except Exception as db_exc:
            logger.error(f"[Worker] Could not update error status: {db_exc}")

        # Release the quota reservation on terminal failure
        if reservation_id:
            try:
                from lasagna.core.quota import get_quota_manager

                quota_mgr = get_quota_manager()
                quota_mgr.release_upload_reservation(
                    session_id,
                    reserved_file_size,
                    reservation_id=reservation_id,
                )
            except Exception as q_exc:
                logger.error(f"[Worker] Could not release quota reservation: {q_exc}")

        raise


# ---------------------------------------------------------------------------
# Celery task definition (only when Celery is installed)
# ---------------------------------------------------------------------------

try:
    from celery import shared_task
except ImportError:
    shared_task = None  # type: ignore[assignment,misc]


if shared_task is not None:

    @shared_task(
        name="rag.process_document",
        bind=True,
        max_retries=2,
        default_retry_delay=30,
        acks_late=True,
    )
    def process_document_task(
        self,
        document_id: str,
        session_id: str,
        file_path: str,
        reservation_id: str = "",
        reserved_file_size: int = 0,
    ) -> dict:
        """Celery task wrapper — delegates to _process_document_impl."""
        ctx = TaskRetryContext(
            retries=self.request.retries,
            max_retries=self.max_retries,
            retry_fn=lambda exc: self.retry(exc=exc),
        )
        return _process_document_impl(
            document_id,
            session_id,
            file_path,
            reservation_id=reservation_id,
            reserved_file_size=reserved_file_size,
            retry_ctx=ctx,
        )

else:
    # Celery not installed — define a no-op placeholder so imports don't break
    process_document_task = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# GC Celery tasks (only when Celery is installed)
# ---------------------------------------------------------------------------

if shared_task is not None:

    @shared_task(
        name="knowledge.gc.run_full_cycle",
        bind=False,
        max_retries=0,
        acks_late=True,
    )
    def run_gc_cycle_task(session_id: Optional[str] = None) -> dict:
        """Celery task: run a full garbage-collection cycle.

        This task is typically scheduled via Celery Beat (daily at 03:00 UTC).
        It runs all 6 GC phases and writes results to the ``gc_audit_log`` table.

        Args:
            session_id: If provided, limit GC to a single session.

        Returns:
            Dict summary of the GC report.
        """
        try:
            from lasagna.api.rag_backend import get_rag_backend

            backend = get_rag_backend()
            report = backend.run_gc(session_id=session_id)
            logger.info(
                f"GC cycle completed: cleaned={report.total_docs_cleaned}, "
                f"bytes={report.total_bytes_freed}, errors={report.total_errors}"
            )
            return {
                "run_id": report.run_id,
                "total_docs_cleaned": report.total_docs_cleaned,
                "total_bytes_freed": report.total_bytes_freed,
                "total_errors": report.total_errors,
                "aborted": report.aborted,
            }
        except Exception as exc:
            logger.exception(f"GC cycle task failed: {exc}")
            return {"error": str(exc)}

    @shared_task(
        name="knowledge.gc.orphan_scan",
        bind=False,
        max_retries=0,
        acks_late=True,
    )
    def run_orphan_scan_task(dry_run: bool = True) -> dict:
        """Celery task: scan for orphaned data between DB and vector store.

        This task is typically scheduled via Celery Beat (weekly on Sunday
        at 04:00 UTC).  In ``dry_run=True`` mode it only logs the report.
        Set ``dry_run=False`` to auto-repair found orphans.

        Args:
            dry_run: If True, detect only; if False, also repair.

        Returns:
            Dict summary of the orphan detection/repair report.
        """
        try:
            from lasagna.api.rag_backend import get_rag_backend

            backend = get_rag_backend()
            report = backend.detect_orphans()
            db_only_count = report.db_only_count
            vec_only_count = report.vector_only_count
            total = db_only_count + vec_only_count
            logger.info(f"Orphan scan: db_only={db_only_count}, vec_only={vec_only_count}")
            if total > 0 and not dry_run:
                repair = backend.repair_orphans(report, dry_run=False)
                logger.info(
                    f"Orphan repair: db_deleted={repair.repaired_db_only}, "
                    f"vec_deleted={repair.repaired_vector_only}, errors={repair.errors}"
                )
                return {
                    "db_only_found": db_only_count,
                    "vec_only_found": vec_only_count,
                    "db_records_deleted": repair.repaired_db_only,
                    "vectors_deleted": repair.repaired_vector_only,
                    "repair_errors": repair.errors,
                }
            return {
                "dry_run": dry_run,
                "db_only_found": db_only_count,
                "vec_only_found": vec_only_count,
            }
        except Exception as exc:
            logger.exception(f"Orphan scan task failed: {exc}")
            return {"error": str(exc)}

else:
    # Celery not installed — define no-op placeholders
    run_gc_cycle_task = None  # type: ignore[assignment]
    run_orphan_scan_task = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Thread pool — lightweight async path when Celery is unavailable
# ---------------------------------------------------------------------------

_doc_executor: Optional[concurrent.futures.ThreadPoolExecutor] = None
_doc_executor_lock = threading.Lock()
_doc_semaphore: Optional[threading.Semaphore] = None
_doc_semaphore_lock = threading.Lock()


def _get_doc_executor(max_workers: int = 5) -> concurrent.futures.ThreadPoolExecutor:
    """Get or create the shared document processing thread pool."""
    global _doc_executor
    if _doc_executor is None:
        with _doc_executor_lock:
            if _doc_executor is None:
                _doc_executor = concurrent.futures.ThreadPoolExecutor(
                    max_workers=max_workers,
                    thread_name_prefix="doc-process",
                )
                atexit.register(_doc_executor.shutdown, wait=False)
    return _doc_executor


def _get_doc_semaphore(max_workers: int = 5) -> threading.Semaphore:
    """Get or create the backpressure semaphore for the document thread pool."""
    global _doc_semaphore
    if _doc_semaphore is None:
        with _doc_semaphore_lock:
            if _doc_semaphore is None:
                _doc_semaphore = threading.Semaphore(max_workers)
    return _doc_semaphore


def _thread_pool_wrapper(
    document_id: str,
    session_id: str,
    file_path: str,
    reservation_id: str,
    reserved_file_size: int,
    semaphore: threading.Semaphore,
) -> None:
    """Run document processing in a thread pool thread with retries.

    Mirrors Celery's ``max_retries=2, default_retry_delay=30`` behaviour.
    The semaphore is released in ``finally`` to guarantee backpressure
    accounting stays correct regardless of outcome.
    """
    max_retries = 2
    retry_delay = 30

    try:
        for attempt in range(max_retries + 1):
            try:
                _process_document_impl(
                    document_id,
                    session_id,
                    file_path,
                    reservation_id=reservation_id,
                    reserved_file_size=reserved_file_size,
                    # No retry_fn — we handle retries in this loop
                    retry_ctx=TaskRetryContext(
                        retries=attempt,
                        max_retries=max_retries,
                        retry_fn=None,
                    ),
                )
                return  # success
            except Exception as exc:
                if attempt < max_retries and not _is_permanent(exc):
                    logger.info(
                        f"[ThreadPool] Retrying document {document_id} "
                        f"(attempt {attempt + 1}/{max_retries}) in {retry_delay}s"
                    )
                    time.sleep(retry_delay)
                else:
                    # Terminal failure — _process_document_impl already
                    # marked ERROR in DB and released quota reservation.
                    logger.error(
                        f"[ThreadPool] Document {document_id} failed after {attempt} retries"
                    )
                    return
    finally:
        semaphore.release()


# ---------------------------------------------------------------------------
# Dispatcher — public API used by the document pipeline / REST layer
# ---------------------------------------------------------------------------


def dispatch_processing(
    document_id: str,
    session_id: str,
    file_path: str,
    *,
    reservation_id: str = "",
    reserved_file_size: int = 0,
) -> bool:
    """Dispatch document processing to a background worker.

    Three-tier dispatch with explicit fallback semantics:

    1. **Celery** — ``celery_enabled=true`` and broker reachable → enqueue.
       If Celery dispatch *fails* (broker down), log WARNING and fall
       through to tier 2.
    2. **Thread pool** — ``async_processing=true`` (default) and semaphore
       available → submit to ``ThreadPoolExecutor``.  If the pool is at
       capacity (semaphore full), log WARNING and fall through to tier 3.
    3. **Synchronous** — return ``False``; caller processes inline.

    Args:
        document_id: Unique document identifier.
        session_id: Owning session.
        file_path: Path to the stored file on disk.
        reservation_id: Quota reservation token from ``check_and_reserve_upload()``.
        reserved_file_size: File size that was reserved (for rollback on failure).

    Returns:
        ``True`` if the task was dispatched (Celery or thread pool),
        ``False`` if the caller should process synchronously.
    """
    from lasagna.core.config import rag_config

    # --- Tier 1: Celery ---
    celery_enabled = getattr(rag_config, "celery_enabled", False)
    if celery_enabled and process_document_task is not None:
        try:
            process_document_task.delay(
                document_id,
                session_id,
                file_path,
                reservation_id=reservation_id,
                reserved_file_size=reserved_file_size,
            )
            logger.info(f"Document {document_id} dispatched to Celery for background processing")
            return True
        except Exception as exc:
            logger.warning(
                f"Celery dispatch failed for {document_id}, falling through to thread pool: {exc}"
            )
            # Fall through to tier 2

    # --- Tier 2: Thread pool ---
    async_enabled = getattr(rag_config, "async_processing", True)
    if not async_enabled:
        return False

    max_workers = getattr(rag_config, "max_concurrent_uploads", 5)
    semaphore = _get_doc_semaphore(max_workers)

    if not semaphore.acquire(blocking=False):
        logger.warning(
            f"Thread pool at capacity ({max_workers}), document {document_id} "
            f"will be processed synchronously"
        )
        return False

    try:
        executor = _get_doc_executor(max_workers)
        executor.submit(
            _thread_pool_wrapper,
            document_id,
            session_id,
            file_path,
            reservation_id,
            reserved_file_size,
            semaphore,
        )
        logger.info(f"Document {document_id} dispatched to thread pool for background processing")
        return True
    except Exception as exc:
        semaphore.release()
        logger.warning(
            f"Thread pool dispatch failed for {document_id}, will process synchronously: {exc}"
        )
        return False
