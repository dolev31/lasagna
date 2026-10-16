"""Background document processing tasks via Celery or ThreadPoolExecutor.

Three-tier dispatch for document processing:

1. **Celery** — When ``celery_enabled = true`` and a worker answers on the broker,
   tasks are enqueued to Celery workers for fully distributed processing.
2. **Thread pool** — When Celery is unavailable and ``async_processing = true``
   (default), a bounded ``ThreadPoolExecutor`` processes documents in
   background threads.  The pool size and backpressure limit are controlled
   by ``max_concurrent_uploads`` (default 5).
3. **Synchronous** — When both async paths are unavailable or at capacity,
   processing happens inline in the caller's thread. A caller that must not
   block (the REST upload route) passes ``refuse_when_busy=True`` and gets
   `ProcessingAtCapacity` instead.

On shutdown, `drain_document_processing` waits for the thread pool up to
``processing_timeout`` and marks what is still unfinished as failed, so no
document is left in "processing" by a process that no longer exists.

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
from typing import Callable, Dict, List, Optional, Tuple

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
    collection that holds documents. So are a model that is not downloaded
    while offline, a missing or unreadable file, a missing package, and a
    rejected credential. Retrying those (twice, 30 s apart) held a
    `lasagna upload` for over a minute before it reported the failure, and
    kept the Python interpreter from exiting for as long.
    """
    from lasagna.api._document_pipeline import is_dim_mismatch_error
    from lasagna.core.model_downloads import ModelNotDownloaded

    permanent = (
        ValueError,
        TypeError,
        ModelNotDownloaded,
        FileNotFoundError,
        PermissionError,
        ImportError,
    )
    return isinstance(exc, permanent) or is_dim_mismatch_error(exc) or _is_auth_error(exc)


def _is_auth_error(exc: BaseException) -> bool:
    """A provider refused the credential (HTTP 401 or 403, or its SDK's auth error)."""
    if getattr(exc, "status_code", None) in (401, 403):
        return True
    return any(
        "Authentication" in cls.__name__ or "PermissionDenied" in cls.__name__
        for cls in type(exc).__mro__
    )


def _failure_reason(exc: BaseException, session_id: str) -> str:
    """The reason recorded on a failed document, with the fix when there is one."""
    from lasagna.core.model_downloads import ModelNotDownloaded, fix_for_model

    if isinstance(exc, ModelNotDownloaded):
        return f"{exc} {fix_for_model(session_id, exc.model)}"
    return str(exc)


#: Set when the interpreter starts exiting; ends every wait for a retry.
_exiting = threading.Event()


def _wait_before_retry(delay: float) -> bool:
    """Wait ``delay`` seconds before a retry; False when the interpreter is exiting.

    A ThreadPoolExecutor's threads are joined when Python exits, so a retry
    asleep for 30 s held a finished script for up to a minute.
    """
    return not _exiting.wait(delay)


def _end_retry_waits_at_exit() -> None:
    """Have the interpreter's exit end the retry waits before it joins the threads.

    Python joins a ThreadPoolExecutor's threads in a threading-atexit hook,
    which runs before ``atexit`` handlers and before the main thread counts
    as finished. Those hooks run last registered first, so this one,
    registered after the executor's module was imported, runs before the join.
    """
    try:
        threading._register_atexit(_exiting.set)
    except Exception as exc:  # noqa: BLE001 - already exiting, or another Python
        logger.debug(f"Retry waits will not end early at exit: {exc}")


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

        # Get session-scoped config. This may be another process than the one
        # that took the upload; process_document embeds with this config's
        # model and makes its collection ready.
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
        # The command that waits on the document (`lasagna upload`) reports
        # the failure itself, and hides this line by its `document_failure` key.
        logger.bind(document_failure=document_id).error(
            f"[Worker] Failed to process document {document_id}: {exc}"
        )

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
                error_message=_failure_reason(exc, session_id),
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
            if report.incomplete:
                logger.warning(
                    f"Orphan scan incomplete, nothing repaired: {report.incomplete_reason}"
                )
                return {"dry_run": dry_run, "incomplete": report.incomplete_reason}
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

#: Documents submitted to the thread pool and not finished:
#: document_id -> (future, session_id, reservation_id, reserved_file_size).
_inflight: Dict[str, Tuple[concurrent.futures.Future, str, str, int]] = {}
_inflight_lock = threading.Lock()


class ProcessingAtCapacity(RuntimeError):
    """Every document-processing thread is busy and the caller may not wait.

    The REST upload route answers 503 with it. Processing inline instead held
    the request, and the worker serving it, for as long as the document took.
    """


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
                _end_retry_waits_at_exit()
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

    Mirrors Celery's ``max_retries=2, default_retry_delay=30`` behaviour. A
    document stays "processing" between attempts and is marked failed only
    when no attempt is left: it used to be marked failed after the first,
    so a client polling for it saw a failure that a retry then reversed.
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
                    # Asks this loop for the retry instead of failing the document.
                    retry_ctx=TaskRetryContext(
                        retries=attempt,
                        max_retries=max_retries,
                        retry_fn=_RetryRequested,
                    ),
                )
                return  # success
            except Exception as exc:
                retry = isinstance(exc, _RetryRequested) or (
                    attempt < max_retries and not _is_permanent(exc)
                )
                if not retry:
                    # Terminal failure: _process_document_impl marked the
                    # document failed with the reason and released the quota.
                    return
                logger.info(
                    f"[ThreadPool] Retrying document {document_id} "
                    f"(attempt {attempt + 1}/{max_retries}) in {retry_delay}s"
                )
                if not _wait_before_retry(retry_delay):
                    _fail_at_exit(document_id, session_id, reservation_id, reserved_file_size)
                    return
    finally:
        semaphore.release()


def _fail_at_exit(
    document_id: str, session_id: str, reservation_id: str, reserved_file_size: int
) -> None:
    """Mark a document failed whose retry the exiting interpreter will not run."""
    from lasagna.core.database import get_rag_database
    from lasagna.core.schema import DocumentStatus

    try:
        get_rag_database().update_document_status(
            document_id=document_id,
            status=DocumentStatus.ERROR.value,
            error_message=(
                "The process exited before processing could be retried. Upload the "
                "document again, or reindex it."
            ),
        )
    except Exception as exc:
        logger.warning(f"Could not mark {document_id} failed at exit: {exc}")
    if reservation_id:
        try:
            from lasagna.core.quota import get_quota_manager

            get_quota_manager().release_upload_reservation(
                session_id, reserved_file_size, reservation_id=reservation_id
            )
        except Exception as exc:
            logger.warning(f"Could not release the quota reservation of {document_id}: {exc}")


# ---------------------------------------------------------------------------
# Celery workers — is anyone consuming the queue?
# ---------------------------------------------------------------------------

#: Seconds a worker probe's answer is reused, so uploads do not each wait on a ping.
WORKER_PROBE_TTL_S = 30.0
_worker_probe: Tuple[float, bool] = (float("-inf"), False)
_worker_probe_lock = threading.Lock()
_last_fallback_reason: Optional[str] = None


def _ping_celery_workers(timeout: float = 1.0) -> bool:
    """Whether at least one Celery worker answers a ping on the broker."""
    try:
        from lasagna.infrastructure.tasks.celery_app import celery_app

        if celery_app is None:
            return False
        # limit=1: return on the first reply instead of waiting out the timeout.
        return bool(celery_app.control.ping(timeout=timeout, limit=1))
    except Exception as exc:  # noqa: BLE001 - an unreachable broker is "no worker"
        logger.debug(f"Celery worker ping failed: {exc}")
        return False


def celery_workers_answer(now: Optional[float] = None) -> bool:
    """`_ping_celery_workers`, reusing an answer for `WORKER_PROBE_TTL_S` seconds."""
    global _worker_probe
    now = time.monotonic() if now is None else now
    with _worker_probe_lock:
        checked_at, answered = _worker_probe
        if now - checked_at < WORKER_PROBE_TTL_S:
            return answered
        answered = _ping_celery_workers()
        _worker_probe = (now, answered)
        return answered


def choose_dispatch(
    celery_enabled: bool,
    celery_installed: bool,
    workers_answer: Callable[[], bool],
    broker: str = "",
) -> Tuple[str, Optional[str]]:
    """Where a document is processed: ``"celery"`` or ``"local"``, and why not Celery.

    The reason is None when Celery was not asked for, or is used. With
    ``RAG_CELERY_ENABLED`` on and Redis reachable, ``.delay()`` used to
    succeed with no worker running (the compose stack without its ``celery``
    profile), so every upload waited in the queue for good. A queue only
    gets a document when a worker answers on the broker.
    """
    if not celery_enabled:
        return "local", None
    if not celery_installed:
        return "local", (
            "RAG_CELERY_ENABLED is true but celery is not installed, so uploads are processed "
            'in this process. Install it with pip install "lasagna-rag[production]", or set '
            "RAG_CELERY_ENABLED=false."
        )
    if not workers_answer():
        where = f" on {broker}" if broker else ""
        return "local", (
            f"RAG_CELERY_ENABLED is true but no Celery worker answered{where}, so uploads are "
            "processed in this process. Start a worker (with compose: --profile celery), or "
            "set RAG_CELERY_ENABLED=false."
        )
    return "celery", None


def _note_fallback(reason: Optional[str]) -> None:
    """Log a change of dispatch decision once, not on every upload."""
    global _last_fallback_reason
    if reason == _last_fallback_reason:
        return
    _last_fallback_reason = reason
    if reason:
        logger.warning(reason)
    else:
        logger.info("A Celery worker answered; uploads are queued for it again.")


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
    refuse_when_busy: bool = False,
    wait_when_busy: float = 0.0,
) -> bool:
    """Dispatch document processing to a background worker.

    Three-tier dispatch with explicit fallback semantics:

    1. **Celery** — ``celery_enabled=true`` and a worker answers on the
       broker (`choose_dispatch`) → enqueue. With no worker, or if Celery
       dispatch *fails*, log WARNING and fall through to tier 2.
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
        refuse_when_busy: Raise `ProcessingAtCapacity` instead of returning
            ``False`` when the thread pool is full.
        wait_when_busy: Seconds to wait for a thread to free up when the pool
            is full, then raise `ProcessingAtCapacity`. A batch upload waits,
            so it is processed in full by the pool's threads; 0 does not wait.

    Returns:
        ``True`` if the task was dispatched (Celery or thread pool),
        ``False`` if the caller should process synchronously.
    """
    from lasagna.core.config import rag_config

    # --- Tier 1: Celery ---
    from lasagna.core.provenance import redact_url

    celery_enabled = bool(getattr(rag_config, "celery_enabled", False))
    route, reason = choose_dispatch(
        celery_enabled,
        process_document_task is not None,
        celery_workers_answer,
        broker=redact_url(getattr(rag_config, "celery_broker_url", "")) if celery_enabled else "",
    )
    if celery_enabled:
        _note_fallback(reason)
    if route == "celery":
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

    if not semaphore.acquire(blocking=False) and not (
        wait_when_busy > 0 and semaphore.acquire(timeout=wait_when_busy)
    ):
        if wait_when_busy > 0:
            raise ProcessingAtCapacity(
                f"All {max_workers} document-processing threads stayed busy for "
                f"{wait_when_busy:.0f} s, so this file was not accepted. Send it again, "
                f"or raise RAG_MAX_CONCURRENT_UPLOADS."
            )
        if refuse_when_busy:
            raise ProcessingAtCapacity(
                f"All {max_workers} document-processing threads are busy, so the upload "
                f"was not accepted. Retry shortly, or raise RAG_MAX_CONCURRENT_UPLOADS."
            )
        logger.warning(
            f"Thread pool at capacity ({max_workers}), document {document_id} "
            f"will be processed synchronously"
        )
        return False

    try:
        executor = _get_doc_executor(max_workers)
        future = executor.submit(
            _thread_pool_wrapper,
            document_id,
            session_id,
            file_path,
            reservation_id,
            reserved_file_size,
            semaphore,
        )
        with _inflight_lock:
            _inflight[document_id] = (future, session_id, reservation_id, reserved_file_size)
        future.add_done_callback(lambda f, doc=document_id: _forget(doc, f))
        logger.info(f"Document {document_id} dispatched to thread pool for background processing")
        return True
    except Exception as exc:
        semaphore.release()
        logger.warning(
            f"Thread pool dispatch failed for {document_id}, will process synchronously: {exc}"
        )
        return False


def _forget(document_id: str, future: concurrent.futures.Future) -> None:
    with _inflight_lock:
        entry = _inflight.get(document_id)
        if entry is not None and entry[0] is future:
            del _inflight[document_id]


def drain_document_processing(timeout: float) -> List[str]:
    """Wait up to ``timeout`` seconds for in-flight documents; fail the rest.

    Shutdown used to leave every document still in the thread pool in
    "processing" for good: the process exited, nothing would finish it, and
    clients polling its status waited forever. A document still running after
    ``timeout`` is marked failed with the reason, and its quota reservation is
    released. Returns the ids marked failed.
    """
    with _inflight_lock:
        pending = dict(_inflight)
    if not pending:
        return []
    logger.info(f"Waiting up to {timeout:g} s for {len(pending)} document(s) still processing")
    futures = {entry[0]: doc_id for doc_id, entry in pending.items()}
    _done, not_done = concurrent.futures.wait(list(futures), timeout=timeout)
    unfinished = sorted(futures[f] for f in not_done)
    if not unfinished:
        return []

    from lasagna.core.database import get_rag_database
    from lasagna.core.quota import get_quota_manager
    from lasagna.core.schema import DocumentStatus

    reason = (
        f"The server shut down before processing finished (it waited {timeout:g} s, "
        f"processing_timeout). Upload the document again, or reindex it."
    )
    for doc_id in unfinished:
        _future, session_id, reservation_id, reserved_file_size = pending[doc_id]
        try:
            get_rag_database().update_document_status(
                document_id=doc_id, status=DocumentStatus.ERROR.value, error_message=reason
            )
        except Exception as exc:
            logger.warning(f"Could not mark {doc_id} failed at shutdown: {exc}")
        if reservation_id:
            try:
                get_quota_manager().release_upload_reservation(
                    session_id, reserved_file_size, reservation_id=reservation_id
                )
            except Exception as exc:
                logger.warning(f"Could not release the quota reservation of {doc_id}: {exc}")
    logger.warning(f"Marked {len(unfinished)} unfinished document(s) failed at shutdown")
    return unfinished
