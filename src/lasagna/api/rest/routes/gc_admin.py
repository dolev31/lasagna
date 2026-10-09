"""GC / Lifecycle admin endpoints.

Protected by ``RAG_ADMIN_TOKEN`` environment variable (``X-Admin-Token`` header).
If the variable is not set, the endpoints are **disabled** (503 Service Unavailable).

Routes:
- ``POST   /api/v1/admin/gc/run``               — trigger a full GC cycle
- ``GET    /api/v1/admin/gc/orphans``           — run orphan detection (read-only)
- ``POST   /api/v1/admin/gc/repair``            — repair orphans (?dry_run=true default)
- ``GET    /api/v1/admin/gc/policy``            — return the current lifecycle policy
- ``PATCH  /api/v1/admin/gc/policy``            — update lifecycle policy fields at runtime
- ``DELETE /api/v1/admin/sessions/{session_id}``— bulk-delete an entire session
"""

from __future__ import annotations

import dataclasses
import secrets
from typing import Any, Dict, Optional

from fastapi import APIRouter, Body, Depends, Header, HTTPException, Query
from fastapi.concurrency import run_in_threadpool
from loguru import logger

from lasagna.api.rag_backend import RAGBackend
from ..dependencies import get_backend

router = APIRouter(prefix="/admin", tags=["admin"])

# Allowlist of fields that can be updated via PATCH /admin/gc/policy
_ALLOWED_POLICY_KEYS = {
    "soft_delete_grace_hours",
    "document_ttl_days",
    "error_retention_days",
    "session_idle_days",
    "gc_cycle_hours",
    "orphan_scan_hours",
    "safety_threshold_percent",
    "batch_size",
    "reconcile_quota_on_startup",
}


def validate_policy_updates(body: Dict[str, Any]) -> Optional[str]:
    """Validate type and range of policy update fields.

    Returns an error string if validation fails, or None if valid.
    """
    errors = []

    def _check_non_neg_int(key: str, *, min_val: int = 0) -> None:
        v = body.get(key)
        if v is None:
            return
        if not isinstance(v, int) or isinstance(v, bool):
            errors.append(f"'{key}' must be an integer (got {type(v).__name__})")
        elif v < min_val:
            errors.append(f"'{key}' must be >= {min_val} (got {v})")

    def _check_pos_int(key: str) -> None:
        _check_non_neg_int(key, min_val=1)

    if "soft_delete_grace_hours" in body:
        _check_non_neg_int("soft_delete_grace_hours")
    if "error_retention_days" in body:
        _check_non_neg_int("error_retention_days")
    if "session_idle_days" in body:
        _check_non_neg_int("session_idle_days")
    if "gc_cycle_hours" in body:
        _check_pos_int("gc_cycle_hours")
    if "orphan_scan_hours" in body:
        _check_pos_int("orphan_scan_hours")
    if "batch_size" in body:
        _check_pos_int("batch_size")

    # document_ttl_days: None or 0 (both mean disabled), or a positive int.
    if "document_ttl_days" in body:
        v = body["document_ttl_days"]
        if v is not None:
            if not isinstance(v, int) or isinstance(v, bool):
                errors.append(
                    f"'document_ttl_days' must be a non-negative integer or null (got {type(v).__name__})"
                )
            elif v < 0:
                errors.append(f"'document_ttl_days' must be >= 0 or null (got {v})")

    # safety_threshold_percent: float strictly between 0 and 1
    if "safety_threshold_percent" in body:
        v = body["safety_threshold_percent"]
        if v is None:
            errors.append("'safety_threshold_percent' cannot be null")
        elif not isinstance(v, (int, float)) or isinstance(v, bool):
            errors.append(f"'safety_threshold_percent' must be a float (got {type(v).__name__})")
        elif not (0.0 < float(v) <= 1.0):
            errors.append(f"'safety_threshold_percent' must be in (0, 1] (got {v})")

    # reconcile_quota_on_startup: bool
    if "reconcile_quota_on_startup" in body:
        v = body["reconcile_quota_on_startup"]
        if not isinstance(v, bool):
            errors.append(
                f"'reconcile_quota_on_startup' must be a boolean (got {type(v).__name__})"
            )

    return "; ".join(errors) if errors else None


# ---------------------------------------------------------------------------
# Authorization helper
# ---------------------------------------------------------------------------


def _require_admin_token(x_admin_token: Optional[str] = Header(default=None)) -> None:
    """Verify the X-Admin-Token header against RAG_ADMIN_TOKEN env var.

    If ``RAG_ADMIN_TOKEN`` is not configured, all admin endpoints are disabled.
    """
    from lasagna.api.rest.auth_mode import configured_secret

    expected = configured_secret("RAG_ADMIN_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail=(
                "Admin endpoints are disabled (RAG_ADMIN_TOKEN not configured, or "
                "set to a placeholder or a value shorter than 32 characters)."
            ),
        )
    # Constant-time, like `routes/sessions.py:37` -- which has a test asserting
    # it. This path carries MORE privilege (it can delete any session) and had
    # the weaker comparison, so a token could be recovered a byte at a time.
    if not x_admin_token or not secrets.compare_digest(x_admin_token, expected):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing X-Admin-Token header.",
        )


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------


def _get_live_policy(backend: RAGBackend):
    """Return the live LifecyclePolicy from the backend (or load fresh)."""
    try:
        mgr = getattr(backend, "_lifecycle_manager", None)
        if mgr is not None:
            return mgr._policy
    except Exception:
        pass
    from lasagna.lifecycle.policy import LifecyclePolicy

    return LifecyclePolicy.load()


def _set_live_policy(backend: RAGBackend, policy) -> None:
    """Update the live LifecyclePolicy in the backend (runtime-only, not persisted).

    Propagates the update to:
    - ``_lifecycle_manager._policy``
    - ``_lifecycle_manager._gc._policy`` (GarbageCollector)
    - ``_lifecycle_scheduler.update_policy()`` (reschedules APScheduler jobs if cadence changed)
    """
    try:
        mgr = getattr(backend, "_lifecycle_manager", None)
        if mgr is not None:
            mgr._policy = policy
            # Also propagate to the GC instance if it exists
            gc = getattr(mgr, "_gc", None)
            if gc is not None:
                gc._policy = policy
        # Reschedule APScheduler jobs if cadence fields changed
        scheduler = getattr(backend, "_lifecycle_scheduler", None)
        if scheduler is not None:
            scheduler.update_policy(policy)
    except Exception as exc:
        logger.warning(f"Could not set live policy on backend: {exc}")


# ---------------------------------------------------------------------------
# GC run
# ---------------------------------------------------------------------------


class GCRunRequest(Dict[str, Any]):
    pass


@router.post("/gc/run")
async def run_gc(
    session_id: Optional[str] = Query(None, description="Limit GC to a single session"),
    _: None = Depends(_require_admin_token),
    backend: RAGBackend = Depends(get_backend),
) -> Dict[str, Any]:
    """Trigger a manual full garbage-collection cycle.

    Runs all 6 GC phases:

    1. Hard-delete expired soft-deleted documents
    2. Expire documents past their TTL
    3. Clean up old ERROR documents
    4. Clean up expired citations
    5. Clean up idle sessions
    6. Reconcile quota counters

    Idempotent — safe to call multiple times.  Returns per-phase results.
    """
    try:
        report = await run_in_threadpool(backend.run_gc, session_id=session_id)
        return _gc_report_to_dict(report)
    except Exception as exc:
        logger.error(f"GC run failed: {exc}")
        raise HTTPException(status_code=500, detail=f"GC run failed: {exc}")


# ---------------------------------------------------------------------------
# Orphan detection / repair
# ---------------------------------------------------------------------------


@router.get("/gc/orphans")
async def detect_orphans(
    session_id: Optional[str] = Query(None, description="Limit scan to a single session"),
    _: None = Depends(_require_admin_token),
    backend: RAGBackend = Depends(get_backend),
) -> Dict[str, Any]:
    """Detect orphaned data between the metadata DB and vector store.

    Returns a consistency report with:
    - ``db_only``: Documents in the DB but missing from the vector store
      (likely a failed ingestion).
    - ``vec_only``: Chunks in the vector store with no DB record
      (likely a leaked vector after a hard-delete failure).
    - ``scan_limited``: True if the vector scan hit ``vector_scan_limit`` and
      results may be incomplete.

    This endpoint is **read-only** — it does not modify any data.
    Pass the report to ``POST /admin/gc/repair`` to fix the issues.
    """
    try:
        report = await run_in_threadpool(backend.detect_orphans, session_id)
        return {
            "session_id": session_id,
            "db_only_count": report.db_only_count,
            "vec_only_count": report.vector_only_count,
            "scan_limited": report.scan_limited,
            "vector_scan_limit": report.vector_scan_limit,
            "estimated_unscanned": report.estimated_unscanned,
            "db_only": [
                {"document_id": o.document_id, "kind": o.kind}
                for o in report.orphans
                if o.kind == "db_only"
            ],
            "vec_only": [
                {"document_id": o.document_id, "kind": o.kind}
                for o in report.orphans
                if o.kind == "vector_only"
            ],
        }
    except Exception as exc:
        logger.error(f"Orphan detection failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Orphan detection failed: {exc}")


@router.post("/gc/repair")
async def repair_orphans(
    session_id: Optional[str] = Query(None, description="Limit repair to a single session"),
    dry_run: bool = Query(
        True, description="Preview changes without applying them (default: true)"
    ),
    _: None = Depends(_require_admin_token),
    backend: RAGBackend = Depends(get_backend),
) -> Dict[str, Any]:
    """Detect and repair orphaned data.

    First runs orphan detection, then repairs found orphans.

    Args:
        dry_run: When ``true`` (default), returns the repair plan without
            making any changes.  Set to ``false`` to apply repairs.
    """
    try:
        report = await run_in_threadpool(backend.detect_orphans, session_id)
        repair = await run_in_threadpool(backend.repair_orphans, report, dry_run=dry_run)
        return {
            "dry_run": dry_run,
            "db_only_found": report.db_only_count,
            "vec_only_found": report.vector_only_count,
            "scan_limited": report.scan_limited,
            "db_records_deleted": repair.repaired_db_only,
            "vectors_deleted": repair.repaired_vector_only,
            "errors": repair.errors,
        }
    except Exception as exc:
        logger.error(f"Orphan repair failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Orphan repair failed: {exc}")


# ---------------------------------------------------------------------------
# Lifecycle policy management
# ---------------------------------------------------------------------------


@router.get("/gc/policy")
async def get_gc_policy(
    _: None = Depends(_require_admin_token),
    backend: RAGBackend = Depends(get_backend),
) -> Dict[str, Any]:
    """Return the currently active lifecycle garbage-collection policy.

    The policy governs soft-delete grace window, TTL expiry, GC scheduling
    intervals, and safety thresholds.  Changes made via ``PATCH /admin/gc/policy``
    are reflected here immediately but are **not persisted** across restarts.
    """
    policy = await run_in_threadpool(_get_live_policy, backend)
    return dataclasses.asdict(policy)


@router.patch("/gc/policy")
async def patch_gc_policy(
    body: Dict[str, Any] = Body(...),
    _: None = Depends(_require_admin_token),
    backend: RAGBackend = Depends(get_backend),
) -> Dict[str, Any]:
    """Update one or more lifecycle policy fields at runtime.

    Changes take effect immediately for the current process but are **not
    persisted** — the server reverts to its configured policy on restart.
    To make changes permanent, update ``[rag.lifecycle]`` in
    ``settings.lasagna.toml`` or the corresponding ``RAG_GC_*`` env vars.

    Unknown field names are rejected with HTTP 422.
    """
    unknown = set(body.keys()) - _ALLOWED_POLICY_KEYS
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown lifecycle policy fields: {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_POLICY_KEYS)}",
        )

    # Validate types and ranges before applying
    validation_error = validate_policy_updates(body)
    if validation_error:
        raise HTTPException(status_code=422, detail=validation_error)

    try:
        policy = await run_in_threadpool(_get_live_policy, backend)

        # Allow explicit null or 0 for document_ttl_days (both mean TTL disabled → None).
        # For all other fields, skip None so omitted fields aren't overwritten.
        def _normalize_ttl(k, v):
            if k == "document_ttl_days":
                return None if not v else v  # 0 and None → None
            return v

        updated = dataclasses.replace(
            policy,
            **{
                k: _normalize_ttl(k, v)
                for k, v in body.items()
                if v is not None or k == "document_ttl_days"
            },
        )
        await run_in_threadpool(_set_live_policy, backend, updated)
        return dataclasses.asdict(updated)
    except Exception as exc:
        logger.error(f"Policy patch failed: {exc}")
        raise HTTPException(status_code=500, detail=f"Policy patch failed: {exc}")


# ---------------------------------------------------------------------------
# Bulk session delete
# ---------------------------------------------------------------------------


@router.delete("/sessions/{session_id}")
async def bulk_delete_session(
    session_id: str,
    _: None = Depends(_require_admin_token),
    backend: RAGBackend = Depends(get_backend),
) -> Dict[str, Any]:
    """Hard-delete ALL documents in a session.

    This is a **destructive, irreversible** operation.  All document records,
    vectors, files, and citations in the session are permanently removed.

    Requires ``X-Admin-Token`` authentication.
    """
    try:
        result = await run_in_threadpool(backend.bulk_delete_session, session_id, confirm=True)
        return {
            "session_id": session_id,
            "requested": result.requested,
            "deleted": result.deleted,
            "bytes_freed": result.bytes_freed,
            "chunks_freed": result.chunks_freed,
            "errors": result.errors,
        }
    except Exception as exc:
        logger.error(f"Bulk session delete failed for {session_id}: {exc}")
        raise HTTPException(status_code=500, detail=f"Bulk delete failed: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _gc_report_to_dict(report) -> Dict[str, Any]:
    """Serialize a GCReport to a JSON-serialisable dict."""
    phases = []
    for phase_result in getattr(report, "phases", []):
        phases.append(
            {
                "phase": phase_result.phase,
                "docs_scanned": phase_result.docs_scanned,
                "docs_cleaned": phase_result.docs_cleaned,
                "bytes_freed": phase_result.bytes_freed,
                "duration_ms": phase_result.duration_ms,
                "errors": phase_result.errors,
                # safety_abort lives in details, not as a top-level field on GCPhaseResult
                "safety_aborted": phase_result.details.get("safety_abort", False),
            }
        )
    return {
        "run_id": getattr(report, "run_id", ""),
        "started_at": str(getattr(report, "started_at", "")),
        "completed_at": str(getattr(report, "completed_at", "")),
        "total_docs_cleaned": getattr(report, "total_docs_cleaned", 0),
        "total_bytes_freed": getattr(report, "total_bytes_freed", 0),
        "total_errors": getattr(report, "total_errors", 0),
        "aborted": getattr(report, "aborted", False),
        "phases": phases,
    }
