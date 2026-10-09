"""Health, liveness, and readiness probes.

- ``GET /api/v1/health``       — full health check (all dependencies)
- ``GET /api/v1/health/live``  — liveness (is the process alive?)
- ``GET /api/v1/health/ready`` — readiness (can it handle requests?)
- ``GET /api/v1/stats``        — system statistics
"""

from __future__ import annotations

from time import perf_counter
from typing import Dict, Tuple

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from loguru import logger

from ..auth_mode import require_auth
from ..dependencies import get_backend
from ..models import (
    HealthCheckDetail,
    HealthResponse,
    LivenessResponse,
    StatsResponse,
)

router = APIRouter(tags=["health"])


def _run_health_checks(backend) -> Tuple[Dict[str, HealthCheckDetail], str]:
    """Run all dependency health checks and return (checks_dict, overall_status).

    Shared between readiness and full_health to avoid duplicated logic.
    """
    checks: dict[str, HealthCheckDetail] = {}
    overall = "healthy"

    # Check embedding model
    try:
        t0 = perf_counter()
        model_name = backend.embedding_manager.model_name
        dim = backend.embedding_manager.dimension
        ms = (perf_counter() - t0) * 1000
        checks["embedding_model"] = HealthCheckDetail(
            status="up",
            latency_ms=round(ms, 2),
            detail=f"{model_name} (dim={dim})",
        )
    except Exception as exc:
        checks["embedding_model"] = HealthCheckDetail(status="down", detail=str(exc))
        overall = "unhealthy"

    # Check vector backend
    try:
        t0 = perf_counter()
        vb = backend.get_vector_backend()
        backend_name = vb.backend_name
        ms = (perf_counter() - t0) * 1000
        checks["vector_backend"] = HealthCheckDetail(
            status="up",
            latency_ms=round(ms, 2),
            detail=backend_name,
        )
    except Exception as exc:
        checks["vector_backend"] = HealthCheckDetail(status="down", detail=str(exc))
        overall = "unhealthy"

    # Check metadata database
    try:
        from lasagna.core.database import get_rag_database

        t0 = perf_counter()
        db = get_rag_database()
        db.get_documents("__health_check__")
        ms = (perf_counter() - t0) * 1000
        db_detail = getattr(db, "database_backend", "sqlite")
        checks["database"] = HealthCheckDetail(
            status="up",
            latency_ms=round(ms, 2),
            detail=db_detail,
        )
    except Exception as exc:
        checks["database"] = HealthCheckDetail(status="down", detail=str(exc))
        overall = "degraded" if overall == "healthy" else overall

    return checks, overall


@router.get("/health/live", response_model=LivenessResponse)
async def liveness():
    """Liveness probe — is this process alive?

    Always returns 200 unless the process is deadlocked.
    Used by Kubernetes to decide whether to restart the container.
    """
    return LivenessResponse(status="alive")


@router.get("/health/ready", response_model=HealthResponse)
async def readiness(backend=Depends(get_backend)):
    """Readiness probe — can this instance handle requests?

    Checks that the RAG backend is initialised and core dependencies
    (vector DB, embedding model) are operational.  Returns 503 if any
    critical dependency is down so that the load balancer stops routing
    traffic to this instance.
    """
    checks, overall = _run_health_checks(backend)
    resp = HealthResponse(status=overall, checks=checks, require_auth=require_auth())

    if overall != "healthy":
        return JSONResponse(
            status_code=503,
            content=resp.model_dump(),
        )
    return resp


@router.get("/health", response_model=HealthResponse)
async def full_health(backend=Depends(get_backend)):
    """Full health check — reports dependency status but always returns 200.

    Use this for dashboards and monitoring; use ``/health/ready`` for
    orchestrator probes that need to remove unhealthy instances.
    """
    checks, overall = _run_health_checks(backend)
    return HealthResponse(status=overall, checks=checks, require_auth=require_auth())


@router.get("/stats", response_model=StatsResponse)
async def system_stats(backend=Depends(get_backend)):
    """System statistics — embedding model, features, cache stats."""
    try:
        raw = backend.get_stats()
        return StatsResponse(
            embedding_model=raw.get("embedding_model"),
            embedding_dim=raw.get("embedding_dim"),
            collection_name=raw.get("collection_name"),
            vector_backend=raw.get("vector_backend"),
            features=raw.get("features", {}),
            cache=raw.get("cache", {}),
        )
    except Exception as exc:
        logger.warning(f"Failed to collect stats: {exc}")
        return StatsResponse()
