"""Agentic pipeline discovery endpoint.

- ``GET /api/v1/pipelines`` — list all registered agentic pipelines
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from loguru import logger

from lasagna.core.config import rag_config
from lasagna.core.registry import registry
from ..models import PipelineInfo, PipelineListResponse

router = APIRouter(prefix="/pipelines", tags=["pipelines"])


@router.get("", response_model=PipelineListResponse)
async def list_pipelines():
    """List all registered agentic pipelines.

    Returns metadata for every pipeline registered in the component
    registry, including built-in and custom (entry-point or decorator)
    pipelines.  The ``active`` field shows which pipeline the current
    config resolves to.
    """
    try:
        raw = registry.list_pipelines()
    except Exception as exc:
        logger.error(f"Failed to list pipelines from registry: {exc}")
        raise HTTPException(status_code=500, detail="Failed to list pipelines") from exc

    pipelines = [
        PipelineInfo(
            name=p.get("name", "unknown"),
            description=p.get("description", ""),
            class_name=p.get("class", "unknown"),
        )
        for p in raw
    ]

    try:
        active = rag_config.agentic_pipeline
    except Exception as exc:
        logger.warning(f"Failed to read active pipeline config, defaulting to legacy_react: {exc}")
        active = "legacy_react"

    return PipelineListResponse(
        pipelines=pipelines,
        active=active,
        total=len(pipelines),
    )
