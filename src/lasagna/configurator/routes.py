"""Configurator API routes.

- ``GET  /configurator/api/schema``   — parameter definitions (types, ranges, help text)
- ``POST /configurator/api/validate`` — validate a config without saving
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from lasagna.api.rest.dependencies import get_session_id

from .schema import get_schema_dict
from .validation import validate_config

# With auth on, a verified caller only, like every other route in the app.
# Validation looks custom pipeline names up in the registry, which answers
# whether a name is registered on this instance.
router = APIRouter(tags=["configurator"], dependencies=[Depends(get_session_id)])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ValidateRequest(BaseModel):
    config: dict[str, Any]
    baseline: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/api/schema")
async def get_schema():
    """Return the full parameter schema for the configurator UI."""
    return get_schema_dict()


@router.post("/api/validate")
async def validate(body: ValidateRequest):
    """Validate a configuration and return errors, warnings, and reindex info."""
    result = validate_config(body.config, body.baseline)
    return result.to_dict()
