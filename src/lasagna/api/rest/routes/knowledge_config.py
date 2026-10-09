"""Knowledge configuration endpoints for the configurator UI.

These endpoints are mounted at ``/api/config/knowledge`` (not under ``/api/v1/``)
to match the configurator frontend's hardcoded API paths.

All endpoints delegate to :class:`KnowledgeConfigManager` which provides
full profile CRUD, session-scoped config, and import/export.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from lasagna.core.session_config import get_knowledge_config_manager
from ..dependencies import get_session_id, get_user_id
from ..models import (
    DeleteProfileRequest,
    ImportConfigRequest,
    KnowledgeConfigFullResponse,
    ProfileDetail,
    RenameProfileRequest,
    ResetConfigRequest,
    SaveProfileRequest,
    SetActiveProfileRequest,
)

router = APIRouter(prefix="/api/config/knowledge", tags=["configurator"])


def _to_response(data: Dict[str, Any]) -> KnowledgeConfigFullResponse:
    """Convert KnowledgeConfigState.to_dict() to API response."""
    profiles = []
    for p in data.get("profiles", []):
        if isinstance(p, dict):
            profiles.append(
                ProfileDetail(
                    name=p.get("name", ""),
                    description=p.get("description", ""),
                    config=p.get("config", {}),
                )
            )
    return KnowledgeConfigFullResponse(
        schema_version=data.get("schema_version", 1),
        defaults_source=data.get("defaults_source", "settings.lasagna.toml"),
        profiles=profiles,
        active_profile=data.get("active_profile", "default"),
        session_overrides=data.get("session_overrides", {}),
        rag_enabled=data.get("rag_enabled", True),
    )


@router.get("", response_model=KnowledgeConfigFullResponse)
async def get_knowledge_config(
    session_id: str = Depends(get_session_id),
):
    """Get the full knowledge configuration state for a session.

    Returns all profiles, active profile, session overrides, and RAG status.
    This is the primary endpoint the configurator UI calls on load.
    """
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        data = mgr.get_config(session_id)
        return _to_response(data)
    except Exception as exc:
        logger.error(f"Failed to get knowledge config for session={session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/active", response_model=KnowledgeConfigFullResponse)
async def set_active_profile(
    body: SetActiveProfileRequest,
    session_id: str = Depends(get_session_id),
):
    """Set the active profile and optionally apply config overrides."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        data = mgr.set_active_profile(
            profile_name=body.active_profile,
            config=body.config or None,
            session_id=session_id,
        )
        return _to_response(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to set active profile '{body.active_profile}': {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/profile", response_model=KnowledgeConfigFullResponse)
async def save_profile(
    body: SaveProfileRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
):
    """Save or update a named profile globally."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        data = mgr.save_profile(
            profile_name=body.profile_name,
            config=body.config,
            session_id=session_id,
            set_active=body.set_active,
            user_id=user_id,
        )
        return _to_response(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to save profile '{body.profile_name}': {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/profile/delete", response_model=KnowledgeConfigFullResponse)
async def delete_profile(
    body: DeleteProfileRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
):
    """Delete a custom profile (built-in profiles are protected)."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        data = mgr.delete_profile(
            profile_name=body.profile_name,
            session_id=session_id,
            user_id=user_id,
        )
        return _to_response(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to delete profile '{body.profile_name}': {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/profile/rename", response_model=KnowledgeConfigFullResponse)
async def rename_profile(
    body: RenameProfileRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
):
    """Rename a custom profile (built-in profiles are protected)."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        data = mgr.rename_profile(
            old_name=body.old_name,
            new_name=body.new_name,
            session_id=session_id,
            user_id=user_id,
        )
        return _to_response(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to rename profile '{body.old_name}' -> '{body.new_name}': {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reset", response_model=KnowledgeConfigFullResponse)
async def reset_config(
    body: ResetConfigRequest,
    session_id: str = Depends(get_session_id),
):
    """Reset configuration to defaults.

    Scope ``"active"`` clears session overrides only.
    Scope ``"all"`` resets profiles and active profile too.
    """
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        data = mgr.reset_config(scope=body.scope, session_id=session_id)
        return _to_response(data)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to reset config (scope={body.scope}): {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/export", response_model=KnowledgeConfigFullResponse)
async def export_config(
    session_id: str = Depends(get_session_id),
):
    """Export the full session configuration as JSON (for download)."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        data = mgr.export_config(session_id)
        return _to_response(data)
    except Exception as exc:
        logger.error(f"Failed to export config: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/import", response_model=KnowledgeConfigFullResponse)
async def import_config(
    data: ImportConfigRequest,
    session_id: str = Depends(get_session_id),
):
    """Import configuration from uploaded JSON."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        result = mgr.import_config(
            import_data=data.model_dump(exclude_none=True), session_id=session_id
        )
        return _to_response(result)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        logger.error(f"Failed to import config: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
