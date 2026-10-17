"""Knowledge configuration endpoints for the configurator UI.

These endpoints are mounted at ``/api/config/knowledge`` (not under ``/api/v1/``)
to match the configurator frontend's hardcoded API paths. Deprecated for any
other caller, which should use ``/api/v1/config``; they move under ``/api/v1``
in 0.2.0.

All endpoints delegate to :class:`KnowledgeConfigManager` which provides
full profile CRUD, session-scoped config, and import/export.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from lasagna.api._profile_guard import EmbeddingSizeConflict
from lasagna.core.session_config import get_knowledge_config_manager
from lasagna.core.session_config._data_models import SettingsNotSaved
from lasagna.core.store_locks import StoreLockTimeout
from ..dependencies import get_profile_owner, get_session_id
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

#: Answered by the app with a 503 and the reason (see ``app._answer_switch_in_progress``):
#: caught below as any other error, a store lock held past its wait, or a change
#: the store did not save, was a 500.
STORE_UNAVAILABLE = (StoreLockTimeout, SettingsNotSaved)


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
    user_id: str = Depends(get_profile_owner),
):
    """Get the full knowledge configuration state for a session.

    Returns the profiles this caller may see, active profile, session
    overrides, and RAG status. This is the primary endpoint the configurator
    UI calls on load.
    """
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        return _to_response(mgr.get_config(session_id, user_id=user_id))
    except STORE_UNAVAILABLE:
        raise
    except Exception as exc:
        logger.error(f"Failed to get knowledge config for session={session_id}: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.put("/active", response_model=KnowledgeConfigFullResponse)
async def set_active_profile(
    body: SetActiveProfileRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_profile_owner),
):
    """Set the active profile and optionally apply config overrides.

    With ``reindex``, the session's documents are reindexed with the new
    profile first, and the switch happens only once every one is.
    """
    from lasagna.api._profile_switch import ProfileCannotLoad, ReindexFailed, switch_profile

    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        report = await asyncio.to_thread(
            switch_profile,
            mgr,
            body.active_profile,
            config=body.config or None,
            session_id=session_id,
            user_id=user_id,
            force=body.force,
            reindex=body.reindex,
            reset=body.reset,
        )
        response = _to_response(mgr.get_config(session_id, user_id=user_id))
        if body.reindex:
            response.reindex = report.to_dict()
        return response
    except (EmbeddingSizeConflict, ProfileCannotLoad, ReindexFailed) as conflict:
        raise HTTPException(status_code=409, detail=str(conflict))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except STORE_UNAVAILABLE:
        raise
    except Exception as exc:
        logger.error(f"Failed to set active profile '{body.active_profile}': {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/profile", response_model=KnowledgeConfigFullResponse)
async def save_profile(
    body: SaveProfileRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_profile_owner),
):
    """Save or update a named profile globally."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        mgr.save_profile(
            profile_name=body.profile_name,
            config=body.config,
            session_id=session_id,
            set_active=body.set_active,
            user_id=user_id,
        )
        return _to_response(mgr.get_config(session_id, user_id=user_id))
    except EmbeddingSizeConflict as conflict:
        raise HTTPException(status_code=409, detail=str(conflict))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except STORE_UNAVAILABLE:
        raise
    except Exception as exc:
        logger.error(f"Failed to save profile '{body.profile_name}': {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/profile/delete", response_model=KnowledgeConfigFullResponse)
async def delete_profile(
    body: DeleteProfileRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_profile_owner),
):
    """Delete a custom profile (built-in profiles are protected)."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        mgr.delete_profile(
            profile_name=body.profile_name,
            session_id=session_id,
            user_id=user_id,
        )
        return _to_response(mgr.get_config(session_id, user_id=user_id))
    except EmbeddingSizeConflict as conflict:
        raise HTTPException(status_code=409, detail=str(conflict))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except STORE_UNAVAILABLE:
        raise
    except Exception as exc:
        logger.error(f"Failed to delete profile '{body.profile_name}': {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/profile/rename", response_model=KnowledgeConfigFullResponse)
async def rename_profile(
    body: RenameProfileRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_profile_owner),
):
    """Rename a custom profile (built-in profiles are protected)."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        mgr.rename_profile(
            old_name=body.old_name,
            new_name=body.new_name,
            session_id=session_id,
            user_id=user_id,
        )
        return _to_response(mgr.get_config(session_id, user_id=user_id))
    except EmbeddingSizeConflict as conflict:  # a switch of the session is running
        raise HTTPException(status_code=409, detail=str(conflict))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except STORE_UNAVAILABLE:
        raise
    except Exception as exc:
        logger.error(f"Failed to rename profile '{body.old_name}' -> '{body.new_name}': {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/reset", response_model=KnowledgeConfigFullResponse)
async def reset_config(
    body: ResetConfigRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_profile_owner),
):
    """Reset configuration to defaults.

    Scope ``"active"`` clears session overrides only.
    Scope ``"all"`` resets profiles and active profile too.
    """
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        mgr.reset_config(scope=body.scope, session_id=session_id)
        return _to_response(mgr.get_config(session_id, user_id=user_id))
    except EmbeddingSizeConflict as conflict:
        raise HTTPException(status_code=409, detail=str(conflict))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except STORE_UNAVAILABLE:
        raise
    except Exception as exc:
        logger.error(f"Failed to reset config (scope={body.scope}): {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/export", response_model=KnowledgeConfigFullResponse)
async def export_config(
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_profile_owner),
):
    """Export the session configuration as JSON (for download)."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        return _to_response(mgr.export_config(session_id, user_id=user_id))
    except STORE_UNAVAILABLE:
        raise
    except Exception as exc:
        logger.error(f"Failed to export config: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/import", response_model=KnowledgeConfigFullResponse)
async def import_config(
    data: ImportConfigRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_profile_owner),
):
    """Import configuration from uploaded JSON."""
    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        mgr.import_config(import_data=data.model_dump(exclude_none=True), session_id=session_id)
        return _to_response(mgr.get_config(session_id, user_id=user_id))
    except EmbeddingSizeConflict as conflict:
        raise HTTPException(status_code=409, detail=str(conflict))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except STORE_UNAVAILABLE:
        raise
    except Exception as exc:
        logger.error(f"Failed to import config: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
