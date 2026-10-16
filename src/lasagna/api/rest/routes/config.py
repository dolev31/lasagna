"""Configuration and profile endpoints.

- ``GET  /api/v1/config``               — get current config
- ``GET  /api/v1/config/profiles``      — list available profiles
- ``PUT  /api/v1/config/profiles/{name}`` — apply a preset profile
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import ValidationError

from lasagna.api._profile_guard import EmbeddingSizeConflict
from lasagna.core.session_config._profiles import BUILTIN_PROFILE_DESCRIPTIONS
from lasagna.api.rag_backend import RAGBackend
from ..dependencies import get_backend, get_profile_owner, get_session_id, get_user_id
from ..models import ConfigPatchRequest, ConfigResponse, ProfileInfo, ProfileListResponse


router = APIRouter(prefix="/config", tags=["config"])

# Flat REST keys → nested session override structure for document_awareness.
_DA_FLAT_TO_NESTED = {
    "document_awareness_enabled": ("document_awareness", "enabled"),
    "enrichment_strategy": ("document_awareness", "enrichment_strategy"),
    "first_tokens_count": ("document_awareness", "first_tokens_count"),
}


def _canonicalize_patch(patch_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Translate flat document_awareness keys to nested structure.

    REST API accepts ``document_awareness_enabled``, ``enrichment_strategy``,
    ``first_tokens_count`` as top-level keys for ergonomics.  RAGConfig reads
    them under ``document_awareness.*``, so we restructure before persisting.
    """
    result = dict(patch_dict)
    for flat_key, (parent, nested_key) in _DA_FLAT_TO_NESTED.items():
        if flat_key in result:
            result.setdefault(parent, {})[nested_key] = result.pop(flat_key)
    return result


_PROFILES = {
    name: description
    for name, description in BUILTIN_PROFILE_DESCRIPTIONS.items()
    if name != "default"
}


@router.get("", response_model=ConfigResponse)
async def get_config(
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Get the current RAG configuration for this session."""
    cfg = backend.get_session_config(session_id)
    try:
        return _config_response(cfg)
    except ValidationError as exc:
        # A value stored before writes were validated. Say which and how to
        # recover, rather than a 500 that leaves the session un-readable.
        keys = sorted({str(e["loc"][0]) for e in exc.errors() if e.get("loc")})
        raise HTTPException(
            status_code=422,
            detail=(
                f"This session's stored configuration has invalid values for "
                f"{', '.join(keys)}. Correct them with PATCH /api/v1/config, or "
                f"clear them with POST /api/config/knowledge/reset."
            ),
        )


def _config_response(cfg) -> ConfigResponse:
    return ConfigResponse(
        vector_backend=getattr(cfg, "vector_backend", "milvus"),
        embedding_model=cfg.embedding_model,
        embedding_dim=cfg.embedding_dim,
        chunking_strategy=cfg.chunking_strategy,
        enable_hybrid_search=cfg.enable_hybrid_search,
        enable_reranking=cfg.enable_reranking,
        enable_mmr=cfg.enable_mmr,
        enable_hyde=cfg.enable_hyde,
        enable_hierarchical=cfg.enable_hierarchical,
        default_top_k=cfg.default_top_k,
        rerank_top_k=cfg.rerank_top_k,
        similarity_threshold=cfg.similarity_threshold,
        hybrid_alpha=cfg.hybrid_alpha,
        mmr_lambda=cfg.mmr_lambda,
        use_agentic_rag=cfg.use_agentic_rag,
        enable_prometheus=cfg.enable_prometheus,
        enable_tracing=cfg.enable_tracing,
        # Structured graph settings
        use_structured_graph=getattr(cfg, "use_structured_graph", False),
        agentic_pipeline=getattr(cfg, "agentic_pipeline", "legacy_react"),
        enable_hallucination_check=getattr(cfg, "enable_hallucination_check", False),
        enable_answer_quality_check=getattr(cfg, "enable_answer_quality_check", False),
        max_quality_check_attempts=getattr(cfg, "max_quality_check_attempts", 2),
        coverage_scorer_mode=getattr(cfg, "coverage_scorer_mode", "keyword"),
        final_search_mode=getattr(cfg, "final_search_mode", "merge"),
        # Processing / Chunking
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        semantic_threshold=cfg.semantic_threshold,
        min_chunk_size=cfg.min_chunk_size,
        max_chunk_size=cfg.max_chunk_size,
        parent_chunk_size=cfg.parent_chunk_size,
        child_chunk_size=cfg.child_chunk_size,
        auto_merge_threshold=cfg.auto_merge_threshold,
        preserve_tables=cfg.preserve_tables,
        preserve_headers=cfg.preserve_headers,
        # OCR
        enable_ocr=cfg.enable_ocr,
        ocr_engine=cfg.ocr_engine,
        # Docling
        docling_enabled=cfg.docling_enabled,
        docling_mode=cfg.docling_mode,
        docling_ocr_enabled=cfg.docling_ocr_enabled,
        docling_table_structure=cfg.docling_table_structure,
        docling_max_tokens=cfg.docling_max_tokens,
        # Query classification & routing
        enable_query_classification=getattr(cfg, "enable_query_classification", True),
        enable_query_routing=getattr(cfg, "enable_query_routing", True),
        routing_strategy=getattr(cfg, "routing_strategy", "rule_based"),
        routing_default_tier=getattr(cfg, "routing_default_tier", "standard"),
        routing_confidence_threshold=getattr(cfg, "routing_confidence_threshold", 0.5),
        # Retrieval features
        enable_context_reordering=getattr(cfg, "enable_context_reordering", True),
        enable_contextual_retrieval=getattr(cfg, "enable_contextual_retrieval", False),
        enable_rag_fusion=getattr(cfg, "enable_rag_fusion", False),
        enable_crag=getattr(cfg, "enable_crag", False),
        # Hybrid search internals
        rrf_k=getattr(cfg, "rrf_k", 60),
        sparse_method=getattr(cfg, "sparse_method", "bm25"),
        chunk_refinement_method=getattr(cfg, "chunk_refinement_method", "auto"),
        # Context compression
        enable_context_compression=getattr(cfg, "enable_context_compression", False),
        compression_method=getattr(cfg, "compression_method", "extractive"),
        compression_ratio=getattr(cfg, "compression_ratio", 0.3),
        # Document awareness
        document_awareness_enabled=getattr(cfg, "document_awareness_enabled", True),
        enrichment_strategy=getattr(cfg, "enrichment_strategy", "first_tokens"),
        first_tokens_count=getattr(cfg, "first_tokens_count", 150),
        # Warnings
        warnings=(
            ["SPLADE is experimental; BM25 fallback may be used silently"]
            if getattr(cfg, "sparse_method", "bm25") == "splade"
            else []
        ),
    )


@router.get("/profiles", response_model=ProfileListResponse)
async def list_profiles(
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
):
    """List available configuration profiles."""
    profiles = [ProfileInfo(name=k, description=v) for k, v in _PROFILES.items()]

    # Detect active profile and include custom profiles
    active = None
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        active = mgr.get_active_profile(session_id)
    except Exception:
        pass

    return ProfileListResponse(profiles=profiles, active=active)


@router.patch("", response_model=ConfigResponse)
async def patch_config(
    patch: ConfigPatchRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Patch session-scoped RAG configuration.

    Accepts a partial config object: only non-null keys are updated.
    Unknown keys are rejected with 422.

    Example body: ``{"enable_hyde": true, "similarity_threshold": 0.5}``
    """
    try:
        from lasagna.core.session_config import get_knowledge_config_manager
        from lasagna.providers import get_session_provider

        patch_dict = _canonicalize_patch(patch.model_dump(exclude_none=True))

        # Write to KnowledgeConfigManager (persistence + validation).
        mgr = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        mgr.patch_config(patch_dict, session_id=session_id)

        # Write-through to session provider so RAGConfig reads the
        # overrides at search time.  KnowledgeConfigManager and
        # InMemorySessionProvider maintain separate stores; without
        # this second write, PATCH /config has no runtime effect.
        #
        # PersistentSessionProvider delegates to the manager already,
        # so skip the write-through to avoid double validation/persist.
        sp = get_session_provider()
        if not getattr(sp, "_delegates_to_manager", False):
            # What the manager kept: a server setting it dropped (its value
            # was the one in force) is not the session's either.
            from lasagna.core.session_config._session_keys import session_key_problem

            kept = {k: v for k, v in patch_dict.items() if session_key_problem(k, v) is None}
            sp.deep_merge_config_overrides(session_id, kept)
    except EmbeddingSizeConflict as conflict:
        raise HTTPException(status_code=409, detail=str(conflict))
    except ValueError as ve:
        logger.warning(f"Validation error patching config: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        logger.error(f"Failed to patch config: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    return await get_config(session_id=session_id, backend=backend)


@router.put("/profiles/{profile_name}", response_model=ConfigResponse)
async def apply_profile(
    profile_name: str,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_profile_owner),
    backend: RAGBackend = Depends(get_backend),
):
    """Apply a preset configuration profile for this session."""
    if profile_name not in _PROFILES:
        raise HTTPException(
            status_code=404,
            detail=f"Profile '{profile_name}' not found. Available: {list(_PROFILES.keys())}",
        )

    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        mgr.set_active_profile(profile_name, session_id=session_id, user_id=user_id)
    except EmbeddingSizeConflict as conflict:
        raise HTTPException(status_code=409, detail=str(conflict))
    except ValueError as ve:
        logger.warning(f"Validation error applying profile {profile_name}: {ve}")
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as exc:
        logger.error(f"Failed to apply profile {profile_name}: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")

    # Return updated config
    return await get_config(session_id=session_id, backend=backend)
