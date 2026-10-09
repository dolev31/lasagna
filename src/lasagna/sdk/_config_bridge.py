"""Session-safe config bridge: KnowledgeConfig -> backend RAGConfig.

Maps SDK ``KnowledgeConfig`` fields to backend config keys and provides
functions to extract only user-changed (non-default) fields for injection.

Two tiers:
    - **Init-time fields**: Consumed during ``RAGBackend._do_init()`` once.
      Must be set in ``rag_config._env_overrides`` *before* the backend
      singleton is created.  Inherently process-wide.
    - **Runtime fields**: Read per-request via ``_get_session_config(session_id)``.
      Written via ``SessionConfigProvider.set_config_overrides()`` for
      per-session safety.

Only fields whose value differs from the SDK dataclass default are bridged.
This prevents masking profile / TOML settings with SDK defaults.

Adding a new field:
    1. Add the SDK field to ``KnowledgeConfig`` in ``config.py``.
    2. Add the mapping here (``INIT_TIME_FIELDS`` or ``RUNTIME_FIELD_MAP``).
    3. If the backend key differs from the SDK key, add to ``SPECIAL_MAPPINGS``.
    4. If the backend uses a nested dict, add to ``NESTED_MAPPINGS``.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from .config import KnowledgeConfig


# =============================================================================
# Init-Time Fields (process-wide, applied BEFORE backend singleton creation)
# =============================================================================

INIT_TIME_FIELDS: Set[str] = {
    # Vector backend
    "vector_backend",
    "vector_backend_config",
    # Observability
    "enable_prometheus",
    "enable_tracing",
    # Multi-tenancy
    "tenant_isolation",
    "quota_enabled",
    # Rate limiting
    "rate_limiting_enabled",
    # Database
    "database_backend",
    "database_dsn",
    "database_pool_min",
    "database_pool_max",
    # Cache
    "cache_backend",
    "redis_url",
    "redis_pool_size",
    # Celery
    "celery_enabled",
    "celery_broker_url",
    "celery_result_backend",
    # Connection pool
    "connection_pool_size",
    # Storage
    "storage_backend",
    "storage_backend_config",
    # Embedding.
    #
    # These are init-time because the EmbeddingManager is constructed when the
    # backend is (`api/rag_backend.py`), from global config. They were mapped
    # only as RUNTIME fields, so `KnowledgeClient(config=KnowledgeConfig(
    # embedding_model=...))` wrote a per-session override that could not
    # possibly change a model already loaded -- the SDK's primary configuration
    # surface silently did nothing for the one setting that costs 5.4 GB to get
    # wrong, and the README documented it as a known limitation.
    #
    # They remain in RUNTIME_FIELD_MAP as well: the evaluation harness sweeps
    # embedding models per config and pairs each with its own `collection_name`,
    # which is a different mechanism and still needs the session path.
    "embedding_model",
    "embedding_model_fallback",
    "embedding_dim",
    "embedding_truncate_dim",
    "embedding_provider",
    "normalize_embeddings",
}


# =============================================================================
# Runtime Field Map (SDK field name -> backend config key, 1:1)
# =============================================================================

RUNTIME_FIELD_MAP: Dict[str, str] = {
    # Search toggles
    "enable_hybrid_search": "enable_hybrid_search",
    "enable_reranking": "enable_reranking",
    "enable_mmr": "enable_mmr",
    "enable_hyde": "enable_hyde",
    "enable_query_expansion": "enable_query_expansion",
    "enable_query_decomposition": "enable_query_decomposition",
    "enable_context_reordering": "enable_context_reordering",
    "enable_context_compression": "enable_context_compression",
    "compression_method": "compression_method",
    "compression_ratio": "compression_ratio",
    "enable_graph_retrieval": "enable_graph_retrieval",
    "graph_max_hops": "graph_max_hops",
    # Reranker config knobs
    "custom_reranker": "custom_reranker",
    "llm_reranker_model": "llm_reranker_model",
    "llm_reranker_window_size": "llm_reranker_window_size",
    "colbert_model": "colbert_model",
    # Sparse method
    "sparse_method": "sparse_method",
    # GraphRAG extraction
    "graph_entity_types": "graph_entity_types",
    "graph_extraction_model": "graph_extraction_model",
    "enable_contextual_retrieval": "enable_contextual_retrieval",
    "enable_rag_fusion": "enable_rag_fusion",
    "enable_crag": "enable_crag",
    "crag_web_search_fallback": "crag_web_search_fallback",
    # Search parameters
    "rrf_k": "rrf_k",
    "coverage_scorer_mode": "coverage_scorer_mode",
    "hybrid_alpha": "hybrid_alpha",
    "default_top_k": "default_top_k",
    "rerank_top_k": "rerank_top_k",
    "similarity_threshold": "similarity_threshold",
    "mmr_lambda": "mmr_lambda",
    "rag_fusion_n_queries": "rag_fusion_n_queries",
    # HyDE
    "hyde_model": "hyde_model",
    "hyde_n_docs": "hyde_n_docs",
    "max_query_variations": "max_query_variations",
    # Embedding
    "embedding_model": "embedding_model",
    "embedding_model_fallback": "embedding_model_fallback",
    "embedding_dim": "embedding_dim",
    "embedding_truncate_dim": "embedding_truncate_dim",
    "embedding_provider": "embedding_provider",
    "embedding_backend": "embedding_backend",
    "normalize_embeddings": "normalize_embeddings",
    "use_gpu": "use_gpu",
    "batch_size": "batch_size",
    # Reranking
    "reranker_model": "reranker_model",
    "reranker_batch_size": "reranker_batch_size",
    # Chunking
    "chunking_strategy": "chunking_strategy",
    "chunk_size": "chunk_size",
    "chunk_overlap": "chunk_overlap",
    "semantic_threshold": "semantic_threshold",
    "min_chunk_size": "min_chunk_size",
    "max_chunk_size": "max_chunk_size",
    "enable_hierarchical": "enable_hierarchical",
    "parent_chunk_size": "parent_chunk_size",
    "child_chunk_size": "child_chunk_size",
    "auto_merge_threshold": "auto_merge_threshold",
    "preserve_tables": "preserve_tables",
    "preserve_headers": "preserve_headers",
    "max_content_length": "max_content_length",
    # OCR
    "enable_ocr": "enable_ocr",
    "ocr_engine": "ocr_engine",
    "ocr_languages": "ocr_languages",
    "ocr_confidence_threshold": "ocr_confidence_threshold",
    # Cache
    "enable_embedding_cache": "enable_embedding_cache",
    "embedding_cache_size": "embedding_cache_size",
    "enable_search_cache": "enable_search_cache",
    "search_cache_ttl": "search_cache_ttl",
    "search_cache_max_size": "search_cache_max_size",
    # Citations
    "citation_format": "citation_format",
    "citation_id_length": "citation_id_length",
    "verify_citations": "verify_citations",
    "track_citations_db": "track_citations_db",
    "show_page_numbers": "show_page_numbers",
    "show_confidence_scores": "show_confidence_scores",
    "citation_db_max_age_hours": "citation_db_max_age_hours",
    "citation_db_max_per_session": "citation_db_max_per_session",
    # Observability (runtime-readable)
    "log_level": "log_level",
    # Docling
    "docling_enabled": "docling_enabled",
    # Agentic
    "use_agentic_rag": "use_agentic_rag",
    "max_agent_iterations": "max_agent_iterations",
    "chunk_refinement_method": "chunk_refinement_method",
    "refinement_model": "refinement_model",
    "agent_temperature": "agent_temperature",
    # Structured graph / quality gates
    "use_structured_graph": "use_structured_graph",
    "enable_hallucination_check": "enable_hallucination_check",
    "enable_answer_quality_check": "enable_answer_quality_check",
    # Query Routing
    "enable_query_routing": "enable_query_routing",
    "routing_strategy": "routing_strategy",
    "routing_default_tier": "routing_default_tier",
    "routing_confidence_threshold": "routing_confidence_threshold",
    # Search parameters
    "mmr_score_normalization": "mmr_score_normalization",
    # Query classification
    "enable_query_classification": "enable_query_classification",
    # Agentic pipeline selection
    "agentic_pipeline": "agentic_pipeline",
}


# =============================================================================
# Special Mappings (SDK field name != backend key name)
# =============================================================================

SPECIAL_MAPPINGS: Dict[str, str] = {
    "hyde_strategy": "agentic_hyde_strategy",
}


# =============================================================================
# Nested Mappings (SDK flat fields -> backend nested dict)
# =============================================================================

NESTED_MAPPINGS: Dict[str, Dict[str, str]] = {
    "chunk_cite_retrieval": {
        "chunk_retrieval_strategy": "strategy",
        "chunk_retrieval_threshold": "threshold_score",
        "chunk_retrieval_max": "max_chunks",
        "chunk_retrieval_min": "min_chunks",
        "score_gap_buffer": "score_gap_buffer",
        "min_gap_ratio": "min_gap_ratio",
    },
    "document_awareness": {
        "document_awareness_enabled": "enabled",
        "enrichment_strategy": "enrichment_strategy",
        "first_tokens_count": "first_tokens_count",
        "max_documents_in_context": "max_documents_in_context",
    },
}


# =============================================================================
# Core Functions
# =============================================================================


def get_non_default_fields(config: "KnowledgeConfig") -> Dict[str, Any]:
    """Return only fields whose value differs from the dataclass default.

    This prevents the bridge from writing SDK defaults as session overrides,
    which would mask profile / TOML / env var settings.
    """
    result: Dict[str, Any] = {}
    for f in dataclasses.fields(config):
        value = getattr(config, f.name)
        if f.default is not dataclasses.MISSING:
            if value != f.default:
                result[f.name] = value
        elif f.default_factory is not dataclasses.MISSING:
            if value != f.default_factory():
                result[f.name] = value
        else:
            # No default at all — always include
            result[f.name] = value
    return result


def _coerce_value(value: Any) -> Any:
    """Convert enum values to their string representation for the backend."""
    if hasattr(value, "value"):
        return value.value
    return value


# Tracks which init-time keys the bridge has written to rag_config._env_overrides.
# Module-level because _env_overrides is process-wide (backend is a singleton).
_applied_init_keys: Set[str] = set()


def get_init_overrides(config: "KnowledgeConfig") -> Dict[str, Any]:
    """Extract init-time overrides from SDK config.

    Only returns fields that:
    1. Are in INIT_TIME_FIELDS
    2. Differ from the SDK dataclass default

    These must be applied to ``rag_config._env_overrides`` BEFORE
    ``get_rag_backend()`` is called.
    """
    non_defaults = get_non_default_fields(config)
    overrides: Dict[str, Any] = {}
    for sdk_key in INIT_TIME_FIELDS:
        if sdk_key in non_defaults:
            overrides[sdk_key] = _coerce_value(non_defaults[sdk_key])
    return overrides


def apply_init_overrides(env_overrides: Dict[str, Any], config: "KnowledgeConfig") -> None:
    """Apply init-time overrides to env_overrides, cleaning up stale keys.

    This replaces the simple ``env_overrides.update(get_init_overrides(config))``
    pattern.  It also removes any keys that were previously bridged but are
    no longer non-default, preventing leakage across clients.

    Args:
        env_overrides: The ``rag_config._env_overrides`` dict (mutated in place).
        config: SDK ``KnowledgeConfig`` to extract overrides from.
    """
    global _applied_init_keys

    new_overrides = get_init_overrides(config)

    # Remove stale keys from previous bridge application
    stale = _applied_init_keys - set(new_overrides.keys())
    for key in stale:
        env_overrides.pop(key, None)

    # Apply new overrides
    env_overrides.update(new_overrides)

    # Update tracking
    _applied_init_keys = set(new_overrides.keys())


def get_runtime_overrides(config: "KnowledgeConfig") -> Dict[str, Any]:
    """Extract runtime overrides from SDK config.

    Only returns fields that:
    1. Are in RUNTIME_FIELD_MAP, SPECIAL_MAPPINGS, or NESTED_MAPPINGS
    2. Differ from the SDK dataclass default

    These are written via ``SessionConfigProvider.set_config_overrides()``
    for per-session safety.
    """
    non_defaults = get_non_default_fields(config)
    overrides: Dict[str, Any] = {}

    # 1:1 mappings
    for sdk_key, backend_key in RUNTIME_FIELD_MAP.items():
        if sdk_key in non_defaults:
            overrides[backend_key] = _coerce_value(non_defaults[sdk_key])

    # Special key-name mappings
    for sdk_key, backend_key in SPECIAL_MAPPINGS.items():
        if sdk_key in non_defaults:
            overrides[backend_key] = _coerce_value(non_defaults[sdk_key])

    # Nested dict mappings (e.g., chunk_cite_retrieval)
    for nested_key, field_map in NESTED_MAPPINGS.items():
        nested: Dict[str, Any] = {}
        for sdk_key, inner_key in field_map.items():
            if sdk_key in non_defaults:
                nested[inner_key] = _coerce_value(non_defaults[sdk_key])
        if nested:
            overrides[nested_key] = nested

    return overrides
