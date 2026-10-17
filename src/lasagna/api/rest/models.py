"""Pydantic request/response models for the REST API.

These models define the HTTP contract.  They intentionally mirror (but
are separate from) the core schema models in ``core/schema.py`` to
decouple the public HTTP interface from internal data structures.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=10000, description="Search query")
    top_k: int = Field(default=5, ge=1, le=100, description="Number of results")
    min_score: Optional[float] = Field(
        default=None,
        description="Drop results whose score is below this. Scores are absolute: a "
        "cross-encoder's probability when the profile reranks, else the retrieval score "
        "((cosine similarity + 1) / 2 for dense search). Default: no threshold, as in "
        "the CLI, the MCP server and the Python SDK.",
    )
    document_ids: Optional[List[str]] = Field(
        default=None, description="Filter to specific documents"
    )
    use_hyde: bool = Field(default=True, description="Enable HyDE query expansion")
    use_reranking: bool = Field(default=True, description="Enable cross-encoder reranking")
    routing_tier: Optional[str] = Field(
        default=None,
        pattern="^(fast|standard|comprehensive)$",
        description="Force a specific routing tier (fast, standard, comprehensive)",
    )


class ChunkResponse(BaseModel):
    chunk_id: str
    content: str
    document_id: str
    chunk_index: int
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class SearchResultResponse(BaseModel):
    chunk: ChunkResponse
    score: float = Field(
        description="Absolute relevance: the reranker's score when the profile reranks, "
        "else the retrieval score ((cosine similarity + 1) / 2 for dense search, the "
        "fused score for hybrid search)"
    )
    rank: int = Field(default=0, description="Position in the results, from 1")
    document_id: str
    filename: str


class RoutingInfo(BaseModel):
    """Routing metadata included in search responses when query routing is active."""

    tier: str
    strategy: str
    confidence: float
    rationale: str
    query_type: Optional[str] = None
    estimated_complexity: Optional[str] = None


#: Documented on every search response.
_DEGRADED_STAGES = Field(
    default_factory=list,
    description=(
        "Pipeline stages that failed and fell back to a lesser result during this "
        "search (hyde, graphrag, hybrid, reranker, query_expansion, agentic, routing). Empty "
        "when every configured stage ran."
    ),
)


#: Documented on every search response.
_NOTE = Field(
    default=None,
    description=(
        "Why nothing was searched, when the session has no documents: the sessions "
        "that have some, or how to add one. Absent when the search ran."
    ),
)


class SearchResponse(BaseModel):
    results: List[SearchResultResponse]
    query: str
    total_results: int
    routing: Optional[RoutingInfo] = None
    degraded_stages: List[str] = _DEGRADED_STAGES
    note: Optional[str] = _NOTE


class CitationResponse(BaseModel):
    citation_id: str
    document_id: str
    document_name: str
    chunk_id: str
    chunk_index: int
    page_number: Optional[int] = None
    score: float
    content: str
    highlight: Optional[str] = None
    highlights: Optional[List[str]] = None
    highlight_ranges: Optional[List[List[int]]] = None


class SearchWithCitationsResponse(BaseModel):
    results: List[SearchResultResponse]
    citations: List[CitationResponse]
    context: str
    query: str
    total_results: int
    routing: Optional[RoutingInfo] = None
    degraded_stages: List[str] = _DEGRADED_STAGES
    note: Optional[str] = _NOTE


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


class DocumentResponse(BaseModel):
    document_id: str
    session_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunks_count: int
    upload_date: datetime
    error_message: Optional[str] = None


class DocumentListResponse(BaseModel):
    documents: List[DocumentResponse]
    total: int


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunks_count: int
    #: The session already held this content: the document returned instead.
    duplicate_of: Optional[str] = None


class BatchUploadItemResponse(BaseModel):
    document_id: str
    filename: str
    file_type: str = ""
    file_size: int = 0
    status: str
    chunks_count: int = 0
    error: Optional[str] = None
    duplicate_of: Optional[str] = None


class BatchUploadResponse(BaseModel):
    results: List[BatchUploadItemResponse]
    total: int
    succeeded: int
    failed: int


class DeleteResponse(BaseModel):
    success: bool
    document_id: str
    message: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


def _package_version() -> str:
    from lasagna.sdk._version import __version__

    return __version__


class HealthCheckDetail(BaseModel):
    status: str  # "up" | "down" | "degraded"
    latency_ms: Optional[float] = None
    detail: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # "healthy" | "degraded" | "unhealthy"
    checks: Dict[str, HealthCheckDetail] = Field(default_factory=dict)
    #: The installed package's version; a constant "1.0.0" was reported for 0.1.0a1.
    version: str = Field(default_factory=lambda: _package_version())

    #: Whether this instance treats ``session_id`` as an authorization claim.
    #: There was previously no way to ask, so an operator could not tell a
    #: hardened instance from an open one without attempting the attack. Not
    #: sensitive to disclose: a caller learns it from their first request.
    require_auth: bool = False

    #: The checks that are ``down``; a readiness 503 names them here.
    failing: List[str] = Field(default_factory=list)

    #: The checks that are ``degraded``: working on a fallback (Redis down).
    degraded: List[str] = Field(default_factory=list)


class LivenessResponse(BaseModel):
    status: str = "alive"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


class ConfigResponse(BaseModel):
    """RAG configuration exposed via the API."""

    vector_backend: str = "milvus"
    embedding_model: str
    embedding_dim: int
    chunking_strategy: str
    enable_hybrid_search: bool
    enable_reranking: bool
    enable_mmr: bool
    enable_hyde: bool
    enable_hierarchical: bool
    default_top_k: int
    rerank_top_k: int
    similarity_threshold: float
    hybrid_alpha: float
    mmr_lambda: float
    use_agentic_rag: bool
    enable_prometheus: bool
    enable_tracing: bool
    enable_query_classification: bool = True
    enable_query_routing: bool = True
    routing_strategy: str = "rule_based"
    routing_default_tier: str = "standard"
    routing_confidence_threshold: float = 0.5
    # Retrieval features
    enable_context_reordering: bool = True
    enable_contextual_retrieval: bool = False
    enable_rag_fusion: bool = False
    enable_crag: bool = False
    # Structured graph settings
    use_structured_graph: bool = False
    agentic_pipeline: str = "legacy_react"
    enable_hallucination_check: bool = False
    enable_answer_quality_check: bool = False
    max_quality_check_attempts: int = 2
    coverage_scorer_mode: str = "keyword"
    final_search_mode: str = "merge"
    # Hybrid search internals
    rrf_k: int = 60
    sparse_method: str = "bm25"
    chunk_refinement_method: str = "auto"
    # Context compression
    enable_context_compression: bool = False
    compression_method: str = "extractive"
    compression_ratio: float = 0.3
    # Document awareness
    document_awareness_enabled: bool = True
    enrichment_strategy: str = "first_tokens"
    first_tokens_count: int = 150
    # Warnings (e.g. SPLADE experimental status)
    warnings: list[str] = Field(default_factory=list)
    reindex: Optional[Dict[str, Any]] = Field(
        default=None,
        description="With reindex=true on a profile switch: the documents reindexed first.",
    )
    # Processing / Chunking
    chunk_size: int = 800
    chunk_overlap: int = 0
    semantic_threshold: float = 0.75
    min_chunk_size: int = 200
    max_chunk_size: int = 1500
    parent_chunk_size: int = 2000
    child_chunk_size: int = 500
    auto_merge_threshold: float = 0.5
    preserve_tables: bool = True
    preserve_headers: bool = True
    # OCR
    enable_ocr: bool = False
    ocr_engine: str = "easyocr"
    # Docling
    docling_enabled: bool = True
    docling_mode: str = "auto"
    docling_ocr_enabled: bool = True
    docling_table_structure: bool = True
    docling_max_tokens: int = 512


class ProfileInfo(BaseModel):
    name: str
    description: str


class ProfileListResponse(BaseModel):
    profiles: List[ProfileInfo]
    active: Optional[str] = None


# ---------------------------------------------------------------------------
# Pipelines
# ---------------------------------------------------------------------------


class PipelineInfo(BaseModel):
    """Metadata about a registered agentic pipeline."""

    name: str = Field(description="Registry key for this pipeline")
    description: str = Field(description="Human-readable description")
    class_name: str = Field(description="Implementing class or factory path")


class PipelineListResponse(BaseModel):
    """Response listing all registered agentic pipelines."""

    pipelines: List[PipelineInfo]
    active: str = Field(description="Currently active pipeline name (from config)")
    total: int = Field(description="Total registered pipelines")


# ---------------------------------------------------------------------------
# Admin / Stats
# ---------------------------------------------------------------------------


class StatsResponse(BaseModel):
    embedding_model: Optional[str] = None
    embedding_dim: Optional[int] = None
    collection_name: Optional[str] = None
    vector_backend: Optional[str] = None
    features: Dict[str, bool] = Field(default_factory=dict)
    cache: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def _bounds(key: str) -> Dict[str, Any]:
    """``ge`` and ``le`` for a numeric session setting, from the tables every surface uses.

    The smallest valid value is the settings check's (``_VALID_RANGES``, which
    the environment and the settings file meet), else the configurator
    schema's, which every session write and every arm is validated against;
    the largest is what a session may set (``SESSION_MAXIMUMS``). This model
    had bounds of its own: ``chunk_size`` from 50 where arms and the settings
    file take 1, ``default_top_k`` at most 100 where a session may set 1000.
    """
    from lasagna.configurator.schema import PARAMETERS
    from lasagna.core.config import _VALID_RANGES
    from lasagna.core.session_config._session_keys import SESSION_MAXIMUMS

    if key in _VALID_RANGES:
        low = _VALID_RANGES[key][0]
    else:
        low = next(p.min_value for p in PARAMETERS if p.key == key and p.min_value is not None)

    def plain(number: float) -> Any:
        return int(number) if float(number).is_integer() else number

    return {"ge": plain(low), "le": plain(SESSION_MAXIMUMS[key])}


class ConfigPatchRequest(BaseModel):
    """Partial config update. Only provided keys are applied.

    All fields are Optional: only non-None values are patched.
    Unknown fields are rejected by Pydantic (model_config forbids extras).
    """

    model_config = {"extra": "forbid"}

    vector_backend: Optional[str] = Field(
        default=None,
        pattern="^(milvus|qdrant|pinecone|weaviate|pgvector|opensearch)$",
        description="Vector database backend to use",
    )
    enable_hybrid_search: Optional[bool] = None
    enable_reranking: Optional[bool] = None
    enable_mmr: Optional[bool] = None
    enable_hyde: Optional[bool] = None
    enable_hierarchical: Optional[bool] = None
    use_agentic_rag: Optional[bool] = None
    enable_prometheus: Optional[bool] = None
    enable_tracing: Optional[bool] = None
    default_top_k: Optional[int] = Field(default=None, **_bounds("default_top_k"))
    rerank_top_k: Optional[int] = Field(default=None, **_bounds("rerank_top_k"))
    similarity_threshold: Optional[float] = Field(default=None, **_bounds("similarity_threshold"))
    hybrid_alpha: Optional[float] = Field(default=None, **_bounds("hybrid_alpha"))
    mmr_lambda: Optional[float] = Field(default=None, **_bounds("mmr_lambda"))
    enable_query_classification: Optional[bool] = None
    enable_query_routing: Optional[bool] = None
    routing_strategy: Optional[str] = Field(default=None, pattern="^(rule_based|adaptive)$")
    routing_default_tier: Optional[str] = Field(
        default=None, pattern="^(fast|standard|comprehensive)$"
    )
    routing_confidence_threshold: Optional[float] = Field(
        default=None, **_bounds("routing_confidence_threshold")
    )
    # Retrieval features
    enable_context_reordering: Optional[bool] = None
    enable_contextual_retrieval: Optional[bool] = None
    enable_rag_fusion: Optional[bool] = None
    enable_crag: Optional[bool] = None
    use_structured_graph: Optional[bool] = None
    agentic_pipeline: Optional[str] = Field(default=None, min_length=1, max_length=100)
    enable_hallucination_check: Optional[bool] = None
    enable_answer_quality_check: Optional[bool] = None
    max_quality_check_attempts: Optional[int] = Field(
        default=None, **_bounds("max_quality_check_attempts")
    )
    coverage_scorer_mode: Optional[str] = Field(default=None, pattern="^(keyword|llm)$")
    final_search_mode: Optional[str] = Field(default=None, pattern="^(merge|replace)$")
    # Hybrid search internals
    rrf_k: Optional[int] = Field(default=None, **_bounds("rrf_k"))
    sparse_method: Optional[str] = Field(default=None, pattern="^(bm25|splade)$")
    chunk_refinement_method: Optional[str] = Field(default=None, pattern="^(vanilla|refined|auto)$")
    # Context compression
    enable_context_compression: Optional[bool] = None
    compression_method: Optional[str] = Field(default=None, pattern="^(extractive|abstractive)$")
    compression_ratio: Optional[float] = Field(default=None, **_bounds("compression_ratio"))
    # Document awareness (flat keys — canonicalized to nested in route handler)
    document_awareness_enabled: Optional[bool] = None
    enrichment_strategy: Optional[str] = Field(
        default=None, pattern="^(first_tokens|llm_summary|hybrid)$"
    )
    first_tokens_count: Optional[int] = Field(
        default=None, **_bounds("document_awareness.first_tokens_count")
    )
    # Processing / Chunking
    chunking_strategy: Optional[str] = Field(
        default=None, pattern="^(fixed|semantic|recursive|sentence_window|hierarchical)$"
    )
    chunk_size: Optional[int] = Field(default=None, **_bounds("chunk_size"))
    chunk_overlap: Optional[int] = Field(default=None, **_bounds("chunk_overlap"))
    semantic_threshold: Optional[float] = Field(default=None, **_bounds("semantic_threshold"))
    min_chunk_size: Optional[int] = Field(default=None, **_bounds("min_chunk_size"))
    max_chunk_size: Optional[int] = Field(default=None, **_bounds("max_chunk_size"))
    parent_chunk_size: Optional[int] = Field(default=None, **_bounds("parent_chunk_size"))
    child_chunk_size: Optional[int] = Field(default=None, **_bounds("child_chunk_size"))
    auto_merge_threshold: Optional[float] = Field(default=None, **_bounds("auto_merge_threshold"))
    preserve_tables: Optional[bool] = None
    preserve_headers: Optional[bool] = None
    # OCR
    enable_ocr: Optional[bool] = None
    ocr_engine: Optional[str] = Field(default=None, pattern="^(easyocr|tesseract|paddleocr)$")
    # Docling
    docling_enabled: Optional[bool] = None
    docling_mode: Optional[str] = Field(default=None, pattern="^(auto|local|production)$")
    docling_ocr_enabled: Optional[bool] = None
    docling_table_structure: Optional[bool] = None
    docling_max_tokens: Optional[int] = Field(default=None, **_bounds("docling_max_tokens"))


# ---------------------------------------------------------------------------
# Configurator (knowledge config management)
# ---------------------------------------------------------------------------


class ProfileDetail(BaseModel):
    name: str
    description: str
    config: Dict[str, Any] = Field(default_factory=dict)


class KnowledgeConfigFullResponse(BaseModel):
    """Full config state returned by the configurator API."""

    schema_version: int
    defaults_source: str
    profiles: List[ProfileDetail]
    active_profile: str
    session_overrides: Dict[str, Any] = Field(default_factory=dict)
    rag_enabled: bool
    reindex: Optional[Dict[str, Any]] = Field(
        default=None,
        description="With reindex=true: the documents reindexed before the switch.",
    )


class SetActiveProfileRequest(BaseModel):
    active_profile: str = Field(..., min_length=1, max_length=100)
    config: Dict[str, Any] = Field(default_factory=dict)
    force: bool = Field(
        default=False,
        description="Switch even when the session's documents would not be searched; "
        "reindex them afterwards (POST /api/knowledge/reindex).",
    )
    reindex: bool = Field(
        default=False,
        description="Reindex the session's documents with the new profile first, and switch "
        "only when every one is reindexed; if one fails, the session keeps its profile.",
    )
    reset: bool = Field(
        default=False,
        description="Drop the session's own settings (its overrides), so the profile's apply "
        "as they are.",
    )


class SaveProfileRequest(BaseModel):
    profile_name: str = Field(..., min_length=1, max_length=100)
    config: Dict[str, Any] = Field(default_factory=dict)
    set_active: bool = True


class DeleteProfileRequest(BaseModel):
    profile_name: str = Field(..., min_length=1, max_length=100)


class RenameProfileRequest(BaseModel):
    old_name: str = Field(..., min_length=1, max_length=100)
    new_name: str = Field(..., min_length=1, max_length=100)


class ResetConfigRequest(BaseModel):
    scope: str = Field(default="active", pattern="^(active|all)$")


class ReindexRequest(BaseModel):
    session_id: str = Field(
        default="",
        max_length=256,
        description="Deprecated: session is now extracted from auth headers. "
        "This field will be removed in a future version.",
        json_schema_extra={"deprecated": True},
    )

    @field_validator("session_id")
    @classmethod
    def warn_deprecated_session_id(cls, v: str) -> str:
        if v:
            import warnings

            warnings.warn(
                "ReindexRequest.session_id is deprecated and ignored. "
                "Session is now extracted from auth headers.",
                DeprecationWarning,
                stacklevel=2,
            )
        return v


class ReindexResponse(BaseModel):
    status: str = Field(description="'completed' or 'partial'")
    documents_reindexed: int
    message: Optional[str] = None
    errors: Optional[List[str]] = None


def _check_depth(obj: Any, max_depth: int, current: int = 0) -> None:
    """Raise ValueError if a nested dict/list exceeds max_depth."""
    if current > max_depth:
        raise ValueError(f"Nested structure exceeds maximum depth of {max_depth}")
    if isinstance(obj, dict):
        for v in obj.values():
            _check_depth(v, max_depth, current + 1)
    elif isinstance(obj, list):
        for item in obj:
            _check_depth(item, max_depth, current + 1)


class ImportConfigRequest(BaseModel):
    """Validated request body for configuration import."""

    model_config = {"extra": "forbid"}

    schema_version: Optional[int] = Field(default=None, ge=1, le=100)
    profiles: Optional[List[Dict[str, Any]]] = Field(default=None, max_length=50)
    active_profile: Optional[str] = Field(default=None, min_length=1, max_length=100)
    session_overrides: Optional[Dict[str, Any]] = Field(default=None)
    rag_enabled: Optional[bool] = None

    @field_validator("profiles")
    @classmethod
    def validate_profile_size(
        cls, v: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if v is None:
            return v
        import json as _json

        _MAX_PROFILE_BYTES = 10_000
        _MAX_DEPTH = 5
        for i, profile in enumerate(v):
            serialized = _json.dumps(profile)
            if len(serialized) > _MAX_PROFILE_BYTES:
                raise ValueError(
                    f"Profile at index {i} exceeds maximum size "
                    f"({len(serialized)} > {_MAX_PROFILE_BYTES} bytes)"
                )
            _check_depth(profile, _MAX_DEPTH)
        return v

    @field_validator("session_overrides")
    @classmethod
    def validate_overrides_size(cls, v: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if v is None:
            return v
        import json as _json

        serialized = _json.dumps(v)
        if len(serialized) > 10_000:
            raise ValueError(f"session_overrides exceeds maximum size ({len(serialized)} bytes)")
        _check_depth(v, max_depth=5)
        return v


class CitationListResponse(BaseModel):
    citations: List[CitationResponse]
    session_id: str
    total: int


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str  # "processing" | "ready" | "error"
    chunks_count: int
    error_message: Optional[str] = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorResponse(BaseModel):
    detail: str
    error_code: Optional[str] = None
