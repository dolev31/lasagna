"""Knowledge SDK Configuration.

Provides type-safe configuration with comprehensive options for all
RAG system components. Use KnowledgeConfig to customize behavior
or preset functions for common configurations.

Example:
    ```python
    from lasagna.sdk import (
        KnowledgeConfig,
        ChunkingStrategy,
        config_for_quality,
    )

    # Custom configuration
    config = KnowledgeConfig(
        enable_reranking=True,
        enable_hyde=True,
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        chunk_size=800,
    )

    # Or use presets
    config = config_for_quality()  # All features enabled
    config = config_for_speed()    # Minimal processing
    config = config_for_balanced() # Recommended default
    ```
"""

import functools
import os
from dataclasses import dataclass, field, asdict, fields
from typing import Optional, List, Dict, Any, Union
from enum import Enum
from pathlib import Path

from .types import (
    ChunkingStrategy,
    EmbeddingProvider,
    OCREngine,
    HyDEStrategy,
    ChunkRetrievalStrategy,
    EnrichmentStrategy,
    CitationFormat,
    VectorBackend,
)


def _get_default_model() -> str:
    """Resolve default model name from env var or fallback."""
    return os.environ.get("MODEL_NAME", "gpt-4o-mini")


# =============================================================================
# KNOWLEDGE CONFIG
# =============================================================================


@dataclass
class KnowledgeConfig:
    """Comprehensive configuration for the Knowledge SDK.

    All fields have sensible defaults - only specify the ones you want to change.
    Configuration can be applied at client creation or updated dynamically.

    Example:
        ```python
        # Create with custom config
        config = KnowledgeConfig(
            enable_reranking=True,
            enable_hyde=True,
            chunking_strategy=ChunkingStrategy.SEMANTIC,
        )
        client = KnowledgeClient(config=config)

        # Or update configuration dynamically
        await client.set_config(enable_hyde=True)
        ```
    """

    # =========================================================================
    # Storage Paths
    # =========================================================================
    db_path: Optional[str] = None
    """Where the databases live: a directory, or a file path whose directory is used (the databases in it have fixed names). Applied when the client is first used, if the process has not loaded its configuration yet; otherwise a warning says where they are. Default: LASAGNA_DB_DIR or RAG_DB_DIR, else ~/.lasagna/dbs"""

    upload_dir: Optional[str] = None
    """Directory for uploaded files, applied the same way as db_path. Default: LASAGNA_UPLOAD_DIR or RAG_UPLOAD_DIR, else ~/.lasagna/uploads"""

    # =========================================================================
    # Feature Toggles
    # =========================================================================
    enable_hybrid_search: bool = True
    """Enable hybrid search combining vector + BM25 keyword matching. Default: True"""

    enable_reranking: bool = True
    """Enable cross-encoder reranking for better accuracy. Default: True"""

    enable_mmr: bool = True
    """Enable Max Marginal Relevance for result diversity. Default: True"""

    enable_hyde: bool = False
    """Enable HyDE (Hypothetical Document Embeddings) query expansion. Default: False"""

    enable_ocr: bool = False
    """Enable OCR for scanned PDFs and images. Default: False"""

    enable_query_expansion: bool = False
    """Generate alternative query phrasings. Default: False"""

    enable_query_decomposition: bool = False
    """Break complex queries into sub-questions. Default: False"""

    enable_embedding_cache: bool = True
    """Cache computed embeddings for faster repeat queries. Default: True"""

    enable_search_cache: bool = True
    """Cache search results with TTL. Default: True"""

    # =========================================================================
    # Vector Backend
    # =========================================================================
    vector_backend: VectorBackend = VectorBackend.MILVUS
    """Vector database backend to use. Default: milvus"""

    vector_backend_config: Optional[Dict[str, Any]] = None
    """Backend-specific configuration (api_key, host, etc.)."""

    # =========================================================================
    # Embedding Model
    # =========================================================================
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE
    """Embedding model provider. Default: huggingface"""

    embedding_model: str = "BAAI/bge-m3"
    """Embedding model name. Default: BAAI/bge-m3 (1024 dim, dense+sparse+ColBERT)"""

    embedding_model_fallback: Optional[str] = "BAAI/bge-large-en-v1.5"
    """Fallback model if primary fails. Default: BAAI/bge-large-en-v1.5"""

    embedding_dim: int = 1024
    """Embedding vector dimension (must match model). Default: 1024"""

    embedding_truncate_dim: Optional[int] = None
    """Matryoshka truncation dimension. None = use full dim; int = truncate+renorm. Default: None"""

    normalize_embeddings: bool = True
    """Normalize embeddings for cosine similarity. Default: True"""

    use_gpu: bool = True
    """Use GPU for embedding computation if available. Default: True"""

    batch_size: int = 32
    """Batch size for embedding computation. Default: 32"""

    # =========================================================================
    # Chunking
    # =========================================================================
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.SEMANTIC
    """Chunking strategy for document processing. Default: semantic"""

    chunk_size: int = 800
    """Target chunk size in characters. Default: 800"""

    chunk_overlap: int = 0
    """Overlap between consecutive chunks. Default: 0 (research: no measurable benefit, arXiv:2601.14123)"""

    semantic_threshold: float = 0.75
    """Similarity threshold for semantic chunking (0-1). Default: 0.75"""

    min_chunk_size: int = 200
    """Minimum chunk size in characters. Default: 200"""

    max_chunk_size: int = 1500
    """Maximum chunk size in characters. Default: 1500"""

    enable_hierarchical: bool = True
    """Enable hierarchical chunking (parent/child). Default: True"""

    parent_chunk_size: int = 2000
    """Parent chunk size for hierarchical chunking. Default: 2000"""

    child_chunk_size: int = 500
    """Child chunk size for hierarchical chunking. Default: 500"""

    auto_merge_threshold: float = 0.5
    """Ratio of retrieved children / total siblings to trigger parent merge.
    When enough child chunks from the same parent are in results, they are
    replaced with the parent's full content for richer LLM context.
    Range: 0.0-1.0. Lower = more aggressive merging. Default: 0.5"""

    preserve_tables: bool = True
    """Preserve table structure during chunking. Default: True"""

    preserve_headers: bool = True
    """Track section headers for context. Default: True"""

    max_content_length: int = 4000
    """Maximum content length per chunk. Default: 4000"""

    # =========================================================================
    # Search Parameters
    # =========================================================================
    default_top_k: int = 20
    """Number of chunks to retrieve before reranking. Default: 20"""

    rerank_top_k: int = 5
    """Number of chunks to return after reranking (ceiling for adaptive mode). Default: 5"""

    similarity_threshold: float = 0.3
    """Minimum similarity score for results (0-1). Default: 0.3"""

    hybrid_alpha: float = 0.7
    """Weight between vector (1.0) and keyword (0.0) search. Default: 0.7"""

    mmr_score_normalization: str = "sigmoid"
    """MMR score normalization: 'sigmoid', 'minmax', 'sigmoid_minmax', 'none'. Default: sigmoid"""

    rrf_k: int = 60
    """RRF (Reciprocal Rank Fusion) constant. Higher = more weight to lower ranks. Range: 1-200. Default: 60"""

    coverage_scorer_mode: str = "keyword"
    """Coverage scoring for agentic RAG: 'keyword' (fast) or 'llm' (accurate). Default: keyword"""

    # =========================================================================
    # Reranking
    # =========================================================================
    reranker_model: str = "mixedbread-ai/mxbai-rerank-large-v1"
    """Cross-encoder model for reranking. Default: mixedbread-ai/mxbai-rerank-large-v1 (BEIR NDCG@10 = 57.49)"""

    reranker_batch_size: int = 16
    """Batch size for reranking. Default: 16"""

    mmr_lambda: float = 0.5
    """MMR diversity weight (0=diversity, 1=relevance). Default: 0.5 (SOTA standard)"""

    # =========================================================================
    # SOTA Retrieval Features
    # =========================================================================
    enable_context_reordering: bool = True
    """Reorder retrieved chunks to place most relevant near top/bottom (lost-in-the-middle mitigation). Default: True"""

    enable_context_compression: bool = False
    """Compress chunks to reduce noise and token budget after reranking. Default: False"""

    compression_method: str = "extractive"
    """Compression method: 'extractive' (sentence filtering) or 'abstractive' (LLM). Default: extractive"""

    compression_ratio: float = 0.3
    """Target compression ratio (0.3 = keep ~30% of content). Default: 0.3"""

    enable_graph_retrieval: bool = False
    """Enable GraphRAG entity-linked retrieval for multi-hop reasoning. Default: False"""

    graph_max_hops: int = 2
    """Maximum traversal depth in knowledge graph. Default: 2"""

    graph_entity_types: Optional[List[str]] = None
    """Entity types for graph extraction. Default: None (uses built-in list)"""

    graph_extraction_model: str = ""
    """Model for entity extraction in GraphRAG.

    Empty means "use the configured default LLM". Set a provider-specific
    identifier to pin one (e.g. ``"gpt-4o-mini"``, or a gateway-prefixed name
    such as ``"openai/gpt-4o-mini"``). Default: empty.
    """

    custom_reranker: str = ""
    """Registry name for custom reranker (empty = use default). Default: empty"""

    llm_reranker_model: str = ""
    """Model for the LLM listwise reranker.

    Empty means "use the configured default LLM". Default: empty.
    """

    llm_reranker_window_size: int = 20
    """Sliding window size for listwise reranking (RankGPT's). Default: 20"""

    colbert_model: str = "colbert-ir/colbertv2.0"
    """ColBERT late-interaction model name. Default: colbert-ir/colbertv2.0"""

    sparse_method: str = "bm25"
    """Sparse retrieval method: 'bm25' or 'splade'. Default: bm25"""

    enable_contextual_retrieval: bool = False
    """Add LLM-generated context to each chunk before embedding (Anthropic technique). Default: False"""

    enable_rag_fusion: bool = False
    """Enable RAG Fusion: generate multiple query variations, search each, merge with RRF. Default: False"""

    rag_fusion_n_queries: int = 3
    """Number of alternative queries for RAG Fusion. Default: 3"""

    enable_crag: bool = False
    """Enable Corrective RAG: verify retrieval quality and fall back to web search. Default: False"""

    crag_web_search_fallback: bool = False
    """Allow CRAG to fall back to web search when retrieval quality is low. Default: False"""

    docling_enabled: bool = True
    """Enable Docling for advanced document parsing (tables, layouts, OCR). Default: True, as in the shipped settings; without the ``docling`` extra installed, documents go through the built-in extractors"""

    docling_mode: str = "auto"
    """Docling deployment mode: 'auto' (detect GPU), 'local' (CPU-only), 'production' (GPU). Default: auto"""

    docling_pdf_backend: str = "docling_parse"
    """Docling PDF backend: 'docling_parse' (production) or 'pypdfium2' (lightweight). Default: docling_parse"""

    docling_ocr_enabled: bool = True
    """Enable OCR within Docling pipeline for scanned pages. Default: True"""

    docling_table_structure: bool = True
    """Enable TableFormer table structure extraction (97.9% accuracy). Default: True"""

    docling_max_tokens: int = 512
    """Max tokens per chunk for Docling HybridChunker. Default: 512"""

    docling_timeout: int = 300
    """Per-document processing timeout in seconds. Default: 300"""

    embedding_backend: str = "auto"
    """Embedding backend: 'auto' (detect from model name), 'flag-embedding', 'sentence-transformers'. Default: auto"""

    # =========================================================================
    # Query Routing (Adaptive-RAG)
    # =========================================================================
    enable_query_routing: bool = True
    """Enable intelligent query routing to FAST/STANDARD/COMPREHENSIVE tiers. Default: True"""

    routing_strategy: str = "rule_based"
    """Routing strategy: 'rule_based' (deterministic) or 'adaptive' (confidence-aware). Default: rule_based"""

    routing_default_tier: str = "standard"
    """Default tier when routing is disabled: 'fast', 'standard', 'comprehensive'. Default: standard"""

    routing_confidence_threshold: float = 0.5
    """Confidence threshold below which adaptive strategy escalates tier. Default: 0.5"""

    enable_query_classification: bool = True
    """Enable adaptive query-type classification for routing and parameter tuning. Default: True"""

    # =========================================================================
    # Query Enhancement (HyDE)
    # =========================================================================
    hyde_model: str = "default"
    """Model for HyDE generation. Default: uses LLM provider default."""

    hyde_strategy: HyDEStrategy = HyDEStrategy.ADAPTIVE
    """HyDE application strategy. Default: adaptive"""

    hyde_n_docs: int = 3
    """Number of hypothetical documents to generate. Default: 3"""

    max_query_variations: int = 3
    """Number of query expansions to generate. Default: 3"""

    # =========================================================================
    # Agentic RAG
    # =========================================================================
    use_agentic_rag: bool = True
    """Enable agentic multi-hop search. Default: True"""

    max_agent_iterations: int = 3
    """Maximum search iterations for agentic RAG. Default: 3"""

    chunk_refinement_method: str = "auto"
    """Chunk refinement: 'vanilla', 'refined', 'auto'. Default: auto"""

    refinement_model: str = field(default_factory=lambda: _get_default_model())
    """Model for chunk refinement. Uses MODEL_NAME env var or gpt-4o-mini."""

    agent_temperature: float = 0.1
    """Temperature for agent model. Default: 0.1"""

    use_structured_graph: bool = False
    """Use structured StateGraph instead of flat ReAct agent. Default: False"""

    agentic_pipeline: Optional[str] = None
    """Registered agentic pipeline name. Overrides use_structured_graph when set.
    "default" = structured graph, "legacy_react" = ReAct agent, or custom name.
    Default: None (auto-selects based on use_structured_graph flag)."""

    enable_hallucination_check: bool = False
    """Enable post-generation hallucination grounding check (ISSUP). Requires use_structured_graph. Default: False"""

    enable_answer_quality_check: bool = False
    """Enable post-generation answer quality check (ISUSE). Requires use_structured_graph. Default: False"""

    # =========================================================================
    # Chunk Retrieval Strategy
    # =========================================================================
    chunk_retrieval_strategy: ChunkRetrievalStrategy = ChunkRetrievalStrategy.ADAPTIVE
    """Strategy for determining how many chunks to retrieve. Default: adaptive"""

    chunk_retrieval_threshold: float = 0.5
    """Score threshold for ABOVE_THRESHOLD strategy. Default: 0.5"""

    chunk_retrieval_max: int = 20
    """Maximum chunks for adaptive strategy. Default: 20"""

    chunk_retrieval_min: int = 2
    """Minimum chunks for adaptive strategy. Default: 2"""

    score_gap_buffer: int = 2
    """Extra chunks past the score gap as safety margin (adaptive strategy). Default: 2"""

    min_gap_ratio: float = 0.15
    """Minimum relative gap to trigger cutoff in adaptive strategy. Default: 0.15"""

    # =========================================================================
    # Citation Tracking
    # =========================================================================
    enable_citations: bool = True
    """Enable citation tracking and generation. When disabled, search results
    won't include citation IDs and get_context() returns plain text. Default: True"""

    citation_format: CitationFormat = CitationFormat.STABLE_ID
    """Citation format style. Default: stable_id"""

    citation_id_length: int = 8
    """Length of citation hash. Default: 8"""

    verify_citations: bool = True
    """Verify citations exist in database. Default: True"""

    track_citations_db: bool = True
    """Store citations in database. Default: True"""

    show_page_numbers: bool = True
    """Include page numbers in citations. Default: True"""

    citation_db_max_age_hours: int = 168
    """Maximum age of citations in hours (1 week). Default: 168"""

    citation_db_max_per_session: int = 1000
    """Maximum citations per session. Default: 1000"""

    # =========================================================================
    # Document Awareness
    # =========================================================================
    document_awareness_enabled: bool = True
    """Enable document awareness in agent prompts. Default: True"""

    enrichment_strategy: EnrichmentStrategy = EnrichmentStrategy.FIRST_TOKENS
    """Document enrichment strategy. Default: first_tokens"""

    first_tokens_count: int = 150
    """Number of tokens to extract for FIRST_TOKENS strategy. Default: 150"""

    max_documents_in_context: int = 10
    """Maximum documents to include in context. Default: 10"""

    # =========================================================================
    # OCR Settings
    # =========================================================================
    ocr_engine: OCREngine = OCREngine.EASYOCR
    """OCR engine to use. Default: easyocr"""

    ocr_languages: List[str] = field(default_factory=lambda: ["en"])
    """Languages for OCR. Default: ['en']"""

    ocr_confidence_threshold: float = 0.5
    """Minimum OCR confidence (0-1). Default: 0.5"""

    # =========================================================================
    # Caching
    # =========================================================================
    embedding_cache_size: int = 10000
    """Maximum cached embeddings. Default: 10000"""

    search_cache_ttl: int = 300
    """Search cache TTL in seconds. Default: 300 (5 minutes)"""

    search_cache_max_size: int = 1000
    """Maximum search cache entries. Default: 1000"""

    # =========================================================================
    # Logging and Monitoring
    # =========================================================================
    log_level: str = "INFO"
    """Logging level. Default: INFO"""

    enable_prometheus: bool = False
    """Enable Prometheus metrics endpoint. Default: False"""

    enable_tracing: bool = False
    """Enable OpenTelemetry distributed tracing. Default: False"""

    # =========================================================================
    # Infrastructure (Production Mode)
    # =========================================================================
    database_backend: str = "sqlite"
    """Metadata database: 'sqlite' (default) or 'postgresql'. Default: sqlite"""

    database_dsn: Optional[str] = None
    """PostgreSQL connection string. Required when database_backend='postgresql'."""

    cache_backend: str = "memory"
    """Cache backend: 'memory' (default) or 'redis'. Default: memory"""

    redis_url: Optional[str] = None
    """Redis connection URL. Required when cache_backend='redis'."""

    celery_enabled: bool = False
    """Enable Celery background document processing. Default: False"""

    tenant_isolation: str = "none"
    """Tenant isolation: 'none' (default) or 'partition'. Default: none"""

    quota_enabled: bool = False
    """Enable per-tenant resource quotas. Default: False"""

    rate_limiting_enabled: bool = False
    """Enable adaptive rate limiting. Default: False"""

    connection_pool_size: int = 1
    """Vector backend connection pool size. Default: 1 (no pooling)"""

    # =========================================================================
    # File Storage Backend
    # =========================================================================
    storage_backend: str = "local"
    """File storage backend: 'local' (default), 's3', or 'fsspec'.

    - 'local': Files stored on local filesystem (zero-config, built-in)
    - 's3': S3-compatible (AWS S3, MinIO, R2, Backblaze, Wasabi, 15+ services)
    - 'fsspec': Universal (GCS, Azure, FTP, SFTP, HDFS, 30+ backends)
    """

    storage_backend_config: Optional[Dict[str, Any]] = None
    """Backend-specific storage configuration.

    For S3: {"bucket": "my-docs", "prefix": "rag", "endpoint_url": "http://minio:9000"}
    For fsspec: {"protocol": "gcs", "base_path": "my-bucket/rag"}
    """

    database_pool_min: int = 2
    """Minimum PostgreSQL connection pool size. Default: 2"""

    database_pool_max: int = 10
    """Maximum PostgreSQL connection pool size. Default: 10"""

    celery_broker_url: Optional[str] = None
    """Celery broker URL (typically Redis). Default: derived from redis_url."""

    celery_result_backend: Optional[str] = None
    """Celery result backend URL (typically Redis). Default: derived from redis_url."""

    redis_pool_size: int = 10
    """Redis connection pool size. Default: 10"""

    # =========================================================================
    # METHODS
    # =========================================================================

    def __post_init__(self):
        """Validate configuration on creation."""
        from .config_validation import validate_static

        validate_static(self)

    def __setattr__(self, name: str, value: Any) -> None:
        object.__setattr__(self, name, value)
        # After construction, assigning a field is a choice the caller made.
        # A new frozenset rather than an in-place add, so a shallow copy never
        # shares the record with its original.
        explicit = self.__dict__.get("_explicit_fields")
        if explicit is not None and name in _FIELD_NAMES:
            object.__setattr__(self, "_explicit_fields", explicit | {name})

    @property
    def explicit_fields(self) -> frozenset:
        """Names of the fields the caller set, by keyword or by assignment.

        The bridge to the backend sends these even when the value equals the
        dataclass default: ``docling_enabled=False`` is a choice, and the
        shipped settings turn Docling on. ``with_updates`` keeps the record;
        ``dataclasses.replace`` passes every field, so it marks them all set.
        """
        return self.__dict__.get("_explicit_fields", frozenset())

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary.

        Handles enum conversion for serialization.
        All fields are included, including None values.

        Returns:
            Dictionary representation of config
        """
        result = {}
        for key, value in asdict(self).items():
            if isinstance(value, Enum):
                result[key] = value.value
            else:
                result[key] = value
        return result

    def with_updates(self, **kwargs) -> "KnowledgeConfig":
        """Create new config with specified updates.

        Args:
            **kwargs: Configuration options to update

        Returns:
            New KnowledgeConfig with updates applied

        Example:
            ```python
            new_config = config.with_updates(
                enable_hyde=True,
                chunk_size=1000,
            )
            ```
        """
        current = self.to_dict()
        # Handle enum string values
        for key, value in kwargs.items():
            if isinstance(value, str) and hasattr(self, key):
                field_value = getattr(self, key)
                if isinstance(field_value, Enum):
                    # Convert string to enum
                    enum_class = type(field_value)
                    try:
                        value = enum_class(value)
                    except ValueError:
                        pass  # Keep as string if not valid enum
            current[key] = value
        updated = KnowledgeConfig(**current)
        # Every field was passed to rebuild it; only these were chosen.
        object.__setattr__(updated, "_explicit_fields", self.explicit_fields | set(kwargs))
        return updated

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgeConfig":
        """Create config from dictionary.

        Args:
            data: Dictionary with configuration values

        Returns:
            KnowledgeConfig instance
        """
        # Convert string enum values to enums
        enum_fields = {
            "chunking_strategy": ChunkingStrategy,
            "embedding_provider": EmbeddingProvider,
            "ocr_engine": OCREngine,
            "hyde_strategy": HyDEStrategy,
            "chunk_retrieval_strategy": ChunkRetrievalStrategy,
            "enrichment_strategy": EnrichmentStrategy,
            "citation_format": CitationFormat,
            "vector_backend": VectorBackend,
        }

        processed = {}
        for key, value in data.items():
            if key in enum_fields and isinstance(value, str):
                try:
                    processed[key] = enum_fields[key](value)
                except ValueError:
                    processed[key] = value
            else:
                processed[key] = value

        return cls(**processed)


_FIELD_NAMES = frozenset(f.name for f in fields(KnowledgeConfig))


def _recording_set_fields(init):
    """Wrap the generated ``__init__`` to remember which fields the caller passed.

    The value alone cannot say whether a field was chosen: a value equal to the
    default looks the same as no value. Fields assigned inside ``__post_init__``
    are not recorded, since the record starts after the generated init returns.
    """

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        init(self, *args, **kwargs)
        positional = [f.name for f in fields(self) if f.init][: len(args)]
        object.__setattr__(self, "_explicit_fields", frozenset(positional) | frozenset(kwargs))

    return __init__


KnowledgeConfig.__init__ = _recording_set_fields(KnowledgeConfig.__init__)


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================


def config_for_speed() -> KnowledgeConfig:
    """Configuration optimized for speed with minimal processing.

    Disables all expensive pipeline stages: reranking, HyDE, MMR, hybrid
    search, and agentic RAG. Uses smallest embedding model.
    Target latency: ~50-100ms.

    Returns:
        KnowledgeConfig optimized for speed

    Example:
        ```python
        client = KnowledgeClient(config=config_for_speed())
        ```
    """
    return KnowledgeConfig(
        # Disable all expensive features
        enable_reranking=False,
        # The small one, for when reranking is turned on for this preset.
        reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        enable_hyde=False,
        enable_mmr=False,
        enable_hybrid_search=False,
        enable_query_expansion=False,
        enable_query_decomposition=False,
        use_agentic_rag=False,
        # SOTA features off for speed
        docling_enabled=False,
        enable_context_reordering=False,
        enable_contextual_retrieval=False,
        enable_rag_fusion=False,
        enable_crag=False,
        agentic_pipeline="legacy_react",
        # Routing disabled — can't deliver higher tiers without SOTA features
        enable_query_routing=False,
        routing_default_tier="fast",
        enable_query_classification=False,
        # Fastest embedding model
        embedding_model="all-MiniLM-L6-v2",
        embedding_dim=384,
        # Smaller retrieval pool (no reranking to filter)
        default_top_k=20,
        rerank_top_k=5,
        similarity_threshold=0.3,
        # Fixed-k chunk selection, as the speed profile does.
        chunk_retrieval_strategy=ChunkRetrievalStrategy.FIXED,
        chunk_retrieval_max=10,
        # Structure-aware: headers and tables kept, no hierarchical overhead
        chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,
        chunk_overlap=0,
        enable_hierarchical=False,
        parent_chunk_size=2000,
        child_chunk_size=500,
        auto_merge_threshold=0.5,
    )


def config_for_standard() -> KnowledgeConfig:
    """Standard configuration with SOTA features and BGE-M3 embedding model.

    Enables core SOTA features (hybrid search, cross-encoder reranking, MMR,
    agentic RAG, context reordering, Docling parsing) with the BGE-M3 model.
    Recommended starter configuration for production quality.
    Target latency: ~200-400ms.

    Returns:
        KnowledgeConfig with standard settings

    Example:
        ```python
        client = KnowledgeClient(config=config_for_standard())
        ```
    """
    return KnowledgeConfig(
        # Core SOTA features enabled
        enable_reranking=True,
        enable_mmr=True,
        enable_hybrid_search=True,
        use_agentic_rag=True,
        # Expensive features disabled for speed
        enable_hyde=False,
        enable_query_expansion=False,
        enable_query_decomposition=False,
        # SOTA features
        docling_enabled=True,
        enable_context_reordering=True,
        enable_rag_fusion=False,
        enable_crag=False,
        agentic_pipeline="default",
        reranker_model="mixedbread-ai/mxbai-rerank-large-v1",
        # Query routing
        enable_query_routing=True,
        routing_default_tier="standard",
        # BGE-M3 embedding model (multilingual, 8K context)
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        # Semantic chunking — no hierarchical overhead
        chunking_strategy=ChunkingStrategy.STRUCTURE_AWARE,  # what runs, as in the profile
        chunk_size=800,
        chunk_overlap=0,
        enable_hierarchical=False,
        parent_chunk_size=2000,
        child_chunk_size=500,
        auto_merge_threshold=0.5,
        # SOTA search parameters
        default_top_k=20,
        rerank_top_k=5,
        mmr_lambda=0.5,
        mmr_score_normalization="sigmoid",
        hybrid_alpha=0.7,
        similarity_threshold=0.3,
        max_agent_iterations=3,
        chunk_refinement_method="auto",
        chunk_retrieval_max=15,
    )


def config_for_balanced() -> KnowledgeConfig:
    """Balanced configuration with SOTA features and BGE-M3 embedding model.

    Enables core SOTA features (hybrid search, cross-encoder reranking, MMR,
    agentic RAG, context reordering, Docling, hierarchical chunking) with
    the BGE-M3 model. Best quality-to-speed tradeoff for production use.
    Target latency: ~300-600ms.

    Returns:
        KnowledgeConfig with balanced settings

    Example:
        ```python
        client = KnowledgeClient(config=config_for_balanced())
        ```
    """
    return KnowledgeConfig(
        # Core SOTA features enabled
        enable_reranking=True,
        enable_mmr=True,
        enable_hybrid_search=True,
        use_agentic_rag=True,
        use_structured_graph=True,
        enable_hallucination_check=True,
        # Expensive features disabled for speed
        enable_hyde=False,
        enable_query_expansion=False,
        enable_query_decomposition=False,
        # SOTA features
        docling_enabled=True,
        enable_context_reordering=True,
        enable_rag_fusion=False,
        enable_crag=False,
        agentic_pipeline="default",
        reranker_model="mixedbread-ai/mxbai-rerank-large-v1",
        # Query routing — adaptive strategy
        enable_query_routing=True,
        routing_strategy="adaptive",
        routing_confidence_threshold=0.5,
        # BGE-M3 embedding model (multilingual, 8K context)
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        # Semantic chunking with hierarchical auto-merge
        chunking_strategy=ChunkingStrategy.HIERARCHICAL,  # what runs, as in the profile
        chunk_size=800,
        chunk_overlap=0,
        enable_hierarchical=True,
        parent_chunk_size=2000,
        child_chunk_size=500,
        auto_merge_threshold=0.5,
        # SOTA search parameters
        default_top_k=20,
        rerank_top_k=5,
        mmr_lambda=0.5,
        mmr_score_normalization="sigmoid",
        hybrid_alpha=0.7,
        similarity_threshold=0.3,
        max_agent_iterations=3,
        chunk_refinement_method="auto",
        chunk_retrieval_max=20,
    )


def config_for_max_quality() -> KnowledgeConfig:
    """Configuration optimized for maximum retrieval quality.

    Enables all advanced features including HyDE, reranking, query expansion,
    RAG Fusion, CRAG, contextual retrieval, and agentic RAG with more
    iterations. Uses BGE-M3 embedding model for maximum recall.
    Target latency: ~1-3 seconds.

    Returns:
        KnowledgeConfig optimized for maximum quality

    Example:
        ```python
        client = KnowledgeClient(config=config_for_max_quality())
        ```
    """
    return KnowledgeConfig(
        # Enable all quality features
        enable_reranking=True,
        enable_hyde=True,
        enable_mmr=True,
        enable_hybrid_search=True,
        enable_query_expansion=True,
        enable_query_decomposition=False,
        use_agentic_rag=True,
        use_structured_graph=True,
        enable_hallucination_check=True,
        enable_answer_quality_check=True,
        # All SOTA features enabled
        docling_enabled=True,
        enable_context_reordering=True,
        enable_rag_fusion=True,
        enable_crag=True,
        enable_contextual_retrieval=True,
        agentic_pipeline="default",
        reranker_model="mixedbread-ai/mxbai-rerank-large-v1",
        # Routing disabled — always use comprehensive tier
        enable_query_routing=False,
        routing_default_tier="comprehensive",
        routing_confidence_threshold=0.3,
        # BGE-M3 embedding model (multilingual, 8K context)
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        # Semantic chunking with hierarchical auto-merge
        chunking_strategy=ChunkingStrategy.HIERARCHICAL,  # what runs, as in the profile
        chunk_size=800,
        chunk_overlap=0,
        enable_hierarchical=True,
        parent_chunk_size=2000,
        child_chunk_size=500,
        auto_merge_threshold=0.4,  # More aggressive merging for max recall
        # More iterations for complex queries
        max_agent_iterations=5,
        # Adaptive HyDE
        hyde_strategy=HyDEStrategy.ADAPTIVE,
        # SOTA search parameters
        default_top_k=20,
        rerank_top_k=5,
        mmr_lambda=0.5,
        mmr_score_normalization="sigmoid",
        hybrid_alpha=0.7,
        similarity_threshold=0.2,
        # More context for thorough answers
        chunk_retrieval_max=30,
        chunk_refinement_method="auto",
    )


# Backward compatibility alias
config_for_quality = config_for_max_quality


def config_for_standalone(
    db_path: Optional[Union[str, Path]] = None,
    upload_dir: Optional[Union[str, Path]] = None,
) -> KnowledgeConfig:
    """Configuration for standalone usage standalone environment.

    Provides explicit paths for databases and uploads.
    Uses balanced profile settings as the base configuration.

    Args:
        db_path: Path to vector database. Default: ./knowledge_data/vectors.db
        upload_dir: Directory for uploads. Default: ./knowledge_data/uploads

    Returns:
        KnowledgeConfig configured for standalone use

    Example:
        ```python
        config = config_for_standalone(
            db_path="./my_data/vectors.db",
            upload_dir="./my_data/uploads",
        )
        client = KnowledgeClient(config=config)
        ```
    """
    base_dir = Path.cwd() / "knowledge_data"

    return KnowledgeConfig(
        db_path=str(db_path) if db_path else str(base_dir / "vectors.db"),
        upload_dir=str(upload_dir) if upload_dir else str(base_dir / "uploads"),
        # Balanced defaults
        enable_reranking=True,
        enable_hyde=False,
        enable_mmr=True,
        enable_hybrid_search=True,
        use_agentic_rag=True,
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        embedding_model="BAAI/bge-m3",
        reranker_model="mixedbread-ai/mxbai-rerank-large-v1",
        chunk_overlap=0,
        # SOTA search parameters
        default_top_k=20,
        rerank_top_k=5,
        mmr_lambda=0.5,
        similarity_threshold=0.3,
        chunk_retrieval_max=20,
    )


def config_for_production(
    database_dsn: str = "postgresql://localhost:5432/rag",
    redis_url: str = "redis://localhost:6379",
    vector_backend: VectorBackend = VectorBackend.MILVUS,
    vector_backend_config: Optional[Dict[str, Any]] = None,
    storage_backend: str = "local",
    storage_backend_config: Optional[Dict[str, Any]] = None,
    connection_pool_size: int = 10,
) -> KnowledgeConfig:
    """Production-ready configuration for multi-tenant deployment.

    Enables all scaling features: PostgreSQL, Redis cache, Celery background
    processing, partition-based tenant isolation, rate limiting, Prometheus
    metrics, and OpenTelemetry tracing.

    Target: thousands of concurrent users with hard tenant isolation.

    All production infrastructure dependencies (PostgreSQL, Redis, Celery)
    are optional imports — they only need to be installed when this profile
    is used.

    **Important:** This config sets init-time fields (database_backend,
    cache_backend, vector_backend, etc.) that are consumed during backend
    singleton creation.  It must be passed at ``KnowledgeClient`` creation
    time — calling ``set_config()`` after the backend is initialized will
    not apply init-time changes.

    Args:
        database_dsn: PostgreSQL connection string.
        redis_url: Redis connection URL.
        vector_backend: Vector DB backend (milvus, qdrant, pinecone, weaviate).
        vector_backend_config: Backend-specific config (host, api_key, etc.).
        storage_backend: File storage backend ('local', 's3', 'fsspec').
        storage_backend_config: Storage backend config (bucket, endpoint, etc.).
        connection_pool_size: Number of vector backend connections.

    Returns:
        KnowledgeConfig configured for production deployment.

    Examples::

        # Local team with PostgreSQL
        config = config_for_production()

        # SaaS with MinIO object storage
        config = config_for_production(
            storage_backend="s3",
            storage_backend_config={"bucket": "docs", "endpoint_url": "http://minio:9000"},
        )

        # SaaS with GCS via fsspec
        config = config_for_production(
            storage_backend="fsspec",
            storage_backend_config={"protocol": "gcs", "base_path": "my-bucket/rag"},
        )
    """
    return KnowledgeConfig(
        # --- Infrastructure: Production backends ---
        database_backend="postgresql",
        database_dsn=database_dsn,
        database_pool_min=5,
        database_pool_max=20,
        cache_backend="redis",
        redis_url=redis_url,
        redis_pool_size=10,
        celery_enabled=True,
        celery_broker_url=f"{redis_url}/0",
        celery_result_backend=f"{redis_url}/1",
        connection_pool_size=connection_pool_size,
        # --- File Storage ---
        storage_backend=storage_backend,
        storage_backend_config=storage_backend_config,
        # --- Tenant Isolation ---
        tenant_isolation="partition",
        quota_enabled=True,
        rate_limiting_enabled=True,
        # --- Observability ---
        enable_prometheus=True,
        enable_tracing=True,
        # --- Search Quality (balanced profile as base) ---
        enable_reranking=True,
        enable_mmr=True,
        enable_hybrid_search=True,
        use_agentic_rag=True,
        enable_hyde=False,  # Opt-in (requires LLM API key)
        embedding_model="BAAI/bge-m3",
        embedding_dim=1024,
        reranker_model="mixedbread-ai/mxbai-rerank-large-v1",
        chunk_overlap=0,
        chunking_strategy=ChunkingStrategy.SEMANTIC,
        enable_hierarchical=True,
        default_top_k=20,
        rerank_top_k=5,
        # --- Vector Backend ---
        vector_backend=vector_backend,
        vector_backend_config=vector_backend_config,
    )
