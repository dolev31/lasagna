"""Search orchestration pipeline for RAG retrieval.

Implements the full SOTA search pipeline: query classification, HyDE,
dense search, hybrid fusion (BM25), cross-encoder reranking, MMR diversity,
auto-merge, caching, metrics, and citation extraction.
"""

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from loguru import logger

from lasagna.core.config import rag_config, RAGConfig
from lasagna.core.schema import (
    Document,
    DocumentStatus,
    RecordedDocumentChunk,
    SearchResult,
)
from ..vector_backends import VectorStoreAdapter, FilterExpression
from ..vector_backends.base import SearchParams, MetricType
from ..infrastructure.embedding import EmbeddingManager
from ..infrastructure.cache import get_search_cache
from ..retrieval.search.hybrid import HybridSearcher
from ..retrieval.search.ranker import CombinedReranker, RerankedResult
from ..retrieval.enhancement.enhancer import QueryEnhancer
from ..retrieval.routing import get_query_router, RoutingTier
from ..core.strict import strict_stages
from lasagna.agentic.citation_manager import CitationManager, Citation
from lasagna.utils.monitoring import METRICS, log_json, log_json_debug, hash_text
from lasagna.utils.tracing import TRACER
from lasagna.sdk.config_validation import validate_runtime

from ._validation import is_valid_session_id, is_safe_document_id
from . import _degraded
from ._auto_merger import AutoMerger
from lasagna.core.tenant import get_tenant_manager
from lasagna.core.quota import get_quota_manager

# Thread-local storage for routing decisions so concurrent requests
# don't overwrite each other's metadata.
_routing_local = threading.local()


class SearchScopeUnavailable(RuntimeError):
    """The documents a search is confined to could not be determined.

    Raised instead of searching the whole session; REST answers 503.
    """


#: The arm's components while an agentic pipeline runs. The pipeline issues its
#: own searches (one per hop) and passes only settings, so without this every
#: hop searched with the profile's reranker, enhancer and retriever, and a
#: reranker x agentic comparison measured the agentic loop only.
_INNER_SEARCH_COMPONENTS: contextvars.ContextVar[Dict[str, Any]] = contextvars.ContextVar(
    "lasagna_inner_search_components", default={}
)


def _component_identity(obj: Any) -> str:
    """Return a stable identity string for a component instance.

    Checks common model-name attributes first; falls back to the
    fully-qualified class name so that two distinct classes always
    produce different fingerprints even when neither exposes a name.
    """
    for attr in ("model_name_or_path", "model_name", "name"):
        val = getattr(obj, attr, None)
        if val:
            return str(val)
    cls = type(obj)
    return f"{cls.__module__}.{cls.__qualname__}"


if TYPE_CHECKING:
    from ._collection_manager import CollectionManager
    from ._document_pipeline import DocumentPipeline


class _RawHit:
    """The shape the orchestrator reads off a backend result.

    A ``BaseRetriever`` returns ``ScoredDocument``; everything downstream of the
    dense-search call expects ``.id``/``.score``/``.distance``/``.entity``.
    Adapting here keeps the public contract as small as it looks rather than
    asking component authors to mimic a backend row.
    """

    __slots__ = ("id", "score", "distance", "entity")

    def __init__(self, doc):
        self.id = getattr(doc, "chunk_id", "") or ""
        self.score = float(getattr(doc, "score", 0.0) or 0.0)
        # Distance is the backend's raw metric; a component reports similarity
        # only, so mirror the score rather than invent a distance it never
        # computed.
        self.distance = self.score
        self.entity = {
            "document_id": getattr(doc, "document_id", "") or "",
            "content": getattr(doc, "content", "") or "",
            "metadata": dict(getattr(doc, "metadata", {}) or {}),
            "embedding": None,
        }


def _is_public_enhancer(enhancer: Any) -> bool:
    """A query enhancer written against the public ``BaseQueryEnhancer`` contract."""
    from ..core.base_components import BaseQueryEnhancer

    return isinstance(enhancer, BaseQueryEnhancer) and not hasattr(
        enhancer, "generate_query_variations"
    )


def _query_variations(enhancer: Any, query: str, n: int, config: Any = None) -> List[str]:
    """Up to ``n`` reformulations of ``query`` from either enhancer contract.

    The orchestrator only knew the internal ``generate_query_variations``, so
    a component implementing the documented ``enhance(query)`` was never
    called: with expansion off it was skipped, and with expansion on it raised
    AttributeError and the arm scored as plain search.
    """
    if hasattr(enhancer, "generate_query_variations"):
        return list(enhancer.generate_query_variations(query, n))
    if not hasattr(enhancer, "enhance"):
        raise TypeError(
            f"{type(enhancer).__name__} has neither enhance() nor generate_query_variations()"
        )
    if getattr(enhancer, "_forwards_session_config", False):
        # Built-in adapters read the arm's settings (hyde_model, how many
        # variations); a user's enhancer is shown only the public contract.
        out = enhancer.enhance(query, config=config, n=n)
    else:
        out = enhancer.enhance(query)
    if asyncio.iscoroutine(out):
        out.close()
        raise TypeError(
            f"{type(enhancer).__name__}.enhance() is async; the search pipeline "
            "calls it synchronously, as BaseQueryEnhancer declares"
        )
    seen = {query.strip()}
    variations = []
    for v in out or []:
        if isinstance(v, str) and v.strip() and v.strip() not in seen:
            seen.add(v.strip())
            variations.append(v)
    return variations[:n]


def _adapt_retriever_results(documents) -> list:
    """Convert a ``BaseRetriever``'s output into backend-row shape."""
    from ..core.base_components import ScoredDocument

    adapted = []
    for i, doc in enumerate(documents or []):
        if not isinstance(doc, ScoredDocument):
            raise TypeError(
                f"Retriever returned {type(doc).__name__} at index {i}; a "
                f"BaseRetriever must return ScoredDocument. Run "
                f"`lasagna component validate <module:Class>` to check."
            )
        adapted.append(_RawHit(doc))
    return adapted


@dataclass(frozen=True)
class RetrievalContext:
    """The store a search reads, for a built-in retriever to read it the same way.

    Passed as ``context=`` (beside ``config=``) only to a retriever that sets
    ``_forwards_session_config``. A user's retriever keeps the public contract,
    ``retrieve(query, collection, top_k)``, and never sees it.
    """

    backend: Any
    collection: str
    session_id: str
    #: Session and document filter, as the default vector search applies it.
    filter_expr: Any = None
    #: The session alone, for reading the session's whole corpus.
    session_filter: Any = None
    document_ids: Optional[List[str]] = None
    partition_key: Optional[str] = None
    #: The vector the default search would have used (HyDE's, when it ran).
    query_embedding: Any = None
    embedding_manager: Any = None
    hybrid_searcher: Any = None
    search_params: Any = None


class SearchOrchestrator:
    """Orchestrates the full RAG search pipeline.

    Handles:
    - Query classification for adaptive retrieval
    - HyDE (Hypothetical Document Embeddings) expansion
    - Dense vector search with backend-agnostic filters
    - Hybrid search (dense + BM25) fusion
    - Cross-encoder reranking with MMR diversity
    - Hierarchical auto-merge of child chunks
    - Search result caching
    - Multi-query expansion with RRF fusion
    - Citation extraction and formatting
    - Detailed metrics and observability
    """

    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        hybrid_searcher: HybridSearcher,
        reranker: CombinedReranker,
        query_enhancer: QueryEnhancer,
        citation_manager: CitationManager,
        collection_manager: "CollectionManager",
        document_pipeline: "DocumentPipeline",
        auto_merger: AutoMerger,
        default_backend: VectorStoreAdapter,
    ):
        self.embedding_manager = embedding_manager
        self.hybrid_searcher = hybrid_searcher
        self.reranker = reranker
        self.query_enhancer = query_enhancer
        self.citation_manager = citation_manager
        self._collection_manager = collection_manager
        self._document_pipeline = document_pipeline
        self._auto_merger = auto_merger
        self._default_backend = default_backend

    # =========================================================================
    # Primary Search
    # =========================================================================

    def search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        min_score: float = -100.0,
        document_ids: Optional[List[str]] = None,
        use_hyde: bool = True,
        use_reranking: bool = True,
        trace_id: Optional[str] = None,
        routing_tier: Optional[str] = None,
        _from_fusion: bool = False,
        _from_routing: bool = False,
        _reranker_override: Optional[Any] = None,
        _retriever_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
        _embedder_override: Optional[Any] = None,
        _agentic_pipeline_override: Optional[Any] = None,
        _graph_node_overrides: Optional[Dict[str, Any]] = None,
        query_config_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search with SOTA retrieval pipeline.

        Args:
            query: Search query
            session_id: Session ID
            top_k: Final number of results
            min_score: Minimum score threshold (default -100 to allow cross-encoder logits)
            document_ids: Optional document filter. If focused_document_ids are set
                in the session config, they will override this parameter.
            use_hyde: Whether to use HyDE
            use_reranking: Whether to use reranking
            trace_id: Trace ID for observability
            routing_tier: Optional forced routing tier ("fast", "standard", "comprehensive").
            _from_fusion: Internal flag to prevent recursion from RAG Fusion.
                Do not set this manually; it is used by search_multi_query().
            _from_routing: Internal flag to prevent routing recursion from
                COMPREHENSIVE tier (AgenticRAG) and RAG Fusion sub-searches.
            _reranker_override: Optional reranker instance to use instead of self.reranker.
                Used by KnowledgePipeline to inject custom components.
            _enhancer_override: Optional query enhancer instance to use instead of
                self.query_enhancer. Used by KnowledgePipeline to inject custom components.
            _embedder_override: Optional embedder instance to use instead of
                self.embedding_manager for query encoding. Used by KnowledgePipeline
                to inject custom embedders during ablation studies.
            _agentic_pipeline_override: A ``BaseAgenticPipeline`` instance
                to use instead of the config-resolved pipeline for COMPREHENSIVE tier.
            _graph_node_overrides: Dict mapping graph node names to callables,
                passed to the built-in structured graph pipeline.

        Returns:
            List of search results
        """
        if _from_routing:
            # A search an agentic pipeline issued: the arm's components, unless
            # the pipeline named its own.
            inner = _INNER_SEARCH_COMPONENTS.get()
            _reranker_override = _reranker_override or inner.get("_reranker_override")
            _enhancer_override = _enhancer_override or inner.get("_enhancer_override")
            _retriever_override = _retriever_override or inner.get("_retriever_override")
            _embedder_override = _embedder_override or inner.get("_embedder_override")

        # Resolve effective components (overrides take priority)
        from ..retrieval.search.ranker import adapt_reranker_for_pipeline

        # A component written against the public BaseReranker contract cannot
        # be called with the pipeline's internal signature. Adapt here, where
        # both the configured reranker and a per-query override converge.
        reranker = adapt_reranker_for_pipeline(_reranker_override or self.reranker)
        enhancer = _enhancer_override or self.query_enhancer
        # Get session-aware config to support UI-configured overrides
        config = self._get_session_config(session_id)
        # Apply per-query ephemeral overrides (eval SDK / ablation studies).
        # These take highest priority and are never written to the session store.
        if query_config_overrides:
            config = config.with_query_overrides(query_config_overrides)

        # === QUERY ROUTING ===
        # Route the query BEFORE RAG Fusion and cache to select the optimal
        # retrieval tier. Internal calls (_from_routing, _from_fusion) skip
        # routing to prevent recursion.
        routing_decision = None
        _early_classification = None
        if not _from_routing and not _from_fusion:
            # Classify query early so routing and downstream can both reuse it
            if config.enable_query_classification:
                try:
                    from ..retrieval.enhancement import classify_query

                    _early_classification = classify_query(query)
                except Exception as e:
                    logger.warning(f"Early query classification for routing failed: {e}")

            # CRAG requires the agentic (comprehensive) pipeline — force the
            # tier so that enable_crag=True is not silently ignored when the
            # router would otherwise select STANDARD tier.
            if getattr(config, "enable_crag", False) and not routing_tier:
                routing_tier = "comprehensive"

            try:
                router = get_query_router()
                routing_start = perf_counter()
                with TRACER.span(
                    "query_routing", attributes={"forced_tier": routing_tier or "auto"}
                ):
                    routing_decision = router.route(
                        query=query,
                        document_ids=document_ids,
                        classification=_early_classification,
                        config=config,
                        forced_tier=routing_tier,
                    )
                routing_ms = (perf_counter() - routing_start) * 1000.0
                _routing_local.last_decision = routing_decision

                if getattr(config, "routing_log_decisions", True):
                    log_json(
                        "rag_routing_decision",
                        {
                            **routing_decision.to_dict(),
                            "session_id": session_id,
                            "decision_ms": round(routing_ms, 3),
                        },
                    )
                METRICS.counter(
                    "rag.routing.decisions", 1, tags={"tier": routing_decision.tier.value}
                )
                METRICS.histogram("rag.routing.decision_ms", routing_ms)

                # COMPREHENSIVE → early-return via agentic retrieval
                if routing_decision.tier == RoutingTier.COMPREHENSIVE:
                    return self._search_comprehensive(
                        query=query,
                        session_id=session_id,
                        top_k=top_k,
                        min_score=min_score,
                        document_ids=document_ids,
                        trace_id=trace_id,
                        routing_decision=routing_decision,
                        _reranker_override=_reranker_override,
                        _retriever_override=_retriever_override,
                        _enhancer_override=_enhancer_override,
                        _embedder_override=_embedder_override,
                        _agentic_pipeline_override=_agentic_pipeline_override,
                        _graph_node_overrides=_graph_node_overrides,
                        query_config_overrides=query_config_overrides,
                    )

                # FAST → override heavyweight features
                if routing_decision.tier == RoutingTier.FAST:
                    overrides = routing_decision.overrides
                    use_hyde = overrides.get("use_hyde", use_hyde)
                    use_reranking = overrides.get("use_reranking", use_reranking)
            except Exception as e:
                logger.warning(f"Query routing failed, using standard pipeline: {e}")
                routing_decision = None

        # Track effective values for features that may be overridden by FAST tier.
        # These local vars shadow config properties so the pipeline uses the
        # routing-adjusted values without mutating session config.
        _effective_hybrid_search = config.enable_hybrid_search
        _effective_mmr = config.enable_mmr
        _effective_context_reordering = config.enable_context_reordering
        _effective_hierarchical = config.enable_hierarchical
        if routing_decision and routing_decision.tier == RoutingTier.FAST:
            overrides = routing_decision.overrides
            _effective_hybrid_search = overrides.get(
                "enable_hybrid_search", _effective_hybrid_search
            )
            _effective_mmr = overrides.get("enable_mmr", _effective_mmr)
            _effective_context_reordering = overrides.get(
                "enable_context_reordering", _effective_context_reordering
            )
            _effective_hierarchical = overrides.get("enable_hierarchical", _effective_hierarchical)

        # RAG Fusion / Query Expansion: delegate to multi-query search with RRF
        # when either enable_rag_fusion OR enable_query_expansion is set.
        # Skip for focused/document-specific searches and to prevent recursion.
        _enable_fusion = getattr(config, "enable_rag_fusion", False)
        _enable_expansion = getattr(config, "enable_query_expansion", False)
        # A public-contract enhancer has no other way to run: naming it means
        # using it, whatever the expansion flags say.
        if _is_public_enhancer(enhancer):
            _enable_expansion = True
        if not _from_fusion and (_enable_fusion or _enable_expansion) and not document_ids:
            if _enable_fusion:
                n_queries = getattr(config, "rag_fusion_n_queries", 3)
            else:
                n_queries = getattr(config, "max_query_variations", 3)
            logger.debug(
                f"{'RAG Fusion' if _enable_fusion else 'Query Expansion'} enabled: "
                f"delegating to search_multi_query (n_variations={n_queries})"
            )
            return self.search_multi_query(
                query=query,
                session_id=session_id,
                top_k=top_k,
                document_ids=document_ids,
                n_variations=n_queries,
                trace_id=trace_id,
                _reranker_override=_reranker_override,
                _enhancer_override=_enhancer_override,
                _embedder_override=_embedder_override,
                _from_routing=True,
                query_config_overrides=query_config_overrides,
            )

        # Quota enforcement — early reject if rate limit exceeded
        quota_mgr = get_quota_manager()
        quota_mgr.check_search(session_id)

        # Cache setup — actual lookup happens after query classification
        # so that cache keys use effective (post-classification) values.
        search_cache = None
        _cache_hit = False
        if config.enable_search_cache:
            search_cache = get_search_cache(config)

        # ENFORCEMENT: Check for focused_document_ids in session config
        # If set, they override any document_ids parameter passed to this method
        effective_document_ids = document_ids
        using_focused_docs = False
        focused_doc_count = 0
        try:
            from ..core.session_config import get_knowledge_config_manager

            config_manager = get_knowledge_config_manager()
            focused_ids = config_manager.get_focused_document_ids(session_id)
            if focused_ids:
                logger.debug(
                    f"ENFORCEMENT: Using focused_document_ids {focused_ids} "
                    f"(overriding document_ids={document_ids})"
                )
                effective_document_ids = focused_ids
                using_focused_docs = True
                focused_doc_count = len(focused_ids)
        except Exception as e:
            # Searching on would search every document in the session while the
            # caller asked for a subset: a silent change of scope, answered 200.
            raise SearchScopeUnavailable(
                f"The session's focused documents could not be read ({type(e).__name__}: "
                f"{e}), so the search was not run: it would have searched every "
                f"document in the session instead of the focused ones. Retry, or clear "
                f"the focus (MCP tool set_focused_documents with an empty list)."
            ) from e

        # Use the effective document IDs for filtering
        document_ids = effective_document_ids
        self._maybe_reload_embedding_manager(config)

        # Resolve effective embedder AFTER reload (which may change self.embedding_manager)
        effective_embedder = _embedder_override or self.embedding_manager
        from ..retrieval.search.ranker import bind_embedding_manager

        bind_embedding_manager(reranker, effective_embedder)

        # Compute component fingerprint for cache key isolation.
        # This ensures that swapping components (e.g., during ablation studies)
        # produces different cache keys, preventing stale cached results.
        component_fingerprint = "|".join(
            [
                f"rr:{_component_identity(reranker)}",
                # The retriever too: a retriever arm was served the cached
                # results of an arm that searched the default way.
                f"rt:{_component_identity(_retriever_override) if _retriever_override else ''}",
                f"qe:{_component_identity(enhancer)}",
                f"em:{_component_identity(effective_embedder)}",
                # Every other setting (reranker model, thresholds, depth...):
                # the cache keyed on a dozen fields, so one sweep arm could be
                # served another arm's results.
                f"cfg:{config.settings_fingerprint()}",
            ]
        )

        total_start = perf_counter()
        query_len = len(query) if query else 0
        query_hash = hash_text(query)
        hyde_ms = 0.0
        if top_k is None:
            top_k = config.rerank_top_k

        # Default min_score from config.similarity_threshold when caller
        # didn't provide an explicit value (-100.0 is the sentinel default).
        if min_score <= -100.0:
            _cfg_threshold = getattr(config, "similarity_threshold", None)
            if _cfg_threshold is not None and _cfg_threshold > 0:
                min_score = _cfg_threshold

        # Determine query for embedding
        search_query = query

        # Query classification for adaptive retrieval (P0-3: SOTA feature)
        # Reuse the early classification from routing if available,
        # avoiding a redundant classify_query() call.
        query_classification = _early_classification
        effective_alpha = config.hybrid_alpha
        effective_use_hyde = use_hyde and config.enable_hyde

        if config.enable_query_classification and query_classification is None:
            try:
                from ..retrieval.enhancement import classify_query

                with TRACER.span("query_classification") as _cls_span:
                    query_classification = classify_query(query)
                    _cls_span.set_attribute("query_type", query_classification.query_type.value)
                    _cls_span.set_attribute("confidence", query_classification.confidence)

                logger.debug(
                    f"Query classified as {query_classification.query_type.value} "
                    f"(confidence={query_classification.confidence:.2f}, "
                    f"recommended_alpha={query_classification.recommended_alpha})"
                )

                # Apply recommended alpha — but respect explicit user overrides.
                # query_config_overrides contains per-query settings from experiment
                # configs / API callers; if the user explicitly set hybrid_alpha,
                # the classifier must NOT overwrite it.
                _user_overrides = query_config_overrides or {}
                if not document_ids and "hybrid_alpha" not in _user_overrides:
                    effective_alpha = query_classification.recommended_alpha

                # Apply HyDE recommendation
                if use_hyde:  # Respect explicit disable
                    effective_use_hyde = query_classification.use_hyde
                    # Update hyde_n_docs based on strategy
                    if query_classification.hyde_strategy == "off":
                        effective_use_hyde = False
                    elif query_classification.hyde_strategy == "single":
                        # Will be handled in HyDE section
                        pass
            except Exception as e:
                logger.warning(f"Query classification failed: {e}")

        # Track effective hyde_n_docs for cache key consistency
        effective_hyde_n_docs = config.hyde_n_docs
        if (
            effective_use_hyde
            and query_classification
            and query_classification.hyde_strategy == "single"
        ):
            effective_hyde_n_docs = 1

        # Runtime validation: log when classification overrides user settings
        validate_runtime(
            {
                "query_classification": query_classification,
                "config_hybrid_alpha": config.hybrid_alpha,
                "effective_alpha": effective_alpha,
                "config_enable_hyde": use_hyde and config.enable_hyde,
                "effective_use_hyde": effective_use_hyde,
            }
        )

        # P0-1 FIX: Search cache check — runs AFTER classification so cache
        # keys use effective (post-classification) values consistently.
        if search_cache is not None:
            try:
                cached = search_cache.get(
                    query=query,
                    session_id=session_id,
                    document_ids=document_ids,
                    top_k=top_k,
                    use_hyde=effective_use_hyde,
                    use_reranking=use_reranking and config.enable_reranking,
                    hybrid_alpha=effective_alpha,
                    mmr_lambda=config.mmr_lambda,
                    hyde_n_docs=effective_hyde_n_docs if effective_use_hyde else 0,
                    enable_hybrid_search=_effective_hybrid_search,
                    enable_mmr=_effective_mmr,
                    component_fingerprint=component_fingerprint,
                )
                if cached:
                    cached_results, cached_embedding = cached
                    _cache_hit = True
                    logger.debug(f"Search cache HIT for query (len={len(query)})")
                    METRICS.counter("rag.search.cache_hits", 1)
                    # Reconstruct SearchResult objects from cached data
                    search_results = []
                    for cached_result in cached_results:
                        chunk = RecordedDocumentChunk(
                            chunk_id=cached_result["chunk_id"],
                            content=cached_result["content"],
                            document_id=cached_result["document_id"],
                            chunk_index=cached_result.get("chunk_index", 0),
                            page_number=cached_result.get("page_number"),
                            metadata=cached_result.get("metadata", {}),
                            created_at=datetime.now(timezone.utc),
                        )
                        doc = Document(
                            document_id=cached_result["document_id"],
                            session_id=session_id,
                            filename=cached_result.get("filename", ""),
                            file_size=cached_result.get("file_size", 0),
                            file_type=cached_result.get("file_type", ""),
                            status=DocumentStatus.READY,
                            chunks_count=cached_result.get("chunks_count", 0),
                            upload_date=datetime.now(timezone.utc),
                        )
                        search_results.append(
                            SearchResult(chunk=chunk, score=cached_result["score"], document=doc)
                        )
                    log_json(
                        "rag_search_metrics",
                        {
                            "session_id": session_id,
                            "query_hash": hash_text(query),
                            "query_len": len(query) if query else 0,
                            "results_count_returned": len(search_results),
                            "cache_hit": True,
                            "total_ms": 0.1,  # Near instant
                        },
                    )
                    return search_results
            except Exception as e:
                logger.warning(f"Search cache lookup failed: {e}")
                METRICS.counter("rag.search.cache_errors", 1)

        if search_cache is not None:
            METRICS.counter("rag.search.cache_misses", 1)

        # Apply HyDE if enabled (respects query classification recommendations)
        # Note: Using sync call here since search() is synchronous.
        hyde_embedding = None  # For multi-doc HyDE
        if effective_use_hyde:
            # Determine number of docs - query classification can override to single
            hyde_n_docs = config.hyde_n_docs
            if query_classification and query_classification.hyde_strategy == "single":
                hyde_n_docs = 1  # Force single-doc HyDE based on classification

            logger.debug(
                "HyDE enabled for search "
                f"(model={config.hyde_model}, n_docs={hyde_n_docs}, query_len={query_len})"
            )
            hyde_start = perf_counter()
            # HyDE is its own stage: a swapped-in public enhancer only
            # replaces query reformulation, so HyDE stays with the built-in.
            hyde_enhancer = (
                enhancer if hasattr(enhancer, "get_hyde_embedding") else self.query_enhancer
            )
            try:
                with TRACER.span(
                    "hyde", attributes={"n_docs": hyde_n_docs, "model": config.hyde_model}
                ) as _hyde_span:
                    if hyde_n_docs > 1:
                        # Multi-document HyDE: generate multiple hypothetical docs and average embeddings
                        hyde_embedding = hyde_enhancer.get_hyde_embedding(
                            query,
                            embedding_manager=effective_embedder,
                            n_docs=hyde_n_docs,
                            model_ref=config.hyde_model,
                            trace_id=trace_id,
                            strategy=config.hyde_embedding_strategy,
                        )
                        if hyde_embedding is not None:
                            logger.debug(f"Using multi-HyDE embedding from {hyde_n_docs} docs")
                        else:
                            logger.debug("Multi-HyDE failed, will use original query embedding")
                    else:
                        # Single-doc HyDE
                        hyde_doc = hyde_enhancer.hyde.generate_hypothetical_document_sync(
                            query,
                            model_ref=config.hyde_model,
                            trace_id=trace_id,
                        )
                        if hyde_doc:
                            search_query = hyde_doc
                            logger.debug("Using single HyDE document for search")
                    _hyde_span.set_attribute(
                        "success", hyde_embedding is not None or search_query != query
                    )
            except Exception as e:
                if strict_stages():
                    raise RuntimeError(f"HyDE failed: {e}") from e
                logger.warning(f"HyDE failed, using original query: {e}")
            if strict_stages() and hyde_embedding is None and search_query == query:
                raise RuntimeError("HyDE produced no hypothetical document (is an LLM configured?)")
            if hyde_embedding is None and search_query == query:
                _degraded.record("hyde", "no hypothetical document; searched the query")
            hyde_ms = (perf_counter() - hyde_start) * 1000.0
        else:
            logger.debug(
                "HyDE disabled for search "
                f"(effective_use_hyde={effective_use_hyde}, classification={query_classification.query_type.value if query_classification else 'N/A'})"
            )

        # Generate query embedding with proper prefix
        cache_before = self.embedding_manager.get_cache_stats() or {}
        encode_start = perf_counter()
        # Use multi-HyDE embedding if available, otherwise encode the search query
        with TRACER.span(
            "embed_query", attributes={"model": config.embedding_model}
        ) as _embed_span:
            if hyde_embedding is not None:
                query_embedding = hyde_embedding
                logger.debug("Using pre-computed multi-HyDE embedding")
                _embed_span.set_attribute("source", "hyde_precomputed")
            else:
                query_embedding = effective_embedder.encode_query(search_query)
                _embed_span.set_attribute("source", "encode")
        encode_ms = (perf_counter() - encode_start) * 1000.0
        cache_after = self.embedding_manager.get_cache_stats() or {}
        cache_hits_delta = None
        cache_misses_delta = None
        if "hits" in cache_before and "hits" in cache_after:
            cache_hits_delta = cache_after["hits"] - cache_before["hits"]
        if "misses" in cache_before and "misses" in cache_after:
            cache_misses_delta = cache_after["misses"] - cache_before["misses"]
        query_cache_hit = None
        if cache_hits_delta is not None or cache_misses_delta is not None:
            query_cache_hit = bool(cache_hits_delta)

        # Resolve the vector backend for this session (may differ from default)
        backend = self._collection_manager.resolve_backend(session_id)

        # Resolve tenant partition key for partition-based isolation
        tenant_mgr = get_tenant_manager()
        partition_key = tenant_mgr.resolve_partition_key(session_id)
        if partition_key:
            tenant_mgr.ensure_partition(backend, config.collection_name, partition_key)

        # Register session backend with hybrid searcher if it differs from default
        if backend is not self._default_backend:
            self.hybrid_searcher.set_session_backend(session_id, backend, config.collection_name)

        # Build filter using backend-agnostic FilterExpression
        # SECURITY: All user-provided values must be validated
        filter_parts = []

        # Only add session_id filter if provided and valid
        session_filter = None
        if session_id:
            if not is_valid_session_id(session_id):
                raise ValueError(f"Invalid session_id format: {session_id[:50]}...")
            session_filter = backend.session_filter(config.collection_name, session_id)
            filter_parts.append(session_filter)

        if document_ids:
            # Validate each document_id before using in filter
            valid_doc_ids = [doc_id for doc_id in document_ids if is_safe_document_id(doc_id)]
            if valid_doc_ids:
                # HIGH PRIORITY FIX: Limit document IDs to prevent filter expression overflow
                max_doc_ids = config.max_document_ids_filter
                if len(valid_doc_ids) > max_doc_ids:
                    raise ValueError(
                        f"Too many document_ids ({len(valid_doc_ids)}). "
                        f"Maximum allowed: {max_doc_ids}. Use fewer IDs or paginate."
                    )
                filter_parts.append(FilterExpression.in_list("document_id", valid_doc_ids))
            elif document_ids:
                # All document_ids were invalid
                logger.warning(f"All document_ids failed validation: {document_ids}")
                return []

        # Build combined filter expression
        filter_expr = FilterExpression.and_all(filter_parts) if filter_parts else None

        # Dense search
        initial_top_k = config.default_top_k
        collection_name = config.collection_name
        logger.debug(
            f"Searching vector backend ({backend.backend_name}): "
            f"collection={collection_name}, filter={'yes' if filter_expr else 'none'}, limit={initial_top_k}"
        )

        dense_start = perf_counter()
        # P1-3 FIX: Include 'embedding' in output_fields to reuse for MMR
        # This avoids re-embedding chunks in the reranker, saving ~50-100ms

        # Use backend-agnostic search via VectorStoreAdapter
        search_params = SearchParams(
            metric_type=MetricType(config.metric_type),
            params=config.search_params,
        )
        with TRACER.span(
            "dense_search", attributes={"top_k": initial_top_k, "backend": backend.backend_name}
        ) as _span:
            if _retriever_override is not None:
                # A registered retriever replaces the vector-store call
                # entirely. It owns *how* candidates are found -- that is the
                # stage under test -- so filters and search params are not
                # imposed on top of it; a component silently constrained by
                # settings it never saw is not the component being measured.
                retrieve_kwargs: Dict[str, Any] = {}
                if getattr(_retriever_override, "_forwards_session_config", False):
                    # A built-in retriever reads this session's store with the
                    # arm's settings; it cannot find candidates without them.
                    retrieve_kwargs["config"] = config
                    retrieve_kwargs["context"] = RetrievalContext(
                        backend=backend,
                        collection=collection_name,
                        session_id=session_id,
                        filter_expr=filter_expr,
                        session_filter=session_filter,
                        document_ids=document_ids,
                        partition_key=partition_key,
                        query_embedding=query_embedding,
                        embedding_manager=effective_embedder,
                        hybrid_searcher=self.hybrid_searcher,
                        search_params=search_params,
                    )
                dense_results_raw = _adapt_retriever_results(
                    _retriever_override.retrieve(
                        query=query,
                        collection=collection_name,
                        top_k=initial_top_k,
                        **retrieve_kwargs,
                    )
                )
                # Its list is the first stage. Fusing it with BM25 below (a
                # profile with hybrid search on) would score a retriever the arm
                # did not name: `bm25` would come back as BM25 fused with BM25.
                _effective_hybrid_search = False
            else:
                dense_results_raw = backend.search(
                    collection_name=collection_name,
                    query_embedding=query_embedding.tolist(),
                    vector_field="embedding",
                    filter_expr=filter_expr,
                    limit=initial_top_k,
                    output_fields=["document_id", "content", "metadata", "embedding"],
                    search_params=search_params,
                    partition_key=partition_key,
                )
        # Convert VectorSearchResult list to legacy nested-list format for downstream compatibility
        dense_results = (
            [
                [
                    {
                        "id": r.id,
                        "score": r.score,
                        "distance": r.distance,
                        "entity": r.entity,
                    }
                    for r in dense_results_raw
                ]
            ]
            if dense_results_raw
            else [[]]
        )
        dense_ms = (perf_counter() - dense_start) * 1000.0

        logger.debug(
            f"Dense search returned: {len(dense_results[0]) if dense_results and dense_results[0] else 0} results"
        )

        if not dense_results or not dense_results[0]:
            total_ms = (perf_counter() - total_start) * 1000.0
            filter_payload = {
                "document_ids_filter_count": len(document_ids) if document_ids else 0,
            }
            if document_ids and len(document_ids) <= 5:
                filter_payload["document_ids_filter"] = document_ids
            overhead_ms = total_ms - (dense_ms + encode_ms + hyde_ms)
            log_json(
                "rag_search_metrics",
                {
                    "session_id": session_id,
                    "query_hash": query_hash,
                    "query_len": query_len,
                    "collection_name": collection_name,
                    "filter": str(filter_expr) if filter_expr else "",
                    "results_count_returned": 0,
                    "returned_document_ids_count": 0,
                    "dense_search_ms": round(dense_ms, 3),
                    "embedding_encode_ms": round(encode_ms, 3),
                    "hyde_ms": round(hyde_ms, 3),
                    "sparse_ms": 0.0,
                    "merge_ms": 0.0,
                    "rerank_ms": 0.0,
                    "mmr_ms": 0.0,
                    "postprocess_ms": 0.0,
                    "overhead_ms": round(overhead_ms, 3),
                    "total_ms": round(total_ms, 3),
                    "hybrid_mode": "dense_only",
                    "query_embedding_cache_hits_count": cache_hits_delta,
                    "query_embedding_cache_misses_count": cache_misses_delta,
                    "query_embedding_cache_hit": query_cache_hit,
                    "doc_embedding_cache_hits_count": None,
                    "doc_embedding_cache_misses_count": None,
                    "embedding_cache_global_hits": cache_after.get("hits"),
                    "embedding_cache_global_misses": cache_after.get("misses"),
                    "embedding_cache_global_hit_rate": cache_after.get("hit_rate"),
                    **filter_payload,
                },
            )
            log_json_debug(
                "rag_search_query",
                {
                    "session_id": session_id,
                    "query_hash": query_hash,
                    "query_len": query_len,
                },
            )
            METRICS.histogram("rag.search.dense_ms", dense_ms)
            METRICS.histogram("rag.search.encode_ms", encode_ms)
            METRICS.histogram("rag.search.hyde_ms", hyde_ms)
            METRICS.histogram("rag.search.total_ms", total_ms)
            METRICS.counter("rag.search.calls", 1)
            logger.info(
                f"No results from dense search for session_id={session_id}, query_hash={query_hash}"
            )
            return []

        # Convert to list format
        dense_hits = list(dense_results[0])

        sparse_ms = 0.0
        merge_ms = 0.0
        hybrid_mode = "dense_only"

        # Apply hybrid search (dense + BM25)
        if _effective_hybrid_search:
            # Finding 4 trace: log requested vs effective alpha so experiments
            # can verify the config value actually reached the hybrid searcher.
            logger.debug(
                f"Hybrid search: config.hybrid_alpha={config.hybrid_alpha}, "
                f"effective_alpha={effective_alpha}, "
                f"classification_enabled={config.enable_query_classification}, "
                f"alpha_override={'yes' if effective_alpha != config.hybrid_alpha else 'no'}"
            )
            # The BM25 index lives in memory, filled at upload. In a process that
            # did not upload (the next CLI call, a server), hybrid search ran
            # dense-only with a DEBUG line: same query, same store, different
            # results. Fill it from the store once per session and collection.
            if session_filter is not None:
                self.hybrid_searcher.hydrate_session_index(
                    session_id,
                    backend,
                    collection_name,
                    session_filter,
                    partition_key=partition_key,
                )
            with TRACER.span(
                "hybrid_fusion", attributes={"alpha": effective_alpha}
            ) as _hybrid_span:
                hybrid_results, hybrid_metrics = self.hybrid_searcher.search_hybrid(
                    query=query,
                    dense_results=dense_hits,
                    session_id=session_id,
                    top_k=initial_top_k,
                    query_embedding=query_embedding.tolist(),
                    document_ids=document_ids,
                    return_metrics=True,
                    config=config,
                    partition_key=partition_key,
                    alpha_override=effective_alpha
                    if effective_alpha != config.hybrid_alpha
                    else None,
                    embedding_manager=self.embedding_manager,
                )
            sparse_ms = hybrid_metrics.get("sparse_ms", 0.0)
            merge_ms = hybrid_metrics.get("merge_ms", 0.0)
            hybrid_mode = hybrid_metrics.get("hybrid_mode", "dense_only")

            # Convert back to dict format for reranking
            # P1-3 FIX: Include embedding for MMR to avoid re-embedding
            hits_for_rerank = [
                {
                    "id": hr.chunk_id,
                    "chunk_id": hr.chunk_id,
                    "document_id": hr.document_id,
                    "content": hr.content,
                    "metadata": hr.metadata,
                    "score": hr.combined_score,
                    "embedding": hr.embedding,  # Pre-fetched from Milvus
                    "highlights": hr.highlights,  # Backend highlight fragments
                    "entity": {
                        "document_id": hr.document_id,
                        "content": hr.content,
                        "metadata": hr.metadata,
                        "embedding": hr.embedding,
                        "highlights": hr.highlights,
                    },
                }
                for hr in hybrid_results
            ]
        else:
            # P1-3 FIX: Include embedding for MMR to avoid re-embedding
            hits_for_rerank = [
                {
                    "id": hit.get("id") or hit.get("chunk_id"),
                    "chunk_id": str(hit.get("id") or hit.get("chunk_id") or ""),
                    "document_id": hit.get("entity", {}).get("document_id", ""),
                    "embedding": hit.get("entity", {}).get("embedding"),
                    "content": hit.get("entity", {}).get("content", ""),
                    "metadata": hit.get("entity", {}).get("metadata", {}),
                    "score": hit.get("score", hit.get("distance", 0.0)),
                    "highlights": hit.get("entity", {}).get("highlights"),
                    "entity": hit.get("entity", {}),
                }
                for hit in dense_hits
            ]

        rerank_ms = 0.0
        mmr_ms = 0.0
        rerank_model = None
        mmr_lambda = None

        # GraphRAG expansion: add related chunks via entity graph traversal
        if getattr(config, "enable_graph_retrieval", False) and session_id:
            try:
                from lasagna.retrieval.graph import GraphRetriever

                graph_retriever = GraphRetriever(embedding_manager=self.embedding_manager)
                existing_ids = {h.get("id") or h.get("chunk_id", "") for h in hits_for_rerank}
                new_chunk_ids = graph_retriever.expand(
                    query=query,
                    existing_chunk_ids=existing_ids,
                    session_id=session_id,
                    max_hops=getattr(config, "graph_max_hops", 2),
                )
                if new_chunk_ids:
                    # Fetch chunk data from vector backend
                    graph_hits = self._fetch_chunks_by_ids(new_chunk_ids, session_id)
                    hits_for_rerank.extend(graph_hits)
                    logger.debug(f"GraphRAG added {len(graph_hits)} chunks to candidate set")
            except Exception as e:
                logger.warning(f"GraphRAG expansion failed: {e}")
                _degraded.record("graphrag", e)

        # Apply reranking
        if use_reranking and config.enable_reranking:
            # Check if adaptive-k retrieval is enabled
            chunk_cite_cfg = config.chunk_cite_retrieval
            use_adaptive = chunk_cite_cfg.get("strategy") == "adaptive"

            adaptive_kwargs = {}
            if use_adaptive:
                adaptive_kwargs = dict(
                    adaptive=True,
                    adaptive_min_k=chunk_cite_cfg.get("min_chunks", 2),
                    adaptive_max_k=chunk_cite_cfg.get("max_chunks", 20),
                    adaptive_buffer=chunk_cite_cfg.get("score_gap_buffer", 2),
                    adaptive_min_gap_ratio=chunk_cite_cfg.get("min_gap_ratio", 0.15),
                )

            with TRACER.span(
                "rerank", attributes={"top_k": top_k, "adaptive": use_adaptive}
            ) as _rerank_span:
                reranked, rerank_metrics = reranker.rerank(
                    query=query,
                    results=hits_for_rerank,
                    top_k=top_k,
                    return_metrics=True,
                    config=config,
                    **adaptive_kwargs,
                )
            rerank_ms = rerank_metrics.get("rerank_ms", 0.0)
            mmr_ms = rerank_metrics.get("mmr_ms", 0.0)
            rerank_model = reranker.cross_encoder.model_name
            if _effective_mmr:
                mmr_lambda = config.mmr_lambda
        else:
            # Convert to RerankedResult without reranking
            # Compute normalized scores so downstream filtering works correctly
            raw_scores = [h["score"] for h in hits_for_rerank[:top_k]]
            from lasagna.utils.normalization import normalize_scores_minmax

            norm_scores = normalize_scores_minmax(raw_scores) if raw_scores else []
            reranked = [
                RerankedResult(
                    chunk_id=h["chunk_id"],
                    document_id=h["document_id"],
                    content=h["content"],
                    metadata=h["metadata"],
                    original_score=h["score"],
                    rerank_score=h["score"],
                    normalized_score=norm_scores[i] if i < len(norm_scores) else 0.0,
                    original_rank=i,
                    final_rank=i,
                    highlights=h.get("highlights"),
                )
                for i, h in enumerate(hits_for_rerank[:top_k])
            ]

        # Lost-in-the-middle reordering: place best results at start and end
        # LLMs attend better to the beginning and end of context (arXiv:2406.16008)
        if _effective_context_reordering and len(reranked) > 2:
            reranked = self._reorder_for_context(reranked)

        # Context compression: reduce chunk content to query-relevant portions
        if getattr(config, "enable_context_compression", False) and reranked:
            reranked = self._compress_results(reranked, query, config)

        # Convert to SearchResult format
        postprocess_start = perf_counter()

        # OPTIMIZATION: Batch fetch all document metadata in single query
        # FIX: Use normalized_score for filtering when min_score >= 0.
        # Raw rerank_score can be negative (cross-encoder logits are unbounded),
        # so comparing against min_score=0.0 would filter out valid results.
        # normalized_score is always in [0, 1] and comparable to min_score thresholds.
        use_normalized_filter = min_score >= 0.0
        document_ids_to_fetch = list(
            {
                result.document_id
                for result in reranked
                if (
                    result.normalized_score >= min_score
                    if use_normalized_filter
                    else result.rerank_score >= min_score
                )
            }
        )
        doc_metadata_map = self._document_pipeline.get_documents_metadata_batch(
            document_ids_to_fetch
        )

        search_results = []
        # Chunks retrieved and ranked, then discarded because their parent
        # document has no metadata row. Counted rather than ignored: dropping
        # every result silently is indistinguishable from retrieving nothing,
        # and the caller sees a well-formed empty result either way.
        missing_metadata_ids: List[str] = []
        for result in reranked:
            score_for_filter = (
                result.normalized_score if use_normalized_filter else result.rerank_score
            )
            if score_for_filter < min_score:
                continue

            chunk = RecordedDocumentChunk(
                chunk_id=result.chunk_id,
                content=result.content,
                document_id=result.document_id,
                chunk_index=result.metadata.get("chunk_index", 0),
                page_number=result.metadata.get("page_number"),  # Extract page number from metadata
                metadata=result.metadata,
                created_at=datetime.now(timezone.utc),
            )

            doc_metadata = doc_metadata_map.get(result.document_id)
            if doc_metadata:
                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=result.normalized_score,  # Use normalized score for display
                        document=doc_metadata,
                    )
                )
            else:
                missing_metadata_ids.append(result.document_id)

        if missing_metadata_ids:
            unique_missing = sorted(set(missing_metadata_ids))
            # Loud when it swallowed the entire result set, because that is the
            # case a caller cannot tell apart from "nothing matched": retrieval,
            # reranking and MMR all succeeded and the answer is still empty.
            level = logger.error if not search_results else logger.warning
            level(
                f"Dropped {len(missing_metadata_ids)} retrieved chunk(s) from "
                f"{len(unique_missing)} document(s) with no metadata row"
                + (
                    " — returning 0 results despite successful retrieval"
                    if not search_results
                    else ""
                )
                + f". document_ids={unique_missing[:5]}"
                + (f" (+{len(unique_missing) - 5} more)" if len(unique_missing) > 5 else "")
                + ". The chunks are in the vector store but the documents were "
                "never registered, so results cannot be attributed to a document."
            )

        # Auto-merge: replace child chunks with parent content when enough siblings retrieved
        auto_merge_ms = 0.0
        if _effective_hierarchical and search_results:
            auto_merge_start = perf_counter()
            with TRACER.span(
                "auto_merge", attributes={"input_count": len(search_results)}
            ) as _merge_span:
                search_results = self._auto_merger.merge_results(search_results, config=config)
                _merge_span.set_attribute("output_count", len(search_results))
            auto_merge_ms = (perf_counter() - auto_merge_start) * 1000.0

        postprocess_ms = (perf_counter() - postprocess_start) * 1000.0

        total_ms = (perf_counter() - total_start) * 1000.0
        overhead_ms = total_ms - (
            dense_ms
            + sparse_ms
            + merge_ms
            + rerank_ms
            + mmr_ms
            + encode_ms
            + hyde_ms
            + postprocess_ms
        )
        cache_stats = cache_after or {}
        returned_doc_ids = sorted({r.document.document_id for r in search_results})
        filter_payload = {
            "document_ids_filter_count": len(document_ids) if document_ids else 0,
        }
        if document_ids and len(document_ids) <= 5:
            filter_payload["document_ids_filter"] = document_ids
        returned_payload = {
            "returned_document_ids_count": len(returned_doc_ids),
        }
        if len(returned_doc_ids) <= 5:
            returned_payload["returned_document_ids"] = returned_doc_ids

        log_payload = {
            "session_id": session_id,
            "query_hash": query_hash,
            "query_len": query_len,
            "routing_tier": routing_decision.tier.value if routing_decision else None,
            "collection_name": collection_name,
            "filter": str(filter_expr) if filter_expr else "",
            "results_count_returned": len(search_results),
            "using_focused_docs": using_focused_docs,
            "focused_doc_count": focused_doc_count,
            "dense_search_ms": round(dense_ms, 3),
            "embedding_encode_ms": round(encode_ms, 3),
            "hyde_ms": round(hyde_ms, 3),
            "sparse_ms": round(sparse_ms, 3),
            "merge_ms": round(merge_ms, 3),
            "rerank_ms": round(rerank_ms, 3),
            "mmr_ms": round(mmr_ms, 3),
            "auto_merge_ms": round(auto_merge_ms, 3),
            "postprocess_ms": round(postprocess_ms, 3),
            "overhead_ms": round(overhead_ms, 3),
            "total_ms": round(total_ms, 3),
            "hybrid_mode": hybrid_mode,
            "rerank_model": rerank_model,
            "mmr_lambda": mmr_lambda,
            "query_embedding_cache_hits_count": cache_hits_delta,
            "query_embedding_cache_misses_count": cache_misses_delta,
            "query_embedding_cache_hit": query_cache_hit,
            "doc_embedding_cache_hits_count": None,
            "doc_embedding_cache_misses_count": None,
            "embedding_cache_global_hits": cache_stats.get("hits"),
            "embedding_cache_global_misses": cache_stats.get("misses"),
            "embedding_cache_global_hit_rate": cache_stats.get("hit_rate"),
            "embedding_cache_size": cache_stats.get("size"),
            **filter_payload,
            **returned_payload,
        }
        log_json("rag_search_metrics", log_payload)
        log_json_debug(
            "rag_search_query",
            {
                "session_id": session_id,
                "query_hash": query_hash,
                "query_len": query_len,
            },
        )

        METRICS.histogram("rag.search.dense_ms", dense_ms)
        METRICS.histogram("rag.search.encode_ms", encode_ms)
        METRICS.histogram("rag.search.hyde_ms", hyde_ms)
        METRICS.histogram("rag.search.sparse_ms", sparse_ms)
        METRICS.histogram("rag.search.merge_ms", merge_ms)
        METRICS.histogram("rag.search.rerank_ms", rerank_ms)
        METRICS.histogram("rag.search.mmr_ms", mmr_ms)
        METRICS.histogram("rag.search.auto_merge_ms", auto_merge_ms)
        METRICS.histogram("rag.search.postprocess_ms", postprocess_ms)
        METRICS.histogram("rag.search.overhead_ms", overhead_ms)
        METRICS.histogram("rag.search.total_ms", total_ms)
        METRICS.counter("rag.search.calls", 1)

        logger.debug(
            f"Search: {len(dense_hits)} dense -> {len(hits_for_rerank)} candidates "
            f"({hybrid_mode}) -> {len(search_results)} final"
        )

        # Warn if focused docs caused empty results
        if using_focused_docs and len(search_results) == 0:
            logger.warning(
                f"Search returned 0 results while filtering by {focused_doc_count} "
                f"focused document(s). The focused documents may not contain "
                f"relevant content for this query."
            )

        # P0-1 FIX: Populate search cache with results for future queries
        if search_cache and search_results:
            try:
                # Serialize results for caching
                cacheable_results = []
                for result in search_results:
                    cacheable_results.append(
                        {
                            "chunk_id": result.chunk.chunk_id,
                            "content": result.chunk.content,
                            "document_id": result.chunk.document_id,
                            "chunk_index": result.chunk.chunk_index,
                            "page_number": result.chunk.page_number,
                            "metadata": result.chunk.metadata,
                            "score": result.score,
                            "filename": result.document.filename if result.document else "",
                            "file_size": result.document.file_size if result.document else 0,
                            "file_type": result.document.file_type if result.document else "",
                            "chunks_count": result.document.chunks_count if result.document else 0,
                        }
                    )
                search_cache.put(
                    query=query,
                    session_id=session_id,
                    results=cacheable_results,
                    query_embedding=query_embedding.tolist()
                    if hasattr(query_embedding, "tolist")
                    else list(query_embedding),
                    document_ids=document_ids,
                    top_k=top_k,
                    use_hyde=effective_use_hyde,
                    use_reranking=use_reranking and config.enable_reranking,
                    hybrid_alpha=effective_alpha,
                    mmr_lambda=config.mmr_lambda,
                    hyde_n_docs=effective_hyde_n_docs if effective_use_hyde else 0,
                    enable_hybrid_search=_effective_hybrid_search,
                    enable_mmr=_effective_mmr,
                    component_fingerprint=component_fingerprint,
                )
                logger.debug(f"Search results cached (count={len(search_results)})")
            except Exception as e:
                logger.warning(f"Failed to cache search results: {e}")

        # Record successful search for quota tracking
        quota_mgr.record_search(session_id)

        return search_results

    # =========================================================================
    # Async Search
    # =========================================================================

    async def async_search(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        min_score: float = -100.0,
        document_ids: Optional[List[str]] = None,
        use_hyde: bool = True,
        use_reranking: bool = True,
        trace_id: Optional[str] = None,
        routing_tier: Optional[str] = None,
        _reranker_override: Optional[Any] = None,
        _retriever_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
    ) -> List[SearchResult]:
        """Async version of search().  Offloads CPU-bound work to thread pool.

        Suitable for use in FastAPI async endpoints.  Uses the same SOTA
        pipeline as the synchronous ``search()`` method.
        """
        loop = asyncio.get_running_loop()

        # Capture the routing decision from the executor thread via closure.
        # threading.local() is per-thread, so we must relay the decision back
        # to the event-loop thread where the REST handler reads it.
        _routing_holder: Dict[str, Any] = {}

        def _run() -> List[SearchResult]:
            result = self.search(
                query=query,
                session_id=session_id,
                top_k=top_k,
                min_score=min_score,
                document_ids=document_ids,
                use_hyde=use_hyde,
                use_reranking=use_reranking,
                trace_id=trace_id,
                routing_tier=routing_tier,
                _reranker_override=_reranker_override,
                _enhancer_override=_enhancer_override,
            )
            _routing_holder["decision"] = getattr(_routing_local, "last_decision", None)
            return result

        # In the caller's context: the request id on every log line, and the
        # degraded stages the REST response reports, live in context variables
        # that run_in_executor alone does not carry into the thread.
        results = await loop.run_in_executor(None, contextvars.copy_context().run, _run)
        # Relay to instance attribute — safe: no yield between this write
        # and the REST handler's read of _get_routing_info().
        self._last_routing_decision = _routing_holder.get("decision")
        return results

    async def async_search_with_citations(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
        routing_tier: Optional[str] = None,
        _reranker_override: Optional[Any] = None,
        _retriever_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
    ) -> Tuple[List[SearchResult], List[Citation], str]:
        """Async version of search_with_citations()."""
        loop = asyncio.get_running_loop()

        _routing_holder: Dict[str, Any] = {}

        def _run() -> Tuple[List[SearchResult], List[Citation], str]:
            result = self.search_with_citations(
                query=query,
                session_id=session_id,
                top_k=top_k,
                document_ids=document_ids,
                trace_id=trace_id,
                routing_tier=routing_tier,
                _reranker_override=_reranker_override,
                _enhancer_override=_enhancer_override,
            )
            _routing_holder["decision"] = getattr(_routing_local, "last_decision", None)
            return result

        results = await loop.run_in_executor(None, contextvars.copy_context().run, _run)
        self._last_routing_decision = _routing_holder.get("decision")
        return results

    # =========================================================================
    # Search with Citations
    # =========================================================================

    def search_with_citations(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        trace_id: Optional[str] = None,
        routing_tier: Optional[str] = None,
        _reranker_override: Optional[Any] = None,
        _retriever_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
    ) -> Tuple[List[SearchResult], List[Citation], str]:
        """Search and return results with citations and formatted context.

        Args:
            query: Search query
            session_id: Session ID
            top_k: Number of results
            document_ids: Optional document filter
            trace_id: Trace ID for observability
            routing_tier: Optional forced routing tier.
            _reranker_override: Optional reranker instance override.
            _enhancer_override: Optional query enhancer instance override.

        Returns:
            Tuple of (results, citations, formatted_context)
        """
        results = self.search(
            query=query,
            session_id=session_id,
            top_k=top_k,
            document_ids=document_ids,
            trace_id=trace_id,
            routing_tier=routing_tier,
            _reranker_override=_reranker_override,
            _enhancer_override=_enhancer_override,
        )

        # Create citations with highlight extraction
        query_hash = hash_text(query)
        citation_start = perf_counter()
        with TRACER.span(
            "citation_extraction", attributes={"result_count": len(results)}
        ) as _cite_span:
            # Reset citation sequence so each new query starts from [1]
            self.citation_manager.reset_session_sequence(session_id)
            citations = self.citation_manager.create_citations(
                search_results=results,
                session_id=session_id,
                query=query,  # Pass query for highlight extraction
            )

            # Format context for LLM
            formatted_context = self.citation_manager.format_context_with_citations(
                citations=citations
            )
            _cite_span.set_attribute("citation_count", len(citations))
        citation_ms = (perf_counter() - citation_start) * 1000.0

        log_json(
            "rag_citation_metrics",
            {
                "session_id": session_id,
                "query_hash": query_hash,
                "query_len": len(query) if query else 0,
                "results_count_returned": len(results),
                "citations_count": len(citations),
                "citation_ms": round(citation_ms, 3),
            },
        )
        log_json_debug(
            "rag_citation_query",
            {
                "session_id": session_id,
                "query_hash": query_hash,
                "query_len": len(query) if query else 0,
            },
        )
        METRICS.histogram("rag.search.citation_ms", citation_ms)

        return results, citations, formatted_context

    # =========================================================================
    # Multi-Query Search
    # =========================================================================

    def search_multi_query(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
        n_variations: int = 3,
        trace_id: Optional[str] = None,
        _reranker_override: Optional[Any] = None,
        _retriever_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
        _embedder_override: Optional[Any] = None,
        _from_routing: bool = False,
        query_config_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Search with query expansion and RRF-based result fusion.

        This is a SOTA technique that:
        1. Generates query variations using the QueryExpander
        2. Searches with each query variation (original + variations)
        3. Fuses results using Reciprocal Rank Fusion (RRF) with original query weighted higher

        P1-1 FIX: Optimized to avoid running full HyDE pipeline for each variation.
        P2-3 FIX: Original query is weighted 2x in RRF scoring.

        Args:
            query: Original search query
            session_id: Session ID
            top_k: Number of final results
            document_ids: Optional document filter
            n_variations: Number of query variations to generate
            trace_id: Trace ID for observability
            _reranker_override: Optional reranker instance override.
            _enhancer_override: Optional query enhancer instance override.
            _embedder_override: Optional embedder instance override. Forwarded
                to each sub-search so that ablation studies use the correct
                embedder for query encoding.

        Returns:
            Fused search results
        """
        # Resolve effective enhancer for query expansion
        enhancer = _enhancer_override or self.query_enhancer
        config = self._get_session_config(session_id)
        # Apply per-query ephemeral overrides (eval SDK / ablation path) so that
        # enable_query_expansion, max_query_variations, etc. set via config_overrides
        # are visible here.
        if query_config_overrides:
            config = config.with_query_overrides(query_config_overrides)
        top_k = top_k or config.rerank_top_k
        start_time = perf_counter()

        expansion_on = config.enable_query_expansion or _is_public_enhancer(enhancer)
        if not expansion_on or n_variations <= 0:
            # Fall back to regular search (_from_fusion prevents re-entering RAG Fusion)
            return self.search(
                query,
                session_id,
                top_k,
                document_ids=document_ids,
                trace_id=trace_id,
                _from_fusion=True,
                _from_routing=True,
                _reranker_override=_reranker_override,
                _enhancer_override=_enhancer_override,
                _embedder_override=_embedder_override,
                query_config_overrides=query_config_overrides,
            )

        # Generate query variations using the sync helper, which calls
        # llm.invoke() directly and therefore touches no event loop.
        #
        # This previously bridged to the async expander via asyncio.run(). That
        # created a fresh event loop per call, so a cached LLM client bound to
        # an earlier loop raised RuntimeError; the broad `except` below turned
        # it into a one-line warning and set variations = []. Downstream RRF
        # then fused a single query, making every query_expansion and
        # rag_fusion configuration silently identical to plain hybrid search.
        try:
            variations = _query_variations(enhancer, query, n_variations, config)
        except Exception as e:
            if strict_stages():
                raise RuntimeError(f"Query enhancement failed: {e}") from e
            logger.warning(f"Query expansion failed [{type(e).__name__}]: {e}")
            logger.exception("Query expansion failed; falling back to single query")
            _degraded.record("query_expansion", e)
            variations = []
        if not variations:
            # Nothing to fuse: the plain search is the answer. Fusing a single
            # query used to come back empty and score 0.0.
            #
            # Not a failure, even under evaluation: a technique may have nothing
            # to add for a query ("photosynthesis" has no stopwords to drop).
            # An enhancer that cannot run at all (no LLM) raises above, and an
            # arm whose enhancer never changed a ranking is flagged by the
            # isolation check from its ranked lists.
            return self.search(
                query,
                session_id,
                top_k,
                document_ids=document_ids,
                trace_id=trace_id,
                _from_fusion=True,
                _from_routing=True,
                _reranker_override=_reranker_override,
                _enhancer_override=_enhancer_override,
                _embedder_override=_embedder_override,
                query_config_overrides=query_config_overrides,
            )

        all_queries = [query] + variations[:n_variations]
        logger.debug(f"Multi-query search with {len(all_queries)} queries")

        # OpenSearch _msearch batch optimization: batch N hybrid searches
        # into a single HTTP round-trip, cutting network latency by ~Nx.
        backend = self._collection_manager.resolve_backend(session_id)
        if (
            backend.backend_name == "opensearch"
            and hasattr(backend, "multi_hybrid_search")
            and len(all_queries) > 1
        ):
            try:
                return self._search_multi_query_msearch(
                    all_queries,
                    session_id,
                    config,
                    top_k,
                    document_ids,
                    backend,
                    _reranker_override,
                    _embedder_override,
                    start_time,
                )
            except Exception as e:
                logger.warning(f"_msearch batch path failed, falling through to threaded path: {e}")

        # P1-1 FIX: Disable HyDE for variations to avoid 4x LLM calls
        # HyDE is already capturing query intent - variations are for keyword diversity
        all_results_by_chunk: Dict[int, Dict[str, Any]] = {}
        rrf_k = config.rrf_k

        # P2-3 FIX: Track query index for weighted scoring
        # Original query (index 0) gets weight 2.0, variations get weight 1.0
        ORIGINAL_QUERY_WEIGHT = 2.0
        VARIATION_QUERY_WEIGHT = 1.0

        # HIGH PRIORITY FIX: Parallelize multi-query searches using ThreadPoolExecutor
        # This reduces latency from O(n * search_time) to O(search_time)
        def _search_query(args: Tuple[int, str]) -> Tuple[int, List["SearchResult"]]:
            """Helper function to search a single query variant."""
            query_idx, q = args
            try:
                # P1-1 FIX: Only use HyDE for original query, not variations
                use_hyde_for_query = query_idx == 0
                results = self.search(
                    query=q,
                    session_id=session_id,
                    top_k=top_k,
                    document_ids=document_ids,
                    use_hyde=use_hyde_for_query,
                    trace_id=trace_id,
                    _from_fusion=True,
                    _from_routing=True,
                    _reranker_override=_reranker_override,
                    _enhancer_override=_enhancer_override,
                    _embedder_override=_embedder_override,
                    # Without these each variation searched under the session
                    # config, not the arm's (another embedder, another
                    # collection), failed, and the fused list came back empty.
                    query_config_overrides=query_config_overrides,
                )
                return query_idx, results
            except Exception as e:
                logger.warning(f"Search for query variation {query_idx} failed: {e}")
                return query_idx, []

        # Execute searches in parallel
        max_workers = min(len(all_queries), 4)  # Cap at 4 to avoid overwhelming the backend
        # HIGH PRIORITY FIX: Add timeout to prevent indefinite hanging on slow queries
        # 30 seconds per query is generous but prevents complete system stall
        QUERY_TIMEOUT_SECONDS = 30.0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Each worker runs in a copy of this context, not a fresh one.
            #
            # A bare `submit` starts the thread with an empty context, so any
            # contextvar set by the caller is invisible below here -- including
            # the token UsageMeter. Every LLM call made inside a multi-query
            # search (the variations, and HyDE within each branch) was therefore
            # uncounted, which is most of what an enhanced query actually
            # spends. `asyncio.to_thread` copies the context for this reason; a
            # raw ThreadPoolExecutor does not.
            futures = {
                executor.submit(contextvars.copy_context().run, _search_query, (i, q)): i
                for i, q in enumerate(all_queries)
            }

            # Queries run in parallel, so total wall time ≈ one query's latency.
            # Add small buffer for thread scheduling overhead.
            total_timeout = QUERY_TIMEOUT_SECONDS + 5.0
            for future in as_completed(futures, timeout=total_timeout):
                try:
                    # HIGH PRIORITY FIX: Add per-future timeout as additional safety
                    query_idx, results = future.result(timeout=QUERY_TIMEOUT_SECONDS)
                    query_weight = (
                        ORIGINAL_QUERY_WEIGHT if query_idx == 0 else VARIATION_QUERY_WEIGHT
                    )

                    for rank, result in enumerate(results):
                        chunk_id = result.chunk.chunk_id
                        if chunk_id not in all_results_by_chunk:
                            all_results_by_chunk[chunk_id] = {
                                "result": result,
                                "ranks": [],  # (rank, weight) tuples
                            }
                        all_results_by_chunk[chunk_id]["ranks"].append((rank, query_weight))
                except TimeoutError:
                    logger.warning(f"Search query timed out after {QUERY_TIMEOUT_SECONDS}s")
                except Exception as e:
                    logger.warning(f"Error processing search future: {e}")

        if not all_results_by_chunk:
            return []

        # P2-3 FIX: RRF fusion with weighted original query
        # Formula: score = sum(weight / (rrf_k + rank))
        fused_scores = []
        for chunk_id, data in all_results_by_chunk.items():
            # Weighted RRF: original query contributes more
            rrf_score = sum(weight / (rrf_k + rank) for rank, weight in data["ranks"])
            fused_scores.append((rrf_score, data["result"]))

        # Sort by fused score
        fused_scores.sort(key=lambda x: x[0], reverse=True)

        total_ms = (perf_counter() - start_time) * 1000.0
        logger.debug(
            f"Multi-query fusion completed: {len(all_queries)} queries, "
            f"{len(all_results_by_chunk)} unique chunks, {total_ms:.1f}ms "
            f"(original_weight={ORIGINAL_QUERY_WEIGHT}, variation_weight={VARIATION_QUERY_WEIGHT})"
        )

        # Update scores in results
        final_results = []
        for i, (fused_score, result) in enumerate(fused_scores[:top_k]):
            # Update the result's score to reflect fusion.
            # SearchResult is a Pydantic model — create a new instance with
            # the fused score rather than mutating rerank_score (which is not
            # a field on SearchResult; it lives on the internal RerankResult).
            final_results.append(
                SearchResult(
                    chunk=result.chunk,
                    score=fused_score,
                    document=result.document,
                )
            )

        logger.debug(
            f"Multi-query fusion: {len(all_queries)} queries -> "
            f"{len(all_results_by_chunk)} unique chunks -> "
            f"{len(final_results)} final results"
        )

        return final_results

    # =========================================================================
    # OpenSearch _msearch Batch Path for RAG Fusion
    # =========================================================================

    def _search_multi_query_msearch(
        self,
        all_queries: List[str],
        session_id: str,
        config: "RAGConfig",
        top_k: int,
        document_ids: Optional[List[str]],
        backend: "VectorStoreAdapter",
        _reranker_override: Optional[Any],
        _embedder_override: Optional[Any],
        start_time: float,
    ) -> List["SearchResult"]:
        """Batch N hybrid searches into a single OpenSearch _msearch round-trip.

        This replaces the ThreadPoolExecutor approach when the backend is
        OpenSearch, cutting network latency from O(N) to O(1).

        Pipeline: batch embed → batch hybrid search → RRF fusion → rerank → convert.
        """
        # Quota enforcement — must match the standard search() path
        quota_mgr = get_quota_manager()
        quota_mgr.check_search(session_id)

        effective_embedder = _embedder_override or self.embedding_manager

        # Batch embed all queries (returns List[np.ndarray]). Only the built-in
        # manager has a batch method; a component embedder answers one query
        # at a time through the public contract.
        if hasattr(effective_embedder, "encode_queries_batch"):
            raw_embeddings = effective_embedder.encode_queries_batch(all_queries)
        else:
            raw_embeddings = [effective_embedder.encode_query(q) for q in all_queries]
        all_embeddings = [emb.tolist() for emb in raw_embeddings]

        # ENFORCEMENT: Check for focused_document_ids in session config
        # Must match search() behavior — focused docs override caller's document_ids
        effective_document_ids = document_ids
        try:
            from ..core.session_config import get_knowledge_config_manager

            config_manager = get_knowledge_config_manager()
            focused_ids = config_manager.get_focused_document_ids(session_id)
            if focused_ids:
                logger.debug(
                    f"ENFORCEMENT (msearch): Using focused_document_ids {focused_ids} "
                    f"(overriding document_ids={document_ids})"
                )
                effective_document_ids = focused_ids
        except Exception as e:
            logger.warning(f"Could not check focused_document_ids in msearch path: {e}")

        # Build filter — backend is always OpenSearch in this code path
        filter_parts = []
        if session_id:
            if not is_valid_session_id(session_id):
                raise ValueError(f"Invalid session_id format: {session_id[:50]}...")
            filter_parts.append(FilterExpression.eq("session_id", session_id))
        if effective_document_ids:
            # Validate each document_id before using in filter
            valid_doc_ids = [
                doc_id for doc_id in effective_document_ids if is_safe_document_id(doc_id)
            ]
            if valid_doc_ids:
                max_doc_ids = config.max_document_ids_filter
                if len(valid_doc_ids) > max_doc_ids:
                    raise ValueError(
                        f"Too many document_ids ({len(valid_doc_ids)}). "
                        f"Maximum allowed: {max_doc_ids}. Use fewer IDs or paginate."
                    )
                filter_parts.append(FilterExpression.in_list("document_id", valid_doc_ids))
            elif effective_document_ids:
                logger.warning(
                    f"All document_ids failed validation in msearch path: {effective_document_ids}"
                )
                return []
        filter_expr = FilterExpression.and_all(filter_parts) if filter_parts else None

        tenant_mgr = get_tenant_manager()
        partition_key = tenant_mgr.resolve_partition_key(session_id)

        # Batch hybrid search
        all_vsr_lists = backend.multi_hybrid_search(
            collection_name=config.collection_name,
            query_texts=all_queries,
            query_embeddings=all_embeddings,
            filter_expr=filter_expr,
            limit=top_k,
            alpha=getattr(config, "hybrid_alpha", 0.7),
            output_fields=["document_id", "content", "metadata", "embedding"],
            partition_key=partition_key,
        )

        # Weighted RRF fusion
        ORIGINAL_QUERY_WEIGHT = 2.0
        VARIATION_QUERY_WEIGHT = 1.0
        rrf_k = config.rrf_k

        chunk_data: Dict[str, Dict[str, Any]] = {}
        for query_idx, vsr_list in enumerate(all_vsr_lists):
            weight = ORIGINAL_QUERY_WEIGHT if query_idx == 0 else VARIATION_QUERY_WEIGHT
            for rank, vsr in enumerate(vsr_list):
                cid = vsr.id
                if cid not in chunk_data:
                    chunk_data[cid] = {"vsr": vsr, "ranks": []}
                chunk_data[cid]["ranks"].append((rank, weight))

        if not chunk_data:
            return []

        fused = sorted(
            [(sum(w / (rrf_k + r) for r, w in d["ranks"]), d["vsr"]) for d in chunk_data.values()],
            key=lambda x: x[0],
            reverse=True,
        )[:top_k]

        # Rerank the fused results for quality parity with the threaded path
        from ..retrieval.search.ranker import adapt_reranker_for_pipeline

        effective_reranker = adapt_reranker_for_pipeline(_reranker_override or self.reranker)
        from ..retrieval.search.ranker import bind_embedding_manager

        bind_embedding_manager(effective_reranker, self.embedding_manager)
        original_query = all_queries[0]

        hits_for_rerank = [
            {
                "id": vsr.id,
                "chunk_id": vsr.id,
                "document_id": vsr.entity.get("document_id", ""),
                "content": vsr.entity.get("content", ""),
                "metadata": vsr.entity.get("metadata", {}),
                "score": fused_score,
                "embedding": vsr.entity.get("embedding"),
                "highlights": vsr.entity.get("highlights"),
                "entity": vsr.entity,
            }
            for fused_score, vsr in fused
        ]

        reranked = effective_reranker.rerank(
            query=original_query,
            results=hits_for_rerank,
            top_k=top_k,
            config=config,
        )

        # Convert reranked results to SearchResult objects
        doc_ids = list({r.document_id for r in reranked})
        doc_map = self._document_pipeline.get_documents_metadata_batch(doc_ids)

        final_results = []
        for result in reranked:
            doc = doc_map.get(result.document_id)
            if not doc:
                continue
            chunk = RecordedDocumentChunk(
                chunk_id=result.chunk_id,
                content=result.content,
                document_id=result.document_id,
                chunk_index=result.metadata.get("chunk_index", 0),
                page_number=result.metadata.get("page_number"),
                metadata=result.metadata,
                created_at=datetime.now(timezone.utc),
            )
            final_results.append(
                SearchResult(chunk=chunk, score=result.normalized_score, document=doc)
            )

        total_ms = (perf_counter() - start_time) * 1000.0
        logger.debug(
            f"_msearch batch fusion: {len(all_queries)} queries -> "
            f"{len(chunk_data)} unique chunks -> {len(final_results)} reranked results "
            f"({total_ms:.1f}ms)"
        )

        # Record successful search for quota tracking (matches standard path)
        quota_mgr.record_search(session_id)

        return final_results

    # =========================================================================
    # Comprehensive (Agentic) Search
    # =========================================================================

    def _search_comprehensive(
        self,
        query: str,
        session_id: str,
        top_k: Optional[int],
        min_score: float,
        document_ids: Optional[List[str]],
        trace_id: Optional[str],
        routing_decision: Any,
        _reranker_override: Optional[Any] = None,
        _retriever_override: Optional[Any] = None,
        _enhancer_override: Optional[Any] = None,
        _embedder_override: Optional[Any] = None,
        _agentic_pipeline_override: Optional[Any] = None,
        _graph_node_overrides: Optional[Dict[str, Any]] = None,
        query_config_overrides: Optional[Dict[str, Any]] = None,
    ) -> List[SearchResult]:
        """Run agentic multi-hop retrieval for complex queries.

        Preserves quota enforcement and metrics parity with the standard
        search pipeline.  Called as an early-return from ``search()`` when the
        routing tier is COMPREHENSIVE.

        Args:
            _agentic_pipeline_override: A ``BaseAgenticPipeline`` instance
                to use instead of the config-resolved pipeline.
            _graph_node_overrides: Dict mapping node names to callables,
                passed through to the built-in structured graph.
        """
        total_start = perf_counter()
        config = self._get_session_config(session_id)
        # The per-query settings outrank the session's here too; the agentic
        # path used to read only the session store, which sweep arms share.
        if query_config_overrides:
            config = config.with_query_overrides(query_config_overrides)

        # Quota enforcement (same as standard path)
        quota_mgr = get_quota_manager()
        quota_mgr.check_search(session_id)

        # Callers already inside an event loop MUST use async_search(), which
        # offloads search() to a worker thread with no running loop, so the
        # asyncio.run() below creates a fresh one cleanly. This is checked
        # before the try on purpose: inside it, the broad fallback handler
        # caught the violation and quietly served standard results under an
        # agentic label, which is how every SDK agentic search degraded.
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass  # no running loop — the expected case
        else:
            raise RuntimeError(
                "search() was called from inside a running event loop. "
                "Use `await async_search(...)` instead: it offloads to a "
                "worker thread so the agentic pipeline gets a clean loop."
            )

        try:
            from lasagna.agentic.agentic_rag import AgenticRAG
            from lasagna.providers import get_llm_provider

            llm = get_llm_provider().get_llm()
            max_iterations = getattr(config, "max_agent_iterations", 3)
            agent = AgenticRAG(
                llm=llm,
                max_iterations=max_iterations,
                session_config=config,
                _pipeline_override=_agentic_pipeline_override,
                _node_overrides=_graph_node_overrides,
            )

            # AgenticRAG.retrieve is async — run it synchronously on one loop.
            #
            # This is deliberately a single asyncio.run() rather than the former
            # "running loop -> ThreadPoolExecutor + asyncio.run, else
            # asyncio.run" pair. Both branches ended up calling asyncio.run,
            # so the only thing the split added was a second code path that
            # could hand the agent a different loop than the one its cached LLM
            # client was bound to -- surfacing as "bound to a different event
            # loop" and, because the failure was swallowed upstream, as CRAG
            # runs whose RAGAS scores came back empty with no error message.
            #
            # (The running-loop contract is checked before the try above.)

            effective_top_k = top_k or config.rerank_top_k
            _given = {
                "_reranker_override": _reranker_override,
                "_enhancer_override": _enhancer_override,
                "_retriever_override": _retriever_override,
                "_embedder_override": _embedder_override,
            }
            _token = _INNER_SEARCH_COMPONENTS.set(
                {k: v for k, v in _given.items() if v is not None}
            )
            try:
                result = asyncio.run(
                    agent.retrieve(
                        question=query,
                        session_id=session_id,
                        document_ids=document_ids,
                        top_k=effective_top_k,
                        trace_id=trace_id,
                        routing_decision=routing_decision,
                    )
                )
            finally:
                _INNER_SEARCH_COMPONENTS.reset(_token)

            # Convert AgenticRAGResult chunks to List[SearchResult],
            # preserving the agentic answer in the first chunk's metadata.
            search_results = self._convert_agentic_results(result, session_id)

        except (KeyError, ValueError, TypeError):
            # Configuration errors (unknown pipeline, bad node overrides, etc.)
            # must fail loudly — don't mask misconfig behind a silent fallback.
            raise
        except Exception as e:
            if strict_stages():
                # Under evaluation the standard results would be scored as the
                # agentic arm's: fail the task with the reason instead.
                raise RuntimeError(f"Agentic search failed: {e}") from e
            logger.error(f"Comprehensive search failed, falling back to standard: {e}")
            _degraded.record("agentic", e)
            # Fallback to standard search on transient/runtime failure
            fallback_results = self.search(
                query=query,
                session_id=session_id,
                top_k=top_k,
                min_score=min_score,
                document_ids=document_ids,
                trace_id=trace_id,
                _from_routing=True,  # prevent re-routing
                _reranker_override=_reranker_override,
                _enhancer_override=_enhancer_override,
            )
            # Mark the substitution so an evaluation can tell these standard
            # results from agentic ones instead of scoring them as agentic.
            if fallback_results:
                fallback_results[0].chunk.metadata["_routing_tier_used"] = "standard_fallback"
            return fallback_results

        # Metrics parity — include pipeline identity for observability
        pipeline_identity = (
            getattr(_agentic_pipeline_override, "name", type(_agentic_pipeline_override).__name__)
            if _agentic_pipeline_override
            else getattr(config, "agentic_pipeline", "legacy_react")
        )
        total_ms = (perf_counter() - total_start) * 1000.0
        log_json(
            "rag_search_metrics",
            {
                "session_id": session_id,
                "routing_tier": "comprehensive",
                "agentic_pipeline": pipeline_identity,
                "results_count_returned": len(search_results),
                "total_ms": round(total_ms, 3),
                "query_hash": hash_text(query),
                "query_len": len(query) if query else 0,
            },
        )
        METRICS.histogram(
            "rag.search.total_ms",
            total_ms,
            tags={"tier": "comprehensive", "pipeline": pipeline_identity},
        )
        METRICS.counter("rag.search.calls", 1)

        quota_mgr.record_search(session_id)
        return search_results

    def _convert_agentic_results(
        self,
        agentic_result: Any,
        session_id: str,
    ) -> List[SearchResult]:
        """Convert AgenticRAGResult to List[SearchResult] for type compatibility.

        Preserves the agentic pipeline's generated answer (``agent_summary``)
        and formatted context in the first SearchResult's chunk metadata so
        that callers (e.g. the evaluation SDK) can retrieve and evaluate them.
        """
        search_results = []
        chunks = getattr(agentic_result, "chunks", [])
        for i, chunk_data in enumerate(chunks):
            if isinstance(chunk_data, SearchResult):
                search_results.append(chunk_data)
                continue

            # Handle dict-like chunk data from agentic results
            if isinstance(chunk_data, dict):
                chunk = RecordedDocumentChunk(
                    chunk_id=chunk_data.get("chunk_id", f"agentic_{i}"),
                    content=chunk_data.get("content", ""),
                    document_id=chunk_data.get("document_id", ""),
                    chunk_index=chunk_data.get("chunk_index", i),
                    page_number=chunk_data.get("page_number"),
                    metadata=chunk_data.get("metadata", {}),
                    created_at=datetime.now(timezone.utc),
                )
                doc = Document(
                    document_id=chunk_data.get("document_id", "") or f"agentic_{i}",
                    session_id=session_id,
                    filename=chunk_data.get("filename", chunk_data.get("document_name", "")),
                    file_type=chunk_data.get("file_type", ""),
                    file_size=0,
                    status=DocumentStatus.READY,
                    chunks_count=0,
                    upload_date=datetime.now(timezone.utc),
                )
                search_results.append(
                    SearchResult(
                        chunk=chunk,
                        score=chunk_data.get("score", 0.0),
                        document=doc,
                    )
                )

        # Stash the agentic pipeline's generated answer and formatted context
        # in the first result's chunk metadata.  This allows the evaluation
        # SDK to extract and evaluate the agentic answer separately.
        agent_summary = getattr(agentic_result, "agent_summary", "")
        formatted_context = getattr(agentic_result, "formatted_context", "")

        if not search_results and agent_summary:
            # Finding 3 fix: Agentic pipeline produced an answer but returned
            # zero chunks (e.g. all chunks filtered out, answer from LLM
            # knowledge).  Create a synthetic SearchResult to carry the
            # metadata so the answer is not silently lost.
            synthetic_chunk = RecordedDocumentChunk(
                chunk_id="agentic_synthetic_0",
                content="",
                document_id="",
                chunk_index=0,
                metadata={},
                created_at=datetime.now(timezone.utc),
            )
            synthetic_doc = Document(
                document_id="agentic_synthetic",
                session_id=session_id,
                filename="",
                file_type="",
                file_size=0,
                status=DocumentStatus.READY,
                chunks_count=0,
                upload_date=datetime.now(timezone.utc),
            )
            search_results.append(
                SearchResult(
                    chunk=synthetic_chunk,
                    score=0.0,
                    document=synthetic_doc,
                )
            )

        if search_results:
            first_meta = search_results[0].chunk.metadata
            first_meta["_routing_tier_used"] = "comprehensive"
            if agent_summary:
                first_meta["_agentic_answer"] = agent_summary
            if formatted_context:
                first_meta["_agentic_formatted_context"] = formatted_context

        return search_results

    # =========================================================================
    # Internal Helpers
    # =========================================================================

    @staticmethod
    def _reorder_for_context(results: List) -> List:
        """Reorder results to mitigate lost-in-the-middle effect.

        LLMs attend better to the beginning and end of context windows.
        This interleaves results so that even-indexed items (ranks 0, 2, 4, ...)
        stay at the front and odd-indexed items (ranks 1, 3, 5, ...) are
        appended in reverse order at the end. The best results thus occupy
        the high-attention positions at the start and end of the context.

        Reference: arXiv:2406.16008 (+15 percentage points improvement)
        """
        start = results[0::2]  # Ranks 0, 2, 4, ... (best at start)
        end = results[1::2]  # Ranks 1, 3, 5, ... (placed at end, reversed)
        reordered = start + end[::-1]
        logger.debug(
            f"Context reordering: {len(results)} results reordered (start={len(start)}, end={len(end)})"
        )
        return reordered

    def _fetch_chunks_by_ids(
        self,
        chunk_ids: List[str],
        session_id: str,
    ) -> List[Dict[str, Any]]:
        """Fetch chunk data from vector backend by chunk_ids.

        Used by GraphRAG to retrieve chunks found via entity traversal.
        Returns results in the same dict format as dense search hits.
        """
        if not chunk_ids:
            return []

        try:
            from lasagna.vector_backends.filters import FilterExpression

            backend = self._collection_manager.resolve_backend(session_id)
            config = self._get_session_config(session_id)
            collection_name = config.collection_name

            # Batch fetch: use IN filter for all chunk_ids, scoped to session
            chunk_filter = FilterExpression.in_list("chunk_id", list(chunk_ids))
            # Add session_id filter to prevent cross-session leakage
            session_filter = backend.session_filter(collection_name, session_id)
            filter_expr = FilterExpression.and_all([chunk_filter, session_filter])
            hits = backend.query(
                collection_name=collection_name,
                filter_expr=filter_expr,
                output_fields=["chunk_id", "document_id", "content", "metadata"],
                limit=len(chunk_ids),
            )

            results = []
            for hit in hits:
                results.append(
                    {
                        "id": hit.get("chunk_id", ""),
                        "chunk_id": hit.get("chunk_id", ""),
                        "document_id": hit.get("document_id", ""),
                        "content": hit.get("content", ""),
                        "metadata": hit.get("metadata", {}),
                        "score": 0.5,  # Neutral score; reranker will re-score
                    }
                )
            return results
        except Exception as e:
            logger.warning(f"Failed to fetch graph chunks: {e}")
            return []

    def _compress_results(
        self,
        results: List,
        query: str,
        config,
    ) -> List:
        """Compress chunk content to reduce noise and token budget.

        Two modes:
        - extractive: Keep query-relevant sentences via query-term overlap scoring.
        - abstractive: LLM summarizes each chunk to extract relevant facts.
        """
        import re as _re

        method = getattr(config, "compression_method", "extractive")
        ratio = getattr(config, "compression_ratio", 0.3)

        if method == "abstractive":
            return self._compress_abstractive(results, query, ratio)

        # Extractive: sentence-level query-term overlap filtering
        for result in results:
            original = result.content
            if not original or len(original) < 100:
                continue

            # Split into sentences (simple regex splitter)
            sentences = _re.split(r"(?<=[.!?])\s+", original)
            if len(sentences) <= 2:
                continue

            # Score sentences by query term overlap
            query_terms = set(query.lower().split())
            scored = []
            for i, sent in enumerate(sentences):
                sent_terms = set(sent.lower().split())
                overlap = len(query_terms & sent_terms)
                # Boost first sentence (often contains key info)
                boost = 1.5 if i == 0 else 1.0
                scored.append((i, sent, overlap * boost))

            # Keep top-ratio sentences in original order
            n_keep = max(1, int(len(sentences) * ratio))
            top_indices = sorted(
                sorted(range(len(scored)), key=lambda i: scored[i][2], reverse=True)[:n_keep]
            )
            result.content = " ".join(sentences[i] for i in top_indices)

        return results

    def _compress_abstractive(self, results: List, query: str, ratio: float) -> List:
        """LLM-based abstractive compression of chunk content."""
        try:
            from lasagna.providers.llm_provider import SimpleLLMProvider

            llm = SimpleLLMProvider().get_llm(temperature=0)
        except Exception as e:
            logger.warning(f"Abstractive compression failed to init LLM: {e}")
            return results

        prompt_template = (
            'Extract only the facts relevant to this query: "{query}"\n'
            "Be concise. Keep approximately {pct}% of the content.\n\n"
            "Text:\n{text}\n\n"
            "Relevant facts:"
        )

        for result in results:
            if not result.content or len(result.content) < 100:
                continue
            try:
                prompt = prompt_template.format(
                    query=query,
                    pct=int(ratio * 100),
                    text=result.content[:3000],
                )
                response = llm.invoke(prompt)
                compressed = response.content if hasattr(response, "content") else str(response)
                if compressed and len(compressed) > 20:
                    result.content = compressed
            except Exception as e:
                logger.warning(f"Abstractive compression failed for chunk {result.chunk_id}: {e}")

        return results

    @staticmethod
    def _get_session_config(session_id: Optional[str] = None) -> RAGConfig:
        """Get a session-aware RAGConfig instance."""
        if session_id:
            return rag_config.with_session(session_id)
        return rag_config

    def _maybe_reload_embedding_manager(self, config: RAGConfig) -> None:
        """Reload embedding manager if the configured model changes."""
        self._collection_manager.maybe_reload_embedding_manager(config)
        # Sync back any changes to local references
        self.embedding_manager = self._collection_manager.embedding_manager
