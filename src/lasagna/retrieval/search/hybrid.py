"""Hybrid Search for RAG System - SOTA Implementation.

This module provides hybrid search combining:
- Dense retrieval (vector similarity)
- Sparse retrieval (BM25 keyword matching)
- Reciprocal Rank Fusion (RRF) for score combination

Supports two modes:
1. **Milvus Native BM25** (Standalone/Distributed): Uses Milvus's built-in
   sparse vectors and hybrid_search with RRF ranking. More efficient and
   scalable.
2. **rank_bm25 Fallback** (Milvus Lite): Uses the rank_bm25 library for
   in-memory BM25 search. Required for Milvus Lite which doesn't support
   native full-text search.

The mode is automatically detected based on Milvus deployment type.
"""

import re
import time
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, TYPE_CHECKING
from loguru import logger

# Try importing rank_bm25 for fallback mode
try:
    from rank_bm25 import BM25Okapi

    BM25_FALLBACK_AVAILABLE = True
except ImportError:
    BM25_FALLBACK_AVAILABLE = False
    logger.info("rank_bm25 not installed. Will use Milvus native BM25 if available.")

# Try importing NLTK stemmer for BM25 improvement
try:
    from nltk.stem import PorterStemmer

    STEMMER_AVAILABLE = True
except ImportError:
    STEMMER_AVAILABLE = False
    logger.warning(
        "nltk is not installed, so BM25 runs without stemming; BM25 and hybrid "
        "scores differ from a run with it (pip install 'lasagna-rag[eval]')."
    )

# Try importing Milvus hybrid search components
try:
    from pymilvus import AnnSearchRequest, RRFRanker, WeightedRanker  # noqa: F401

    MILVUS_HYBRID_AVAILABLE = True
except ImportError:
    MILVUS_HYBRID_AVAILABLE = False
    logger.info("Milvus hybrid search components not available.")

from lasagna.core.model_licenses import warn_if_restricted
from lasagna.core.config import rag_config

if TYPE_CHECKING:
    from pymilvus import MilvusClient


class SPLADESparseEncoder:
    """SPLADE-max learned sparse encoder for query-time scoring.

    Encodes queries into sparse term weights using a masked language model.
    At query time, scores documents via dot-product of shared terms.
    Replaces BM25 in the RRF merge for semantically-aware sparse retrieval.
    """

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self._model_name = model_name
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _ensure_loaded(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from transformers import AutoModelForMaskedLM, AutoTokenizer

                        warn_if_restricted(self._model_name, component="SPLADE sparse encoder")
                        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                        self._model = AutoModelForMaskedLM.from_pretrained(self._model_name)
                        self._model.eval()
                        logger.info(f"SPLADE model loaded: {self._model_name}")
                    except ImportError:
                        raise ImportError(
                            "transformers is required for SPLADE. "
                            "Install with: pip install transformers"
                        )

    def encode_query(self, query: str) -> Dict[str, float]:
        """Encode query to sparse {term: weight} mapping."""
        self._ensure_loaded()
        import torch

        inputs = self._tokenizer(query, return_tensors="pt", truncation=True, max_length=256)
        with torch.no_grad():
            logits = self._model(**inputs).logits
        weights = torch.log1p(torch.relu(logits)).max(dim=1).values.squeeze()
        non_zero = (weights > 0).nonzero().squeeze(-1)
        return {self._tokenizer.decode([int(idx)]): float(weights[idx]) for idx in non_zero}

    def score_documents(
        self, query_sparse: Dict[str, float], documents: List[Dict[str, Any]]
    ) -> List[Tuple[int, float]]:
        """Score documents against sparse query. Returns (doc_idx, score) tuples."""
        results = []
        for idx, doc in enumerate(documents):
            text = doc.get("content", "")
            if not text:
                continue
            doc_terms = set(text.lower().split())
            score = sum(w for term, w in query_sparse.items() if term.lower() in doc_terms)
            if score > 0:
                results.append((idx, score))
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results


# Module-level SPLADE encoder singleton (lazy-loaded)
_splade_encoder: Optional[SPLADESparseEncoder] = None
_splade_lock = threading.Lock()


def _get_splade_encoder() -> SPLADESparseEncoder:
    """Get or create the SPLADE encoder singleton."""
    global _splade_encoder
    if _splade_encoder is None:
        with _splade_lock:
            if _splade_encoder is None:
                _splade_encoder = SPLADESparseEncoder()
    return _splade_encoder


def _escape_milvus_value(s: str) -> str:
    """Escape a string value for use in Milvus filter expressions.

    For the native hybrid search path which requires raw filter strings
    (AnnSearchRequest cannot use FilterExpression). For all other filter
    construction, use FilterExpression from vector_backends.filters.
    """
    return s.replace("\\", "\\\\").replace('"', '\\"')


@dataclass
class HybridSearchResult:
    """Result from hybrid search with combined scores."""

    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    dense_score: float = 0.0
    sparse_score: float = 0.0
    combined_score: float = 0.0
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    # P1-3 FIX: Store pre-fetched embedding to avoid re-embedding in MMR
    embedding: Optional[List[float]] = None
    # Backend highlight fragments (e.g. from OpenSearch BM25 matching)
    highlights: Optional[List[str]] = None


class MilvusNativeHybridSearcher:
    """Hybrid search using Milvus native sparse vectors and hybrid_search.

    This is the preferred method when using Milvus Standalone or Distributed.
    It uses Milvus's built-in BM25 function and RRF ranking for efficient
    hybrid search without needing external libraries.

    Note: NOT supported in Milvus Lite.
    """

    def __init__(
        self, milvus_client: "MilvusClient", collection_name: str, alpha: Optional[float] = None
    ):
        """Initialize native hybrid searcher.

        Args:
            milvus_client: Milvus client instance
            collection_name: Name of the collection
            alpha: Weight for dense vs sparse (1.0 = pure dense, 0.0 = pure sparse)
        """
        self.client = milvus_client
        self.collection_name = collection_name
        self.alpha = alpha if alpha is not None else rag_config.hybrid_alpha
        self._native_bm25_supported: Optional[bool] = None

    def is_native_bm25_supported(self) -> bool:
        """Check if native BM25 is supported by the current Milvus deployment.

        Returns:
            True if native BM25/hybrid search is available
        """
        if self._native_bm25_supported is not None:
            return self._native_bm25_supported

        if not MILVUS_HYBRID_AVAILABLE:
            self._native_bm25_supported = False
            return False

        # Check if collection has sparse_embedding field
        try:
            collection_info = self.client.describe_collection(self.collection_name)
            fields = collection_info.get("fields", [])
            has_sparse = any(f.get("name") == "sparse_embedding" for f in fields)
            self._native_bm25_supported = has_sparse
            if has_sparse:
                logger.info("Native Milvus BM25 hybrid search is available")
            else:
                logger.info("Collection doesn't have sparse_embedding field")
            return has_sparse
        except Exception as e:
            logger.debug(f"Could not check native BM25 support: {e}")
            self._native_bm25_supported = False
            return False

    def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        session_id: str,
        top_k: int = 10,
        document_ids: Optional[List[str]] = None,
        config=None,
    ) -> List[HybridSearchResult]:
        """Perform native Milvus hybrid search.

        Args:
            query: Raw text query (for BM25)
            query_embedding: Dense embedding vector
            session_id: Session filter
            top_k: Number of results
            document_ids: Optional document filter

        Returns:
            List of hybrid search results
        """
        if not self.is_native_bm25_supported():
            logger.warning("Native BM25 not supported, returning empty results")
            return []

        # Build filter expression - use double quotes for JSON key access in Milvus
        # Note: config is resolved first since we need max_document_ids_filter
        config = config or rag_config
        # Validate session_id format before using in filter expressions
        if not re.match(r"^[a-zA-Z0-9_\-.:]+$", session_id):
            raise ValueError("Invalid session_id format: contains disallowed characters")
        # Escape values to prevent filter expression injection
        filters = [f'metadata["session_id"] == "{_escape_milvus_value(session_id)}"']
        if document_ids:
            limited_doc_ids = document_ids
            max_doc_ids = config.max_document_ids_filter
            if len(document_ids) > max_doc_ids:
                logger.warning(
                    f"Native hybrid search: document ID filter truncated: {len(document_ids)} -> {max_doc_ids}"
                )
                limited_doc_ids = document_ids[:max_doc_ids]
            doc_filter = " || ".join(
                [f'document_id == "{_escape_milvus_value(d)}"' for d in limited_doc_ids]
            )
            filters.append(f"({doc_filter})")
        filter_expr = " && ".join(filters)

        # Dense search request

        dense_request = AnnSearchRequest(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": config.metric_type, **config.search_params},
            limit=top_k * 2,  # Get more for fusion
        )

        # Sparse BM25 search request
        sparse_request = AnnSearchRequest(
            data=[query],  # Raw text for BM25
            anns_field="sparse_embedding",
            param={"metric_type": "BM25"},
            limit=top_k * 2,
        )

        # Use RRF ranker for fusion
        ranker = RRFRanker()

        try:
            results = self.client.hybrid_search(
                collection_name=self.collection_name,
                reqs=[dense_request, sparse_request],
                ranker=ranker,
                filter=filter_expr,
                limit=top_k,
                output_fields=["document_id", "content", "metadata"],
            )

            # Convert to HybridSearchResult
            # NOTE: In native hybrid mode, Milvus returns RRF-fused results
            # Individual dense/sparse ranks are not available - set to None
            # The combined_score reflects the RRF fusion ranking
            hybrid_results = []
            for rank, hit in enumerate(results[0] if results else []):
                hybrid_results.append(
                    HybridSearchResult(
                        chunk_id=str(hit.get("id") or hit.get("chunk_id") or ""),
                        document_id=hit.get("entity", {}).get("document_id", ""),
                        content=hit.get("entity", {}).get("content", ""),
                        metadata=hit.get("entity", {}).get("metadata", {}),
                        combined_score=hit.get("distance", 0.0),
                        # In native mode, we only have the fused rank, not individual ranks
                        dense_rank=None,  # Not available from native hybrid search
                        sparse_rank=None,  # Not available from native hybrid search
                    )
                )

            logger.debug(f"Native hybrid search returned {len(hybrid_results)} results")
            return hybrid_results

        except Exception as e:
            logger.error(f"Native hybrid search failed: {e}")
            return []


class BM25FallbackIndex:
    """BM25 sparse retrieval index using rank_bm25 library.

    This is the fallback for Milvus Lite which doesn't support native BM25.
    Maintains an in-memory BM25 index for keyword-based search.

    Features:
    - Optional Porter stemming for better recall
    - Lazy index building for efficiency
    """

    def __init__(self, enable_stemming: bool = True):
        """Initialize empty BM25 index.

        Args:
            enable_stemming: Whether to apply Porter stemming (improves recall)
        """
        self.documents: List[Dict[str, Any]] = []
        self.tokenized_corpus: List[List[str]] = []
        self.bm25: Optional[BM25Okapi] = None
        self._is_built = False

        # Version counter for CAS-safe concurrent builds.
        # Incremented on every mutation (add/remove/clear).
        # build_index() snapshots _version before building and only commits
        # if _version is unchanged — prevents stale-built state when a
        # concurrent thread adds documents during the build.
        self._version: int = 0
        self._built_version: int = -1

        # Per-index lock that serializes build_index() calls so only one
        # thread does the expensive BM25Okapi construction at a time.
        # This is separate from HybridSearcher._session_lock (which
        # protects the session map) — held only during build, not during
        # add_documents or search.
        self._build_lock = threading.Lock()

        # Observability counters
        self._build_count: int = 0
        self._total_build_ms: float = 0.0

        # Stemming support
        self._enable_stemming = enable_stemming and STEMMER_AVAILABLE
        self._stemmer = PorterStemmer() if self._enable_stemming else None
        if self._enable_stemming:
            logger.debug("BM25 Porter stemming enabled")

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text for BM25 with optional stemming.

        Args:
            text: Text to tokenize

        Returns:
            List of tokens (stemmed if enabled)
        """
        # Basic tokenization
        tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())
        tokens = [t for t in tokens if len(t) > 1 or t.isdigit()]

        # Apply stemming if enabled
        if self._enable_stemming and self._stemmer:
            tokens = [self._stemmer.stem(t) for t in tokens]

        return tokens

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the index.

        Args:
            documents: List of document dicts with 'content' field
        """
        for doc in documents:
            content = doc.get("content", "")
            tokens = self.tokenize(content)
            self.documents.append(doc)
            self.tokenized_corpus.append(tokens)
        self._version += 1
        self._is_built = False

    def build_index(self) -> bool:
        """Build the BM25 index from added documents.

        Serialized by ``_build_lock`` so only one thread performs the
        expensive BM25Okapi construction at a time.  Uses a CAS
        (compare-and-swap) pattern on ``_version``: snapshots before
        building and only commits if no concurrent mutation occurred.

        Returns:
            True if the index was successfully built and committed,
            False if skipped (no data, unavailable, already up-to-date,
            another thread is building) or CAS rejected (stale).
        """
        if not BM25_FALLBACK_AVAILABLE:
            logger.debug("rank_bm25 not available - index not built")
            return False

        if not self.tokenized_corpus:
            logger.debug("No documents to index")
            return False

        # Non-blocking acquire: if another thread already holds the lock
        # (i.e., is building), skip immediately instead of blocking all
        # search threads behind an expensive build.
        acquired = self._build_lock.acquire(blocking=False)
        if not acquired:
            logger.debug("BM25 build already in progress — skipping")
            return False

        try:
            # Re-check under lock: another thread may have completed
            # a build while we were waiting / between the outer check
            # and acquiring the lock.
            if self.is_current():
                return False

            snap_version = self._version
            t0 = time.perf_counter()
            bm25 = BM25Okapi(self.tokenized_corpus)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            if self._version == snap_version:
                # No concurrent mutations — commit the build.
                self.bm25 = bm25
                self._is_built = True
                self._built_version = snap_version
                self._build_count += 1
                self._total_build_ms += elapsed_ms
                logger.debug(
                    f"BM25 fallback index built with {len(self.documents)} documents "
                    f"({elapsed_ms:.1f}ms, build #{self._build_count})"
                )
                return True
            else:
                # Concurrent mutation detected — discard stale build.
                logger.debug(
                    f"BM25 build discarded: version changed {snap_version}→{self._version} "
                    f"during {elapsed_ms:.1f}ms build"
                )
                return False
        finally:
            self._build_lock.release()

    def search(self, query: str, top_k: int = 20) -> List[Tuple[int, float]]:
        """Search the BM25 index.

        Args:
            query: Search query
            top_k: Number of results

        Returns:
            List of (doc_index, score) tuples
        """
        if not BM25_FALLBACK_AVAILABLE or not self._is_built or not self.bm25:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        scores = self.bm25.get_scores(query_tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        return [(idx, scores[idx]) for idx in top_indices if scores[idx] > 0]

    def remove_document(self, document_id: str) -> int:
        """Remove all chunks for a document from the index.

        Args:
            document_id: Document ID to remove

        Returns:
            Number of chunks removed
        """
        if not self.documents:
            return 0

        # Find indices to remove
        indices_to_remove = [
            i for i, doc in enumerate(self.documents) if doc.get("document_id") == document_id
        ]

        if not indices_to_remove:
            return 0

        # Remove in reverse order to maintain indices
        for idx in reversed(indices_to_remove):
            del self.documents[idx]
            del self.tokenized_corpus[idx]

        # Invalidate the BM25 index object to free memory
        self._version += 1
        self.bm25 = None
        self._is_built = False

        return len(indices_to_remove)

    def clear(self):
        """Clear the index."""
        self.documents = []
        self.tokenized_corpus = []
        self.bm25 = None
        self._version += 1
        self._built_version = -1
        self._is_built = False

    def is_current(self) -> bool:
        """Check if the built index reflects all mutations.

        Returns True if the index is built and no documents have been
        added, removed, or cleared since the last successful build.
        """
        return self._is_built and self._built_version == self._version


class HybridSearcher:
    """Unified hybrid search combining dense and sparse retrieval.

    Automatically selects the best available method:
    1. Milvus native hybrid search (if available)
    2. rank_bm25 fallback (for Milvus Lite)

    Uses Reciprocal Rank Fusion (RRF) to combine results.
    """

    # Session cleanup configuration (defaults; overridden by config at __init__)
    SESSION_MAX_AGE_SECONDS = 3600 * 24  # 24 hours absolute max age from creation
    CLEANUP_INTERVAL_SECONDS = 3600  # Run periodic cleanup every hour

    def __init__(
        self,
        alpha: Optional[float] = None,
        rrf_k: Optional[int] = None,
        max_sessions: Optional[int] = None,
        session_ttl_seconds: Optional[int] = None,
        enable_periodic_cleanup: bool = True,
    ):
        """Initialize hybrid searcher.

        Args:
            alpha: Weight for dense vs sparse (1.0 = pure dense, 0.0 = pure sparse)
            rrf_k: RRF constant (higher = more weight to lower ranks, default from config)
            max_sessions: Maximum number of session indices to keep in memory (default from config)
            session_ttl_seconds: Time-to-live for session indices in seconds (default from config)
            enable_periodic_cleanup: Enable automatic periodic cleanup of expired sessions
        """
        self.alpha = alpha if alpha is not None else rag_config.hybrid_alpha
        self.rrf_k = rrf_k if rrf_k is not None else rag_config.rrf_k
        self.max_sessions = (
            max_sessions if max_sessions is not None else rag_config.bm25_max_sessions
        )
        self.session_ttl_seconds = (
            session_ttl_seconds
            if session_ttl_seconds is not None
            else rag_config.bm25_session_ttl_seconds
        )

        # Fallback BM25 indices per session with access/creation timestamps
        self._session_indices: Dict[str, BM25FallbackIndex] = {}
        self._session_timestamps: Dict[str, float] = {}  # session_id -> last_access_time
        self._session_creation_times: Dict[str, float] = {}  # session_id -> creation_time
        self._session_lock = threading.Lock()  # Thread-safe session management
        self._last_cleanup_time = time.time()  # MEDIUM PRIORITY FIX: Track last cleanup

        # Native searcher (initialized lazily) - for Milvus-specific hybrid search
        self._native_searcher: Optional[MilvusNativeHybridSearcher] = None

        # Vector backend reference for backend-agnostic hybrid search (BYOVDB)
        self._vector_backend: Optional[Any] = None
        self._collection_name: Optional[str] = None

        # Per-session backend overrides for multi-backend support
        # session_id -> (backend, collection_name)
        self._session_backends: Dict[str, tuple] = {}

        # MEDIUM PRIORITY FIX: Periodic cleanup configuration
        self._enable_periodic_cleanup = enable_periodic_cleanup

    def set_milvus_client(self, client: "MilvusClient", collection_name: str):
        """Set Milvus client for native hybrid search.

        DEPRECATED: Use set_vector_backend instead for BYOVDB support.

        Args:
            client: Milvus client instance (or VectorStoreAdapter)
            collection_name: Collection name
        """
        self._native_searcher = MilvusNativeHybridSearcher(
            milvus_client=client, collection_name=collection_name, alpha=self.alpha
        )

    def set_vector_backend(self, backend: Any, collection_name: str):
        """Set vector backend for hybrid search (BYOVDB support).

        This method supports any VectorStoreAdapter backend. If the backend
        supports native hybrid search (via capabilities), it will be used.
        Otherwise, falls back to in-memory BM25.

        Args:
            backend: VectorStoreAdapter instance
            collection_name: Collection name
        """
        self._vector_backend = backend
        self._collection_name = collection_name

        # Check if backend supports native hybrid search via capabilities
        if hasattr(backend, "capabilities"):
            caps = backend.capabilities
            if caps.supports_hybrid_search:
                logger.info(f"Backend '{backend.backend_name}' supports native hybrid search")
            elif caps.supports_native_bm25:
                logger.info(f"Backend '{backend.backend_name}' supports native BM25")
            else:
                logger.info(
                    f"Backend '{backend.backend_name}' does not support native hybrid search, "
                    f"using in-memory BM25 fallback"
                )

        # Only initialize native Milvus hybrid searcher for Milvus backends
        # Non-Milvus backends use in-memory BM25 fallback
        if hasattr(backend, "backend_name") and backend.backend_name == "milvus":
            self._native_searcher = MilvusNativeHybridSearcher(
                milvus_client=backend, collection_name=collection_name, alpha=self.alpha
            )
        else:
            self._native_searcher = None
            logger.info(
                f"Skipping MilvusNativeHybridSearcher for '{getattr(backend, 'backend_name', 'unknown')}' backend"
            )

    def set_session_backend(
        self,
        session_id: str,
        backend: Any,
        collection_name: str,
    ):
        """Set a session-specific vector backend for hybrid search.

        When a session uses a different backend than the default, this method
        registers it so hybrid search uses the correct backend for that session.

        Args:
            session_id: Session identifier
            backend: VectorStoreAdapter instance for this session
            collection_name: Collection name on this backend
        """
        with self._session_lock:
            self._session_backends[session_id] = (backend, collection_name)
            logger.debug(
                f"Registered session backend: session={session_id[:8]}..., "
                f"backend={getattr(backend, 'backend_name', 'unknown')}"
            )

    def _get_session_backend(self, session_id: Optional[str] = None) -> tuple:
        """Get the (backend, collection_name) for a session.

        Returns session-specific backend if registered, else the default.
        """
        if session_id:
            with self._session_lock:
                override = self._session_backends.get(session_id)
                if override:
                    return override
        return (self._vector_backend, self._collection_name)

    def _backend_supports_native_hybrid(self, session_id: Optional[str] = None) -> bool:
        """Check if the session's backend supports native hybrid search.

        Args:
            session_id: Optional session identifier (for per-session backend lookup)

        Returns:
            True if native hybrid search is available via the backend
        """
        backend, _ = self._get_session_backend(session_id)
        if backend is None:
            return False

        if hasattr(backend, "capabilities"):
            caps = backend.capabilities
            return caps.supports_hybrid_search or caps.supports_native_bm25

        return False

    def is_native_mode_available(self) -> bool:
        """Check if native Milvus hybrid search is available.

        Returns:
            True if native mode can be used
        """
        if self._native_searcher is None:
            return False
        return self._native_searcher.is_native_bm25_supported()

    def _get_or_create_session_locked(self, session_id: str) -> BM25FallbackIndex:
        """Get or create a session index (must hold _session_lock).

        Handles periodic cleanup, TTL expiry, and max-session enforcement.
        Shared by get_session_index() and index_chunks() to ensure all
        session creation paths respect the same limits.

        Args:
            session_id: Session ID

        Returns:
            BM25FallbackIndex for the session
        """
        current_time = time.time()

        # Run periodic cleanup regardless of whether session exists
        if self._enable_periodic_cleanup:
            if current_time - self._last_cleanup_time >= self.CLEANUP_INTERVAL_SECONDS:
                removed = self._cleanup_expired_sessions_locked(current_time)
                self._last_cleanup_time = current_time
                if removed > 0:
                    logger.info(f"Periodic session cleanup removed {removed} expired indices")

        # Update timestamp for existing session or create new one
        if session_id in self._session_indices:
            self._session_timestamps[session_id] = current_time
            return self._session_indices[session_id]

        # Cleanup expired sessions and enforce max limit before creating new
        self._cleanup_expired_sessions_locked(current_time)

        # Create new session index with stemming from config
        enable_stemming = rag_config.enable_bm25_stemming
        self._session_indices[session_id] = BM25FallbackIndex(enable_stemming=enable_stemming)
        self._session_timestamps[session_id] = current_time
        self._session_creation_times[session_id] = current_time
        return self._session_indices[session_id]

    def get_session_index(self, session_id: str) -> BM25FallbackIndex:
        """Get or create BM25 fallback index for a session.

        Thread-safe with automatic cleanup of expired/excess sessions.

        Args:
            session_id: Session ID

        Returns:
            BM25FallbackIndex for the session
        """
        with self._session_lock:
            return self._get_or_create_session_locked(session_id)

    def _cleanup_expired_sessions_locked(self, current_time: float) -> int:
        """Remove expired sessions (must hold _session_lock).

        Args:
            current_time: Current timestamp

        Returns:
            Number of sessions removed
        """
        removed = 0
        idle_threshold = current_time - self.session_ttl_seconds
        max_age_threshold = current_time - self.SESSION_MAX_AGE_SECONDS

        # Find sessions expired by idle TTL or absolute max age
        expired_sessions = [
            sid
            for sid, ts in self._session_timestamps.items()
            if ts < idle_threshold or self._session_creation_times.get(sid, 0) < max_age_threshold
        ]

        for sid in expired_sessions:
            del self._session_indices[sid]
            del self._session_timestamps[sid]
            self._session_creation_times.pop(sid, None)
            self._session_backends.pop(sid, None)
            removed += 1

        # If still over limit, remove oldest sessions
        while len(self._session_indices) >= self.max_sessions:
            oldest_sid = min(
                self._session_timestamps.keys(), key=lambda k: self._session_timestamps[k]
            )
            del self._session_indices[oldest_sid]
            del self._session_timestamps[oldest_sid]
            self._session_creation_times.pop(oldest_sid, None)
            self._session_backends.pop(oldest_sid, None)
            removed += 1

        if removed > 0:
            logger.debug(f"Cleaned up {removed} expired session indices")

        return removed

    def cleanup_expired_sessions(self) -> int:
        """Manually trigger cleanup of expired session indices.

        Returns:
            Number of sessions removed
        """
        with self._session_lock:
            return self._cleanup_expired_sessions_locked(time.time())

    def get_session_count(self) -> int:
        """Get current number of active session indices.

        Returns:
            Number of session indices in memory
        """
        with self._session_lock:
            return len(self._session_indices)

    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        session_id: str,
        *,
        build_now: bool = True,
    ):
        """Index chunks for BM25 fallback search.

        Only needed when using rank_bm25 fallback (Milvus Lite).
        For native Milvus BM25, sparse vectors are indexed at insert time.

        Args:
            chunks: List of chunk dicts with content and metadata
            session_id: Session ID for the chunks
            build_now: If True (default), build BM25 index immediately after
                adding documents. If False, defer the build — the index will
                be built lazily on the first search via _ensure_index_built(),
                or explicitly via build_session_index(). Use False during bulk
                ingestion to avoid O(N²) repeated rebuilds.
        """
        if self.is_native_mode_available():
            logger.debug("Using native BM25 - no fallback indexing needed")
            return

        if not BM25_FALLBACK_AVAILABLE:
            logger.debug("BM25 fallback not available - skipping indexing")
            return

        # Hold _session_lock for get-or-create + add_documents atomically
        # to prevent concurrent uploads from interleaving list mutations.
        # Uses _get_or_create_session_locked to share cleanup/max-session
        # enforcement with get_session_index().
        with self._session_lock:
            index = self._get_or_create_session_locked(session_id)
            index.add_documents(chunks)

        if build_now:
            # Build outside lock — expensive BM25Okapi construction.
            # CAS in build_index() prevents stale-built state.
            index.build_index()

        if build_now:
            logger.info(f"Indexed {len(chunks)} chunks for session {session_id} (fallback mode)")
        else:
            logger.debug(f"Appended {len(chunks)} chunks for session {session_id} (deferred build)")

    def search_hybrid(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        session_id: str,
        top_k: int = 10,
        query_embedding: Optional[List[float]] = None,
        document_ids: Optional[List[str]] = None,
        return_metrics: bool = False,
        config=None,
        partition_key: Optional[str] = None,
        **kwargs,
    ) -> List[HybridSearchResult]:
        """Perform hybrid search combining dense and sparse results.

        Args:
            query: Search query
            dense_results: Results from dense vector search
            session_id: Session ID
            top_k: Number of final results
            query_embedding: Query embedding (for native mode)
            document_ids: Optional document filter
            alpha_override: Optional override for hybrid alpha (from query classification)

        Returns:
            Combined and re-ranked results
        """
        metrics = {
            "sparse_ms": 0.0,
            "merge_ms": 0.0,
            "hybrid_mode": "dense_only",
        }

        config = config or rag_config
        # Support alpha_override for adaptive query-type classification
        # Use `is not None` to allow alpha_override=0.0 (pure BM25)
        alpha_override = kwargs.get("alpha_override")
        alpha = alpha_override if alpha_override is not None else config.hybrid_alpha

        # If alpha is 1.0, skip BM25 entirely
        if alpha >= 1.0 or not config.enable_hybrid_search:
            results = self._convert_dense_results(dense_results)
            return (results, metrics) if return_metrics else results

        # Try backend-agnostic native hybrid search first (BYOVDB).
        # Skip native path when SPLADE is requested — native backends use
        # their own BM25, so SPLADE can only run via the fallback path.
        _effective_sparse = getattr(config, "sparse_method", "bm25") if config else "bm25"
        if (
            self._backend_supports_native_hybrid(session_id)
            and query_embedding is not None
            and _effective_sparse != "splade"
        ):
            backend, coll_name = self._get_session_backend(session_id)
            metrics["hybrid_mode"] = f"native_{backend.backend_name}"
            timer = time.perf_counter()

            # Build filter expression for session/document filtering
            # Use backend-appropriate filter syntax (Milvus stores session_id in JSON metadata)
            from lasagna.vector_backends.filters import FilterExpression

            if backend.backend_name == "milvus":
                filters = [FilterExpression.json_eq("metadata", "session_id", session_id)]
            else:
                filters = [FilterExpression.eq("session_id", session_id)]
            if document_ids:
                # Limit document IDs to prevent filter overflow
                max_doc_ids = config.max_document_ids_filter
                limited_doc_ids = (
                    document_ids[:max_doc_ids] if len(document_ids) > max_doc_ids else document_ids
                )
                if len(document_ids) > max_doc_ids:
                    logger.warning(
                        f"Hybrid search: document ID filter truncated: {len(document_ids)} -> {max_doc_ids}"
                    )
                filters.append(FilterExpression.in_list("document_id", limited_doc_ids))

            filter_expr = FilterExpression.and_all(filters)

            try:
                # Compute sparse query tokens for 3-way hybrid when available
                hybrid_kwargs: Dict[str, Any] = {}
                if backend.backend_name == "opensearch":
                    try:
                        from lasagna.vector_backends.opensearch_backend import (
                            _neural_sparse_enabled,
                        )

                        embedding_mgr = kwargs.get("embedding_manager")
                        if (
                            _neural_sparse_enabled()
                            and embedding_mgr
                            and hasattr(embedding_mgr, "encode_sparse")
                        ):
                            sparse_result = embedding_mgr.encode_sparse([query])
                            if sparse_result and sparse_result[0]:
                                hybrid_kwargs["query_tokens"] = sparse_result[0]
                    except Exception:
                        pass  # Fall through to 2-way hybrid

                backend_results = backend.hybrid_search(
                    collection_name=coll_name,
                    query_text=query,
                    query_embedding=query_embedding,
                    filter_expr=filter_expr,
                    limit=top_k,
                    alpha=alpha,
                    output_fields=["document_id", "content", "metadata", "embedding"],
                    partition_key=partition_key,
                    **hybrid_kwargs,
                )

                # Convert VectorSearchResult to HybridSearchResult
                results = []
                for r in backend_results:
                    results.append(
                        HybridSearchResult(
                            chunk_id=str(r.id),
                            document_id=r.document_id,
                            content=r.content,
                            metadata=r.metadata,
                            combined_score=r.score,
                            embedding=r.embedding,
                            highlights=r.entity.get("highlights"),
                        )
                    )

                metrics["sparse_ms"] = (time.perf_counter() - timer) * 1000.0
                return (results, metrics) if return_metrics else results

            except Exception as e:
                logger.warning(f"Backend hybrid search failed, falling back: {e}")
                # Fall through to legacy native or BM25 fallback

        # Legacy: Try Milvus native mode
        if self.is_native_mode_available() and query_embedding is not None:
            metrics["hybrid_mode"] = "native_milvus"
            timer = time.perf_counter()
            results = self._native_searcher.hybrid_search(
                query=query,
                query_embedding=query_embedding,
                session_id=session_id,
                top_k=top_k,
                document_ids=document_ids,
                config=config,
            )
            metrics["sparse_ms"] = (time.perf_counter() - timer) * 1000.0
            return (results, metrics) if return_metrics else results

        # Fallback to rank_bm25 (or SPLADE)
        results, fallback_metrics = self._fallback_hybrid_search(
            query=query,
            dense_results=dense_results,
            session_id=session_id,
            top_k=top_k,
            alpha=alpha,
            document_ids=document_ids,
            config=config,
        )
        metrics.update(fallback_metrics)
        return (results, metrics) if return_metrics else results

    def _fallback_hybrid_search(
        self,
        query: str,
        dense_results: List[Dict[str, Any]],
        session_id: str,
        top_k: int,
        alpha: float,
        document_ids: Optional[List[str]] = None,
        config=None,
    ) -> tuple[List[HybridSearchResult], Dict[str, Any]]:
        """Fallback hybrid search using rank_bm25 or SPLADE.

        Args:
            query: Search query
            dense_results: Dense search results
            session_id: Session ID
            top_k: Number of results
            document_ids: Optional document filter

        Returns:
            Combined results
        """
        metrics = {
            "sparse_ms": 0.0,
            "merge_ms": 0.0,
            "hybrid_mode": "rank_bm25_fallback",
        }
        # Lazy rebuild: if a document was removed, the index was invalidated.
        # Rebuild before checking _is_built to avoid permanent fallback.
        self._ensure_index_built(session_id)
        index = self.get_session_index(session_id)
        if not index._is_built:
            logger.debug("BM25 index not built - using dense only")
            return self._convert_dense_results(dense_results), metrics

        sparse_timer = time.perf_counter()
        # Use SPLADE or BM25 based on effective config (per-request, not global)
        _effective_config = config or rag_config
        _sparse_method = getattr(_effective_config, "sparse_method", "bm25")
        if _sparse_method == "splade":
            try:
                splade = _get_splade_encoder()
                query_sparse = splade.encode_query(query)
                sparse_results = splade.score_documents(query_sparse, index.documents)
                sparse_results = sparse_results[: top_k * 2]
                metrics["hybrid_mode"] = "splade_fallback"
            except Exception as e:
                logger.warning(f"SPLADE fallback failed, using BM25: {e}")
                sparse_results = index.search(query, top_k=top_k * 2)
        else:
            sparse_results = index.search(query, top_k=top_k * 2)
        metrics["sparse_ms"] = (time.perf_counter() - sparse_timer) * 1000.0

        # Filter sparse results by document_ids BEFORE fusion to avoid
        # wasting RRF computation on results that will be discarded
        if document_ids:
            doc_id_set = set(document_ids)
            sparse_results = [
                (idx, score)
                for idx, score in sparse_results
                if idx < len(index.documents)
                and index.documents[idx].get("document_id") in doc_id_set
            ]

        # Combine using RRF
        merge_timer = time.perf_counter()
        combined = self._reciprocal_rank_fusion(
            dense_results,
            sparse_results,
            index.documents,
            alpha=alpha,
            rrf_k=getattr(config, "rrf_k", None) if config else None,
        )
        metrics["merge_ms"] = (time.perf_counter() - merge_timer) * 1000.0

        combined.sort(key=lambda x: x.combined_score, reverse=True)

        logger.debug(
            f"Fallback hybrid search: {len(dense_results)} dense + "
            f"{len(sparse_results)} sparse -> {len(combined[:top_k])} combined"
        )

        return combined[:top_k], metrics

    def _reciprocal_rank_fusion(
        self,
        dense_results: List[Dict[str, Any]],
        sparse_results: List[Tuple[int, float]],
        indexed_docs: List[Dict[str, Any]],
        alpha: float,
        rrf_k: Optional[int] = None,
    ) -> List[HybridSearchResult]:
        """Combine results using Reciprocal Rank Fusion.

        RRF score = sum(1 / (k + rank)) for each retrieval method

        Args:
            dense_results: Dense search results
            sparse_results: Sparse search results (index, score)
            indexed_docs: Original indexed documents

        Returns:
            Combined results with RRF scores
        """
        results_map: Dict[str, HybridSearchResult] = {}

        # Process dense results
        for rank, result in enumerate(dense_results):
            # Use explicit None check — id=0 is valid (Milvus auto-ID)
            chunk_id = result.get("id")
            if chunk_id is None:
                chunk_id = result.get("chunk_id")
            if chunk_id is None:
                continue
            chunk_id = str(chunk_id)

            dense_score = result.get("score", result.get("distance", 0.0))
            # P1-3 FIX: Preserve embedding from Milvus result
            embedding = result.get("entity", {}).get("embedding")

            if chunk_id not in results_map:
                results_map[chunk_id] = HybridSearchResult(
                    chunk_id=chunk_id,
                    document_id=result.get("entity", {}).get("document_id", ""),
                    content=result.get("entity", {}).get("content", ""),
                    metadata=result.get("entity", {}).get("metadata", {}),
                    dense_score=dense_score,
                    dense_rank=rank,
                    embedding=embedding,
                )
            else:
                results_map[chunk_id].dense_score = dense_score
                results_map[chunk_id].dense_rank = rank
                # Keep embedding if we already have one from dense results
                if embedding and not results_map[chunk_id].embedding:
                    results_map[chunk_id].embedding = embedding

        # Process sparse results
        for rank, (doc_idx, score) in enumerate(sparse_results):
            if doc_idx >= len(indexed_docs):
                continue

            doc = indexed_docs[doc_idx]
            chunk_id = doc.get("chunk_id")
            if chunk_id is None:
                continue
            chunk_id = str(chunk_id)

            if chunk_id not in results_map:
                results_map[chunk_id] = HybridSearchResult(
                    chunk_id=chunk_id,
                    document_id=doc.get("document_id", ""),
                    content=doc.get("content", ""),
                    metadata=doc.get("metadata", {}),
                    sparse_score=score,
                    sparse_rank=rank,
                )
            else:
                results_map[chunk_id].sparse_score = score
                results_map[chunk_id].sparse_rank = rank

        # Calculate combined RRF scores
        _rrf_k = rrf_k if rrf_k is not None else self.rrf_k
        for result in results_map.values():
            dense_rrf = 0.0
            sparse_rrf = 0.0

            if result.dense_rank is not None:
                dense_rrf = 1.0 / (_rrf_k + result.dense_rank)

            if result.sparse_rank is not None:
                sparse_rrf = 1.0 / (_rrf_k + result.sparse_rank)

            result.combined_score = alpha * dense_rrf + (1 - alpha) * sparse_rrf

        return list(results_map.values())

    def _convert_dense_results(
        self, dense_results: List[Dict[str, Any]]
    ) -> List[HybridSearchResult]:
        """Convert dense results to HybridSearchResult format.

        Args:
            dense_results: Dense search results

        Returns:
            List of HybridSearchResult
        """
        results = []
        for rank, result in enumerate(dense_results):
            # Use explicit None check — id=0 is valid (Milvus auto-ID)
            chunk_id = result.get("id")
            if chunk_id is None:
                chunk_id = result.get("chunk_id")
            chunk_id = str(chunk_id) if chunk_id is not None else ""
            dense_score = result.get("score", result.get("distance", 0.0))
            # P1-3 FIX: Preserve embedding from Milvus result to avoid re-embedding
            embedding = result.get("entity", {}).get("embedding")

            results.append(
                HybridSearchResult(
                    chunk_id=chunk_id,
                    document_id=result.get("entity", {}).get("document_id", ""),
                    content=result.get("entity", {}).get("content", ""),
                    metadata=result.get("entity", {}).get("metadata", {}),
                    dense_score=dense_score,
                    sparse_score=0.0,
                    combined_score=dense_score,
                    dense_rank=rank,
                    embedding=embedding,
                )
            )

        return results

    def remove_document(self, document_id: str, session_id: str):
        """Remove a document's chunks from the BM25 index.

        This is called when a document is deleted to prevent stale search results.
        OPTIMIZED: Only marks index as dirty; rebuild happens on next search.

        Args:
            document_id: Document ID to remove
            session_id: Session ID the document belongs to
        """
        with self._session_lock:
            if session_id not in self._session_indices:
                return

            index = self._session_indices[session_id]
            removed_count = index.remove_document(document_id)

            if removed_count > 0:
                # Mark as needing rebuild but don't rebuild immediately
                # The index will be rebuilt lazily on next search
                index._is_built = False
                logger.info(
                    f"Marked {removed_count} chunks for removal from document {document_id} "
                    f"in BM25 fallback index (session {session_id})"
                )

    def _ensure_index_built(self, session_id: str):
        """Ensure the BM25 index for a session is built (lazy rebuild).

        Releases the session lock before the expensive BM25Okapi construction.
        The CAS check in build_index() prevents stale-built state if a
        concurrent mutation occurs while building.

        Args:
            session_id: Session ID
        """
        index = None
        with self._session_lock:
            if session_id in self._session_indices:
                idx = self._session_indices[session_id]
                if not idx._is_built and idx.tokenized_corpus:
                    index = idx
        # Build outside lock — CAS in build_index() handles races.
        if index is not None:
            index.build_index()

    def build_session_index(self, session_id: str) -> bool:
        """Explicitly build/rebuild BM25 index for a session.

        Use after bulk ingestion with build_now=False to trigger a single
        one-shot build. Safe to call concurrently — CAS prevents stale state.

        Args:
            session_id: Session ID

        Returns:
            True if the index was built, False if already up-to-date, empty,
            or session does not exist.
        """
        with self._session_lock:
            index = self._session_indices.get(session_id)
            if not index or not index.tokenized_corpus:
                return False
            if index.is_current():
                return False
        # Build outside lock — CAS handles races.
        return index.build_index()

    def get_session_stats(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Return observability stats for a session's BM25 index.

        Args:
            session_id: Session ID

        Returns:
            Dict with doc_count, is_built, version, built_version,
            build_count, total_build_ms. None if session does not exist.
        """
        with self._session_lock:
            index = self._session_indices.get(session_id)
            if not index:
                return None
            return {
                "doc_count": len(index.documents),
                "is_built": index._is_built,
                "version": index._version,
                "built_version": index._built_version,
                "build_count": index._build_count,
                "total_build_ms": round(index._total_build_ms, 1),
            }

    def clear_session(self, session_id: str):
        """Clear index for a session.

        Args:
            session_id: Session ID to clear
        """
        with self._session_lock:
            if session_id in self._session_indices:
                self._session_indices[session_id].clear()
                del self._session_indices[session_id]
            if session_id in self._session_timestamps:
                del self._session_timestamps[session_id]
            self._session_creation_times.pop(session_id, None)
            self._session_backends.pop(session_id, None)
            logger.info(f"Cleared BM25 fallback index for session {session_id}")

    def get_mode_info(self) -> Dict[str, Any]:
        """Get information about the current search mode.

        Returns:
            Dict with mode information
        """
        native_available = self.is_native_mode_available()
        fallback_available = BM25_FALLBACK_AVAILABLE

        if native_available:
            mode = "native_milvus"
            description = "Using Milvus native BM25 with sparse vectors"
        elif fallback_available:
            mode = "rank_bm25_fallback"
            description = "Using rank_bm25 library (Milvus Lite mode)"
        else:
            mode = "dense_only"
            description = "Hybrid search disabled (no BM25 available)"

        return {
            "mode": mode,
            "description": description,
            "native_milvus_available": native_available,
            "rank_bm25_available": fallback_available,
            "alpha": rag_config.hybrid_alpha,
            "rrf_k": self.rrf_k,
        }


# Module-level singleton
_hybrid_searcher: Optional[HybridSearcher] = None
_hybrid_searcher_lock = threading.Lock()


def get_hybrid_searcher() -> HybridSearcher:
    """Get or create the hybrid searcher singleton (thread-safe)."""
    global _hybrid_searcher
    if _hybrid_searcher is None:
        with _hybrid_searcher_lock:
            if _hybrid_searcher is None:
                _hybrid_searcher = HybridSearcher()
    return _hybrid_searcher


def reset_hybrid_searcher():
    """Reset the hybrid searcher singleton."""
    global _hybrid_searcher
    with _hybrid_searcher_lock:
        _hybrid_searcher = None
