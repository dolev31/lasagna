"""Cross-Encoder Reranking for RAG System - SOTA Implementation.

This module provides:
- Cross-encoder reranking for improved precision
- MMR (Maximal Marginal Relevance) for result diversity
- Configurable reranking strategies
"""

from dataclasses import dataclass
import threading
from collections import OrderedDict
import time
from typing import List, Optional, Tuple, Union, Any, Dict
import numpy as np
from loguru import logger

try:
    from sentence_transformers import CrossEncoder

    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False
    logger.warning("sentence-transformers CrossEncoder not available. Reranking disabled.")

from lasagna.core.accelerator import device_guard
from lasagna.core.config import rag_config
from lasagna.core.model_licenses import warn_if_restricted
from lasagna.utils.normalization import (
    normalize_score_sigmoid as _normalize_score_sigmoid,
    normalize_scores_minmax as _normalize_scores_minmax,
    normalize_scores_for_mmr as _normalize_scores_for_mmr,
    adaptive_k_selection as _adaptive_k_selection,
)

# Cached device for cross-encoder model loading (detected once)
_CACHED_DEVICE: Optional[str] = None


def _detect_device() -> str:
    """Detect and cache the best available device for model inference."""
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE
    # Respect use_gpu config setting
    try:
        from ...core.config import rag_config

        _use_gpu = rag_config.use_gpu
    except Exception:
        _use_gpu = True
    if not _use_gpu:
        _CACHED_DEVICE = "cpu"
        return _CACHED_DEVICE
    try:
        import torch

        if torch.cuda.is_available():
            _CACHED_DEVICE = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            _CACHED_DEVICE = "mps"
        else:
            _CACHED_DEVICE = "cpu"
    except ImportError:
        _CACHED_DEVICE = "cpu"
    return _CACHED_DEVICE


# ---------------------------------------------------------------------------
# Shared result extraction helpers
# ---------------------------------------------------------------------------


def extract_content(result: Dict[str, Any]) -> str:
    """Extract content from various result formats."""
    if "content" in result:
        return result["content"]
    if "entity" in result and "content" in result["entity"]:
        return result["entity"]["content"]
    if "chunk" in result and hasattr(result["chunk"], "content"):
        return result["chunk"].content
    return str(result)


def extract_chunk_id(result: Dict[str, Any]) -> str:
    """Extract chunk ID from result."""
    if "id" in result:
        return str(result["id"])
    if "chunk_id" in result:
        return str(result["chunk_id"])
    if "chunk" in result and hasattr(result["chunk"], "chunk_id"):
        return str(result["chunk"].chunk_id)
    return ""


def extract_document_id(result: Dict[str, Any]) -> str:
    """Extract document ID from result."""
    if "document_id" in result:
        return result["document_id"]
    if "entity" in result and "document_id" in result["entity"]:
        return result["entity"]["document_id"]
    if "chunk" in result and hasattr(result["chunk"], "document_id"):
        return result["chunk"].document_id
    return ""


def extract_metadata(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract metadata from result."""
    if "metadata" in result:
        return result["metadata"]
    if "entity" in result and "metadata" in result["entity"]:
        return result["entity"]["metadata"]
    if "chunk" in result and hasattr(result["chunk"], "metadata"):
        return result["chunk"].metadata
    return {}


def extract_score(result: Dict[str, Any]) -> float:
    """Extract original score from result."""
    if "score" in result:
        return float(result["score"])
    if "distance" in result:
        return float(result["distance"])
    if "combined_score" in result:
        return float(result["combined_score"])
    return 0.0


def extract_embedding(result: Dict[str, Any]) -> Optional[List[float]]:
    """Extract pre-fetched embedding from result."""
    if "embedding" in result and result["embedding"] is not None:
        return result["embedding"]
    if "entity" in result and "embedding" in result["entity"]:
        return result["entity"]["embedding"]
    return None


@dataclass
class RerankedResult:
    """Result after reranking with scores."""

    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    original_score: float
    rerank_score: float  # Raw cross-encoder score (may be negative)
    normalized_score: float = 0.0  # Normalized to [0, 1] for display
    original_rank: int = 0
    final_rank: int = 0
    # P1-3 FIX: Pre-fetched embedding from Milvus to avoid re-embedding in MMR
    embedding: Optional[List[float]] = None
    # Backend highlight fragments (e.g. from OpenSearch BM25)
    highlights: Optional[List[str]] = None


class CrossEncoderReranker:
    """Cross-encoder based reranker for improved precision.

    Cross-encoders jointly encode query and document, providing
    more accurate relevance scores than bi-encoders at the cost
    of being slower (can't pre-compute document embeddings).
    """

    #: Cross-encoders kept loaded at once. Sweep arms that differ in
    #: reranker_model run side by side, and each call picks its own.
    MAX_LOADED_MODELS = 3

    def __init__(self, model_name: Optional[str] = None, config=None):
        """Initialize the reranker.

        Args:
            model_name: Cross-encoder model name (default from config)
            config: Optional RAGConfig instance (lazy-loaded from global if None)
        """
        _config = config or rag_config
        self.model_name = model_name or _config.reranker_model
        # A model named here (an ensemble member) stays; otherwise the
        # session's reranker_model is followed on each call.
        self._pinned = model_name is not None
        self.model: Optional[CrossEncoder] = None
        self.batch_size = _config.reranker_batch_size
        self._config_lock = threading.Lock()
        # Loaded on first use, not here: the reranker is built at backend start,
        # so loading eagerly downloaded the default 870 MB model even for the
        # speed profile, which turns reranking off.
        self._models: "OrderedDict[str, Any]" = OrderedDict()
        self._failed: Dict[str, str] = {}

    def configure(self, config) -> None:
        """Follow the session's reranker settings (thread-safe).

        Only records them; the model is chosen per call in :meth:`rerank`, so
        two sessions with different models never swap one under the other.
        """
        if not config.enable_reranking:
            return
        with self._config_lock:
            if not self._pinned:
                self.model_name = config.reranker_model
            self.batch_size = config.reranker_batch_size

    def ensure_loaded(self) -> bool:
        """Load the model if it is not loaded yet; True when it is usable."""
        try:
            self._model_for(self.model_name)
        except RuntimeError:
            return False
        return True

    def _load_model(self):
        """Load ``self.model_name`` (kept for callers of the old interface)."""
        self.model = self._model_for(self.model_name)

    def _model_for(self, name: str) -> Any:
        """The loaded cross-encoder for ``name``, loading it on first use.

        A model that cannot load raises, once per name: it used to be skipped,
        so an arm with a misspelled model scored the retrieval order, recorded
        the misspelled name as what ran, and retried the load on every query.
        """
        with self._config_lock:
            if name in self._failed:
                raise RuntimeError(self._failed[name])
            model = self._models.get(name)
            if model is not None:
                self._models.move_to_end(name)
                return model
            if not CROSS_ENCODER_AVAILABLE:
                raise RuntimeError(
                    "Reranking is enabled but sentence-transformers is not installed"
                )
            try:
                device = _detect_device()
                logger.info(f"Loading cross-encoder reranker: {name} on {device}")
                warn_if_restricted(name, component="reranker")
                with device_guard(device):
                    model = CrossEncoder(name, device=device)
            except Exception as e:
                self._failed[name] = f"Cannot load reranker model {name!r}: {e}"
                raise RuntimeError(self._failed[name]) from e
            self._models[name] = model
            while len(self._models) > self.MAX_LOADED_MODELS:
                self._models.popitem(last=False)
            if name == self.model_name:
                self.model = model
            logger.info(f"Reranker loaded successfully: {name}")
            return model

    def rerank(
        self, query: str, results: List[Dict[str, Any]], top_k: Optional[int] = None, config=None
    ) -> List[RerankedResult]:
        """Rerank results using cross-encoder.

        Args:
            query: The search query
            results: List of search results with content
            top_k: Number of top results to return (default from config)
            config: The session config; its ``reranker_model`` picks the model
                unless this reranker was built for a named one.

        Returns:
            Reranked results

        Raises:
            RuntimeError: reranking is enabled and the model cannot load or
                score. Returning the retrieval order instead put the control's
                number under the treatment's name.
        """
        if not results:
            return []

        if config:
            top_k = top_k or config.rerank_top_k
        else:
            top_k = top_k or rag_config.rerank_top_k

        enabled = config.enable_reranking if config else rag_config.enable_reranking
        if not enabled:
            return self._convert_results(results, top_k)

        # The model for this call, held locally: the shared attribute can be
        # switched by another session between choosing a model and scoring.
        name = self.model_name if (self._pinned or config is None) else config.reranker_model
        batch_size = getattr(config, "reranker_batch_size", None) or self.batch_size
        model = self._model_for(name)

        # Prepare query-document pairs
        pairs = []
        for result in results:
            content = self._extract_content(result)
            pairs.append([query, content])

        # Get reranking scores
        try:
            with device_guard(_detect_device()):
                scores = model.predict(pairs, batch_size=batch_size, show_progress_bar=False)
        except Exception as e:
            raise RuntimeError(f"Reranking with {name!r} failed: {e}") from e

        raw_scores = [float(s) for s in scores]

        # DIAGNOSTIC: Log cross-encoder score distribution (DEBUG to avoid production noise)
        score_variance = max(raw_scores) - min(raw_scores) if len(raw_scores) > 1 else 0.0
        logger.debug(
            f"[RERANK DIAG] Cross-encoder scores: "
            f"min={min(raw_scores):.3f} max={max(raw_scores):.3f} "
            f"mean={sum(raw_scores) / len(raw_scores):.3f} spread={score_variance:.3f} "
            f"n={len(raw_scores)}"
        )

        # Create reranked results
        reranked = []
        for i, (result, score) in enumerate(zip(results, scores)):
            reranked.append(
                RerankedResult(
                    chunk_id=self._extract_chunk_id(result),
                    document_id=self._extract_document_id(result),
                    content=self._extract_content(result),
                    metadata=self._extract_metadata(result),
                    original_score=self._extract_score(result),
                    rerank_score=float(score),
                    normalized_score=0.0,  # Will be set after sorting
                    original_rank=i,
                    final_rank=0,  # Will be set after sorting
                    # P1-3 FIX: Preserve embedding from search result for MMR
                    embedding=self._extract_embedding(result),
                    highlights=result.get("highlights"),
                )
            )

        # Sort by rerank score
        reranked.sort(key=lambda x: x.rerank_score, reverse=True)

        # Assign final ranks and normalize using sigmoid (preserves cross-encoder calibration)
        for i, result in enumerate(reranked):
            result.final_rank = i
            result.normalized_score = _normalize_score_sigmoid(result.rerank_score)

        best_raw = f"{reranked[0].rerank_score:.3f}" if reranked else "N/A"
        best_normalized = f"{reranked[0].normalized_score:.3f}" if reranked else "N/A"
        logger.debug(
            f"Reranked {len(results)} results -> top {top_k} "
            f"(best raw: {best_raw}, normalized: {best_normalized})"
        )

        # DIAGNOSTIC: Log rank changes (DEBUG to avoid production noise)
        rank_changes = []
        for r in reranked[:top_k]:
            delta = r.original_rank - r.final_rank
            direction = f"+{delta}" if delta > 0 else str(delta) if delta < 0 else "="
            rank_changes.append(f"#{r.original_rank}->{r.final_rank}({direction})")
        logger.debug(f"[RERANK DIAG] Rank changes (top {top_k}): {', '.join(rank_changes)}")

        return reranked[:top_k]

    def _extract_content(self, result: Dict[str, Any]) -> str:
        return extract_content(result)

    def _extract_chunk_id(self, result: Dict[str, Any]) -> str:
        return extract_chunk_id(result)

    def _extract_document_id(self, result: Dict[str, Any]) -> str:
        return extract_document_id(result)

    def _extract_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return extract_metadata(result)

    def _extract_score(self, result: Dict[str, Any]) -> float:
        return extract_score(result)

    def _extract_embedding(self, result: Dict[str, Any]) -> Optional[List[float]]:
        return extract_embedding(result)

    def _convert_results(self, results: List[Dict[str, Any]], top_k: int) -> List[RerankedResult]:
        """Convert results without reranking."""
        converted = []
        raw_scores = [self._extract_score(result) for result in results[:top_k]]
        normalized = _normalize_scores_minmax(raw_scores) if raw_scores else []

        for i, result in enumerate(results[:top_k]):
            score = self._extract_score(result)
            norm_score = normalized[i] if i < len(normalized) else _normalize_score_sigmoid(score)
            converted.append(
                RerankedResult(
                    chunk_id=self._extract_chunk_id(result),
                    document_id=self._extract_document_id(result),
                    content=self._extract_content(result),
                    metadata=self._extract_metadata(result),
                    original_score=score,
                    rerank_score=score,
                    normalized_score=norm_score,
                    original_rank=i,
                    final_rank=i,
                    # P1-3 FIX: Preserve embedding for MMR
                    embedding=self._extract_embedding(result),
                    highlights=result.get("highlights"),
                )
            )
        return converted


class MMRReranker:
    """Maximal Marginal Relevance for result diversity.

    MMR balances relevance and diversity by penalizing results
    that are too similar to already-selected results.
    """

    def __init__(self, lambda_param: Optional[float] = None, embedding_manager=None):
        """Initialize MMR reranker.

        Args:
            lambda_param: Balance between relevance (1) and diversity (0)
            embedding_manager: Embedding manager for similarity computation
        """
        self.lambda_param = lambda_param if lambda_param is not None else rag_config.mmr_lambda
        self.embedding_manager = embedding_manager

    def rerank_mmr(
        self, query: str, results: List[RerankedResult], top_k: int = 5, config=None
    ) -> List[RerankedResult]:
        """Rerank results using MMR for diversity.

        MMR Score = λ * Relevance(d) - (1-λ) * max(Similarity(d, d'))

        IMPORTANT: Relevance scores are normalized to [0, 1] before MMR calculation
        to ensure they are comparable to the cosine similarity diversity term.
        This fixes the bug where raw cross-encoder logits (unbounded, potentially
        negative) were mixed with bounded similarity values.

        Args:
            query: Search query
            results: Pre-scored results
            top_k: Number of diverse results to return

        Returns:
            Diverse subset of results
        """
        if config:
            lambda_param = config.mmr_lambda
            mmr_enabled = config.enable_mmr
            mmr_normalization = getattr(config, "mmr_score_normalization", "sigmoid")
        else:
            lambda_param = self.lambda_param
            mmr_enabled = rag_config.enable_mmr
            mmr_normalization = getattr(rag_config, "mmr_score_normalization", "sigmoid")

        if not results or not mmr_enabled:
            return results[:top_k]

        if not self.embedding_manager:
            logger.debug("No embedding manager for MMR - skipping diversity")
            return results[:top_k]

        # P1-3 FIX: Use pre-fetched embeddings from Milvus when available
        # This avoids re-embedding chunks, saving ~50-100ms per search
        embeddings = []
        need_embedding_indices = []

        for i, r in enumerate(results):
            if r.embedding is not None:
                # Convert list to numpy array if needed
                emb = np.array(r.embedding) if isinstance(r.embedding, list) else r.embedding
                embeddings.append(emb)
            else:
                embeddings.append(None)
                need_embedding_indices.append(i)

        # Only re-embed chunks that don't have pre-fetched embeddings
        if need_embedding_indices:
            contents_to_embed = [results[i].content for i in need_embedding_indices]
            new_embeddings = self.embedding_manager.encode_documents_batch(
                contents_to_embed, show_progress=False
            )
            for i, idx in enumerate(need_embedding_indices):
                embeddings[idx] = new_embeddings[i]
            logger.debug(
                f"MMR: Reused {len(results) - len(need_embedding_indices)}/{len(results)} "
                f"pre-fetched embeddings, computed {len(need_embedding_indices)} new"
            )
        else:
            logger.debug(f"MMR: Reused all {len(results)} pre-fetched embeddings (0 re-computed)")

        # CRITICAL FIX: Normalize relevance scores to [0, 1] range
        # This ensures relevance is comparable to cosine similarity (also [0, 1])
        raw_scores = [r.rerank_score for r in results]
        normalized_relevances = _normalize_scores_for_mmr(raw_scores, strategy=mmr_normalization)

        # Create mapping for O(1) lookup during MMR iteration
        relevance_map = {i: normalized_relevances[i] for i in range(len(results))}

        selected = []
        selected_embeddings = []
        remaining = list(range(len(results)))

        # DIAGNOSTIC: Log normalized relevance scores (DEBUG to avoid production log noise)
        logger.debug(
            f"[MMR DIAG] Input: {len(results)} results, lambda={lambda_param}, "
            f"normalization={mmr_normalization}"
        )
        logger.debug(
            f"[MMR DIAG] Raw rerank scores: min={min(raw_scores):.3f} max={max(raw_scores):.3f}"
        )
        logger.debug(
            f"[MMR DIAG] Normalized relevances: "
            f"min={min(normalized_relevances):.3f} max={max(normalized_relevances):.3f}"
        )

        while len(selected) < top_k and remaining:
            best_idx = None
            best_score = float("-inf")
            best_relevance = 0.0
            best_max_sim = 0.0

            for idx in remaining:
                result = results[idx]
                embedding = embeddings[idx]

                # Relevance: use NORMALIZED score (now in [0, 1] range)
                relevance = relevance_map[idx]

                # Diversity: max similarity to already selected (also [0, 1])
                max_sim = 0.0
                if selected_embeddings:
                    for sel_emb in selected_embeddings:
                        sim = self.embedding_manager.similarity(embedding, sel_emb)
                        max_sim = max(max_sim, sim)

                # MMR score - now both terms are in [0, 1] range
                mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim

                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = idx
                    best_relevance = relevance
                    best_max_sim = max_sim

            if best_idx is not None:
                # DIAGNOSTIC: Log each MMR selection (DEBUG to avoid production log noise)
                logger.debug(
                    f"[MMR DIAG] Selected #{len(selected) + 1}: "
                    f"idx={best_idx} relevance={best_relevance:.3f} "
                    f"max_sim={best_max_sim:.3f} "
                    f"mmr_score={best_score:.3f} "
                    f'content="{results[best_idx].content[:60]}..."'
                )
                if best_max_sim < 0:
                    logger.warning(
                        f"[MMR DIAG] NEGATIVE similarity detected! "
                        f"max_sim={best_max_sim:.3f} - diversity penalty is INVERTED"
                    )

                selected.append(best_idx)
                selected_embeddings.append(embeddings[best_idx])
                remaining.remove(best_idx)

        # Return selected results with updated ranks
        diverse_results = []
        for new_rank, idx in enumerate(selected):
            result = results[idx]
            result.final_rank = new_rank
            diverse_results.append(result)

        logger.debug(
            f"MMR diversity: {len(results)} -> {len(diverse_results)} "
            f"(λ={lambda_param}, normalization={mmr_normalization})"
        )

        return diverse_results


class CombinedReranker:
    """Combined reranking pipeline: Cross-encoder + MMR.

    Applies cross-encoder reranking first for precision,
    then MMR for diversity.
    """

    def __init__(self, embedding_manager=None):
        """Initialize combined reranker.

        Args:
            embedding_manager: Embedding manager for MMR
        """
        self.cross_encoder = CrossEncoderReranker()
        self.mmr = MMRReranker(embedding_manager=embedding_manager)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        apply_mmr: bool = True,
        return_metrics: bool = False,
        config=None,
        adaptive: bool = False,
        adaptive_min_k: int = 2,
        adaptive_max_k: int = 20,
        adaptive_buffer: int = 2,
        adaptive_min_gap_ratio: float = 0.15,
    ) -> List[RerankedResult]:
        """Apply full reranking pipeline.

        Args:
            query: Search query
            results: Initial search results
            top_k: Maximum number of results (used as ceiling for adaptive mode,
                    exact count for fixed mode)
            apply_mmr: Whether to apply MMR diversity
            return_metrics: Whether to return timing metrics
            config: Optional session config
            adaptive: When True, use adaptive-k score-gap analysis to dynamically
                      determine the number of results instead of fixed top_k
            adaptive_min_k: Minimum results for adaptive mode
            adaptive_max_k: Maximum results for adaptive mode (overrides top_k)
            adaptive_buffer: Buffer chunks past score gap
            adaptive_min_gap_ratio: Minimum relative gap to trigger cutoff

        Returns:
            Reranked and diversified results
        """
        if config:
            top_k = top_k or config.rerank_top_k
            self.cross_encoder.configure(config)
        else:
            top_k = top_k or rag_config.rerank_top_k

        rerank_ms = 0.0
        mmr_ms = 0.0

        # Step 1: Cross-encoder reranking
        # When adaptive, retrieve more candidates for the gap analysis to work with
        effective_max = adaptive_max_k if adaptive else top_k
        ce_top_k = effective_max * 3 if apply_mmr else effective_max
        rerank_start = time.perf_counter()
        reranked = self.cross_encoder.rerank(query, results, top_k=ce_top_k, config=config)
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        # Step 2: Adaptive-k determination (before MMR)
        # Use score-gap analysis on normalized cross-encoder scores to find
        # the natural cutoff between relevant and irrelevant chunks
        if adaptive and reranked:
            normalized_scores = [r.normalized_score for r in reranked]
            optimal_k = _adaptive_k_selection(
                scores=normalized_scores,
                min_k=adaptive_min_k,
                max_k=adaptive_max_k,
                buffer=adaptive_buffer,
                min_gap_ratio=adaptive_min_gap_ratio,
            )
            logger.info(
                f"[ADAPTIVE-K] Score-gap analysis: {len(reranked)} candidates -> "
                f"optimal_k={optimal_k} (min={adaptive_min_k}, max={adaptive_max_k}, "
                f"buffer={adaptive_buffer})"
            )
            # Use adaptive k as the target for downstream stages
            top_k = optimal_k

        # Step 3: Pre-MMR relevance filtering
        # SOTA: Filter irrelevant results BEFORE MMR to prevent diversity selection
        # from promoting genuinely irrelevant items. Uses sigmoid threshold since
        # cross-encoder logits are unbounded and need calibration.
        if apply_mmr and (config.enable_mmr if config else rag_config.enable_mmr):
            pre_filter_count = len(reranked)
            relevance_floor = 0.1  # sigmoid(score) must be >= 0.1
            filtered = [
                r for r in reranked if _normalize_score_sigmoid(r.rerank_score) >= relevance_floor
            ]
            if not filtered:
                # Safety fallback: if ALL results are below threshold, keep the
                # top-K by rerank score rather than returning nothing.
                logger.warning(
                    f"[RERANK DIAG] Pre-MMR filter removed ALL {pre_filter_count} results "
                    f"(all sigmoid < {relevance_floor}). Falling back to top-{top_k} by score."
                )
                filtered = reranked[:top_k]
            elif pre_filter_count != len(filtered):
                logger.info(
                    f"[RERANK DIAG] Pre-MMR filter: {pre_filter_count} -> {len(filtered)} "
                    f"(removed {pre_filter_count - len(filtered)} with sigmoid < {relevance_floor})"
                )
            reranked = filtered

            # Step 4: MMR for diversity (if enabled)
            mmr_start = time.perf_counter()
            reranked = self.mmr.rerank_mmr(query, reranked, top_k=top_k, config=config)
            mmr_ms = (time.perf_counter() - mmr_start) * 1000.0
        else:
            reranked = reranked[:top_k]

        if return_metrics:
            return reranked, {"rerank_ms": rerank_ms, "mmr_ms": mmr_ms}
        return reranked


@dataclass
class RerankerModelConfig:
    """Configuration for a single reranker model in an ensemble."""

    model_name: str
    weight: float = 1.0
    enabled: bool = True


class EnsembleReranker:
    """Ensemble of multiple cross-encoder rerankers.

    Combines scores from multiple models using weighted average for
    improved reranking quality. This is a SOTA technique that reduces
    model-specific biases and improves robustness.

    Example models to ensemble:
    - 'cross-encoder/ms-marco-MiniLM-L-6-v2' (fast, good quality)
    - 'cross-encoder/ms-marco-electra-base' (slower, higher quality)
    - 'BAAI/bge-reranker-base' (BGE series reranker)
    """

    def __init__(
        self,
        model_configs: Optional[List[RerankerModelConfig]] = None,
        embedding_manager=None,
    ):
        """Initialize ensemble reranker.

        Args:
            model_configs: List of model configurations with weights
            embedding_manager: Embedding manager for MMR
        """
        self.embedding_manager = embedding_manager
        self.rerankers: List[Tuple[CrossEncoderReranker, float]] = []

        if model_configs is None:
            # Default: single model from config
            model_configs = [RerankerModelConfig(model_name=rag_config.reranker_model)]

        # Initialize enabled rerankers
        for config in model_configs:
            if config.enabled:
                try:
                    reranker = CrossEncoderReranker(model_name=config.model_name)
                    if reranker.ensure_loaded():
                        self.rerankers.append((reranker, config.weight))
                        logger.info(
                            f"Ensemble: Added reranker {config.model_name} (weight={config.weight})"
                        )
                except Exception as e:
                    logger.warning(f"Failed to load ensemble reranker {config.model_name}: {e}")

        if not self.rerankers:
            logger.warning("No rerankers loaded for ensemble - falling back to pass-through")

        # MMR reranker for diversity
        self.mmr = MMRReranker(embedding_manager=embedding_manager)

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        apply_mmr: bool = True,
        return_metrics: bool = False,
        config=None,
        adaptive: bool = False,
        adaptive_min_k: int = 2,
        adaptive_max_k: int = 20,
        adaptive_buffer: int = 2,
        adaptive_min_gap_ratio: float = 0.15,
    ) -> List[RerankedResult]:
        """Rerank using ensemble of models.

        Args:
            query: Search query
            results: Initial search results
            top_k: Final number of results
            apply_mmr: Whether to apply MMR diversity
            return_metrics: Whether to return timing metrics
            config: Optional config overrides
            adaptive: When True, use adaptive-k score-gap analysis
            adaptive_min_k: Minimum results for adaptive mode
            adaptive_max_k: Maximum results for adaptive mode
            adaptive_buffer: Buffer chunks past score gap
            adaptive_min_gap_ratio: Minimum relative gap to trigger cutoff

        Returns:
            Reranked and optionally diversified results.
            If return_metrics is True, returns (results, metrics_dict).
        """
        if not results:
            if return_metrics:
                return [], {"rerank_ms": 0.0, "mmr_ms": 0.0}
            return []

        top_k = top_k or (config.rerank_top_k if config else rag_config.rerank_top_k)

        if not self.rerankers:
            # Pass-through mode
            converted = self._convert_results(results, top_k)
            if return_metrics:
                return converted, {"rerank_ms": 0.0, "mmr_ms": 0.0}
            return converted

        rerank_ms = 0.0
        mmr_ms = 0.0

        # Get scores from each reranker
        all_scores: List[Tuple[Dict[int, float], float]] = []  # (chunk_id -> score, weight)
        total_weight = 0.0

        rerank_start = time.perf_counter()
        for reranker, weight in self.rerankers:
            try:
                reranked = reranker.rerank(query, results, top_k=len(results), config=config)
                scores = {r.chunk_id: r.rerank_score for r in reranked}
                all_scores.append((scores, weight))
                total_weight += weight
            except Exception as e:
                logger.warning(f"Ensemble reranker failed: {e}")
        rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        if not all_scores or total_weight == 0:
            converted = self._convert_results(results, top_k)
            if return_metrics:
                return converted, {"rerank_ms": rerank_ms, "mmr_ms": 0.0}
            return converted

        # Combine scores with weighted average
        combined_scores: Dict[int, float] = {}
        for scores, weight in all_scores:
            for chunk_id, score in scores.items():
                if chunk_id not in combined_scores:
                    combined_scores[chunk_id] = 0.0
                combined_scores[chunk_id] += (score * weight) / total_weight

        # Build reranked results
        result_map = {self._extract_chunk_id(r): r for r in results}
        reranked_results = []

        for chunk_id, combined_score in combined_scores.items():
            if chunk_id in result_map:
                original = result_map[chunk_id]
                reranked_results.append(
                    RerankedResult(
                        chunk_id=chunk_id,
                        document_id=self._extract_document_id(original),
                        content=self._extract_content(original),
                        metadata=self._extract_metadata(original),
                        original_score=self._extract_score(original),
                        rerank_score=combined_score,
                        normalized_score=0.0,  # Will be set after sorting
                        original_rank=0,
                        final_rank=0,
                        embedding=self._extract_embedding(original),
                        highlights=original.get("highlights"),
                    )
                )

        # Sort by combined score
        reranked_results.sort(key=lambda x: x.rerank_score, reverse=True)

        # Normalize and assign ranks (sigmoid preserves cross-encoder calibration)
        for i, result in enumerate(reranked_results):
            result.final_rank = i
            result.normalized_score = _normalize_score_sigmoid(result.rerank_score)

        # Adaptive-k determination (before MMR)
        if adaptive and reranked_results:
            normalized_scores = [r.normalized_score for r in reranked_results]
            optimal_k = _adaptive_k_selection(
                scores=normalized_scores,
                min_k=adaptive_min_k,
                max_k=adaptive_max_k,
                buffer=adaptive_buffer,
                min_gap_ratio=adaptive_min_gap_ratio,
            )
            logger.info(
                f"[ADAPTIVE-K] Ensemble score-gap analysis: {len(reranked_results)} candidates -> "
                f"optimal_k={optimal_k} (min={adaptive_min_k}, max={adaptive_max_k})"
            )
            top_k = optimal_k

        # Apply MMR if requested
        if apply_mmr and (config.enable_mmr if config else rag_config.enable_mmr):
            # Pre-MMR relevance filter with safety fallback
            candidates = reranked_results[: top_k * 3]
            relevance_floor = 0.1
            filtered = [
                r for r in candidates if _normalize_score_sigmoid(r.rerank_score) >= relevance_floor
            ]
            if not filtered:
                filtered = candidates[:top_k]
            candidates = filtered
            mmr_start = time.perf_counter()
            reranked_results = self.mmr.rerank_mmr(query, candidates, top_k=top_k, config=config)
            mmr_ms = (time.perf_counter() - mmr_start) * 1000.0
        else:
            reranked_results = reranked_results[:top_k]

        logger.debug(
            f"Ensemble reranking: {len(results)} -> {len(reranked_results)} (models={len(self.rerankers)})"
        )
        if return_metrics:
            return reranked_results, {"rerank_ms": rerank_ms, "mmr_ms": mmr_ms}
        return reranked_results

    @property
    def cross_encoder(self):
        """Compatibility shim for SearchOrchestrator.

        The orchestrator reads reranker.cross_encoder.model_name at L791.
        Return self since self.model_name already returns a descriptive string.
        """
        return self

    @property
    def model_name(self) -> str:
        """Return a descriptive model name for cache fingerprinting.

        This ensures _component_identity() in SearchOrchestrator returns a
        meaningful string (not just the class name), so the search cache key
        differentiates between different ensemble configurations.
        """
        if not self.rerankers:
            return "ensemble:empty"
        return "ensemble:" + "+".join(r.model_name for r, _ in self.rerankers)

    def _extract_chunk_id(self, result: Dict[str, Any]) -> str:
        return extract_chunk_id(result)

    def _extract_document_id(self, result: Dict[str, Any]) -> str:
        return extract_document_id(result)

    def _extract_content(self, result: Dict[str, Any]) -> str:
        return extract_content(result)

    def _extract_metadata(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return extract_metadata(result)

    def _extract_score(self, result: Dict[str, Any]) -> float:
        return extract_score(result)

    def _extract_embedding(self, result: Dict[str, Any]) -> Optional[List[float]]:
        return extract_embedding(result)

    def _convert_results(self, results: List[Dict[str, Any]], top_k: int) -> List[RerankedResult]:
        """Convert results without reranking."""
        converted = []
        for i, result in enumerate(results[:top_k]):
            score = self._extract_score(result)
            converted.append(
                RerankedResult(
                    chunk_id=self._extract_chunk_id(result),
                    document_id=self._extract_document_id(result),
                    content=self._extract_content(result),
                    metadata=self._extract_metadata(result),
                    original_score=score,
                    rerank_score=score,
                    normalized_score=0.0,
                    original_rank=i,
                    final_rank=i,
                    highlights=result.get("highlights"),
                )
            )
        return converted


# Module-level singleton
class ColBERTReranker:
    """ColBERT late-interaction reranker using token-level MaxSim scoring.

    Uses HuggingFace transformers for ColBERT inference (no ragatouille dependency).
    Computes MaxSim: for each query token, find max dot-product with any doc token,
    then sum across query tokens. More accurate than cross-encoder on long documents
    due to fine-grained token interactions.
    """

    def __init__(self, config=None, embedding_manager=None, **kwargs):
        self._config = config or rag_config
        self._model_name = getattr(self._config, "colbert_model", "colbert-ir/colbertv2.0")
        self._model = None
        self._tokenizer = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    try:
                        from transformers import AutoModel, AutoTokenizer

                        warn_if_restricted(self._model_name, component="ColBERT reranker")
                        self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
                        self._model = AutoModel.from_pretrained(self._model_name)
                        self._model.eval()
                        logger.info(f"ColBERT model loaded: {self._model_name}")
                    except Exception as exc:
                        raise ImportError(
                            f"Failed to load ColBERT model '{self._model_name}': {exc}. "
                            "Ensure transformers and torch are installed."
                        )

    def _encode(self, texts: List[str], max_length: int = 512):
        """Encode texts into token-level embeddings for MaxSim."""
        import torch

        inputs = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        # Use last hidden state as token embeddings
        # Shape: (batch, seq_len, hidden_dim)
        embeddings = outputs.last_hidden_state
        # Mask padding tokens
        mask = inputs["attention_mask"].unsqueeze(-1).float()
        return embeddings * mask

    def _maxsim_score(self, query_emb, doc_emb) -> float:
        """Compute ColBERT MaxSim score between query and document embeddings."""
        import torch

        # query_emb: (1, q_len, dim), doc_emb: (1, d_len, dim)
        # Similarity matrix: (q_len, d_len)
        sim = torch.matmul(query_emb.squeeze(0), doc_emb.squeeze(0).T)
        # MaxSim: for each query token, take max similarity across doc tokens
        max_sim = sim.max(dim=1).values
        # Sum across query tokens (excluding padding — zeros contribute 0)
        return float(max_sim.sum())

    @property
    def model_name(self) -> str:
        return f"colbert:{self._model_name}"

    @property
    def cross_encoder(self):
        """Compatibility — orchestrator reads reranker.cross_encoder.model_name."""
        return self

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_metrics: bool = False,
        config=None,
        **kwargs,
    ) -> Union[List[RerankedResult], Tuple[List[RerankedResult], Dict[str, Any]]]:
        """Rerank using ColBERT MaxSim scoring (token-level late interaction)."""
        import math

        self._ensure_model()
        top_k = top_k or len(results)
        texts = [extract_content(r) for r in results]

        t0 = time.time()
        # Encode query once
        query_emb = self._encode([query], max_length=128)
        # Score each document via MaxSim
        scores: List[float] = []
        for text in texts:
            doc_emb = self._encode([text[:2000]], max_length=512)
            scores.append(self._maxsim_score(query_emb, doc_emb))
        rerank_ms = (time.time() - t0) * 1000

        # Sort by score descending
        scored_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        reranked = []
        for rank, orig_idx in enumerate(scored_indices[:top_k]):
            r = results[orig_idx]
            score = scores[orig_idx]
            # Normalize via sigmoid (ColBERT scores are unbounded)
            norm_score = 1.0 / (1.0 + math.exp(-score / 100.0))
            reranked.append(
                RerankedResult(
                    chunk_id=extract_chunk_id(r),
                    document_id=extract_document_id(r),
                    content=extract_content(r),
                    metadata=extract_metadata(r),
                    original_score=extract_score(r),
                    rerank_score=score,
                    normalized_score=norm_score,
                    original_rank=orig_idx,
                    final_rank=rank,
                    embedding=extract_embedding(r),
                    highlights=r.get("highlights") if hasattr(r, "get") else None,
                )
            )

        if return_metrics:
            return reranked[:top_k], {"rerank_ms": rerank_ms, "mmr_ms": 0.0}
        return reranked[:top_k]


def parse_rankgpt_permutation(text: str, n: int) -> Tuple[List[int], bool]:
    """A window's order from a RankGPT response, and whether it named any passage.

    RankGPT's parsing: the identifiers in the order given, minus duplicates and
    out-of-range ones, then every passage the response left out in its current
    order. Two guards it lacked: a reasoning model's ``<think>`` block is
    dropped, and when the response uses ``[n]`` identifiers only those are
    read, so prose such as "the 20 passages ranked:" is not read as passage 20.
    """
    import re

    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.DOTALL)
    numbers = re.findall(r"\[(\d+)\]", text) or re.findall(r"\d+", text)
    seen: set = set()
    order: List[int] = []
    for number in numbers:
        i = int(number) - 1
        if 0 <= i < n and i not in seen:
            seen.add(i)
            order.append(i)
    named = bool(order)
    order.extend(i for i in range(n) if i not in seen)
    return order, named


class LLMListwiseReranker:
    """RankGPT listwise reranking (Sun et al. 2023, "Is ChatGPT Good at Search?").

    The LLM is shown a window of passages as numbered chat turns and asked for a
    permutation (``[2] > [3] > [1]``), with RankGPT's prompt. Windows of
    ``llm_reranker_window_size`` passages (20) slide from the bottom of the list
    to the top in steps of ``llm_reranker_step_size`` (10), each permutation
    reordering its window in place, so a relevant passage can rise across
    windows to the top. Each passage is cut to its first 300 words, as in
    RankGPT. Raises when no LLM is configured, when a call keeps failing, or when
    no response named a passage: returning the retrieval order would put the
    control's number under this arm's name.
    """

    SYSTEM = (
        "You are RankGPT, an intelligent assistant that can rank passages based on "
        "their relevancy to the query."
    )
    PREFIX = (
        "I will provide you with {num} passages, each indicated by number identifier []. "
        "\nRank the passages based on their relevance to query: {query}."
    )
    POST = (
        "Search Query: {query}. \nRank the {num} passages above based on their relevance "
        "to the search query. The passages should be listed in descending order using "
        "identifiers. The most relevant passages should be listed first. The output "
        "format should be [] > [], e.g., [1] > [2]. Only response the ranking results, "
        "do not say any word or explain."
    )
    #: Words of each passage shown to the LLM, as in RankGPT.
    PASSAGE_WORDS = 300

    def __init__(self, config=None, embedding_manager=None, **kwargs):
        self._config = config or rag_config
        self._window_size = getattr(self._config, "llm_reranker_window_size", 20)
        self._step_size = getattr(self._config, "llm_reranker_step_size", 10)
        from lasagna.utils.model_utils import get_default_model_name

        # Empty/unset resolves to the configured provider default. A vendor-
        # prefixed literal here fails for anyone not on that same proxy.
        self._model = getattr(self._config, "llm_reranker_model", "") or get_default_model_name()
        self._llms: Dict[str, Any] = {}
        self._llm_lock = threading.Lock()

    def _llm_for(self, model: str) -> Any:
        with self._llm_lock:
            llm = self._llms.get(model)
            if llm is None:
                from lasagna.providers import LLMProviderNotConfigured, get_llm_provider

                try:
                    provider = get_llm_provider()
                except LLMProviderNotConfigured as exc:
                    raise RuntimeError("llm_listwise needs an LLM, and none is configured") from exc
                llm = self._llms[model] = provider.get_llm(model_name=model, temperature=0)
            return llm

    @property
    def model_name(self) -> str:
        return f"llm_listwise:{self._model}"

    @property
    def cross_encoder(self):
        """Compatibility — orchestrator reads reranker.cross_encoder.model_name."""
        return self

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        return_metrics: bool = False,
        config=None,
        **kwargs,
    ) -> Union[List[RerankedResult], Tuple[List[RerankedResult], Dict[str, Any]]]:
        """RankGPT's sliding-window permutation generation over ``results``."""
        # Read per call: the instance is built once per pipeline, and a sweep's
        # arms differ in these settings.
        effective = config or self._config
        window = int(getattr(effective, "llm_reranker_window_size", None) or self._window_size)
        step = int(getattr(effective, "llm_reranker_step_size", None) or self._step_size)
        model = getattr(effective, "llm_reranker_model", None) or self._model
        if window < 2 or step < 1:
            raise ValueError(
                f"llm_listwise needs a window of at least 2 and a step of at least 1 "
                f"(got window={window}, step={step})"
            )

        t0 = time.time()
        top_k = top_k or len(results)
        order = list(range(len(results)))
        if len(results) > 1:
            llm = self._llm_for(model)
            passages = [" ".join(extract_content(r).split()[: self.PASSAGE_WORDS]) for r in results]
            named_any = False
            # The loop of rank_llm, the RankGPT authors' own library: unlike
            # RankGPT's first version it also ranks a list shorter than a window.
            end, start = len(order), len(order) - window
            while end > 0 and start + step != 0:
                start = max(start, 0)
                current = order[start:end]
                if len(current) > 1:
                    permutation, named = self._rank_window(
                        llm, query, [passages[i] for i in current]
                    )
                    named_any = named_any or named
                    order[start:end] = [current[j] for j in permutation]
                end, start = end - step, start - step
            if not named_any:
                raise RuntimeError(
                    f"llm_listwise: no response from {model!r} named a passage, so "
                    f"nothing was reranked"
                )

        n = len(order)
        reranked = []
        for final_rank, orig_idx in enumerate(order[:top_k]):
            r = results[orig_idx]
            reranked.append(
                RerankedResult(
                    chunk_id=extract_chunk_id(r),
                    document_id=extract_document_id(r),
                    content=extract_content(r),
                    metadata=extract_metadata(r),
                    original_score=extract_score(r),
                    rerank_score=float(n - final_rank),
                    normalized_score=(n - final_rank) / n,
                    original_rank=orig_idx,
                    final_rank=final_rank,
                    embedding=extract_embedding(r),
                    highlights=r.get("highlights"),
                )
            )

        rerank_ms = (time.time() - t0) * 1000
        if return_metrics:
            return reranked, {"rerank_ms": rerank_ms, "mmr_ms": 0.0}
        return reranked

    def _rank_window(self, llm: Any, query: str, passages: List[str]) -> Tuple[List[int], bool]:
        """One RankGPT call: the window's permutation, and whether it named a passage."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        n = len(passages)
        messages: List[Any] = [
            SystemMessage(content=self.SYSTEM),
            HumanMessage(content=self.PREFIX.format(num=n, query=query)),
            AIMessage(content="Okay, please provide the passages."),
        ]
        for i, passage in enumerate(passages, start=1):
            messages.append(HumanMessage(content=f"[{i}] {passage}"))
            messages.append(AIMessage(content=f"Received passage [{i}]."))
        messages.append(HumanMessage(content=self.POST.format(query=query, num=n)))

        for attempt in range(3):
            try:
                response = llm.invoke(messages)
                break
            except Exception as e:
                if attempt == 2:
                    raise RuntimeError(f"llm_listwise: the LLM call failed: {e}") from e
                logger.warning(f"LLM reranker attempt {attempt + 1} failed: {e}")
                time.sleep(2**attempt)
        text = response.content if hasattr(response, "content") else str(response)
        return parse_rankgpt_permutation(str(text), n)


_reranker: Optional[CombinedReranker] = None
_reranker_lock = threading.Lock()


def get_reranker(embedding_manager=None) -> CombinedReranker:
    """Get or create the reranker singleton (thread-safe)."""
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                _reranker = CombinedReranker(embedding_manager=embedding_manager)
    return _reranker


def reset_reranker():
    """Reset the reranker singleton."""
    global _reranker
    with _reranker_lock:
        _reranker = None


# ---------------------------------------------------------------------------
# ABC-compliant adapters
#
# These adapters bridge between the public ``BaseReranker`` ABC (which uses
# ``ScoredDocument``) and the internal reranker classes (which use
# ``Dict[str, Any]`` / ``RerankedResult``).  The search orchestrator
# continues to call the internal classes directly; these adapters are
# used via the component registry for researcher workflows.
# ---------------------------------------------------------------------------

from lasagna.core.base_components import (  # noqa: E402
    BaseReranker,
    ScoredDocument,
    scored_document_from_dict,
    scored_document_to_dict,
    scored_document_from_reranked,
)


class CrossEncoderRerankerAdapter(BaseReranker):
    """Adapts ``CrossEncoderReranker`` to the public ``BaseReranker`` ABC."""

    # Wraps an internal reranker that reads the session config.
    _forwards_session_config = True

    def __init__(self, **kwargs):
        self._inner = CrossEncoderReranker(**kwargs)

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 5,
        **kwargs,
    ) -> List[ScoredDocument]:
        dicts = [scored_document_to_dict(d) for d in documents]
        results = self._inner.rerank(query, dicts, top_k, **kwargs)
        return [scored_document_from_reranked(r) for r in results]


class ColBERTRerankerAdapter(BaseReranker):
    """Adapts ``ColBERTReranker`` (late-interaction MaxSim) to ``BaseReranker``."""

    # Wraps an internal reranker that reads the session config.
    _forwards_session_config = True

    def __init__(self, **kwargs):
        self._inner = ColBERTReranker(**kwargs)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 5,
        **kwargs,
    ) -> List[ScoredDocument]:
        dicts = [scored_document_to_dict(d) for d in documents]
        results = self._inner.rerank(query, dicts, top_k=top_k, config=kwargs.get("config"))
        return [scored_document_from_reranked(r) for r in results]


class LLMListwiseRerankerAdapter(BaseReranker):
    """Adapts ``LLMListwiseReranker`` (RankGPT-style sliding window) to ``BaseReranker``."""

    # Wraps an internal reranker that reads the session config.
    _forwards_session_config = True

    def __init__(self, **kwargs):
        self._inner = LLMListwiseReranker(**kwargs)

    @property
    def model_name(self) -> str:
        return self._inner.model_name

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 5,
        **kwargs,
    ) -> List[ScoredDocument]:
        dicts = [scored_document_to_dict(d) for d in documents]
        results = self._inner.rerank(query, dicts, top_k=top_k, config=kwargs.get("config"))
        return [scored_document_from_reranked(r) for r in results]


class _PublicRerankerModel:
    """Stands in for ``CrossEncoderReranker`` where only its name is read."""

    def __init__(self, name: str):
        self.model_name = name


class PipelineRerankerAdapter:
    """Lets a user's ``BaseReranker`` be called by the search orchestrator.

    Every other adapter in this module runs the other way -- wrapping an
    internal reranker so it satisfies the public ABC. Nothing ran this
    direction, so a component written against the documented contract could be
    registered, validated and swapped in, and then raised ``TypeError`` the
    first time the pipeline actually called it: the orchestrator invokes
    ``rerank(query=..., results=[dict], top_k=..., return_metrics=..., config=...)``
    while ``BaseReranker.rerank`` declares ``(query, documents: [ScoredDocument],
    top_k, **kwargs)``. ``**kwargs`` swallowed ``results``, leaving ``documents``
    unfilled.

    That gap was invisible to every documented check -- contract validation, the
    isolated comparison, and ``pipeline.replace()`` all exercise the component
    outside the pipeline -- so the first thing that failed was the benchmark the
    whole exercise exists to run.
    """

    def __init__(self, component: BaseReranker):
        self._component = component
        # The orchestrator reads `reranker.cross_encoder.model_name` for the
        # response metadata. Report the component's own name instead of
        # pretending to be a cross-encoder.
        name = getattr(component, "name", None) or type(component).__name__
        self.cross_encoder = _PublicRerankerModel(name)

    @property
    def name(self) -> str:
        """The wrapped component's identity, for cache keys and provenance.

        Without it every custom reranker was fingerprinted as this adapter class,
        so two custom rerankers in one process shared search-cache entries and
        the second was served the first one's results.
        """
        inner = self._component
        for attr in ("model_name_or_path", "model_name", "name"):
            value = getattr(inner, attr, None)
            if value and not callable(value):
                return f"{type(inner).__module__}.{type(inner).__qualname__}:{value}"
        return f"{type(inner).__module__}.{type(inner).__qualname__}"

    def rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
        top_k: int = 5,
        return_metrics: bool = False,
        **kwargs,
    ):
        documents = [scored_document_from_dict(r) for r in results]
        started = time.perf_counter()
        # Only `top_k` is forwarded. `config`, `return_metrics` and the adaptive
        # keywords are pipeline-internal, and passing them on would make a
        # correct component look broken for accepting a contract it was never
        # shown. The built-in adapters are the exception: without the session
        # config their cross-encoder ran the global default model, not the
        # arm's reranker_model.
        extra = {}
        if getattr(self._component, "_forwards_session_config", False):
            if kwargs.get("config") is not None:
                extra["config"] = kwargs["config"]
        reranked = self._component.rerank(query, documents, top_k=top_k, **extra)
        elapsed_ms = (time.perf_counter() - started) * 1000

        by_chunk = {
            str(r.get("id") or r.get("chunk_id", "")): (i, r) for i, r in enumerate(results)
        }
        out = []
        for rank, doc in enumerate(reranked):
            original_rank, original = by_chunk.get(doc.chunk_id, (rank, {}))
            out.append(
                RerankedResult(
                    chunk_id=doc.chunk_id,
                    document_id=doc.document_id,
                    content=doc.content,
                    metadata=doc.metadata or {},
                    original_score=float(original.get("score", doc.score) or 0.0),
                    rerank_score=doc.score,
                    normalized_score=doc.score,
                    original_rank=original_rank,
                    final_rank=rank,
                    embedding=original.get("embedding"),
                    highlights=original.get("highlights"),
                )
            )

        if return_metrics:
            return out, {"rerank_ms": elapsed_ms, "mmr_ms": 0.0}
        return out


def bind_embedding_manager(reranker: Any, embedding_manager: Any) -> None:
    """Point a reranker's MMR stage at the embedder the current search uses.

    The reranker is built once with the embedding manager of that moment. When
    a session or a sweep arm switches embedders, MMR kept re-embedding chunks
    with the old model and comparing them with vectors from the new one: with
    a 384-d corpus under a 1024-d default, every reranking arm crashed with
    "shapes (384,) and (1024,) not aligned".
    """
    if embedding_manager is None or not hasattr(embedding_manager, "encode_documents_batch"):
        return
    for candidate in (reranker, getattr(reranker, "_inner", None)):
        mmr = getattr(candidate, "mmr", None)
        if mmr is not None and hasattr(mmr, "embedding_manager"):
            mmr.embedding_manager = embedding_manager


def adapt_reranker_for_pipeline(reranker: Any) -> Any:
    """Wrap a public-contract reranker; leave internal rerankers untouched.

    Called at the point the orchestrator resolves which reranker to use, so it
    covers both the configured component and a per-query override.
    """
    if isinstance(reranker, BaseReranker):
        return PipelineRerankerAdapter(reranker)
    return reranker


class CombinedRerankerAdapter(BaseReranker):
    """Adapts ``CombinedReranker`` to the public ``BaseReranker`` ABC."""

    # Wraps an internal reranker that reads the session config.
    _forwards_session_config = True

    def __init__(self, embedding_manager=None, **kwargs):
        self._inner = CombinedReranker(embedding_manager=embedding_manager)

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 5,
        **kwargs,
    ) -> List[ScoredDocument]:
        dicts = [scored_document_to_dict(d) for d in documents]
        results = self._inner.rerank(query, dicts, top_k=top_k, **kwargs)
        return [scored_document_from_reranked(r) for r in results]


class EnsembleRerankerAdapter(BaseReranker):
    """Adapts ``EnsembleReranker`` to the public ``BaseReranker`` ABC."""

    # Wraps an internal reranker that reads the session config.
    _forwards_session_config = True

    def __init__(self, **kwargs):
        self._inner = EnsembleReranker(**kwargs)

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 5,
        **kwargs,
    ) -> List[ScoredDocument]:
        dicts = [scored_document_to_dict(d) for d in documents]
        results = self._inner.rerank(query, dicts, top_k=top_k, **kwargs)
        return [scored_document_from_reranked(r) for r in results]


class MMRRerankerAdapter(BaseReranker):
    """Adapts ``MMRReranker`` to the public ``BaseReranker`` ABC.

    Maps ``rerank()`` → ``rerank_mmr()`` and converts between
    ``ScoredDocument`` and ``RerankedResult``.
    """

    # Wraps an internal reranker that reads the session config.
    _forwards_session_config = True

    def __init__(self, lambda_param: Optional[float] = None, embedding_manager=None, **kwargs):
        self._inner = MMRReranker(
            lambda_param=lambda_param,
            embedding_manager=embedding_manager,
        )

    def rerank(
        self,
        query: str,
        documents: List[ScoredDocument],
        top_k: int = 5,
        **kwargs,
    ) -> List[ScoredDocument]:
        # Convert ScoredDocument → RerankedResult for MMR input
        reranked_inputs = [
            RerankedResult(
                chunk_id=d.chunk_id,
                document_id=d.document_id,
                content=d.content,
                metadata=d.metadata,
                original_score=d.score,
                rerank_score=d.score,
                normalized_score=d.score,
            )
            for d in documents
        ]
        results = self._inner.rerank_mmr(query, reranked_inputs, top_k, **kwargs)
        return [scored_document_from_reranked(r) for r in results]
