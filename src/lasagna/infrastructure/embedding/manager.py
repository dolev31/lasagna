"""Embedding Manager for RAG System - SOTA Implementation.

This module provides a robust embedding manager with:
- BGE-M3 support via FlagEmbedding (dense+sparse+ColBERT, 8K context, 100+ languages)
- Sentence-transformers fallback for other models
- Auto-detection of backend based on model name
- Batch processing with GPU acceleration
- Embedding caching with LRU eviction
- Query prefixing for improved retrieval
"""

import hashlib
import threading
from typing import Dict, List, Optional, Union
from collections import OrderedDict
import numpy as np
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Embeddings will fail.")

try:
    from FlagEmbedding import BGEM3FlagModel

    FLAG_EMBEDDING_AVAILABLE = True
except ImportError:
    FLAG_EMBEDDING_AVAILABLE = False

from lasagna.core.accelerator import device_guard
from lasagna.core.model_licenses import warn_if_restricted
from lasagna.core.strict import strict_stages
from lasagna.core.config import rag_config, RAGConfig


class EmbeddingCache:
    """Thread-safe LRU cache for embeddings to avoid recomputation.

    CRITICAL FIX: All cache operations are now protected by RLock to prevent
    race conditions in multi-threaded environments (FastAPI, concurrent requests).
    """

    def __init__(self, max_size: int = 10000):
        """Initialize cache with maximum size.

        Args:
            max_size: Maximum number of embeddings to cache
        """
        self.max_size = max_size
        self.cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self.hits = 0
        self.misses = 0
        # Thread safety: RLock allows same thread to acquire multiple times
        self._lock = threading.RLock()

    def _compute_key(self, text: str, prefix: str = "") -> str:
        """Compute cache key from text and prefix."""
        content = f"{prefix}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def get(self, text: str, prefix: str = "") -> Optional[np.ndarray]:
        """Get embedding from cache (thread-safe).

        Args:
            text: Text to look up
            prefix: Query prefix used

        Returns:
            Cached embedding or None
        """
        key = self._compute_key(text, prefix)
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return self.cache[key]
            self.misses += 1
            return None

    def put(self, text: str, embedding: np.ndarray, prefix: str = ""):
        """Store embedding in cache (thread-safe).

        Args:
            text: Original text
            embedding: Computed embedding
            prefix: Query prefix used
        """
        key = self._compute_key(text, prefix)

        with self._lock:
            # Evict oldest if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)

            self.cache[key] = embedding

    def clear(self):
        """Clear the cache (thread-safe)."""
        with self._lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate (thread-safe)."""
        with self._lock:
            total = self.hits + self.misses
            return self.hits / total if total > 0 else 0.0

    def stats(self) -> dict:
        """Get cache statistics (thread-safe)."""
        with self._lock:
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": self.hits / (self.hits + self.misses)
                if (self.hits + self.misses) > 0
                else 0.0,
            }


# ---------------------------------------------------------------------------
# Model-aware prefix resolution
# ---------------------------------------------------------------------------
# Different embedding model families require different query/document prefixes
# for optimal retrieval performance.

_MODEL_PREFIXES = {
    "e5": {"query": "query: ", "document": "passage: "},
    "bge": {
        "query": "Represent this sentence for searching relevant passages: ",
        "document": "",
    },
}
_DEFAULT_PREFIXES = {"query": "", "document": ""}


def _resolve_prefixes(model_name: str) -> dict:
    """Resolve query/document prefixes based on the embedding model name."""
    name_lower = model_name.lower()
    for key, prefixes in _MODEL_PREFIXES.items():
        if key in name_lower:
            return prefixes
    return _DEFAULT_PREFIXES


def _resolve_backend(config: RAGConfig) -> str:
    """Resolve which embedding backend to use.

    Returns 'flag-embedding' or 'sentence-transformers'.
    """
    backend = config.embedding_backend
    if backend == "auto":
        model = config.embedding_model.lower()
        if "bge-m3" in model:
            return "flag-embedding" if FLAG_EMBEDDING_AVAILABLE else "sentence-transformers"
        return "sentence-transformers"
    return backend


def _config_signature(config: RAGConfig) -> tuple:
    return (
        config.embedding_model,
        config.embedding_model_fallback,
        config.embedding_dim,
        config.batch_size,
        config.normalize_embeddings,
        config.use_gpu,
        config.enable_embedding_cache,
        config.embedding_cache_size,
        config.embedding_backend,
        getattr(config, "embedding_truncate_dim", None),
    )


class EmbeddingManager:
    """Manages embedding generation with SOTA practices.

    Features:
    - BGE-M3 via FlagEmbedding (dense+sparse+ColBERT, 8K context, 100+ languages)
    - Sentence-transformers support for other models (BGE-large, MiniLM, etc.)
    - Auto-detection of backend based on model name
    - Query prefixing for asymmetric retrieval
    - Batch processing with configurable batch size
    - Optional GPU acceleration
    - LRU caching for performance
    - Normalized embeddings for cosine similarity
    """

    def __init__(self, config: Optional[RAGConfig] = None):
        """Initialize the embedding manager."""
        self._config = config or rag_config
        self._signature = _config_signature(self._config)
        self._model: Optional[Union[SentenceTransformer, "BGEM3FlagModel"]] = None
        self._load_error: Optional[str] = None
        self._load_lock = threading.RLock()
        # The configured name until the model loads; a fallback replaces it.
        self.model_name: Optional[str] = self._config.embedding_model
        self.embedding_dim: int = self._config.embedding_dim
        self._backend: str = _resolve_backend(self._config)

        # Initialize cache if enabled
        self.cache: Optional[EmbeddingCache] = None
        if self._config.enable_embedding_cache:
            self.cache = EmbeddingCache(max_size=self._config.embedding_cache_size)
            logger.info(
                f"Embedding cache enabled with max size: {self._config.embedding_cache_size}"
            )

        # Thread safety for concurrent encode() calls from the document
        # processing thread pool (document_tasks.py dispatches up to
        # max_concurrent_uploads workers that share this singleton).
        self._encode_lock = threading.Lock()
        # Where the model runs; on MPS every encode also takes the process-wide
        # device lock (see core/accelerator.py).
        self._device: Optional[str] = None

        # The model loads on first use, not here. A backend is built from the
        # settings-file default (bge-m3, 2.3 GB) before any profile applies, and
        # loading it eagerly downloaded that model on the `speed` path sold as
        # keyless and small, and failed every task offline.
        self._set_prefixes()

        # Matryoshka dimension truncation
        self._truncate_dim: Optional[int] = getattr(self._config, "embedding_truncate_dim", None)

    def _maybe_truncate(self, embedding: np.ndarray) -> np.ndarray:
        """Truncate and L2-renormalize for Matryoshka models.

        If ``embedding_truncate_dim`` is set and smaller than the native
        dimension, slices the embedding and re-normalises so that cosine
        similarity remains valid.
        """
        td = self._truncate_dim
        if not td or td >= embedding.shape[-1]:
            return embedding
        truncated = embedding[..., :td]
        norms = np.linalg.norm(truncated, axis=-1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        return truncated / norms

    @property
    def model(self) -> Optional[Union[SentenceTransformer, "BGEM3FlagModel"]]:
        """The loaded model, loading it on first access."""
        if self._model is None:
            self._ensure_loaded()
        return self._model

    @model.setter
    def model(self, value) -> None:
        self._model = value

    @property
    def is_loaded(self) -> bool:
        """Whether the model has been loaded (reading ``model`` loads it)."""
        return self._model is not None

    def _ensure_loaded(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                # Once per manager: a model that cannot load is not retried on
                # every query, each time with a download attempt and a stack trace.
                raise RuntimeError(self._load_error)
            configured = self.model_name
            try:
                self._load_model()
            except Exception as exc:
                self._model = None  # a model that loaded but failed its probe is not kept
                self._load_error = str(exc)
                raise
            if self.model_name != configured:
                self._set_prefixes()  # a fallback model has its own prefixes

    def _set_prefixes(self) -> None:
        prefixes = _resolve_prefixes(self.model_name or self._config.embedding_model)
        self.query_prefix: str = prefixes["query"]
        self.document_prefix: str = prefixes["document"]

    def _load_model(self):
        """Load the embedding model with fallback support."""
        if self._backend == "flag-embedding":
            self._load_flag_embedding()
        else:
            self._load_sentence_transformer()

    def _load_flag_embedding(self):
        """Load BGE-M3 via FlagEmbedding library."""
        if not FLAG_EMBEDDING_AVAILABLE:
            logger.warning("FlagEmbedding not installed, falling back to sentence-transformers")
            self._backend = "sentence-transformers"
            self._load_sentence_transformer()
            return

        model_name = self._config.embedding_model

        # Determine device and fp16 settings
        use_fp16 = False
        device = None  # Let FlagEmbedding auto-detect
        if self._config.use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    use_fp16 = True
                    device = "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    use_fp16 = False  # MPS doesn't support fp16 well
                    device = "mps"
                else:
                    device = "cpu"
            except ImportError:
                device = "cpu"
        else:
            device = "cpu"

        try:
            logger.info(
                f"Loading BGE-M3 via FlagEmbedding: {model_name} on {device} (fp16={use_fp16})"
            )
            self._device = device
            with device_guard(device):
                self.model = BGEM3FlagModel(
                    model_name,
                    use_fp16=use_fp16,
                    device=device,
                )
                # Verify embedding dimension with a test encode
                test_output = self.model.encode(["test"], return_dense=True)
            self.model_name = model_name

            actual_dim = test_output["dense_vecs"].shape[1]

            if actual_dim != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: model produces {actual_dim}, "
                    f"config expects {self.embedding_dim}. Updating config."
                )
                self.embedding_dim = actual_dim

            logger.info(
                f"Loaded {model_name} via FlagEmbedding - dim={actual_dim}, device={device}, fp16={use_fp16}"
            )

        except Exception as e:
            logger.warning(f"Failed to load {model_name} via FlagEmbedding: {e}")
            logger.info("Falling back to sentence-transformers")
            self._backend = "sentence-transformers"
            self._load_sentence_transformer()

    @staticmethod
    def _resolve_local_snapshot(model_name: str) -> str:
        """Return local HuggingFace Hub snapshot path if available, else model_name.

        When a model is cached locally, using its snapshot directory path directly
        avoids network calls to huggingface.co (useful in restricted environments).
        Falls back to the original model_name string on any error.
        """
        try:
            from pathlib import Path

            cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
            slug = "models--" + model_name.replace("/", "--")
            refs_file = cache_dir / slug / "refs" / "main"
            if refs_file.exists():
                snapshot_hash = refs_file.read_text().strip()
                local_path = cache_dir / slug / "snapshots" / snapshot_hash
                # Only use the local path if it looks like a valid ST model dir
                if (local_path / "config_sentence_transformers.json").exists():
                    logger.debug(f"Resolved {model_name} → local snapshot {snapshot_hash[:8]}")
                    return str(local_path)
        except Exception:
            pass
        return model_name

    def _load_sentence_transformer(self):
        """Load model via sentence-transformers (original path)."""
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )

        primary_model = self._config.embedding_model
        fallback_model = self._config.embedding_model_fallback

        # Determine device
        device = "cpu"
        if self._config.use_gpu:
            try:
                import torch

                if torch.cuda.is_available():
                    device = "cuda"
                    logger.info("CUDA available, using GPU")
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    device = "mps"
                    logger.info("MPS (Apple Silicon) available, using GPU")
                else:
                    logger.info("GPU requested but not available, using CPU")
            except ImportError:
                logger.warning("PyTorch not available, using CPU")

        # Prefer local snapshot to avoid network latency/failures in restricted envs
        primary_path = self._resolve_local_snapshot(primary_model)
        try:
            logger.info(f"Loading primary embedding model: {primary_model} on {device}")
            warn_if_restricted(primary_model, component="embedding model")
            self._device = device
            with device_guard(device):
                self.model = SentenceTransformer(primary_path, device=device)
                # Verify embedding dimension
                test_embedding = self.model.encode("test", convert_to_numpy=True)
            self.model_name = primary_model  # keep the canonical name
            actual_dim = len(test_embedding)

            if actual_dim != self.embedding_dim:
                logger.warning(
                    f"Embedding dimension mismatch: model produces {actual_dim}, "
                    f"config expects {self.embedding_dim}. Updating config."
                )
                self.embedding_dim = actual_dim

            logger.info(f"Loaded {primary_model} - dim={actual_dim}, device={device}")

        except Exception as e:
            # An evaluation must not measure a different embedder than the one
            # it names: the fallback is a serving convenience.
            if strict_stages():
                raise RuntimeError(
                    f"Cannot load embedding model {primary_model!r}: {e}. An evaluation "
                    f"does not fall back to {fallback_model!r}, since its numbers would "
                    f"then describe another model."
                ) from e
            logger.warning(f"Failed to load primary model {primary_model}: {e}")
            logger.info(f"Attempting fallback model: {fallback_model}")

            fallback_path = self._resolve_local_snapshot(fallback_model)
            try:
                with device_guard(device):
                    self.model = SentenceTransformer(fallback_path, device=device)
                    test_embedding = self.model.encode("test", convert_to_numpy=True)
                self.model_name = fallback_model
                actual_dim = len(test_embedding)
                self.embedding_dim = actual_dim

                logger.info(f"Loaded fallback {fallback_model} - dim={actual_dim}, device={device}")

            except Exception as fallback_error:
                logger.error(f"Failed to load fallback model: {fallback_error}")
                raise RuntimeError(
                    f"Could not load any embedding model. Primary: {primary_model}, "
                    f"Fallback: {fallback_model}"
                )

    def encode_query(self, query: str, use_cache: bool = True) -> np.ndarray:
        """Encode a query with appropriate prefix for retrieval.

        The prefix is resolved automatically based on the loaded model
        (e.g. BGE uses a long instruction prefix, E5 uses ``"query: "``).

        Args:
            query: The search query
            use_cache: Whether to use cache

        Returns:
            Query embedding as numpy array
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(query, self.query_prefix)
            if cached is not None:
                return cached

        with self._encode_lock, device_guard(self._device):
            if self._backend == "flag-embedding":
                # BGE-M3 handles query prefixing internally
                output = self.model.encode([query], return_dense=True)
                embedding = output["dense_vecs"][0]
                if self._config.normalize_embeddings:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
            else:
                # Apply model-specific query prefix
                prefixed_query = f"{self.query_prefix}{query}" if self.query_prefix else query
                embedding = self.model.encode(
                    prefixed_query,
                    convert_to_numpy=True,
                    normalize_embeddings=self._config.normalize_embeddings,
                )

        # Matryoshka truncation
        embedding = self._maybe_truncate(embedding)

        # Cache the result
        if use_cache and self.cache:
            self.cache.put(query, embedding, self.query_prefix)

        return embedding

    def encode_document(self, text: str, use_cache: bool = True) -> np.ndarray:
        """Encode a document chunk for storage.

        Applies the model-specific document prefix (e.g. ``"passage: "`` for
        E5, empty string for BGE).

        Args:
            text: Document text to encode
            use_cache: Whether to use cache

        Returns:
            Document embedding as numpy array
        """
        # Check cache first
        if use_cache and self.cache:
            cached = self.cache.get(text, self.document_prefix)
            if cached is not None:
                return cached

        with self._encode_lock, device_guard(self._device):
            if self._backend == "flag-embedding":
                output = self.model.encode([text], return_dense=True)
                embedding = output["dense_vecs"][0]
                if self._config.normalize_embeddings:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm
            else:
                # Apply model-specific document prefix (e.g. "passage: " for E5)
                encode_text = f"{self.document_prefix}{text}" if self.document_prefix else text
                embedding = self.model.encode(
                    encode_text,
                    convert_to_numpy=True,
                    normalize_embeddings=self._config.normalize_embeddings,
                )

        # Matryoshka truncation
        embedding = self._maybe_truncate(embedding)

        # Cache the result
        if use_cache and self.cache:
            self.cache.put(text, embedding, self.document_prefix)

        return embedding

    def encode_documents_batch(
        self, texts: List[str], show_progress: bool = True
    ) -> List[np.ndarray]:
        """Encode multiple document chunks in batches.

        Uses batch processing for efficiency with GPU acceleration.

        Args:
            texts: List of document texts
            show_progress: Show progress bar

        Returns:
            List of embeddings
        """
        if not texts:
            return []

        batch_size = self._config.batch_size

        logger.info(f"Encoding {len(texts)} documents in batches of {batch_size}")

        # Check cache for already computed embeddings
        results = [None] * len(texts)
        texts_to_compute = []
        indices_to_compute = []

        if self.cache:
            for i, text in enumerate(texts):
                cached = self.cache.get(text, self.document_prefix)
                if cached is not None:
                    results[i] = cached
                else:
                    texts_to_compute.append(text)
                    indices_to_compute.append(i)

            if len(texts) - len(texts_to_compute) > 0:
                logger.debug(
                    "Cache hit for "
                    f"{len(texts) - len(texts_to_compute)}/{len(texts)} document embeddings"
                )
        else:
            texts_to_compute = texts
            indices_to_compute = list(range(len(texts)))

        # Batch encode remaining texts
        if texts_to_compute:
            with self._encode_lock, device_guard(self._device):
                if self._backend == "flag-embedding":
                    output = self.model.encode(
                        texts_to_compute,
                        return_dense=True,
                        batch_size=batch_size,
                    )
                    embeddings = output["dense_vecs"]
                    if self._config.normalize_embeddings:
                        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                        norms = np.where(norms > 0, norms, 1)
                        embeddings = embeddings / norms
                else:
                    # Apply model-specific document prefix (e.g. "passage: " for E5)
                    encode_texts = (
                        [f"{self.document_prefix}{t}" for t in texts_to_compute]
                        if self.document_prefix
                        else texts_to_compute
                    )
                    embeddings = self.model.encode(
                        encode_texts,
                        convert_to_numpy=True,
                        normalize_embeddings=self._config.normalize_embeddings,
                        batch_size=batch_size,
                        show_progress_bar=show_progress,
                    )

            # Matryoshka truncation (batch)
            if self._truncate_dim and self._truncate_dim < embeddings.shape[-1]:
                embeddings = self._maybe_truncate(embeddings)

            # Store results and update cache
            for idx, embedding in zip(indices_to_compute, embeddings):
                results[idx] = embedding
                if self.cache:
                    self.cache.put(texts[idx], embedding, self.document_prefix)

        return results

    def encode_queries_batch(self, queries: List[str]) -> List[np.ndarray]:
        """Encode multiple queries in batch.

        Useful for multi-query retrieval strategies.

        Args:
            queries: List of queries

        Returns:
            List of query embeddings
        """
        if not queries:
            return []

        with self._encode_lock, device_guard(self._device):
            if self._backend == "flag-embedding":
                output = self.model.encode(
                    queries,
                    return_dense=True,
                    batch_size=self._config.batch_size,
                )
                embeddings = output["dense_vecs"]
                if self._config.normalize_embeddings:
                    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                    norms = np.where(norms > 0, norms, 1)
                    embeddings = embeddings / norms
            else:
                # Apply model-specific query prefix
                prefixed_queries = (
                    [f"{self.query_prefix}{q}" for q in queries] if self.query_prefix else queries
                )
                embeddings = self.model.encode(
                    prefixed_queries,
                    convert_to_numpy=True,
                    normalize_embeddings=self._config.normalize_embeddings,
                    batch_size=self._config.batch_size,
                )

        # Matryoshka truncation (batch)
        if self._truncate_dim:
            embeddings = self._maybe_truncate(np.asarray(embeddings))
        return list(embeddings)

    def encode_documents_late_chunking(
        self,
        full_text: str,
        chunk_texts: List[str],
    ) -> List[np.ndarray]:
        """Late chunking: embed full document, extract per-chunk embeddings.

        Passes the full document text through the transformer to get token-level
        representations, then pools tokens within each chunk's span to produce
        chunk-level embeddings.  This preserves cross-chunk context that is lost
        when chunks are embedded independently.

        Reference: Günther et al., "Late Chunking: Contextual Chunk Embeddings
        Using Long-Context Embedding Models", arXiv:2409.04701.

        Falls back to standard ``encode_documents_batch`` when:
        - The backend is FlagEmbedding (no token-level access)
        - The model doesn't support fast tokenizer / token-level output
        - Any error occurs during token-level encoding

        Args:
            full_text: The complete document text (pre-chunking).
            chunk_texts: Ordered list of chunk texts (substrings of full_text).

        Returns:
            List of chunk embeddings (same length as chunk_texts).
        """
        if not chunk_texts:
            return []

        # Pre-flight checks OUTSIDE the lock so fallback doesn't deadlock
        if self._backend == "flag-embedding":
            logger.info(
                "Late chunking not supported with FlagEmbedding backend, using standard encoding"
            )
            return self.encode_documents_batch(chunk_texts, show_progress=True)

        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model is None:
            logger.info(
                "Late chunking: sentence-transformers not available, using standard encoding"
            )
            return self.encode_documents_batch(chunk_texts, show_progress=True)

        try:
            import torch
        except ImportError:
            logger.info("Late chunking: torch not available, using standard encoding")
            return self.encode_documents_batch(chunk_texts, show_progress=True)

        # Verify model structure: need model[0].auto_model for token-level output
        try:
            transformer_module = self.model[0]
            auto_model = transformer_module.auto_model
        except (IndexError, AttributeError):
            logger.info(
                "Late chunking: model architecture not compatible (no auto_model), using standard encoding"
            )
            return self.encode_documents_batch(chunk_texts, show_progress=True)

        # Verify tokenizer supports offset mapping (fast tokenizers only)
        tokenizer = self.model.tokenizer
        if not getattr(tokenizer, "is_fast", False):
            logger.info(
                "Late chunking: slow tokenizer (no offset mapping support), using standard encoding"
            )
            return self.encode_documents_batch(chunk_texts, show_progress=True)

        # Verify the model has no Dense projection layer after the transformer.
        # If it does, late chunking would produce raw hidden_dim embeddings that
        # don't match the final embedding dimension after the Dense projection.
        # Sentence-transformers pipeline: [0]=Transformer, [1]=Pooling, [2+]=Dense/Normalize
        # Use iter(self.model) which yields ST module wrappers in pipeline order.
        _has_dense = any(type(m).__name__ == "Dense" for idx, m in enumerate(self.model) if idx > 0)
        if _has_dense:
            logger.info(
                "Late chunking: model has Dense projection layer — "
                "late chunking would produce wrong dimensions, using standard encoding"
            )
            return self.encode_documents_batch(chunk_texts, show_progress=True)

        try:
            with self._encode_lock, device_guard(self._device):
                max_len = getattr(self.model, "max_seq_length", 8192)

                # Apply document prefix to full text
                encode_text = (
                    f"{self.document_prefix}{full_text}" if self.document_prefix else full_text
                )
                full_encoding = tokenizer(
                    encode_text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=max_len,
                    return_offsets_mapping=True,
                )

                full_token_count = full_encoding["input_ids"].shape[1]
                if full_token_count >= max_len:
                    logger.warning(
                        f"Late chunking: document ({len(full_text)} chars) truncated "
                        f"to {max_len} tokens. Chunks beyond boundary will use standard encoding."
                    )

                # Get token-level embeddings from the model's transformer
                device = self.model.device
                input_ids = full_encoding["input_ids"].to(device)
                attention_mask = full_encoding["attention_mask"].to(device)

                with torch.no_grad():
                    model_output = auto_model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
                # token_embeddings shape: (1, seq_len, hidden_dim)
                token_embeddings = model_output.last_hidden_state[0].cpu().numpy()
                offsets = full_encoding["offset_mapping"][0].numpy()  # (seq_len, 2)

            # --- Token-to-chunk mapping (no lock needed, pure numpy) ---
            prefix_len = len(self.document_prefix) if self.document_prefix else 0

            chunk_embeddings = []
            fallback_indices = []  # chunks that need standard encoding
            search_start = 0

            for idx, chunk_text in enumerate(chunk_texts):
                # Find chunk's character span in full_text
                chunk_start = full_text.find(chunk_text, search_start)
                if chunk_start == -1:
                    fallback_indices.append(idx)
                    chunk_embeddings.append(None)
                    continue

                chunk_end = chunk_start + len(chunk_text)
                search_start = chunk_end

                # Adjust for prefix offset in tokenized text
                adj_start = chunk_start + prefix_len
                adj_end = chunk_end + prefix_len

                # Find token indices that overlap with chunk's character span
                token_mask = (offsets[:, 1] > adj_start) & (offsets[:, 0] < adj_end)
                # Exclude special tokens (offset 0,0)
                special_mask = (offsets[:, 0] == 0) & (offsets[:, 1] == 0)
                token_mask = token_mask & ~special_mask

                if not token_mask.any():
                    fallback_indices.append(idx)
                    chunk_embeddings.append(None)
                    continue

                # Mean pooling of token embeddings for this chunk
                chunk_token_embs = token_embeddings[token_mask]
                chunk_emb = chunk_token_embs.mean(axis=0)

                # Normalize
                if self._config.normalize_embeddings:
                    norm = np.linalg.norm(chunk_emb)
                    if norm > 0:
                        chunk_emb = chunk_emb / norm

                chunk_embeddings.append(chunk_emb)

            # Batch-encode any fallback chunks (outside the lock)
            if fallback_indices:
                fallback_texts = [chunk_texts[i] for i in fallback_indices]
                logger.debug(
                    f"Late chunking: {len(fallback_indices)}/{len(chunk_texts)} chunks "
                    f"need standard encoding (not found in full text or truncated)"
                )
                fallback_embs = self.encode_documents_batch(fallback_texts, show_progress=False)
                for fi, emb in zip(fallback_indices, fallback_embs):
                    chunk_embeddings[fi] = emb

            # Matryoshka truncation
            chunk_embeddings = [self._maybe_truncate(e) for e in chunk_embeddings]

            logger.info(
                f"Late chunking: {len(chunk_texts) - len(fallback_indices)}/{len(chunk_texts)} "
                f"chunks from {full_token_count}-token document, "
                f"{len(fallback_indices)} fallback"
            )
            return chunk_embeddings

        except Exception as e:
            logger.warning(f"Late chunking failed ({e}), falling back to standard batch encoding")
            return self.encode_documents_batch(chunk_texts, show_progress=True)

    def encode_sparse(self, texts: List[str]) -> Optional[List[Dict[str, float]]]:
        """Encode texts and return sparse (lexical) weights.

        Only available with BGE-M3 via FlagEmbedding. Returns None for
        sentence-transformers backend.

        Args:
            texts: List of texts to encode

        Returns:
            List of {token: weight} dicts, or None if not supported
        """
        if self._backend != "flag-embedding":
            return None

        with self._encode_lock, device_guard(self._device):
            output = self.model.encode(texts, return_sparse=True)
        return output.get("lexical_weights")

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute similarity between two embeddings.

        Uses cosine similarity for normalized embeddings. Clamped to [0, 1]
        to ensure correct behavior in MMR diversity calculations.

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score clamped to [0, 1]
        """
        # For normalized embeddings, cosine similarity = dot product
        # Clamp to [0, 1] to prevent negative values from inverting
        # the MMR diversity penalty
        return float(max(0.0, np.dot(embedding1, embedding2)))

    def get_cache_stats(self) -> Optional[dict]:
        """Get embedding cache statistics."""
        if self.cache:
            return self.cache.stats()
        return None

    def clear_cache(self):
        """Clear the embedding cache."""
        if self.cache:
            self.cache.clear()
            logger.info("Embedding cache cleared")

    @property
    def backend_name(self) -> str:
        """Return the active backend name."""
        return self._backend

    @property
    def dimension(self) -> int:
        """Get effective embedding dimension (truncated if Matryoshka)."""
        td = self._truncate_dim
        if td and td < self.embedding_dim:
            return td
        return self.embedding_dim

    @property
    def signature(self) -> tuple:
        """Return the config signature used for this manager."""
        return self._signature


# Module-level singleton
_embedding_manager: Optional[EmbeddingManager] = None
_embedding_lock = threading.Lock()


def get_embedding_manager(config: Optional[RAGConfig] = None) -> EmbeddingManager:
    """Get or create the embedding manager singleton.

    Thread-safe: both initial creation and config-drift reload are protected
    by ``_embedding_lock``.  On config drift the old manager is replaced but
    NOT explicitly destroyed — in-flight requests may still hold a reference.
    Python GC will reclaim it when all references are dropped.

    Returns:
        EmbeddingManager instance
    """
    global _embedding_manager

    # Capture a local snapshot to avoid races with concurrent reset_embedding_manager().
    # All return paths use a local variable — never the bare global — so that a
    # concurrent reset cannot cause this function to return None.
    mgr = _embedding_manager

    if mgr is None:
        with _embedding_lock:
            if _embedding_manager is None:
                _embedding_manager = EmbeddingManager(config=config)
            return _embedding_manager

    # Config-drift detection — must also be under lock to prevent two
    # concurrent callers from both detecting drift and both loading models.
    if config is not None:
        desired_signature = _config_signature(config)
        # Use local snapshot for the outer check to avoid AttributeError
        # if another thread calls reset_embedding_manager() concurrently.
        if mgr.signature != desired_signature:
            with _embedding_lock:
                # Re-read under lock: may have been reset or reloaded by another thread.
                if _embedding_manager is None:
                    _embedding_manager = EmbeddingManager(config=config)
                elif _embedding_manager.signature != desired_signature:
                    logger.info(
                        "Embedding manager config change detected; reloading. "
                        f"old={_embedding_manager.signature} new={desired_signature}"
                    )
                    _embedding_manager = EmbeddingManager(config=config)

                    # Best-effort GPU memory cleanup (old model will be GC'd
                    # once in-flight references are dropped)
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                return _embedding_manager

    # Return the local snapshot — NOT the global.  A concurrent
    # reset_embedding_manager() may have set _embedding_manager to None
    # between the snapshot at the top and this return.  The snapshot is a
    # valid, fully-initialised instance so it is safe to return.
    return mgr


def reset_embedding_manager():
    """Reset the embedding manager singleton.

    Useful for testing or when configuration changes.
    """
    global _embedding_manager
    with _embedding_lock:
        if _embedding_manager is not None:
            logger.info("Resetting embedding manager")
            _embedding_manager = None


# ---------------------------------------------------------------------------
# ABC-compliant adapter
# ---------------------------------------------------------------------------

from lasagna.core.base_components import BaseEmbedder  # noqa: E402


class EmbeddingManagerAdapter(BaseEmbedder):
    """Adapts ``EmbeddingManager`` to the public ``BaseEmbedder`` ABC.

    Uses the module-level singleton so that model loading is shared
    with the rest of the pipeline.
    """

    def __init__(self, config=None, **kwargs):
        self._inner = get_embedding_manager(config)

    def encode(self, texts: List[str], **kwargs) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension))
        results = self._inner.encode_documents_batch(texts, show_progress=False)
        # Guard against None entries from partial cache misses / encoding failures
        valid = [r for r in results if r is not None]
        if len(valid) != len(results):
            logger.warning(
                f"Embedding batch had {len(results) - len(valid)} failed entries "
                f"out of {len(results)} — replacing with zero vectors"
            )
            dim = self.dimension
            results = [r if r is not None else np.zeros(dim) for r in results]
        return np.array(results)

    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        return self._inner.encode_query(query)

    @property
    def dimension(self) -> int:
        return self._inner.dimension
