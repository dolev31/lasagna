"""Embedding Manager for RAG System.

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
from lasagna._quiet import loading_bar_hidden
from lasagna.core import model_downloads

#: Fewer batches than this encode without a progress bar.
_PROGRESS_MIN_BATCHES = 8


class EmbeddingCache:
    """Thread-safe LRU cache for embeddings to avoid recomputation.

    All cache operations are now protected by RLock to prevent
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

_WEB_SEARCH = "Given a web search query, retrieve relevant passages that answer the query"
_REPRESENT = "Represent this sentence for searching relevant passages: "

#: (substrings that must all occur in the lower-cased model name, prefixes),
#: each from its model card. First match wins, so a specific family precedes
#: the general one it would also match: "e5-mistral" before "e5", "bge-m3"
#: before "bge". Documents take no prefix unless stated.
_MODEL_PREFIX_RULES = (
    # Qwen3-Embedding: no space after "Query:" (its own prompt template).
    (("qwen3-embedding",), {"query": f"Instruct: {_WEB_SEARCH}\nQuery:", "document": ""}),
    (("gte-qwen",), {"query": f"Instruct: {_WEB_SEARCH}\nQuery: ", "document": ""}),
    (("e5-mistral",), {"query": f"Instruct: {_WEB_SEARCH}\nQuery: ", "document": ""}),
    (("e5", "instruct"), {"query": f"Instruct: {_WEB_SEARCH}\nQuery: ", "document": ""}),
    (("e5",), {"query": "query: ", "document": "passage: "}),
    # BGE-M3 is trained without a query instruction; adding one moves its vectors.
    (("bge-m3",), {"query": "", "document": ""}),
    (("bge", "-zh"), {"query": "为这个句子生成表示以用于检索相关文章：", "document": ""}),
    (("bge",), {"query": _REPRESENT, "document": ""}),
    (("nomic-embed-text",), {"query": "search_query: ", "document": "search_document: "}),
    (("mxbai-embed",), {"query": _REPRESENT, "document": ""}),
    (("arctic-embed", "v2"), {"query": "query: ", "document": ""}),
    (("arctic-embed",), {"query": _REPRESENT, "document": ""}),
)
_DEFAULT_PREFIXES = {"query": "", "document": ""}


def _resolve_prefixes(model_name: str) -> dict:
    """Resolve query/document prefixes based on the embedding model name."""
    name_lower = model_name.lower()
    for needles, prefixes in _MODEL_PREFIX_RULES:
        if all(needle in name_lower for needle in needles):
            return dict(prefixes)
    return dict(_DEFAULT_PREFIXES)


def _special_frame(tokenizer) -> tuple:
    """The special tokens a tokenizer puts before and after a single sequence.

    Read off an encoding rather than asked for: ``build_inputs_with_special_tokens``
    is gone from transformers 5's tokenizers.
    """
    bare = tokenizer("a", add_special_tokens=False)["input_ids"]
    full = tokenizer("a", add_special_tokens=True)["input_ids"]
    for i in range(len(full) - len(bare) + 1):
        if full[i : i + len(bare)] == bare:
            return list(full[:i]), list(full[i + len(bare) :])
    raise ValueError("cannot locate the text between the tokenizer's special tokens")


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
    """Manages embedding generation: batching, caching and device placement.

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

    def _device_guard(self):
        """The device lock for one encode, decided after the model has loaded.

        ``device_guard(self._device)`` read the device before the lazy load had
        set it, so the first encode on each manager (the load and its first
        batch) ran on MPS without the lock, and concurrent arms still crashed.
        """
        if self._model is None:
            self._ensure_loaded()
        return device_guard(self._device)

    def _ensure_loaded(self) -> None:
        with self._load_lock:
            if self._model is not None:
                return
            if self._load_error is not None:
                # Once per manager: a model that cannot load is not retried on
                # every query, each time with a download attempt and a stack trace.
                raise RuntimeError(self._load_error)
            configured = self.model_name
            # A server refuses a model it would have to download; not recorded
            # as a load error, so it loads once the model is on disk.
            model_downloads.check(configured)
            self._refuse_if_offline_and_missing(configured)
            try:
                self._load_model()
            except Exception as exc:
                self._model = None  # a model that loaded but failed its probe is not kept
                self._load_error = str(exc)
                raise
            if self.model_name != configured:
                self._set_prefixes()  # a fallback model has its own prefixes

    def _refuse_if_offline_and_missing(self, configured: Optional[str]) -> None:
        """Offline, a model that is not on disk is refused before any load is tried.

        Trying cost two load attempts (primary and fallback), a warning and an
        error line each, and then "Could not load any embedding model", which
        named neither the cause nor the fix. Not recorded as a load error, so
        it loads once the model is on disk. A cached fallback is still used
        where falling back is allowed.
        """
        if not configured or not model_downloads.offline():
            return
        if model_downloads.is_cached(configured):
            return
        fallback = self._config.embedding_model_fallback
        if fallback and not strict_stages() and model_downloads.is_cached(fallback):
            return
        raise model_downloads.ModelNotDownloaded(configured, offline=True)

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
            with device_guard(device), model_downloads.one_weights_copy():
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
            with (
                device_guard(device),
                loading_bar_hidden(primary_path),
                model_downloads.one_weights_copy(),
            ):
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
                with (
                    device_guard(device),
                    loading_bar_hidden(fallback_path),
                    model_downloads.one_weights_copy(),
                ):
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

        with self._encode_lock, self._device_guard():
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

        with self._encode_lock, self._device_guard():
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
            with self._encode_lock, self._device_guard():
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
                        # A bar for a few batches flashes by and is all an
                        # upload of one document printed; a long ingest keeps it.
                        show_progress_bar=show_progress
                        and len(encode_texts) > _PROGRESS_MIN_BATCHES * batch_size,
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

        with self._encode_lock, self._device_guard():
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

    def _late_chunking_unavailable(self, reason: str, chunk_texts: List[str]) -> List[np.ndarray]:
        """Standard encoding when serving; an error when evaluating.

        An arm that enables late chunking and quietly gets standard chunk
        embeddings measures the control under the technique's name.
        """
        if strict_stages():
            raise RuntimeError(
                f"Late chunking cannot run: {reason}. An evaluation does not fall back to "
                f"standard encoding, since the arm would then measure plain chunk "
                f"embeddings under the late-chunking name."
            )
        logger.info(f"Late chunking: {reason}; using standard encoding")
        return self.encode_documents_batch(chunk_texts, show_progress=True)

    def _warn_if_not_mean_pooled(self) -> None:
        """Late chunking mean-pools tokens; say so once if the model pools otherwise."""
        if getattr(self, "_pooling_warned", False):
            return
        self._pooling_warned = True
        pooling = next((m for m in self.model if type(m).__name__ == "Pooling"), None)
        # sentence-transformers 6 names it `pooling_mode`; earlier releases had a getter.
        mode = getattr(pooling, "pooling_mode", None)
        if mode is None and hasattr(pooling, "get_pooling_mode_str"):
            mode = pooling.get_pooling_mode_str()
        if isinstance(mode, str) and mode not in ("mean", "mean_tokens"):
            logger.warning(
                f"Late chunking mean-pools token embeddings, but {self.model_name} was "
                f"trained with {mode!r} pooling, so its late-chunked vectors are not the "
                f"kind it was trained to produce. The paper used mean-pooled models."
            )

    def encode_documents_late_chunking(
        self,
        full_text: str,
        chunk_texts: List[str],
    ) -> List[np.ndarray]:
        """Late chunking: embed the whole document, then pool tokens per chunk.

        The document goes through the transformer once, so every token's
        representation has seen the whole document; each chunk's embedding is
        the mean of its tokens' embeddings. Chunks embedded on their own lose
        that context.

        A document longer than the model's window is processed as overlapping
        macro windows ("long late chunking"): each window holds the model's
        maximum of tokens, consecutive windows share ``late_chunking_overlap``
        tokens (capped at half a window), and an overlapping token keeps the
        embedding from the window where it has left context.

        Reference: Günther et al., "Late Chunking: Contextual Chunk Embeddings
        Using Long-Context Embedding Models", arXiv:2409.04701.

        Needs a sentence-transformers model with a fast tokenizer and no Dense
        projection. Otherwise (and for a chunk that is not a slice of
        ``full_text``) it encodes chunks the standard way when serving, and
        raises during an evaluation.

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
            return self._late_chunking_unavailable(
                "the FlagEmbedding backend gives no token-level output", chunk_texts
            )
        if not SENTENCE_TRANSFORMERS_AVAILABLE or self.model is None:
            return self._late_chunking_unavailable(
                "sentence-transformers is not available", chunk_texts
            )
        try:
            import torch
        except ImportError:
            return self._late_chunking_unavailable("torch is not available", chunk_texts)

        # Verify model structure: need model[0].auto_model for token-level output
        try:
            auto_model = self.model[0].auto_model
        except (IndexError, AttributeError, TypeError):
            return self._late_chunking_unavailable(
                "the model exposes no token-level transformer", chunk_texts
            )

        # Verify tokenizer supports offset mapping (fast tokenizers only)
        tokenizer = self.model.tokenizer
        if not getattr(tokenizer, "is_fast", False):
            return self._late_chunking_unavailable(
                "the tokenizer is not a fast tokenizer, so it has no offsets", chunk_texts
            )

        # Sentence-transformers pipeline: [0]=Transformer, [1]=Pooling, [2+]=Dense/Normalize.
        # Token embeddings come out before a Dense projection, at the wrong size.
        if any(type(m).__name__ == "Dense" for idx, m in enumerate(self.model) if idx > 0):
            return self._late_chunking_unavailable(
                "the model has a Dense projection after pooling", chunk_texts
            )
        self._warn_if_not_mean_pooled()

        try:
            prefix = self.document_prefix or ""
            encode_text = f"{prefix}{full_text}"
            encoding = tokenizer(
                encode_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
                verbose=False,
            )
            ids = list(encoding["input_ids"])
            offsets = np.asarray(encoding["offset_mapping"], dtype=int).reshape(-1, 2)
            head, tail = _special_frame(tokenizer)
            max_len = int(getattr(self.model, "max_seq_length", None) or 512)
            window = max_len - len(head) - len(tail)
            overlap = min(int(getattr(self._config, "late_chunking_overlap", 0) or 0), window // 2)

            token_embeddings: Optional[np.ndarray] = None
            n_windows = 0
            with self._encode_lock, self._device_guard():
                device = self.model.device
                start = 0
                while start < len(ids):
                    end = min(start + window, len(ids))
                    piece = ids[start:end]
                    input_ids = torch.tensor([head + piece + tail], device=device)
                    with torch.no_grad():
                        hidden = auto_model(
                            input_ids=input_ids, attention_mask=torch.ones_like(input_ids)
                        ).last_hidden_state[0]
                    hidden = hidden.float().cpu().numpy()
                    content = hidden[len(head) : len(head) + len(piece)]
                    if token_embeddings is None:
                        token_embeddings = np.zeros((len(ids), content.shape[1]), dtype=np.float32)
                    # The first `overlap` tokens of a later window were embedded by
                    # the previous one with context on both sides; keep those.
                    keep = 0 if start == 0 else overlap
                    token_embeddings[start + keep : end] = content[keep:]
                    n_windows += 1
                    if end == len(ids):
                        break
                    start = end - overlap

            # --- Token-to-chunk mapping (no lock needed, pure numpy) ---
            chunk_embeddings: List[Optional[np.ndarray]] = []
            missing = []
            search_start = 0
            for idx, chunk_text in enumerate(chunk_texts):
                chunk_start = full_text.find(chunk_text, search_start)
                if chunk_start == -1 or token_embeddings is None:
                    missing.append(idx)
                    chunk_embeddings.append(None)
                    continue
                chunk_end = chunk_start + len(chunk_text)
                search_start = chunk_end
                lo, hi = chunk_start + len(prefix), chunk_end + len(prefix)
                token_mask = (offsets[:, 1] > lo) & (offsets[:, 0] < hi)
                if not token_mask.any():
                    missing.append(idx)
                    chunk_embeddings.append(None)
                    continue
                chunk_emb = token_embeddings[token_mask].mean(axis=0)
                if self._config.normalize_embeddings:
                    norm = np.linalg.norm(chunk_emb)
                    if norm > 0:
                        chunk_emb = chunk_emb / norm
                chunk_embeddings.append(chunk_emb)
        except Exception as e:
            if strict_stages():
                raise
            logger.warning(f"Late chunking failed ({e}), falling back to standard batch encoding")
            return self.encode_documents_batch(chunk_texts, show_progress=True)

        if missing:
            # A chunk that is not a slice of the document has no tokens to pool.
            reason = f"{len(missing)} of {len(chunk_texts)} chunks are not slices of the document"
            if strict_stages():
                self._late_chunking_unavailable(reason, chunk_texts)
            logger.info(f"Late chunking: {reason}; encoding those the standard way")
            fallback = self.encode_documents_batch(
                [chunk_texts[i] for i in missing], show_progress=False
            )
            for i, emb in zip(missing, fallback):
                chunk_embeddings[i] = emb

        chunk_embeddings = [self._maybe_truncate(e) for e in chunk_embeddings]
        logger.info(
            f"Late chunking: {len(chunk_texts) - len(missing)}/{len(chunk_texts)} chunks "
            f"from a {len(ids)}-token document in {n_windows} window(s)"
        )
        return chunk_embeddings

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

        with self._encode_lock, self._device_guard():
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

# Managers for models other than the singleton's, by config signature. A sweep
# arm's embedder or semantic chunker loads its model here once, and never
# replaces the singleton the rest of the pipeline is searching with.
_shared_managers: "OrderedDict[tuple, EmbeddingManager]" = OrderedDict()
_SHARED_MANAGERS_MAX = 4


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
                _embedding_manager = _shared_or_new(config)
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
                    _embedding_manager = _shared_or_new(config)
                elif _embedding_manager.signature != desired_signature:
                    logger.info(
                        "Embedding manager config change detected; reloading. "
                        f"old={_embedding_manager.signature} new={desired_signature}"
                    )
                    _embedding_manager = _shared_or_new(config)

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


def _shared_or_new(config: Optional[RAGConfig]) -> EmbeddingManager:
    """A manager already loaded for ``config``'s signature, or a new one (lock held)."""
    found = _shared_managers.get(_config_signature(config or rag_config))
    return found if found is not None else EmbeddingManager(config=config)


def shared_embedding_manager(config: Optional[RAGConfig] = None) -> EmbeddingManager:
    """The embedding manager for ``config``'s model, loaded at most once per process.

    For components that embed with a model of their own: a sweep arm's
    ``sentence_transformers`` embedder, the semantic chunker. Returns the
    pipeline's singleton when the settings match it, so the same model is never
    loaded twice, and otherwise a manager kept beside it. Unlike
    :func:`get_embedding_manager` it never replaces the singleton: an arm naming
    another model must not change the model every other search uses.
    """
    config = config or rag_config
    signature = _config_signature(config)
    mgr = _embedding_manager
    if mgr is not None and mgr.signature == signature:
        return mgr
    with _embedding_lock:
        found = _shared_managers.get(signature)
        if found is None:
            found = EmbeddingManager(config=config)
            _shared_managers[signature] = found
            while len(_shared_managers) > _SHARED_MANAGERS_MAX:
                _shared_managers.popitem(last=False)
        else:
            _shared_managers.move_to_end(signature)
        return found


def reset_embedding_manager():
    """Reset the embedding manager singleton and the shared managers.

    Useful for testing or when configuration changes.
    """
    global _embedding_manager
    with _embedding_lock:
        _shared_managers.clear()
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
            if strict_stages():
                # A zero vector is a document no query can find: under
                # evaluation the corpus would be silently smaller.
                raise RuntimeError(
                    f"{len(results) - len(valid)} of {len(results)} texts could not be embedded"
                )
            logger.warning(
                f"Embedding batch had {len(results) - len(valid)} failed entries "
                f"out of {len(results)}: replacing with zero vectors"
            )
            dim = self.dimension
            results = [r if r is not None else np.zeros(dim) for r in results]
        return np.array(results)

    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        return self._inner.encode_query(query)

    def encode_queries_batch(self, queries: List[str]) -> List[np.ndarray]:
        return self._inner.encode_queries_batch(queries)

    @property
    def dimension(self) -> int:
        return self._inner.dimension
