"""Embedder components: Hugging Face models, BGE-M3, and the OpenAI, Cohere and Voyage APIs.

Every one embeds documents and queries the way its model was trained to be
used: a query instruction or prefix for the families that need one (resolved
from the model name), ``input_type`` for the APIs that distinguish queries from
documents. ``model_name`` names the model that actually ran; a sweep keys its
corpus on it.

The local embedders share loaded models through
:func:`~lasagna.infrastructure.embedding.manager.shared_embedding_manager`,
so naming the model the pipeline already uses loads nothing new. The API
embedders need their package and key, and raise naming both when either is
missing.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from lasagna.core.base_components import BaseEmbedder
from lasagna.core.config import rag_config


def _arm_config(config: Any, **overrides: Any) -> Any:
    """``config`` with ``overrides`` on top of the arm's own per-query overrides."""
    config = config if config is not None else rag_config
    if not overrides:
        return config
    merged = {**(getattr(config, "_query_overrides", None) or {}), **overrides}
    return config.with_query_overrides(merged)


def _as_matrix(vectors: Sequence[np.ndarray], dim: int) -> np.ndarray:
    if not len(vectors):
        return np.empty((0, dim), dtype=np.float32)
    return np.asarray(np.stack([np.asarray(v) for v in vectors]), dtype=np.float32)


# ---------------------------------------------------------------------------
# Local models
# ---------------------------------------------------------------------------


class SentenceTransformersEmbedder(BaseEmbedder):
    """Any sentence-transformers model, named by ``embedding_model``.

    Applies the query/document prefixes the model's card prescribes: E5's
    ``query:``/``passage:``, the BGE v1.5 retrieval instruction, Nomic's
    ``search_query:``/``search_document:``, the instruction templates of
    Qwen3-Embedding, gte-Qwen and E5-Mistral, mxbai and Arctic-Embed's query
    prefixes, none for BGE-M3, GTE and MiniLM. Matryoshka truncation follows
    ``embedding_truncate_dim``; vectors are L2-normalized when
    ``normalize_embeddings`` is set. Always the sentence-transformers backend,
    even for BGE-M3 (see ``bge_m3`` for FlagEmbedding).

    The model loads on first use and is shared with the pipeline when it is
    the same one.

    Settings: ``embedding_model``, ``embedding_truncate_dim``,
    ``normalize_embeddings``.
    """

    # Built from the arm's embedding settings.
    _forwards_session_config = True

    #: The backend this component must run on.
    _backend = "sentence-transformers"

    def __init__(
        self,
        config: Any = None,
        model: Optional[str] = None,
        embedding_manager: Any = None,
        **kwargs: Any,
    ):
        from lasagna.infrastructure.embedding.manager import _resolve_backend

        config = _arm_config(config, **({"embedding_model": model} if model else {}))
        # Overridden only when the arm's setting would resolve to another
        # backend: an identical config shares the pipeline's loaded model.
        if _resolve_backend(config) != self._resolved_backend():
            config = _arm_config(config, embedding_backend=self._backend)
        self._config = config
        self._manager = embedding_manager
        self._configured_model = config.embedding_model

    def _resolved_backend(self) -> str:
        return self._backend

    @property
    def manager(self) -> Any:
        """The embedding manager holding the model, created on first use."""
        if self._manager is None:
            from lasagna.infrastructure.embedding.manager import shared_embedding_manager

            self._manager = shared_embedding_manager(self._config)
        return self._manager

    @property
    def model_name(self) -> str:
        if self._manager is None:
            return self._configured_model
        return self._manager.model_name or self._configured_model

    def encode(self, texts: List[str], **kwargs: Any) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        vectors = self.manager.encode_documents_batch(texts, show_progress=False)
        return _as_matrix(vectors, self.dimension)

    def encode_query(self, query: str, **kwargs: Any) -> np.ndarray:
        return np.asarray(self.manager.encode_query(query), dtype=np.float32)

    def encode_queries_batch(self, queries: List[str]) -> List[np.ndarray]:
        """Query vectors for several queries (the multi-query search path)."""
        return [np.asarray(v, dtype=np.float32) for v in self.manager.encode_queries_batch(queries)]

    def encode_documents_late_chunking(
        self, full_text: str, chunk_texts: List[str]
    ) -> List[np.ndarray]:
        """Late-chunked chunk vectors (see ``EmbeddingManager``) with this model."""
        return self.manager.encode_documents_late_chunking(full_text, chunk_texts)

    @property
    def dimension(self) -> int:
        manager = self.manager
        if not manager.is_loaded:
            _ = manager.model  # the configured embedding_dim is a guess until it loads
        return int(manager.dimension)


class BGEM3Embedder(SentenceTransformersEmbedder):
    """BGE-M3's dense vectors (Chen et al., 2024, arXiv:2402.03216).

    1024-dimensional, 8192-token context, 100+ languages, no query
    instruction. Runs through FlagEmbedding when it is installed and through
    sentence-transformers otherwise; both give the model's dense (CLS-pooled,
    normalized) vectors.

    The pipeline indexes dense vectors only. BGE-M3's other two outputs are
    available for experiments outside it: :meth:`encode_sparse` (lexical
    weights) and :meth:`encode_multivector` (ColBERT token vectors), both of
    which need FlagEmbedding.
    """

    MODEL = "BAAI/bge-m3"
    _backend = "auto"  # FlagEmbedding when installed

    def __init__(self, config: Any = None, embedding_manager: Any = None, **kwargs: Any):
        super().__init__(config=config, model=self.MODEL, embedding_manager=embedding_manager)

    def _resolved_backend(self) -> str:
        from lasagna.infrastructure.embedding.manager import FLAG_EMBEDDING_AVAILABLE

        return "flag-embedding" if FLAG_EMBEDDING_AVAILABLE else "sentence-transformers"

    def _flag_model(self, what: str) -> Any:
        manager = self.manager
        _ = manager.model
        if manager.backend_name != "flag-embedding":
            raise ImportError(
                f"BGE-M3 {what} needs FlagEmbedding: pip install 'lasagna-rag[flag-embedding]'"
            )
        return manager.model

    def encode_sparse(self, texts: List[str]) -> List[Dict[str, float]]:
        """Lexical weights per text: ``{token_id: weight}``."""
        output = self._flag_model("sparse output").encode(list(texts), return_sparse=True)
        return list(output["lexical_weights"])

    def encode_multivector(self, texts: List[str]) -> List[np.ndarray]:
        """ColBERT vectors per text: one ``(n_tokens, 1024)`` array each."""
        output = self._flag_model("multi-vector output").encode(
            list(texts), return_dense=False, return_colbert_vecs=True
        )
        return list(output["colbert_vecs"])


# ---------------------------------------------------------------------------
# API embedders
# ---------------------------------------------------------------------------


class _APIEmbedder(BaseEmbedder):
    """Batching, normalization and honest failure, shared by the API embedders."""

    _forwards_session_config = True

    provider: str = ""
    package: str = ""
    pip_name: str = ""
    key_vars: Tuple[str, ...] = ()
    #: Model-name prefixes of this provider: an ``embedding_model`` that starts
    #: with one is used; anything else (a Hugging Face id) gets ``default_model``.
    families: Tuple[str, ...] = ()
    default_model: str = ""
    native_dims: Dict[str, int] = {}
    #: Models that accept a requested output dimension.
    sizable: Tuple[str, ...] = ()
    max_batch: int = 96

    def __init__(
        self,
        config: Any = None,
        model: Optional[str] = None,
        dimensions: Optional[int] = None,
        client: Any = None,
        **kwargs: Any,
    ):
        self._config = config if config is not None else rag_config
        configured = getattr(self._config, "embedding_model", "") or ""
        if model:
            self._model = model
        elif configured.startswith(self.families):
            self._model = configured
        else:
            self._model = self.default_model
        if dimensions is None:
            dimensions = getattr(self._config, "embedding_truncate_dim", None)
        self._dimensions = int(dimensions) if dimensions else None
        if self._dimensions and not self._model.startswith(self.sizable):
            raise ValueError(
                f"{self._model} has a fixed output size; a dimension of "
                f"{self._dimensions} (embedding_truncate_dim) cannot be requested"
            )
        self._normalize = bool(getattr(self._config, "normalize_embeddings", True))
        self._client = client if client is not None else self._make_client()
        self._dim: Optional[int] = self._dimensions or self.native_dims.get(self._model)

    def _api_key(self) -> str:
        for var in self.key_vars:
            value = os.environ.get(var)
            if value:
                return value
        raise RuntimeError(
            f"the {self.provider} embedder needs an API key: set {' or '.join(self.key_vars)}"
        )

    def _import(self) -> Any:
        import importlib

        try:
            return importlib.import_module(self.package)
        except ImportError as exc:
            raise ImportError(
                f"the {self.provider} embedder needs the {self.pip_name} package: "
                f"pip install {self.pip_name}"
            ) from exc

    def _make_client(self) -> Any:
        raise NotImplementedError

    def _embed(self, texts: List[str], is_query: bool) -> List[List[float]]:
        raise NotImplementedError

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def name(self) -> str:
        return f"{self.provider}:{self._model}"

    def _vectors(self, texts: Sequence[str], is_query: bool) -> np.ndarray:
        # The APIs reject an empty string; a space embeds as near-nothing, as
        # an empty chunk does locally.
        texts = [t if t.strip() else " " for t in texts]
        rows: List[List[float]] = []
        for i in range(0, len(texts), self.max_batch):
            rows.extend(self._embed(texts[i : i + self.max_batch], is_query))
        matrix = np.asarray(rows, dtype=np.float32)
        if self._normalize and len(matrix):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.where(norms == 0, 1, norms)
        if len(matrix) and matrix.shape[1] != self._dim:
            if self._dimensions:
                raise ValueError(
                    f"{self.name} returned {matrix.shape[1]}-dimensional vectors; "
                    f"{self._dimensions} were requested"
                )
            if self._dim is not None:
                logger.warning(
                    f"{self.name} returned {matrix.shape[1]}-dimensional vectors, not the "
                    f"{self._dim} listed for it; using what it returned"
                )
            self._dim = int(matrix.shape[1])
        return matrix

    def encode(self, texts: List[str], **kwargs: Any) -> np.ndarray:
        texts = list(texts)
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        return self._vectors(texts, is_query=False)

    def encode_query(self, query: str, **kwargs: Any) -> np.ndarray:
        return self._vectors([query], is_query=True)[0]

    def encode_queries_batch(self, queries: List[str]) -> List[np.ndarray]:
        """Query vectors for several queries (the multi-query search path)."""
        return list(self._vectors(list(queries), is_query=True))

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self._vectors(["dimension probe"], is_query=False)  # an unlisted model
        return int(self._dim)


class OpenAIEmbedder(_APIEmbedder):
    """OpenAI embeddings: text-embedding-3-small (default), -3-large, ada-002.

    Symmetric: queries and documents are embedded the same way, as OpenAI
    documents. ``embedding_truncate_dim`` becomes the API's ``dimensions``
    (Matryoshka shortening, text-embedding-3 only). An ``embedding_model``
    starting with ``text-embedding-`` picks the model.

    Needs ``pip install openai`` and ``OPENAI_API_KEY`` (``OPENAI_BASE_URL`` is
    honoured by the client).
    """

    provider = "openai"
    package = "openai"
    pip_name = "openai"
    key_vars = ("OPENAI_API_KEY",)
    families = ("text-embedding-",)
    default_model = "text-embedding-3-small"
    native_dims = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }
    sizable = ("text-embedding-3",)
    max_batch = 256

    def _make_client(self) -> Any:
        openai = self._import()
        return openai.OpenAI(api_key=self._api_key())

    def _embed(self, texts: List[str], is_query: bool) -> List[List[float]]:
        extra = {"dimensions": self._dimensions} if self._dimensions else {}
        response = self._client.embeddings.create(model=self._model, input=texts, **extra)
        return [item.embedding for item in sorted(response.data, key=lambda d: d.index)]


class CohereEmbedder(_APIEmbedder):
    """Cohere Embed v3 and v4, with ``input_type`` search_query / search_document.

    embed-english-v3.0 by default; also the multilingual and light v3 models
    and embed-v4.0, whose output size ``embedding_truncate_dim`` can set (256,
    512, 1024 or 1536). An ``embedding_model`` starting with ``embed-`` picks
    the model.

    Needs ``pip install cohere`` and ``CO_API_KEY`` or ``COHERE_API_KEY``.
    """

    provider = "cohere"
    package = "cohere"
    pip_name = "cohere"
    key_vars = ("CO_API_KEY", "COHERE_API_KEY")
    families = ("embed-",)
    default_model = "embed-english-v3.0"
    native_dims = {
        "embed-english-v3.0": 1024,
        "embed-multilingual-v3.0": 1024,
        "embed-english-light-v3.0": 384,
        "embed-multilingual-light-v3.0": 384,
        "embed-v4.0": 1536,
    }
    sizable = ("embed-v4",)
    max_batch = 96  # the API's limit per request

    def _make_client(self) -> Any:
        cohere = self._import()
        client_cls = getattr(cohere, "ClientV2", None) or cohere.Client
        return client_cls(api_key=self._api_key())

    def _embed(self, texts: List[str], is_query: bool) -> List[List[float]]:
        extra = {"output_dimension": self._dimensions} if self._dimensions else {}
        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type="search_query" if is_query else "search_document",
            embedding_types=["float"],
            **extra,
        )
        embeddings = response.embeddings
        floats = getattr(embeddings, "float_", None) or getattr(embeddings, "float", None)
        return list(floats if floats is not None else embeddings)


class VoyageEmbedder(_APIEmbedder):
    """Voyage AI embeddings, voyage-3 family, with ``input_type`` query / document.

    voyage-3 by default; voyage-3-large, voyage-3.5, voyage-3.5-lite and
    voyage-code-3 also take an output size from ``embedding_truncate_dim``.
    Voyage prepends its own retrieval prompt for each input type. An
    ``embedding_model`` starting with ``voyage-`` picks the model.

    Needs ``pip install voyageai`` and ``VOYAGE_API_KEY``.
    """

    provider = "voyage"
    package = "voyageai"
    pip_name = "voyageai"
    key_vars = ("VOYAGE_API_KEY",)
    families = ("voyage-",)
    default_model = "voyage-3"
    native_dims = {
        "voyage-3": 1024,
        "voyage-3-lite": 512,
        "voyage-3-large": 1024,
        "voyage-3.5": 1024,
        "voyage-3.5-lite": 1024,
        "voyage-code-3": 1024,
    }
    sizable = ("voyage-3-large", "voyage-3.5", "voyage-code-3")
    max_batch = 128

    def _make_client(self) -> Any:
        voyageai = self._import()
        return voyageai.Client(api_key=self._api_key())

    def _embed(self, texts: List[str], is_query: bool) -> List[List[float]]:
        extra = {"output_dimension": self._dimensions} if self._dimensions else {}
        response = self._client.embed(
            texts,
            model=self._model,
            input_type="query" if is_query else "document",
            truncation=True,
            **extra,
        )
        return list(response.embeddings)
