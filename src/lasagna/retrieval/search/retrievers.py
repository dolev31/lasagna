"""Built-in first-stage retrievers: dense, BM25, SPLADE and two hybrid fusions.

Each is a ``BaseRetriever`` the search orchestrator calls in place of its own
vector search when an arm names it (``retriever: bm25``). They read the arm's
settings from ``config=`` and the session's store from ``context=`` (a
``RetrievalContext``): the backend, the session and document filters, the query
embedding the default search would have used, and the hybrid searcher whose
per-session BM25 index holds the session's chunks.

Lexical and learned-sparse retrieval score that index's passages exhaustively,
so ``bm25_k1``/``bm25_b`` and ``splade_model`` take effect on every backend,
including ones whose native BM25 fixes its parameters at index time.
"""

from __future__ import annotations

import copy
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from lasagna.core.base_components import BaseRetriever, ScoredDocument
from lasagna.core.config import rag_config

from .hybrid import (
    BM25_FALLBACK_AVAILABLE,
    BM25FallbackIndex,
    _get_splade_encoder,
)


def _context(component: str, kwargs: Dict[str, Any]) -> Any:
    """The search context, or a clear error when called outside a pipeline search."""
    context = kwargs.get("context")
    if context is None:
        raise RuntimeError(
            f"The {component} retriever reads a session's store, which the search "
            "pipeline passes as context=; run it through a pipeline query"
        )
    return context


def _setting(config: Any, key: str) -> Any:
    value = getattr(config, key, None) if config is not None else None
    return value if value is not None else getattr(rag_config, key)


# ---------------------------------------------------------------------------
# The session's corpus
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Corpus:
    """A consistent snapshot of a session's chunks, as the BM25 index holds them."""

    documents: List[Dict[str, Any]]
    tokens: List[List[str]]
    tokenize: Any
    #: Identifies this snapshot, so encodings and BM25 statistics built from it
    #: are reused until a chunk is added or removed.
    key: Tuple[Any, ...]


_private_indices: Dict[Tuple[Any, ...], BM25FallbackIndex] = {}
_private_lock = threading.Lock()


def _private_index(context: Any, config: Any) -> BM25FallbackIndex:
    """The session's chunks read straight from the store, once per process."""
    key = (id(context.backend), context.collection, context.session_id, context.partition_key)
    with _private_lock:
        index = _private_indices.get(key)
        if index is not None:
            return index
        index = BM25FallbackIndex(enable_stemming=bool(_setting(config, "enable_bm25_stemming")))
        rows = context.backend.iter_all(
            collection_name=context.collection,
            filter_expr=context.session_filter,
            output_fields=["chunk_id", "content", "document_id", "metadata"],
            partition_key=context.partition_key,
        )
        index.add_documents(
            [
                {
                    "chunk_id": r.get("chunk_id"),
                    "content": r.get("content", "") or "",
                    "document_id": r.get("document_id", "") or "",
                    "metadata": r.get("metadata", {}) or {},
                }
                for r in rows
                if r.get("chunk_id")
            ]
        )
        if index.documents:  # an empty store is read again next time
            _private_indices[key] = index
        return index


def _session_corpus(context: Any, config: Any) -> _Corpus:
    """The session's chunks, from the hybrid searcher's BM25 index where it has them.

    That index is filled at upload under hybrid profiles and from the store on
    first use otherwise. A backend with native BM25 never fills it, so the
    chunks are then read from the store into an index of this module's own.
    """
    if not BM25_FALLBACK_AVAILABLE:
        raise ImportError("Lexical retrieval needs rank_bm25: pip install rank-bm25")
    searcher = context.hybrid_searcher
    index = None
    if searcher is not None and context.session_filter is not None:
        searcher.hydrate_session_index(
            context.session_id,
            context.backend,
            context.collection,
            context.session_filter,
            partition_key=context.partition_key,
        )
        index = searcher.get_session_index(context.session_id)
        lock = searcher._session_lock  # held by uploads while they append
    if index is None or not index.documents:
        index = _private_index(context, config)
        lock = _private_lock
    with lock:
        documents = list(index.documents)
        tokens = list(index.tokenized_corpus)
        version = index._version
    n = min(len(documents), len(tokens))
    if n == 0:
        raise RuntimeError(f"No chunks are stored for session {context.session_id!r}")
    # An id() can be reused once an evicted index is freed; this name cannot.
    name = index.__dict__.setdefault("_corpus_name", uuid.uuid4().hex)
    return _Corpus(documents[:n], tokens[:n], index.tokenize, (name, version, n))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_bm25_cache: "OrderedDict[Tuple[Any, ...], Any]" = OrderedDict()
_bm25_lock = threading.Lock()


def _bm25_scores(corpus: _Corpus, query: str, k1: float, b: float) -> np.ndarray:
    """Okapi BM25 of ``query`` against every passage of the corpus."""
    from rank_bm25 import BM25Okapi

    key = (*corpus.key, float(k1), float(b))
    with _bm25_lock:
        model = _bm25_cache.get(key)
        if model is None:
            model = BM25Okapi(corpus.tokens, k1=float(k1), b=float(b))
            _bm25_cache[key] = model
            while len(_bm25_cache) > 8:
                _bm25_cache.popitem(last=False)
    terms = corpus.tokenize(query)
    if not terms:
        return np.zeros(len(corpus.documents))
    return np.asarray(model.get_scores(terms), dtype=np.float64)


def _allowed(corpus: _Corpus, document_ids: Optional[Sequence[str]]) -> Optional[np.ndarray]:
    if not document_ids:
        return None
    wanted = set(document_ids)
    return np.array([d.get("document_id") in wanted for d in corpus.documents])


def _top_positive(
    corpus: _Corpus, scores: np.ndarray, k: int, allowed: Optional[np.ndarray]
) -> List[int]:
    """Indices of the ``k`` best positive scores, one per chunk; ties keep corpus order."""
    candidates = scores > 0
    if allowed is not None:
        candidates &= allowed
    idx = np.flatnonzero(candidates)
    seen, out = set(), []
    for i in idx[np.argsort(-scores[idx], kind="stable")]:
        chunk_id = str(corpus.documents[i].get("chunk_id"))
        if chunk_id in seen:  # an index filled twice holds a chunk twice
            continue
        seen.add(chunk_id)
        out.append(int(i))
        if len(out) >= k:
            break
    return out


def _from_corpus(corpus: _Corpus, i: int, score: float) -> ScoredDocument:
    doc = corpus.documents[i]
    return ScoredDocument(
        content=doc.get("content", "") or "",
        score=float(score),
        document_id=doc.get("document_id", "") or "",
        chunk_id=str(doc.get("chunk_id")),
        metadata=copy.deepcopy(doc.get("metadata") or {}),
    )


def _from_hit(hit: Any, score: float) -> ScoredDocument:
    return ScoredDocument(
        content=hit.entity.get("content", "") or "",
        score=float(score),
        document_id=hit.entity.get("document_id", "") or "",
        chunk_id=str(hit.id),
        metadata=copy.deepcopy(hit.entity.get("metadata") or {}),
    )


def _query_vector(context: Any, query: str) -> np.ndarray:
    if context.query_embedding is not None:
        return np.asarray(context.query_embedding, dtype=np.float64)
    return np.asarray(context.embedding_manager.encode_query(query), dtype=np.float64)


def _vector_search(context: Any, collection: str, query: str, limit: int) -> List[Any]:
    """The default dense search: the same call, filter and parameters."""
    return context.backend.search(
        collection_name=collection,
        query_embedding=_query_vector(context, query).tolist(),
        vector_field="embedding",
        filter_expr=context.filter_expr,
        limit=limit,
        output_fields=["document_id", "content", "metadata", "embedding"],
        search_params=context.search_params,
        partition_key=context.partition_key,
    )


# ---------------------------------------------------------------------------
# Retrievers
# ---------------------------------------------------------------------------


class DenseRetriever(BaseRetriever):
    """Dense retrieval: the chunks nearest the query embedding in the vector store.

    The pipeline's default first stage as a component, so a retriever sweep has
    a like-for-like control: the bi-encoder setting of Karpukhin et al. 2020
    (DPR), with the configured embedder, metric and index.
    """

    _forwards_session_config = True

    def retrieve(
        self, query: str, collection: str, top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        context = _context("dense", kwargs)
        return [_from_hit(h, h.score) for h in _vector_search(context, collection, query, top_k)]


class BM25Retriever(BaseRetriever):
    """Okapi BM25 over the session's chunks (Robertson & Zaragoza 2009).

    "The Probabilistic Relevance Framework: BM25 and Beyond". ``bm25_k1`` (1.2)
    and ``bm25_b`` (0.75) are the classic Okapi values, inside the ranges the
    paper recommends. Text is tokenized as the hybrid index tokenizes it
    (lowercased, Porter-stemmed under ``enable_bm25_stemming``); IDF is
    rank_bm25's Okapi form, which floors the negative IDF of a term found in
    more than half the passages. Passages sharing no term with the query are
    not returned.
    """

    _forwards_session_config = True

    def retrieve(
        self, query: str, collection: str, top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        context = _context("bm25", kwargs)
        config = kwargs.get("config")
        corpus = _session_corpus(context, config)
        scores = _bm25_scores(
            corpus, query, _setting(config, "bm25_k1"), _setting(config, "bm25_b")
        )
        top = _top_positive(corpus, scores, top_k, _allowed(corpus, context.document_ids))
        return [_from_corpus(corpus, i, scores[i]) for i in top]


class SPLADERetriever(BaseRetriever):
    """Learned sparse retrieval with SPLADE (Formal et al. 2021, "SPLADE v2").

    Query and passages are encoded by the ``splade_model`` masked language model
    into vocabulary-sized weights, ``max over tokens of log(1 + ReLU(logit))``,
    and scored by their dot product. The default checkpoint is SPLADE++
    CoCondenser-EnsembleDistil (Formal et al. 2022). The session's passages are
    encoded on first use, exhaustively, and kept until a chunk is added or
    removed. Raises if the model cannot load or has no language-model head.
    """

    _forwards_session_config = True

    def retrieve(
        self, query: str, collection: str, top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        context = _context("splade", kwargs)
        config = kwargs.get("config")
        corpus = _session_corpus(context, config)
        encoder = _get_splade_encoder(_setting(config, "splade_model"))
        encoded = encoder.corpus(
            (encoder.model_name, *corpus.key),
            [d.get("content", "") or "" for d in corpus.documents],
        )
        ids, weights = encoder.encode([query])[0]
        scores = encoded.score(ids, weights)
        top = _top_positive(corpus, scores, top_k, _allowed(corpus, context.document_ids))
        return [_from_corpus(corpus, i, scores[i]) for i in top]


def _depth(config: Any, top_k: int) -> int:
    return max(int(_setting(config, "hybrid_fusion_depth")), int(top_k))


class HybridRRFRetriever(BaseRetriever):
    """Dense and BM25 lists fused by Reciprocal Rank Fusion (Cormack et al. 2009).

    Cormack, Clarke & Buettcher, "Reciprocal Rank Fusion outperforms Condorcet
    and individual Rank Learning Methods" (SIGIR 2009): a chunk scores
    ``sum 1 / (rrf_k + rank)`` over the lists it appears in, ranks counted
    from 1, ``rrf_k`` = 60 as in the paper. Each list contributes its top
    ``hybrid_fusion_depth``. Unweighted, as the paper defines it.
    """

    _forwards_session_config = True

    def retrieve(
        self, query: str, collection: str, top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        context = _context("hybrid_rrf", kwargs)
        config = kwargs.get("config")
        depth = _depth(config, top_k)
        k = float(_setting(config, "rrf_k"))

        hits = _vector_search(context, collection, query, depth)
        corpus = _session_corpus(context, config)
        lexical_scores = _bm25_scores(
            corpus, query, _setting(config, "bm25_k1"), _setting(config, "bm25_b")
        )
        lexical = _top_positive(
            corpus, lexical_scores, depth, _allowed(corpus, context.document_ids)
        )

        fused: Dict[str, float] = {}
        source: Dict[str, ScoredDocument] = {}
        for rank, hit in enumerate(hits, start=1):
            chunk_id = str(hit.id)
            if chunk_id not in fused:
                fused[chunk_id] = 1.0 / (k + rank)
                source[chunk_id] = _from_hit(hit, 0.0)
        for rank, i in enumerate(lexical, start=1):
            chunk_id = str(corpus.documents[i].get("chunk_id"))
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
            source.setdefault(chunk_id, _from_corpus(corpus, i, 0.0))

        position = {chunk_id: n for n, chunk_id in enumerate(fused)}
        ranked = sorted(fused, key=lambda c: (-fused[c], position[c]))[:top_k]
        return [_with_score(source[c], fused[c]) for c in ranked]


def _with_score(doc: ScoredDocument, score: float) -> ScoredDocument:
    return ScoredDocument(
        content=doc.content,
        score=float(score),
        document_id=doc.document_id,
        chunk_id=doc.chunk_id,
        metadata=doc.metadata,
    )


#: Normalisations for hybrid_convex, as Bruch et al. name them.
CONVEX_NORMALIZATIONS = ("tmm", "minmax", "zscore")


def _normalize(values: np.ndarray, method: str, floor: Optional[float]) -> np.ndarray:
    """One list's scores normalised over the fused set (Bruch et al. 2023, section 4)."""
    if method == "tmm":
        if floor is None:
            raise ValueError(
                "theoretical min-max needs a similarity with a known minimum: cosine, "
                "or inner product / L2 over normalised embeddings. Set "
                "hybrid_convex_normalization to 'minmax' or 'zscore'"
            )
        span = values.max() - floor
        return (values - floor) / span if span > 0 else np.zeros_like(values)
    if method == "minmax":
        span = values.max() - values.min()
        return (values - values.min()) / span if span > 0 else np.zeros_like(values)
    if method == "zscore":
        sd = values.std()
        return (values - values.mean()) / sd if sd > 0 else np.zeros_like(values)
    raise ValueError(
        f"hybrid_convex_normalization must be one of {CONVEX_NORMALIZATIONS}, not {method!r}"
    )


def _similarity(metric: str, normalized: bool):
    """The dense scoring function and its theoretical minimum (None if unbounded)."""
    metric = str(getattr(metric, "value", metric)).upper()
    if metric == "COSINE":
        return (
            lambda q, e: float(q @ e / ((np.linalg.norm(q) * np.linalg.norm(e)) or 1.0)),
            -1.0,
        )
    if metric == "IP":
        return (lambda q, e: float(q @ e)), (-1.0 if normalized else None)
    if metric == "L2":
        return (lambda q, e: -float(np.linalg.norm(q - e))), (-2.0 if normalized else None)
    raise ValueError(f"hybrid_convex cannot score metric_type {metric!r}")


def _stored_embeddings(context: Any, collection: str, chunk_ids: List[str]) -> Dict[str, Any]:
    """Embeddings the store holds for these chunks, read in batches."""
    from lasagna.vector_backends.filters import FilterExpression

    found: Dict[str, Any] = {}
    for start in range(0, len(chunk_ids), 200):
        batch = chunk_ids[start : start + 200]
        parts = [FilterExpression.in_list("chunk_id", batch)]
        if context.session_filter is not None:
            parts.append(context.session_filter)
        rows = context.backend.query(
            collection_name=collection,
            filter_expr=FilterExpression.and_all(parts),
            output_fields=["chunk_id", "embedding"],
            limit=len(batch),
            partition_key=context.partition_key,
        )
        for row in rows or []:
            if row.get("embedding") is not None:
                found[str(row.get("chunk_id"))] = row["embedding"]
    return found


class HybridConvexRetriever(BaseRetriever):
    """Convex combination of normalised dense and BM25 scores (Bruch et al. 2023).

    Bruch, Gai & Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval"
    (ACM TOIS): ``alpha * phi(semantic) + (1 - alpha) * phi(BM25)`` over the
    union of both top-``hybrid_fusion_depth`` lists, ``alpha`` = ``hybrid_alpha``.
    A chunk missing from one list has that score computed, as the paper does,
    rather than set to zero: its BM25 against the index, or its stored
    embedding's similarity to the query. ``phi`` is
    ``hybrid_convex_normalization``: ``tmm`` (the default and the paper's
    recommendation), ``(s - inf) / (max - inf)`` with ``inf`` the scoring
    function's theoretical minimum, 0 for BM25 and -1 for cosine; ``minmax``;
    or ``zscore``, each over the union.
    """

    _forwards_session_config = True

    def retrieve(
        self, query: str, collection: str, top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        context = _context("hybrid_convex", kwargs)
        config = kwargs.get("config")
        depth = _depth(config, top_k)
        alpha = float(_setting(config, "hybrid_alpha"))
        method = str(_setting(config, "hybrid_convex_normalization")).lower()
        similarity, floor = _similarity(
            _setting(config, "metric_type"), bool(_setting(config, "normalize_embeddings"))
        )
        if method == "tmm" and floor is None:
            _normalize(np.zeros(1), method, floor)  # raises, naming the alternatives

        hits = _vector_search(context, collection, query, depth)
        corpus = _session_corpus(context, config)
        lexical_scores = _bm25_scores(
            corpus, query, _setting(config, "bm25_k1"), _setting(config, "bm25_b")
        )
        lexical = _top_positive(
            corpus, lexical_scores, depth, _allowed(corpus, context.document_ids)
        )
        position = {}
        for i, doc in enumerate(corpus.documents):
            position.setdefault(str(doc.get("chunk_id")), i)

        union: Dict[str, ScoredDocument] = {}
        embeddings: Dict[str, Any] = {}
        for hit in hits:
            chunk_id = str(hit.id)
            if chunk_id not in union:
                union[chunk_id] = _from_hit(hit, 0.0)
                if hit.entity.get("embedding") is not None:
                    embeddings[chunk_id] = hit.entity["embedding"]
        for i in lexical:
            union.setdefault(str(corpus.documents[i].get("chunk_id")), _from_corpus(corpus, i, 0.0))
        if not union:
            return []

        missing = [c for c in union if c not in embeddings]
        if missing:
            embeddings.update(_stored_embeddings(context, collection, missing))
        unstored = [c for c in union if c not in embeddings]
        if unstored:
            # A backend that does not return vectors: embed the text instead.
            logger.debug(
                f"hybrid_convex: embedding {len(unstored)} chunks the store did not return"
            )
            vectors = context.embedding_manager.encode_documents_batch(
                [union[c].content for c in unstored], show_progress=False
            )
            embeddings.update(zip(unstored, vectors))

        q = _query_vector(context, query)
        ids = list(union)
        semantic = np.array(
            [similarity(q, np.asarray(embeddings[c], dtype=np.float64)) for c in ids]
        )
        lexical_union = np.array(
            [lexical_scores[position[c]] if c in position else 0.0 for c in ids]
        )
        fused = alpha * _normalize(semantic, method, floor) + (1.0 - alpha) * _normalize(
            lexical_union, method, 0.0
        )
        order = sorted(range(len(ids)), key=lambda n: (-fused[n], n))[:top_k]
        return [_with_score(union[ids[n]], fused[n]) for n in order]
