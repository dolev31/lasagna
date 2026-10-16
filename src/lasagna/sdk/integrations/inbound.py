"""Bring a LangChain or LlamaIndex component into a Lasagna sweep.

The other direction from :mod:`.langchain`, which hands Lasagna's retriever to
LangChain. Here a component you already have becomes a pipeline stage::

    import lasagna
    from langchain_cohere import CohereRerank

    name = lasagna.from_langchain(CohereRerank(model="rerank-v3.5", top_n=10))
    # name == "langchain.CohereRerank"; a sweep run from Python names it like
    # any reranker:  pipeline: {rerankers: [cross_encoder, langchain.CohereRerank]}

A sweep file run with ``lasagna benchmark experiment`` can instead name the
object where it is defined, as it names a function: with
``cohere = CohereRerank(top_n=10)`` in ``my_rerankers.py``, the arm
``"my_rerankers:cohere"`` is wrapped the same way.

The object is recognized by the methods it has, so neither library is imported
to decide what it is:

=========================================================  ==========
The object has                                             Stage
=========================================================  ==========
``compress_documents`` (LangChain)                         reranker
``embed_documents`` and ``embed_query`` (LangChain)        embedder
``_get_relevant_documents``, called with ``invoke``        retriever
``postprocess_nodes`` (LlamaIndex)                         reranker
``get_text_embedding_batch`` and ``get_query_embedding``   embedder
``retrieve`` (LlamaIndex)                                  retriever
=========================================================  ==========

What the wrapper does to the documents it passes back:

* **Ids.** Retrieval is scored against document ids. A document the wrapped
  object returns gets its id from its metadata: the first of ``id_key``,
  ``doc_id``, ``document_id``, ``_id`` and ``id`` that is set (for LlamaIndex,
  then the node's source document id). When none is, the id is a hash of the
  text (``sha256:`` and 16 hex digits). A hashed id is stable across runs but
  matches no benchmark's relevance judgments, so every such document scores as
  irrelevant. A reranker's documents keep the ids they arrived with.
* **Scores.** A score the object reports is kept: ``relevance_score`` in a
  LangChain compressor's output (what CohereRerank and CrossEncoderReranker
  write), ``relevance_score`` or ``score`` in a LangChain retriever's documents,
  a LlamaIndex node's score. Otherwise the documents are scored 1/rank in the
  order returned.
* **Copies.** The object is handed copies, so nothing it does to them reaches
  another stage.

**A wrapped retriever brings its own index.** It searches whatever the object
was built over, not the corpus Lasagna ingested for the benchmark. For a
benchmark number, its index must hold that benchmark's corpus, with each
corpus id stored under ``id_key`` in the metadata; otherwise every query scores
zero.

The object's own limits still apply. A compressor's ``top_n`` or a retriever's
``k`` caps how many documents come back; set it at least as high as the
pipeline's depth, or the arm is scored on shorter lists than the others (a
warning says so when the cap is visible on the object).

Registration goes through the same path as ``@lasagna.reranker``: built-in
names are refused, registering a name again replaces it, and the component
listing describes the entry as ``wrapped: langchain.<Class>``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from ...core.base_components import BaseEmbedder, BaseReranker, BaseRetriever, ScoredDocument

#: Metadata keys read for a document id, after the caller's ``id_key``.
_ID_KEYS = ("doc_id", "document_id", "_id", "id")
#: Metadata keys read for a retrieved document's score. A compressor's score is
#: read from ``relevance_score`` alone, the key LangChain's rerankers write, so a
#: ``score`` the passage already carried is never taken for the compressor's.
_SCORE_KEYS = ("relevance_score", "score")
#: Where a reranker's input position travels, so a document is matched back to
#: the one it came from even when the compressor rebuilds it.
_POSITION = "_lasagna_position"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _document_id(metadata: Dict[str, Any], text: str, id_key: str, fallback: Any = None) -> str:
    """The id a returned document is scored under; see the module docstring."""
    for key in (id_key, *_ID_KEYS):
        value = metadata.get(key)
        if value not in (None, ""):
            return str(value)
    if fallback not in (None, ""):
        return str(fallback)
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _chunk_id(explicit: Any, metadata: Dict[str, Any], doc_id: str, text: str) -> str:
    for value in (explicit, metadata.get("chunk_id")):
        if value not in (None, ""):
            return str(value)
    return f"{doc_id}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:12]}"


def _score(value: Any, rank: int) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return 1.0 / (rank + 1)


def _metadata_score(metadata: Dict[str, Any]) -> Any:
    for key in _SCORE_KEYS:
        if key in metadata:
            return metadata[key]
    return None


#: How much of a configuration the digest reads: a retriever can hold its whole
#: corpus in a list field, and hashing millions of documents is not identity.
_DIGEST_DEPTH = 3
_DIGEST_ITEMS = 100


def _simple(value: Any, depth: int = 0) -> Any:
    """A JSON-able summary of a configuration value, for the object's digest.

    Plain values are kept; containers and pydantic models (what LangChain and
    LlamaIndex components are) are walked a few levels down, a long container
    by its length and first items; anything else, a loaded model, a client, a
    secret, counts by its type alone.
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if depth >= _DIGEST_DEPTH:
        return type(value).__qualname__
    if isinstance(value, (list, tuple)):
        return [len(value), [_simple(v, depth + 1) for v in value[:_DIGEST_ITEMS]]]
    if isinstance(value, dict):
        items = list(value.items())[:_DIGEST_ITEMS]
        return [len(value), {str(k): _simple(v, depth + 1) for k, v in items}]
    if hasattr(type(value), "model_fields"):
        return {type(value).__qualname__: _public_fields(value, depth + 1)}
    return type(value).__qualname__


def _public_fields(obj: Any, depth: int = 0) -> Dict[str, Any]:
    fields = getattr(obj, "__dict__", None) or {}
    return {k: _simple(v, depth) for k, v in sorted(fields.items()) if not k.startswith("_")}


def _object_digest(obj: Any) -> str:
    """What the wrapped object is: its class and its configuration.

    Part of the component's identity, which keys the search cache and an
    embedder's corpus. Without it, wrapping a second model under the same name
    would be served the first one's cached results and ingested corpus.
    """
    cls = type(obj)
    blob = json.dumps(
        {"class": f"{cls.__module__}.{cls.__qualname__}", "fields": _public_fields(obj)},
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def _depth_cap(obj: Any) -> Optional[Tuple[str, int]]:
    """The object's own result limit, when it exposes one."""
    for attr in ("top_n", "k", "similarity_top_k"):
        value = getattr(obj, attr, None)
        if isinstance(value, int) and not isinstance(value, bool):
            return attr, value
    kwargs = getattr(obj, "search_kwargs", None)
    if isinstance(kwargs, dict):
        # `vectorstore.as_retriever()` leaves k unset, and LangChain's vector
        # stores then return 4 documents.
        k = kwargs.get("k", 4)
        if isinstance(k, int) and not isinstance(k, bool):
            return "search_kwargs['k']", k
    return None


class _Wrapped:
    """Shared by every wrapper: the object, its registered name, its identity."""

    _wrapped: Any = None
    _registered_name: str = ""
    _label: str = ""
    _digest: str = ""
    _id_key: str = "doc_id"
    _warned_depth: bool = False

    @property
    def name(self) -> str:
        return self._registered_name or type(self).__name__

    @property
    def model_name(self) -> str:
        # Read for cache keys and provenance (see `_object_digest`).
        return f"{self._registered_name}=wrapped: {self._label}#{self._digest}"

    def _warn_if_capped(self, wanted: int) -> None:
        cap = _depth_cap(self._wrapped)
        if cap is None or cap[1] >= wanted or type(self)._warned_depth:
            return
        type(self)._warned_depth = True
        attr, value = cap
        logger.warning(
            f"{self._label} returns at most {attr}={value} documents, fewer than the "
            f"{wanted} this search asks for, so its arm is scored on shorter lists than "
            f"the others. Set {attr} to at least {wanted} on the object you wrapped."
        )


def _matched(position: Any, text: str, handed: Sequence[ScoredDocument], used: set) -> Any:
    """Which input a returned document came from: its position, else its text."""
    if isinstance(position, int) and 0 <= position < len(handed) and position not in used:
        return position
    for i, doc in enumerate(handed):
        if i not in used and doc.content == text:
            return i
    return None


def _reranked(
    returned: Sequence[Tuple[str, Dict[str, Any], Any, Any]],
    handed: Sequence[ScoredDocument],
    id_key: str,
) -> List[ScoredDocument]:
    """``(text, metadata, position, score)`` per returned item, as ScoredDocuments."""
    out: List[ScoredDocument] = []
    used: set = set()
    for rank, (text, metadata, position, score) in enumerate(returned):
        i = _matched(position, text, handed, used)
        if i is not None:
            used.add(i)
            doc_id, chunk_id = handed[i].document_id, handed[i].chunk_id
        else:
            doc_id = _document_id(metadata, text, id_key)
            chunk_id = _chunk_id(None, metadata, doc_id, text)
        out.append(
            ScoredDocument(
                content=text,
                score=_score(score, rank),
                document_id=doc_id,
                chunk_id=chunk_id,
                metadata=metadata,
            )
        )
    return out


# ---------------------------------------------------------------------------
# LangChain
# ---------------------------------------------------------------------------


class LangChainReranker(_Wrapped, BaseReranker):
    """A LangChain document compressor (``compress_documents``) as a reranker."""

    def rerank(
        self, query: str, documents: List[ScoredDocument], top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        from langchain_core.documents import Document

        handed = copy.deepcopy(list(documents))
        if not handed:
            return []
        self._warn_if_capped(top_k)
        lc_docs = [
            Document(page_content=d.content, metadata={**d.metadata, _POSITION: i})
            for i, d in enumerate(handed)
        ]
        returned = []
        for doc in self._wrapped.compress_documents(lc_docs, query) or []:
            metadata = copy.deepcopy(dict(getattr(doc, "metadata", None) or {}))
            position = metadata.pop(_POSITION, None)
            text = str(getattr(doc, "page_content", ""))
            returned.append((text, metadata, position, metadata.get("relevance_score")))
        return _reranked(returned, handed, self._id_key)[:top_k]


class LangChainEmbedder(_Wrapped, BaseEmbedder):
    """LangChain embeddings: ``embed_documents`` for passages, ``embed_query`` for queries."""

    _dimension: Optional[int] = None

    def encode(self, texts: List[str], **kwargs):
        import numpy as np

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.asarray(self._wrapped.embed_documents(list(texts)), dtype=np.float32)

    def encode_query(self, query: str, **kwargs):
        import numpy as np

        return np.asarray(self._wrapped.embed_query(query), dtype=np.float32)

    @property
    def dimension(self) -> int:
        dimension = type(self)._dimension
        if dimension is None:
            dimension = type(self)._dimension = len(self._wrapped.embed_query("dimension probe"))
        return dimension


class LangChainRetriever(_Wrapped, BaseRetriever):
    """A LangChain retriever, called with ``invoke(query)``.

    It searches its own index, not the corpus Lasagna ingested; see the module
    docstring for what that means for a benchmark.
    """

    def retrieve(self, query: str, collection: str, top_k: int = 5, **kwargs):
        self._warn_if_capped(top_k)
        out: List[ScoredDocument] = []
        for rank, doc in enumerate(self._wrapped.invoke(query) or []):
            metadata = copy.deepcopy(dict(getattr(doc, "metadata", None) or {}))
            text = str(getattr(doc, "page_content", ""))
            doc_id = _document_id(metadata, text, self._id_key)
            out.append(
                ScoredDocument(
                    content=text,
                    score=_score(_metadata_score(metadata), rank),
                    document_id=doc_id,
                    chunk_id=_chunk_id(getattr(doc, "id", None), metadata, doc_id, text),
                    metadata=metadata,
                )
            )
        return out[:top_k]


# ---------------------------------------------------------------------------
# LlamaIndex
# ---------------------------------------------------------------------------


class _Node:
    """The parts of a LlamaIndex node a postprocessor reads, for an object that
    duck-types LlamaIndex without it being installed."""

    def __init__(self, text: str, metadata: Dict[str, Any], node_id: str):
        self.text, self.metadata, self.node_id, self.id_ = text, metadata, node_id, node_id

    def get_content(self, *args, **kwargs) -> str:
        return self.text


class _NodeWithScore:
    def __init__(self, node: _Node, score: Optional[float]):
        self.node, self.score = node, score

    def get_content(self, *args, **kwargs) -> str:
        return self.node.get_content()


def _nodes(handed: Sequence[ScoredDocument]) -> List[Any]:
    try:
        from llama_index.core.schema import NodeWithScore, TextNode
    except ImportError:  # a duck-typed object; LlamaIndex itself is not installed
        return [
            _NodeWithScore(_Node(d.content, dict(d.metadata), f"lasagna-{i}"), d.score)
            for i, d in enumerate(handed)
        ]
    return [
        NodeWithScore(
            node=TextNode(text=d.content, metadata=dict(d.metadata), id_=f"lasagna-{i}"),
            score=d.score,
        )
        for i, d in enumerate(handed)
    ]


def _node_parts(item: Any) -> Tuple[Any, str, Dict[str, Any]]:
    node = getattr(item, "node", item)
    get_content = getattr(node, "get_content", None)
    text = str(get_content() if callable(get_content) else getattr(node, "text", ""))
    return node, text, copy.deepcopy(dict(getattr(node, "metadata", None) or {}))


def _position(node_id: Any) -> Optional[int]:
    if isinstance(node_id, str) and node_id.startswith("lasagna-"):
        try:
            return int(node_id[len("lasagna-") :])
        except ValueError:
            return None
    return None


class LlamaIndexReranker(_Wrapped, BaseReranker):
    """A LlamaIndex node postprocessor (``postprocess_nodes``) as a reranker."""

    def rerank(
        self, query: str, documents: List[ScoredDocument], top_k: int = 5, **kwargs
    ) -> List[ScoredDocument]:
        handed = copy.deepcopy(list(documents))
        if not handed:
            return []
        self._warn_if_capped(top_k)
        returned = []
        for item in self._wrapped.postprocess_nodes(_nodes(handed), query_str=query) or []:
            node, text, metadata = _node_parts(item)
            position = _position(getattr(node, "node_id", None))
            returned.append((text, metadata, position, getattr(item, "score", None)))
        return _reranked(returned, handed, self._id_key)[:top_k]


class LlamaIndexEmbedder(_Wrapped, BaseEmbedder):
    """A LlamaIndex embedding: ``get_text_embedding_batch`` for passages,
    ``get_query_embedding`` for queries."""

    _dimension: Optional[int] = None

    def encode(self, texts: List[str], **kwargs):
        import numpy as np

        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)
        return np.asarray(self._wrapped.get_text_embedding_batch(list(texts)), dtype=np.float32)

    def encode_query(self, query: str, **kwargs):
        import numpy as np

        return np.asarray(self._wrapped.get_query_embedding(query), dtype=np.float32)

    @property
    def dimension(self) -> int:
        dimension = type(self)._dimension
        if dimension is None:
            dimension = type(self)._dimension = len(
                self._wrapped.get_query_embedding("dimension probe")
            )
        return dimension


class LlamaIndexRetriever(_Wrapped, BaseRetriever):
    """A LlamaIndex retriever, called with ``retrieve(query)``.

    It searches its own index, not the corpus Lasagna ingested; see the module
    docstring for what that means for a benchmark.
    """

    def retrieve(self, query: str, collection: str, top_k: int = 5, **kwargs):
        self._warn_if_capped(top_k)
        out: List[ScoredDocument] = []
        for rank, item in enumerate(self._wrapped.retrieve(query) or []):
            node, text, metadata = _node_parts(item)
            doc_id = _document_id(metadata, text, self._id_key, getattr(node, "ref_doc_id", None))
            out.append(
                ScoredDocument(
                    content=text,
                    score=_score(getattr(item, "score", None), rank),
                    document_id=doc_id,
                    chunk_id=_chunk_id(getattr(node, "node_id", None), metadata, doc_id, text),
                    metadata=metadata,
                )
            )
        return out[:top_k]


# ---------------------------------------------------------------------------
# Detection and registration
# ---------------------------------------------------------------------------

#: (methods the object must have, stage, wrapper, what to call it in an error)
_LANGCHAIN = (
    (("compress_documents",), "reranker", LangChainReranker, "a document compressor"),
    (("embed_documents", "embed_query"), "embedder", LangChainEmbedder, "embeddings"),
    (("_get_relevant_documents", "invoke"), "retriever", LangChainRetriever, "a retriever"),
)
_LLAMAINDEX = (
    (("postprocess_nodes",), "reranker", LlamaIndexReranker, "a node postprocessor"),
    (
        ("get_text_embedding_batch", "get_query_embedding"),
        "embedder",
        LlamaIndexEmbedder,
        "an embedding",
    ),
    (("retrieve",), "retriever", LlamaIndexRetriever, "a retriever"),
)
_LIBRARIES = {"langchain": _LANGCHAIN, "llamaindex": _LLAMAINDEX}


def _has(obj: Any, method: str) -> bool:
    return callable(getattr(obj, method, None))


def _a(stage: str) -> str:
    return f"{'an' if stage[0] in 'aeiou' else 'a'} {stage}"


def _unsupported(obj: Any, library: str) -> TypeError:
    probed = sorted({m for methods, *_ in _LANGCHAIN + _LLAMAINDEX for m in methods})
    found = [m for m in probed if _has(obj, m)]
    kinds = [
        f"{what} ({' and '.join(methods)}) as {_a(stage)}"
        for methods, stage, _, what in _LIBRARIES[library]
    ]
    other = "llamaindex" if library == "langchain" else "langchain"
    hint = ""
    if any(all(_has(obj, m) for m in methods) for methods, *_ in _LIBRARIES[other]):
        hint = f" It looks like a {other} component: use lasagna.from_{other}."
    cls = type(obj).__qualname__
    return TypeError(
        f"Cannot wrap {cls}. from_{library} supports {', '.join(kinds[:-1])} and "
        f"{kinds[-1]}; {cls} has {', '.join(found) if found else 'none of these methods'}."
        f"{hint} For anything else, write a function and register it with @lasagna.<stage>."
    )


def _wrapper(obj: Any, library: str, name: Optional[str], id_key: str) -> Tuple[str, str, type]:
    """``(stage, registered name, wrapper class)`` for ``obj``."""
    if inspect.isclass(obj):
        raise TypeError(
            f"from_{library} takes the object you built, not the class: pass "
            f"{obj.__qualname__}(...), not {obj.__qualname__}."
        )
    for methods, stage, wrapper, _ in _LIBRARIES[library]:
        if all(_has(obj, m) for m in methods):
            break
    else:
        raise _unsupported(obj, library)

    label = f"{library}.{type(obj).__name__}"
    name = name or label
    digest = _object_digest(obj)
    # One class per wrapped object, named after its digest, so the corpus key
    # (registered name, class) of an embedder changes with what it wraps.
    cls_name = f"{wrapper.__name__}_{digest}"
    cls = type(
        cls_name,
        (wrapper,),
        {
            "_wrapped": staticmethod(obj),
            "_registered_name": name,
            "_label": label,
            "_digest": digest,
            "_id_key": id_key,
            "__doc__": f"wrapped: {label}",
            "__module__": __name__,
            "__qualname__": cls_name,
        },
    )
    return stage, name, cls


def _wrap(obj: Any, library: str, name: Optional[str], id_key: str) -> str:
    from ... import components

    stage, name, cls = _wrapper(obj, library, name, id_key)
    components.register(stage, cls, name)
    return name


def wrapper_class(stage: str, obj: Any, library: str, name: str) -> type:
    """The class to register when a sweep names a LangChain or LlamaIndex object
    as ``module:attr`` under ``stage``; refuses one that fills another stage."""
    found, _, cls = _wrapper(obj, library, name, "doc_id")
    if found != stage:
        raise TypeError(
            f"{name} is {type(obj).__qualname__} from {library}, which Lasagna wraps as "
            f"{_a(found)}, not {_a(stage)}. Name it where {_a(found)} goes."
        )
    return cls


def from_langchain(obj: Any, name: Optional[str] = None, id_key: str = "doc_id") -> str:
    """Register a LangChain compressor, embeddings or retriever as a stage.

    Args:
        obj: The object you built: a document compressor (a reranker),
            embeddings (an embedder) or a retriever (it brings its own index).
        name: The registered name; ``langchain.<Class>`` by default.
        id_key: The metadata key holding a returned document's id.

    Returns:
        The registered name, for a sweep's ``rerankers``, ``embedders`` or a
        ``retriever:`` override.
    """
    return _wrap(obj, "langchain", name, id_key)


def from_llamaindex(obj: Any, name: Optional[str] = None, id_key: str = "doc_id") -> str:
    """Register a LlamaIndex node postprocessor, embedding or retriever as a stage.

    Args:
        obj: The object you built: a node postprocessor (a reranker), an
            embedding (an embedder) or a retriever (it brings its own index).
        name: The registered name; ``llamaindex.<Class>`` by default.
        id_key: The metadata key holding a returned node's document id; the
            node's source document id is the fallback.

    Returns:
        The registered name, for a sweep's ``rerankers``, ``embedders`` or a
        ``retriever:`` override.
    """
    return _wrap(obj, "llamaindex", name, id_key)
