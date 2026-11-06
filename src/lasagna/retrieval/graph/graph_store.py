"""The knowledge graph a session's documents build, stored where every process reads it.

The graph used to live in a module-level dict of NetworkX graphs, in the memory
of the process that ingested. In the Compose stack the Celery worker ingests
and the API workers search, so every search expanded through an empty graph,
silently; a restart lost every graph; a benchmark that reused an ingested
corpus had none; and nothing ever removed one (#55).

`DatabaseGraphStore` keeps the graph in the metadata database every process
already shares: SQLite for ``lasagna serve``, the CLI and benchmarks,
PostgreSQL in the stack. It is keyed by the session and the collection the
chunks are in, so two embedders never share a graph. Every row names the
document it came from, so a document's part can be removed. A process loads a
graph into memory once (`InMemoryGraphStore`) and loads it again when the
session's content generation moves, which every upload and delete bumps.
"""

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, NamedTuple, Optional, Protocol, Sequence, Set, Tuple

import numpy as np
from loguru import logger

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

#: Below this cosine similarity, an entity does not match the query.
_MIN_SIMILARITY = 0.3


class GraphEntity(NamedTuple):
    """An entity a chunk mentions."""

    name: str
    entity_type: str
    embedding: Optional[np.ndarray]
    chunk_id: str


class GraphRelationship(NamedTuple):
    """A directed relationship between two entities."""

    source: str
    target: str
    rel_type: str


class GraphStore(Protocol):
    """What graph retrieval needs from a store: one graph per session and collection."""

    def add_document(
        self,
        document_id: str,
        entities: Sequence[GraphEntity],
        relationships: Sequence[GraphRelationship],
    ) -> None: ...

    def find_entities_by_embedding(
        self, query_embedding: np.ndarray, top_k: int = 10
    ) -> List[str]: ...

    def find_entities_by_name(self, query: str, limit: int = 5) -> List[str]: ...

    def get_neighbors(self, entity_id: str, hops: int = 1) -> Set[str]: ...

    def get_chunk_ids_for_entities(self, entity_ids: Set[str]) -> Set[str]: ...

    @property
    def num_entities(self) -> int: ...

    @property
    def num_relationships(self) -> int: ...

    def remove_document(self, document_id: str) -> None: ...


class InMemoryGraphStore:
    """A directed knowledge graph held in this process's memory.

    The graph a `DatabaseGraphStore` loads, and a store on its own for tests.
    Each mention and edge remembers its document, so a document's part can be
    removed.
    """

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required for GraphRAG. Install with: pip install networkx>=3.0"
            )
        self._graph = nx.DiGraph()
        self._entity_embeddings: Dict[str, np.ndarray] = {}
        # Map entity_id -> set of chunk_ids that mention it
        self._entity_chunks: Dict[str, Set[str]] = {}
        #: (entity, chunk) -> the document of the chunk
        self._mention_documents: Dict[Tuple[str, str], str] = {}
        #: (source, target) -> the documents that stated the edge
        self._edge_documents: Dict[Tuple[str, str], Set[str]] = {}
        #: (entity ids, normalized embeddings), built on the first search
        self._matrix: Optional[Tuple[List[str], np.ndarray]] = None
        self._lock = threading.Lock()

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        chunk_id: Optional[str] = None,
        document_id: str = "",
    ) -> None:
        """Add or update an entity node."""
        with self._lock:
            self._graph.add_node(
                entity_id,
                entity_type=entity_type,
                **(properties or {}),
            )
            if embedding is not None:
                self._entity_embeddings[entity_id] = embedding
                self._matrix = None
            if chunk_id:
                self._entity_chunks.setdefault(entity_id, set()).add(chunk_id)
                self._mention_documents[(entity_id, chunk_id)] = document_id

    def add_mention(self, entity_id: str, chunk_id: str, document_id: str = "") -> None:
        """Record that ``chunk_id`` mentions the entity, leaving its type and embedding."""
        with self._lock:
            if not self._graph.has_node(entity_id):
                self._graph.add_node(entity_id, entity_type="UNKNOWN")
            self._entity_chunks.setdefault(entity_id, set()).add(chunk_id)
            self._mention_documents[(entity_id, chunk_id)] = document_id

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
        document_id: str = "",
    ) -> None:
        """Add a directed relationship between entities."""
        with self._lock:
            self._graph.add_edge(
                source_id,
                target_id,
                rel_type=rel_type,
                **(properties or {}),
            )
            self._edge_documents.setdefault((source_id, target_id), set()).add(document_id)

    def add_document(
        self,
        document_id: str,
        entities: Sequence[GraphEntity],
        relationships: Sequence[GraphRelationship],
    ) -> None:
        """Add what one document's chunks mention, replacing what it added before."""
        self.remove_document(document_id)
        for e in entities:
            self.add_entity(
                e.name,
                e.entity_type,
                embedding=e.embedding,
                chunk_id=e.chunk_id,
                document_id=document_id,
            )
        for r in relationships:
            self.add_relationship(r.source, r.target, r.rel_type, document_id=document_id)

    def remove_document(self, document_id: str) -> None:
        """Remove the document's mentions and edges, and entities left with neither."""
        with self._lock:
            for (entity, chunk), doc in list(self._mention_documents.items()):
                if doc == document_id:
                    del self._mention_documents[(entity, chunk)]
                    chunks = self._entity_chunks.get(entity)
                    if chunks is not None:
                        chunks.discard(chunk)
                        if not chunks:
                            del self._entity_chunks[entity]
            for edge, docs in list(self._edge_documents.items()):
                docs.discard(document_id)
                if not docs:
                    del self._edge_documents[edge]
                    if self._graph.has_edge(*edge):
                        self._graph.remove_edge(*edge)
            for node in list(self._graph.nodes()):
                if node not in self._entity_chunks and self._graph.degree(node) == 0:
                    self._graph.remove_node(node)
                    if self._entity_embeddings.pop(node, None) is not None:
                        self._matrix = None

    def get_neighbors(self, entity_id: str, hops: int = 1) -> Set[str]:
        """Get entities reachable within N hops (undirected traversal)."""
        if entity_id not in self._graph:
            return set()
        visited: Set[str] = set()
        frontier = {entity_id}
        for _ in range(hops):
            next_frontier: Set[str] = set()
            for node in frontier:
                for neighbor in self._graph.successors(node):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                for neighbor in self._graph.predecessors(node):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
            visited.update(frontier)
            frontier = next_frontier - visited
        visited.update(frontier)
        visited.discard(entity_id)
        return visited

    def find_entities_by_embedding(self, query_embedding: np.ndarray, top_k: int = 10) -> List[str]:
        """Find closest entities by cosine similarity to query embedding."""
        if not self._entity_embeddings:
            return []

        matrix = self._matrix
        if matrix is None:
            # Built once: a loaded graph does not change, and stacking every
            # embedding again on each search cost 90 ms at 50k entities.
            entity_ids = list(self._entity_embeddings.keys())
            embeddings = np.array([self._entity_embeddings[eid] for eid in entity_ids])
            norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            matrix = self._matrix = (entity_ids, norms)
        entity_ids, emb_norms = matrix

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        similarities = emb_norms @ query_norm

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [entity_ids[i] for i in top_indices if similarities[i] > _MIN_SIMILARITY]

    def find_entities_by_name(self, query: str, limit: int = 5) -> List[str]:
        """Entities whose name appears in the query, or shares a word of it longer than 3."""
        query_lower = query.lower()
        query_terms = set(query_lower.split())
        found: List[str] = []
        for node in self._graph.nodes():
            node_lower = node.lower()
            if node_lower in query_lower or any(t in node_lower for t in query_terms if len(t) > 3):
                found.append(node)
                if len(found) >= limit:
                    break
        return found

    def get_chunk_ids_for_entities(self, entity_ids: Set[str]) -> Set[str]:
        """Get all chunk_ids associated with a set of entities."""
        chunk_ids: Set[str] = set()
        for eid in entity_ids:
            chunk_ids.update(self._entity_chunks.get(eid, set()))
        return chunk_ids

    @property
    def num_entities(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_relationships(self) -> int:
        return self._graph.number_of_edges()


def _embedding_bytes(embedding: Optional[np.ndarray]) -> Optional[bytes]:
    if embedding is None:
        return None
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _embedding_from(raw: Optional[bytes]) -> Optional[np.ndarray]:
    if not raw:
        return None
    return np.frombuffer(bytes(raw), dtype=np.float32)


#: (session_id, collection_name) -> (content generation, the loaded graph), the
#: most recently used last. Bounded: an API worker searches many sessions.
_loaded: "OrderedDict[Tuple[str, str], Tuple[Optional[int], InMemoryGraphStore]]" = OrderedDict()
_loaded_lock = threading.Lock()
_MAX_LOADED = 64
#: One load at a time per graph: searches arriving after a change wait for it.
_load_locks: Dict[Tuple[str, str], threading.Lock] = {}


class DatabaseGraphStore:
    """A session's graph for one collection, kept in the shared metadata database.

    Writes go to the database at once, one transaction per document. Reads use
    the graph loaded into this process, loaded again when the session's
    content generation has moved since (another process uploaded or deleted)
    or this process wrote.
    """

    def __init__(self, session_id: str, collection_name: str, db: Any = None):
        self.session_id = session_id
        self.collection_name = collection_name or ""
        self._db = db

    @property
    def db(self) -> Any:
        if self._db is None:
            from lasagna.core.database import get_rag_database

            self._db = get_rag_database()
        return self._db

    @property
    def _key(self) -> Tuple[str, str]:
        return (self.session_id, self.collection_name)

    def _forget(self) -> None:
        with _loaded_lock:
            _loaded.pop(self._key, None)

    def _generation(self) -> Optional[int]:
        """The session's content generation in this store's database; None if unreadable."""
        read = getattr(self.db, "get_content_generation", None)
        if read is None:
            return None
        try:
            return int(read(self.session_id))
        except Exception as e:
            logger.debug(f"Could not read the content generation of {self.session_id[:8]}: {e}")
            return None

    def _cached(self, generation: Optional[int]) -> Optional[InMemoryGraphStore]:
        with _loaded_lock:
            cached = _loaded.get(self._key)
            if cached is not None and (generation is None or cached[0] == generation):
                _loaded.move_to_end(self._key)
                return cached[1]
        return None

    def snapshot(self) -> InMemoryGraphStore:
        """The graph as this process has it loaded: one read for a whole search."""
        return self._graph()

    def _graph(self) -> InMemoryGraphStore:
        generation = self._generation()
        graph = self._cached(generation)
        if graph is not None:
            return graph
        with _loaded_lock:
            load_lock = _load_locks.setdefault(self._key, threading.Lock())
        with load_lock:
            graph = self._cached(generation)
            if graph is not None:
                return graph
            return self._load(generation)

    def _load(self, generation: Optional[int]) -> InMemoryGraphStore:
        entities, mentions, edges = self.db.load_graph(self.session_id, self.collection_name)
        graph = InMemoryGraphStore()
        for name, entity_type, embedding in entities:
            graph.add_entity(name, entity_type or "UNKNOWN", embedding=_embedding_from(embedding))
        for name, chunk_id, document_id in mentions:
            graph.add_mention(name, chunk_id, document_id)
        for source, target, rel_type, document_id in edges:
            graph.add_relationship(source, target, rel_type or "RELATED", document_id=document_id)
        with _loaded_lock:
            _loaded[self._key] = (generation, graph)
            _loaded.move_to_end(self._key)
            while len(_loaded) > _MAX_LOADED:
                _loaded.popitem(last=False)
        return graph

    def add_document(
        self,
        document_id: str,
        entities: Sequence[GraphEntity],
        relationships: Sequence[GraphRelationship],
        *,
        extraction_model: Optional[str] = None,
        complete: bool = True,
        unparsed_chunks: int = 0,
    ) -> None:
        """Store what one document's chunks mention, replacing what it stored before.

        ``complete`` (the default): every chunk was answered, so the document
        counts as done even when it names no entity. Pass False when a chunk's
        call failed, so it is extracted again later.
        """
        rows: Dict[str, Tuple[str, Optional[bytes]]] = {}
        for e in entities:
            known = rows.get(e.name)
            embedding = _embedding_bytes(e.embedding)
            rows[e.name] = (
                e.entity_type or "UNKNOWN",
                embedding if embedding is not None else (known[1] if known else None),
            )
        # Sorted: two writers then lock the rows they share in the same order.
        self.db.save_graph_document(
            self.session_id,
            self.collection_name,
            document_id,
            [
                (name, entity_type, embedding)
                for name, (entity_type, embedding) in sorted(rows.items())
            ],
            sorted({(e.name, e.chunk_id) for e in entities if e.chunk_id}),
            sorted({(r.source, r.target, r.rel_type or "RELATED") for r in relationships}),
            extraction_model=extraction_model,
            complete=complete,
            unparsed_chunks=unparsed_chunks,
        )
        self._forget()

    def remove_document(self, document_id: str) -> None:
        """Remove the document's part of every graph of the session."""
        self.db.delete_graph_document(self.session_id, document_id)
        drop_loaded_graphs(self.session_id)

    def find_entities_by_embedding(self, query_embedding: np.ndarray, top_k: int = 10) -> List[str]:
        return self._graph().find_entities_by_embedding(query_embedding, top_k=top_k)

    def find_entities_by_name(self, query: str, limit: int = 5) -> List[str]:
        return self._graph().find_entities_by_name(query, limit=limit)

    def get_neighbors(self, entity_id: str, hops: int = 1) -> Set[str]:
        return self._graph().get_neighbors(entity_id, hops=hops)

    def get_chunk_ids_for_entities(self, entity_ids: Set[str]) -> Set[str]:
        return self._graph().get_chunk_ids_for_entities(entity_ids)

    @property
    def num_entities(self) -> int:
        return self._graph().num_entities

    @property
    def num_relationships(self) -> int:
        return self._graph().num_relationships


def graph_for(session_id: str, collection_name: str) -> DatabaseGraphStore:
    """The store of the session's graph over ``collection_name``'s chunks."""
    return DatabaseGraphStore(session_id, collection_name)


def drop_loaded_graphs(session_id: str) -> None:
    """Forget every graph of the session this process has loaded."""
    with _loaded_lock:
        for key in [k for k in _loaded if k[0] == session_id]:
            del _loaded[key]


def drop_all_loaded_graphs() -> None:
    """Forget every graph this process has loaded (the database was reset)."""
    with _loaded_lock:
        _loaded.clear()


def _name(value: Any) -> Optional[str]:
    """An entity name from the extractor, or None for one that cannot be stored."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return value or None


def index_document_graph(
    session_id: str,
    collection_name: str,
    document_id: str,
    extractions: Sequence[Dict[str, Any]],
    chunk_ids: Sequence[str],
    encode: Any,
    *,
    extraction_model: Optional[str] = None,
    complete: bool = True,
    unparsed_chunks: int = 0,
) -> DatabaseGraphStore:
    """Store the entities and relationships extracted from a document's chunks.

    ``extractions[i]`` is what ``EntityExtractor`` found in the chunk whose id
    is ``chunk_ids[i]``; ``encode`` embeds an entity's name. The one place the
    upload pipeline and the benchmark ingest write a graph.
    """
    entities: List[GraphEntity] = []
    relationships: List[GraphRelationship] = []
    embedded: Dict[str, np.ndarray] = {}
    for extraction, chunk_id in zip(extractions, chunk_ids):
        # An LLM can return a null or empty name: dropped here, it cannot
        # fail the whole document's write (PostgreSQL refuses NULL keys).
        for entity in extraction.get("entities", []) or []:
            name = _name((entity or {}).get("name"))
            if name is None:
                continue
            if name not in embedded:
                embedded[name] = encode(name)
            entities.append(
                GraphEntity(name, str(entity.get("type") or "UNKNOWN"), embedded[name], chunk_id)
            )
        for rel in extraction.get("relationships", []) or []:
            source, target = _name((rel or {}).get("source")), _name((rel or {}).get("target"))
            if source is None or target is None:
                continue
            relationships.append(
                GraphRelationship(source, target, str(rel.get("type") or "RELATED"))
            )
    store = graph_for(session_id, collection_name)
    store.add_document(
        document_id,
        entities,
        relationships,
        extraction_model=extraction_model,
        complete=complete,
        unparsed_chunks=unparsed_chunks,
    )
    logger.info(
        f"GraphRAG: stored {len(embedded)} entities and {len(relationships)} relationships "
        f"for document {document_id}"
    )
    return store


# The in-process registry the graph used to live in. Kept for code that built
# a graph by hand; nothing in the package reads it any more.
_session_graphs: Dict[str, InMemoryGraphStore] = {}
_session_graphs_lock = threading.Lock()


def get_or_create_session_graph(session_id: str) -> InMemoryGraphStore:
    """An in-memory graph for the session, in this process only.

    Deprecated: a graph built here is invisible to every other process, and
    graph retrieval no longer reads it. Use `graph_for` (or
    `index_document_graph`), which every process shares.
    """
    warnings.warn(
        "get_or_create_session_graph builds a graph only this process sees, and graph "
        "retrieval no longer reads it; use lasagna.retrieval.graph.graph_for",
        DeprecationWarning,
        stacklevel=2,
    )
    if session_id not in _session_graphs:
        with _session_graphs_lock:
            if session_id not in _session_graphs:
                _session_graphs[session_id] = InMemoryGraphStore()
    return _session_graphs[session_id]


def remove_session_graph(session_id: str) -> None:
    """Remove the session's in-memory graph (see `get_or_create_session_graph`)."""
    with _session_graphs_lock:
        _session_graphs.pop(session_id, None)


def ensure_corpus_graph(
    session_id: str,
    collection_name: str,
    vectors: Any,
    document_ids: Set[str],
    built_now: Set[str],
    config: Any,
    encode: Any,
) -> Dict[str, Any]:
    """Make sure a benchmark corpus has its graph, and say how it got it.

    A corpus a later run reuses from the vector store keeps its graph in the
    metadata database. A corpus ingested before the graph was stored there, or
    whose database was reset, has chunks and no graph: its documents are
    extracted again from the stored chunks, so a graph arm never runs on an
    empty graph unnoticed. So are documents another extraction model did: the
    corpus key holds the model only when an arm names one. A document counts
    as done once every chunk of it was answered (``graph_documents``), even
    when it names no entity.

    The answer goes in the run record: ``state`` is ``built``, ``reused``,
    ``rebuilt``, or ``incomplete`` when some documents still have no graph (no
    LLM, failed calls), with ``missing`` saying how many. ``no_chunks`` counts
    documents with nothing stored to extract, ``unparsed_chunks`` the chunks
    whose reply the extractor could not read.
    """
    from lasagna.core.database import get_rag_database
    from lasagna.core.provenance import redact
    from lasagna.utils.model_utils import get_default_model_name

    db = get_rag_database()
    wanted = set(document_ids)
    model_now = getattr(config, "graph_extraction_model", "") or get_default_model_name()
    done = db.extracted_graph_documents(session_id, collection_name)
    # A document recorded with no model (a caller that did not say) is kept.
    current = {d for d, r in done.items() if r.get("model") in (None, model_now)}
    to_rebuild = sorted((wanted - set(built_now)) - current)
    rebuilt = written = 0
    no_chunks: Set[str] = set()
    error = None
    if to_rebuild:
        from lasagna.vector_backends.filters import FilterExpression

        from .entity_extractor import extract_graph

        chunks: Dict[str, List[Tuple[int, str, str]]] = {}
        rebuilding = set(to_rebuild)
        filters = [vectors.session_filter(collection_name, session_id)]
        filters.append(FilterExpression.in_list("document_id", to_rebuild))
        rows = vectors.iter_all(
            collection_name=collection_name,
            filter_expr=FilterExpression.and_all([f for f in filters if f is not None]),
            output_fields=["chunk_id", "content", "document_id", "metadata"],
        )
        for row in rows:
            doc = str(row.get("document_id") or "")
            if doc not in rebuilding:
                continue
            meta = row.get("metadata") or {}
            index = int(meta.get("chunk_index", 0) or 0) if isinstance(meta, dict) else 0
            chunks.setdefault(doc, []).append(
                (index, str(row.get("chunk_id")), row.get("content") or "")
            )
        for doc in to_rebuild:
            stored = sorted(chunks.get(doc, []))
            if not stored:
                no_chunks.add(doc)
                continue
            try:
                found = extract_graph([text for _, _, text in stored], config)
            except Exception as e:
                # No LLM to extract with: the rest would fail the same way.
                error = str(redact(str(e)))[:200]
                logger.warning(f"GraphRAG: the reused corpus's graph cannot be rebuilt: {error}")
                break
            index_document_graph(
                session_id,
                collection_name,
                doc,
                found.extractions,
                [cid for _, cid, _ in stored],
                encode,
                extraction_model=found.model,
                complete=found.complete,
                unparsed_chunks=found.unparsed,
            )
            written += 1
            # Done now; a document with a failed call kept what was found.
            rebuilt += int(found.complete)
        if written:
            db.bump_content_generation(session_id)
    done = db.extracted_graph_documents(session_id, collection_name)
    entities, _, _ = db.load_graph(session_id, collection_name)
    # Still extracted by another model: not rebuilt (no LLM), so not done.
    have = {d for d in wanted & set(done) if done[d].get("model") in (None, model_now)}
    missing = len(wanted - have - no_chunks)
    if missing:
        state = "incomplete"
    elif rebuilt:
        state = "rebuilt"
    elif built_now:
        state = "built"
    else:
        state = "reused"
    models = sorted({done[d]["model"] for d in have if done[d].get("model")})
    info = {
        "state": state,
        "documents": len(have),
        "missing": missing,
        "entities": len(entities),
        "rebuilt": rebuilt,
        # One model, as a rule; a list when documents were extracted with several.
        "extraction_model": models[0] if len(models) == 1 else (models or None),
        "unparsed_chunks": sum(done[d]["unparsed_chunks"] for d in have),
    }
    if no_chunks:
        info["no_chunks"] = len(no_chunks)
    if error:
        info["error"] = error
    log = logger.warning if missing else logger.info
    log(f"GraphRAG: the corpus graph of {session_id} was {state} ({info})")
    return info
