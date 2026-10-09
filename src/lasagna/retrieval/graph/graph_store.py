"""In-memory knowledge graph store backed by NetworkX.

One graph per session, stored in a module-level dict with thread-safe access.
"""

import threading
from typing import Any, Dict, List, Optional, Set

import numpy as np
from loguru import logger

try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None


class InMemoryGraphStore:
    """In-memory directed knowledge graph for a single session."""

    def __init__(self):
        if not NETWORKX_AVAILABLE:
            raise ImportError(
                "networkx is required for GraphRAG. Install with: pip install networkx>=3.0"
            )
        self._graph = nx.DiGraph()
        self._entity_embeddings: Dict[str, np.ndarray] = {}
        # Map entity_id -> set of chunk_ids that mention it
        self._entity_chunks: Dict[str, Set[str]] = {}
        self._lock = threading.Lock()

    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Optional[Dict[str, Any]] = None,
        embedding: Optional[np.ndarray] = None,
        chunk_id: Optional[str] = None,
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
            if chunk_id:
                self._entity_chunks.setdefault(entity_id, set()).add(chunk_id)

    def add_relationship(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a directed relationship between entities."""
        with self._lock:
            self._graph.add_edge(
                source_id,
                target_id,
                rel_type=rel_type,
                **(properties or {}),
            )

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

        entity_ids = list(self._entity_embeddings.keys())
        embeddings = np.array([self._entity_embeddings[eid] for eid in entity_ids])

        # Cosine similarity
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
        similarities = emb_norms @ query_norm

        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [entity_ids[i] for i in top_indices if similarities[i] > 0.3]

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


# Module-level session graph store
_session_graphs: Dict[str, InMemoryGraphStore] = {}
_session_graphs_lock = threading.Lock()


def get_or_create_session_graph(session_id: str) -> InMemoryGraphStore:
    """Get or create a graph store for a session."""
    if session_id not in _session_graphs:
        with _session_graphs_lock:
            if session_id not in _session_graphs:
                _session_graphs[session_id] = InMemoryGraphStore()
                logger.debug(f"Created graph store for session {session_id}")
    return _session_graphs[session_id]


def remove_session_graph(session_id: str) -> None:
    """Remove graph store for a session."""
    with _session_graphs_lock:
        _session_graphs.pop(session_id, None)
