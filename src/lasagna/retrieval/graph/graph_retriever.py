"""Query-time graph retrieval: entity matching + N-hop traversal.

Expands the retrieval candidate set by finding entities mentioned in the query,
traversing the knowledge graph, and retrieving associated chunks.
"""

from typing import Any, List, Set

from loguru import logger

from .graph_store import get_or_create_session_graph


class GraphRetriever:
    """Retrieves additional chunks via knowledge graph traversal."""

    def __init__(self, embedding_manager: Any = None):
        self._embedding_manager = embedding_manager

    def expand(
        self,
        query: str,
        existing_chunk_ids: Set[str],
        session_id: str,
        max_hops: int = 2,
        max_graph_chunks: int = 10,
    ) -> List[str]:
        """Find additional chunk_ids via graph traversal.

        Args:
            query: Search query.
            existing_chunk_ids: Already-retrieved chunk_ids (for dedup).
            session_id: Session ID to look up graph.
            max_hops: Maximum traversal depth.
            max_graph_chunks: Maximum chunks to add from graph.

        Returns:
            List of new chunk_ids not in existing_chunk_ids.
        """
        try:
            graph = get_or_create_session_graph(session_id)
        except ImportError:
            return []

        if graph.num_entities == 0:
            return []

        # Step 1: Find entities matching the query via embedding similarity
        matching_entities: Set[str] = set()
        if self._embedding_manager:
            try:
                query_emb = self._embedding_manager.encode_query(query)
                matching_entities = set(graph.find_entities_by_embedding(query_emb, top_k=5))
            except Exception as e:
                logger.debug(f"Graph entity embedding search failed: {e}")

        # Step 2: Fallback — keyword matching on entity names
        if not matching_entities:
            query_lower = query.lower()
            query_terms = set(query_lower.split())
            for node in graph._graph.nodes():
                node_lower = node.lower()
                # Match if entity name appears in query or shares terms
                if node_lower in query_lower or any(
                    t in node_lower for t in query_terms if len(t) > 3
                ):
                    matching_entities.add(node)
                    if len(matching_entities) >= 5:
                        break

        if not matching_entities:
            return []

        # Step 3: N-hop traversal to find related entities
        all_related: Set[str] = set(matching_entities)
        for entity_id in matching_entities:
            neighbors = graph.get_neighbors(entity_id, hops=max_hops)
            all_related.update(neighbors)

        # Step 4: Get chunk_ids for related entities
        related_chunks = graph.get_chunk_ids_for_entities(all_related)

        # Step 5: Deduplicate against existing results
        new_chunks = related_chunks - existing_chunk_ids

        logger.debug(
            f"GraphRAG expansion: {len(matching_entities)} query entities, "
            f"{len(all_related)} related entities ({max_hops} hops), "
            f"{len(new_chunks)} new chunks"
        )

        return list(new_chunks)[:max_graph_chunks]
