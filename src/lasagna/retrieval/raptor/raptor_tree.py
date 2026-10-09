"""RAPTOR tree builder and retriever.

Implements the RAPTOR algorithm:
1. Cluster leaf chunks by embedding similarity (KMeans).
2. Summarize each cluster via an LLM.
3. Embed summaries and recurse until a single root node remains.
4. At query time, retrieve from all tree levels to capture both
   fine-grained details (leaf) and high-level themes (root).

Reference: Sarthi et al., arXiv:2401.18059
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class RaptorNode:
    """A node in the RAPTOR tree."""

    node_id: str
    text: str
    embedding: Optional[np.ndarray] = None
    level: int = 0  # 0 = leaf, higher = more abstract
    children: List[str] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


@dataclass
class RaptorTree:
    """Complete RAPTOR tree for a document or collection."""

    nodes: Dict[str, RaptorNode] = field(default_factory=dict)
    root_ids: List[str] = field(default_factory=list)
    max_level: int = 0

    def get_level(self, level: int) -> List[RaptorNode]:
        """Get all nodes at a given level."""
        return [n for n in self.nodes.values() if n.level == level]

    def all_nodes(self) -> List[RaptorNode]:
        """Get all nodes across all levels."""
        return list(self.nodes.values())


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------


class RaptorTreeBuilder:
    """Builds a RAPTOR summary tree from leaf chunks.

    Args:
        embedding_manager: Embedding manager for encoding summaries.
        llm_provider: LLM provider for generating cluster summaries.
        max_cluster_size: Maximum number of chunks per cluster.
        min_cluster_size: Minimum cluster size (below this, merge with nearest).
        max_levels: Maximum tree depth (prevents runaway recursion).
        summary_max_tokens: Max tokens for each summary.
    """

    def __init__(
        self,
        embedding_manager,
        llm_provider=None,
        max_cluster_size: int = 10,
        min_cluster_size: int = 2,
        max_levels: int = 5,
        summary_max_tokens: int = 200,
    ):
        self._embedding_manager = embedding_manager
        self._llm_provider = llm_provider
        self._max_cluster_size = max_cluster_size
        self._min_cluster_size = min_cluster_size
        self._max_levels = max_levels
        self._summary_max_tokens = summary_max_tokens

    def build(
        self,
        texts: List[str],
        embeddings: Optional[List[np.ndarray]] = None,
        metadata_list: Optional[List[Dict]] = None,
    ) -> RaptorTree:
        """Build a RAPTOR tree from leaf-level chunks.

        Args:
            texts: List of chunk texts (leaf nodes).
            embeddings: Pre-computed embeddings (optional; computed if None).
            metadata_list: Per-chunk metadata (optional).

        Returns:
            A fully constructed RaptorTree.
        """
        if not texts:
            return RaptorTree()

        tree = RaptorTree()

        # Create leaf nodes
        if embeddings is None:
            embeddings = self._embedding_manager.encode_documents_batch(texts, show_progress=False)

        leaf_ids = []
        for i, (text, emb) in enumerate(zip(texts, embeddings)):
            node_id = f"leaf_{i}_{uuid.uuid4().hex[:8]}"
            meta = metadata_list[i] if metadata_list else {}
            node = RaptorNode(
                node_id=node_id,
                text=text,
                embedding=emb,
                level=0,
                metadata=meta,
            )
            tree.nodes[node_id] = node
            leaf_ids.append(node_id)

        # Recursive summarization
        current_ids = leaf_ids
        level = 0

        while len(current_ids) > 1 and level < self._max_levels:
            level += 1
            current_embeddings = np.array([tree.nodes[nid].embedding for nid in current_ids])

            # Cluster
            clusters = self._cluster(current_embeddings, current_ids)

            if len(clusters) >= len(current_ids):
                # No reduction — stop
                break

            # Summarize each cluster
            parent_ids = []
            for cluster_ids in clusters:
                cluster_texts = [tree.nodes[nid].text for nid in cluster_ids]
                summary = self._summarize(cluster_texts)
                if not summary:
                    continue

                # Embed the summary
                summary_emb = self._embedding_manager.encode_document(summary, use_cache=False)

                parent_id = f"level{level}_{uuid.uuid4().hex[:8]}"
                parent = RaptorNode(
                    node_id=parent_id,
                    text=summary,
                    embedding=summary_emb,
                    level=level,
                    children=list(cluster_ids),
                    metadata={"cluster_size": len(cluster_ids)},
                )
                tree.nodes[parent_id] = parent
                parent_ids.append(parent_id)

            if not parent_ids:
                break

            current_ids = parent_ids
            logger.info(
                f"RAPTOR level {level}: {len(parent_ids)} summary nodes "
                f"from {len(clusters)} clusters"
            )

        tree.root_ids = current_ids
        tree.max_level = level
        logger.info(
            f"RAPTOR tree built: {len(tree.nodes)} total nodes, "
            f"{level + 1} levels, {len(leaf_ids)} leaves"
        )
        return tree

    def _cluster(
        self,
        embeddings: np.ndarray,
        node_ids: List[str],
    ) -> List[List[str]]:
        """Cluster nodes by embedding similarity.

        Uses scikit-learn KMeans when available, falls back to simple
        round-robin partitioning.
        """
        n = len(node_ids)
        if n <= self._max_cluster_size:
            return [node_ids]

        n_clusters = max(2, n // self._max_cluster_size)

        try:
            from sklearn.cluster import KMeans

            # Use KMeans instead of GMM: GMM with full covariance requires
            # more samples than embedding dimensions (typically 768-1024),
            # which is rarely the case for document chunks.
            km = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10,
                max_iter=100,
            )
            labels = km.fit_predict(embeddings)

        except ImportError:
            # Fallback: simple round-robin partitioning
            logger.debug("sklearn not available, using simple partitioning")
            labels = np.array([i % n_clusters for i in range(n)])

        # Group by cluster label
        clusters: Dict[int, List[str]] = {}
        for idx, label in enumerate(labels):
            clusters.setdefault(int(label), []).append(node_ids[idx])

        # Merge small clusters with nearest neighbor
        result = []
        for cluster_ids in clusters.values():
            if len(cluster_ids) < self._min_cluster_size and result:
                result[-1].extend(cluster_ids)
            else:
                result.append(cluster_ids)

        return result

    def _summarize(self, texts: List[str]) -> str:
        """Summarize a cluster of texts using LLM."""
        if self._llm_provider is None:
            # Fallback: concatenate first 200 chars of each text
            parts = [t[:200] for t in texts]
            return " ".join(parts)[: self._summary_max_tokens * 4]

        combined = "\n\n---\n\n".join(texts)
        # Truncate to avoid exceeding context window
        combined = combined[:8000]

        prompt = (
            "Summarize the following text passages into a single coherent paragraph "
            "that captures the key information. Be concise and factual.\n\n"
            f"Passages:\n{combined}\n\n"
            "Summary:"
        )

        try:
            response = self._llm_provider.invoke(prompt)
            if hasattr(response, "content"):
                return response.content.strip()
            return str(response).strip()
        except Exception as e:
            logger.warning(f"RAPTOR summarization failed: {e}")
            # Fallback: first N chars
            return combined[: self._summary_max_tokens * 4]


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------


class RaptorRetriever:
    """Retrieve from a RAPTOR tree at multiple abstraction levels.

    Supports two retrieval modes:
    - **tree_traversal**: Start at root, traverse down to find relevant branches.
    - **collapsed** (default): Flatten all nodes, search by embedding similarity.

    Args:
        embedding_manager: For encoding queries.
        mode: Retrieval mode ('collapsed' or 'tree_traversal').
    """

    def __init__(
        self,
        embedding_manager,
        mode: str = "collapsed",
    ):
        self._embedding_manager = embedding_manager
        self._mode = mode

    def retrieve(
        self,
        tree: RaptorTree,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[RaptorNode, float]]:
        """Retrieve the most relevant nodes from the RAPTOR tree.

        Args:
            tree: The RAPTOR tree to search.
            query: Search query.
            top_k: Number of results to return.

        Returns:
            List of (node, score) tuples sorted by relevance.
        """
        if not tree.nodes:
            return []

        query_embedding = self._embedding_manager.encode_query(query)

        if self._mode == "tree_traversal":
            return self._retrieve_traversal(tree, query_embedding, top_k)
        else:
            return self._retrieve_collapsed(tree, query_embedding, top_k)

    def _retrieve_collapsed(
        self,
        tree: RaptorTree,
        query_emb: np.ndarray,
        top_k: int,
    ) -> List[Tuple[RaptorNode, float]]:
        """Search all nodes across all levels by similarity."""
        scored = []
        for node in tree.all_nodes():
            if node.embedding is None:
                continue
            score = float(np.dot(query_emb, node.embedding))
            scored.append((node, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _retrieve_traversal(
        self,
        tree: RaptorTree,
        query_emb: np.ndarray,
        top_k: int,
    ) -> List[Tuple[RaptorNode, float]]:
        """Top-down tree traversal: expand most relevant branch at each level."""
        results = []
        # Start from roots, expand top branches
        current_ids = (
            tree.root_ids if tree.root_ids else [n.node_id for n in tree.get_level(tree.max_level)]
        )

        visited = set()
        for _level in range(tree.max_level, -1, -1):
            if not current_ids:
                break

            # Score current level nodes
            level_scored = []
            for nid in current_ids:
                if nid in visited:
                    continue
                visited.add(nid)
                node = tree.nodes.get(nid)
                if node is None or node.embedding is None:
                    continue
                score = float(np.dot(query_emb, node.embedding))
                level_scored.append((node, score))

            level_scored.sort(key=lambda x: x[1], reverse=True)
            results.extend(level_scored[:top_k])

            # Expand children of top nodes for next level
            next_ids = []
            for node, _score in level_scored[: max(2, top_k // 2)]:
                next_ids.extend(node.children)
            current_ids = next_ids

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
