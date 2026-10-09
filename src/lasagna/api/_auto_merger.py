"""Hierarchical auto-merge for RAG search results.

Implements the AutoMergingRetriever algorithm: when enough sibling child chunks
from the same parent are retrieved, replace them with the parent chunk content
for richer LLM context.
"""

from typing import Dict, List, Optional

from loguru import logger

from lasagna.core.config import rag_config, RAGConfig
from lasagna.core.schema import RecordedDocumentChunk, SearchResult


class AutoMerger:
    """Merges child chunks into parent chunks when enough siblings are retrieved.

    Port of LlamaIndex AutoMergingRetriever algorithm (Apache 2.0):
    Groups results by parent_chunk_id, and when the ratio of retrieved children
    to total siblings exceeds the merge threshold, replaces children with the
    parent chunk content for richer LLM context.

    Pre-hierarchical documents (no parent_chunk_id) pass through unchanged.
    """

    def merge_results(
        self,
        results: List[SearchResult],
        config: Optional[RAGConfig] = None,
    ) -> List[SearchResult]:
        """Auto-merge child chunks into parent chunks when enough siblings are retrieved.

        Args:
            results: Search results from the retrieval pipeline.
            config: RAG configuration (for auto_merge_threshold).

        Returns:
            Merged search results — potentially fewer but with richer content.
        """
        from lasagna.core.database import get_rag_database

        config = config or rag_config
        threshold = config.auto_merge_threshold

        # Group results by parent_chunk_id
        parent_groups: Dict[str, List[SearchResult]] = {}
        no_parent: List[SearchResult] = []

        for result in results:
            parent_id = result.chunk.metadata.get("parent_chunk_id")
            if parent_id:
                parent_groups.setdefault(parent_id, []).append(result)
            else:
                no_parent.append(result)

        if not parent_groups:
            return results  # No hierarchical chunks, nothing to merge

        # Determine which parent groups should be merged
        merge_parent_ids = []
        keep_children: List[SearchResult] = []

        for parent_id, children in parent_groups.items():
            total_siblings = children[0].chunk.metadata.get("total_siblings")
            if total_siblings and total_siblings > 0:
                ratio = len(children) / total_siblings
            else:
                # Fallback: merge when >= 2 children from same parent
                ratio = 1.0 if len(children) >= 2 else 0.0

            if ratio >= threshold:
                merge_parent_ids.append(parent_id)
            else:
                keep_children.extend(children)

        if not merge_parent_ids:
            return results  # Nothing exceeds threshold

        # Batch-fetch parent content AND metadata from SQLite
        rag_db = get_rag_database()
        parent_data = rag_db.get_parent_chunks_with_metadata(merge_parent_ids)

        # Build merged results
        merged_results: List[SearchResult] = []
        for parent_id in merge_parent_ids:
            children = parent_groups[parent_id]

            # Use best child's score for the merged result
            best_child = max(children, key=lambda r: r.score)

            parent_info = parent_data.get(parent_id)
            if parent_info:
                content = parent_info["content"]
                parent_chunk_index = parent_info["chunk_index"]
                parent_start_char = parent_info["start_char"]
                parent_end_char = parent_info["end_char"]
                parent_meta = parent_info["metadata"]
                # Primary: use line numbers stored in parent metadata (newly indexed docs)
                line_start = parent_meta.get("line_start")
                line_end = parent_meta.get("line_end")
            else:
                # Parent not found in SQLite — fallback to concatenated children
                logger.warning(
                    f"Parent chunk {parent_id} not found in SQLite, using concatenated children"
                )
                content = "\n\n".join(c.chunk.content for c in children)
                parent_chunk_index = best_child.chunk.chunk_index
                parent_start_char = None
                parent_end_char = None
                line_start = None
                line_end = None

            # Fallback line range: derive from children (pre-existing parents without line numbers)
            if line_start is None:
                child_line_starts = [
                    c.chunk.metadata.get("line_start")
                    for c in children
                    if c.chunk.metadata.get("line_start") is not None
                ]
                line_start = min(child_line_starts) if child_line_starts else None
            if line_end is None:
                child_line_ends = [
                    c.chunk.metadata.get("line_end")
                    for c in children
                    if c.chunk.metadata.get("line_end") is not None
                ]
                line_end = max(child_line_ends) if child_line_ends else None

            # Fallback start_char/end_char: derive from children (only when parent not found)
            if parent_start_char is None:
                child_starts = [
                    c.chunk.metadata.get("start_char")
                    for c in children
                    if c.chunk.metadata.get("start_char") is not None
                ]
                parent_start_char = min(child_starts) if child_starts else 0
            if parent_end_char is None:
                child_ends = [
                    c.chunk.metadata.get("end_char")
                    for c in children
                    if c.chunk.metadata.get("end_char") is not None
                ]
                parent_end_char = max(child_ends) if child_ends else 0

            # Compute page range from all merged children
            child_pages = sorted(
                set(c.chunk.page_number for c in children if c.chunk.page_number is not None)
            )
            page_number = child_pages[0] if child_pages else best_child.chunk.page_number

            merged_chunk = RecordedDocumentChunk(
                chunk_id=best_child.chunk.chunk_id,
                content=content,
                document_id=best_child.chunk.document_id,
                chunk_index=parent_chunk_index,
                page_number=page_number,
                metadata={
                    **best_child.chunk.metadata,
                    # Override with parent-accurate values
                    "auto_merged": True,
                    "merged_child_count": len(children),
                    "merged_chunk_ids": [c.chunk.chunk_id for c in children],
                    "parent_chunk_id": parent_id,
                    "chunk_index": parent_chunk_index,
                    "start_char": parent_start_char,
                    "end_char": parent_end_char,
                    "line_start": line_start,
                    "line_end": line_end,
                    "page_numbers": child_pages if len(child_pages) > 1 else None,
                },
                created_at=best_child.chunk.created_at,
            )

            merged_results.append(
                SearchResult(
                    chunk=merged_chunk,
                    score=best_child.score,
                    document=best_child.document,
                )
            )

        # Combine: merged parents + unmerged children + non-hierarchical results
        all_results = merged_results + keep_children + no_parent
        all_results.sort(key=lambda r: r.score, reverse=True)

        logger.info(
            f"Auto-merge: {len(results)} results -> {len(all_results)} "
            f"(merged {len(merge_parent_ids)} parent groups)"
        )
        return all_results
