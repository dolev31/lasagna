"""Search Manager - Search and retrieval operations.

Provides semantic search, context retrieval, and hybrid search functionality.
Wraps the underlying RAGBackend search operations.

Example:
    ```python
    # Basic search
    results = await client.search.search("quarterly revenue")

    # Get context for LLM
    context = await client.search.get_context("key findings")

    # Search with options
    results = await client.search.search(
        query="revenue",
        top_k=10,
        use_hyde=True,
    )
    ```
"""

from typing import List, Optional, TYPE_CHECKING, AsyncIterator
from datetime import datetime, timezone
import asyncio

from loguru import logger

from lasagna.utils import degraded as _degraded

from ..types import (
    Document,
    DocumentChunk,
    SearchResult,
    SearchResults,
    Citation,
    CitationFormat,
    DocumentStatus,
)
from ..exceptions import SearchError, QueryError

if TYPE_CHECKING:
    from ..client import KnowledgeClient


class SearchManager:
    """Manages search operations with full retrieval pipeline.

    Provides semantic search, hybrid search, and context generation.
    All searches respect the client's focused documents if set.

    Example:
        ```python
        # Basic search
        results = await client.search.search("revenue")

        # Search with filters
        results = await client.search.search(
            query="revenue",
            top_k=10,
            document_ids=["doc_123"],
        )

        # Get context for LLM injection
        context = await client.search.get_context("key findings")
        ```
    """

    def __init__(self, client: "KnowledgeClient"):
        """Initialize search manager.

        Args:
            client: Parent KnowledgeClient instance
        """
        self._client = client

    # =========================================================================
    # Search Operations
    # =========================================================================

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        document_ids: Optional[List[str]] = None,
        use_reranking: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
        routing_tier: Optional[str] = None,
    ) -> SearchResults:
        """Perform semantic search with full pipeline.

        The search uses the configured pipeline which may include:
        - Hybrid search (vector + BM25) if enabled
        - HyDE query expansion if enabled
        - Cross-encoder reranking if enabled
        - MMR diversity if enabled
        - Intelligent query routing (Adaptive-RAG)

        Args:
            query: Natural language search query
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold (default allows all)
            document_ids: Optional filter by document IDs (overrides focus)
            use_reranking: Override config's enable_reranking
            use_hyde: Override config's enable_hyde
            routing_tier: Force a routing tier ("fast", "standard",
                "comprehensive"). If None, the router auto-selects.

        Returns:
            SearchResults containing ranked chunks with scores

        Raises:
            SearchError: If search operation fails
            QueryError: If query is invalid

        Example:
            ```python
            results = await client.search.search("What is the revenue?")
            for result in results:
                print(f"[{result.score:.2f}] {result.chunk.content[:100]}")

            # With options
            results = await client.search.search(
                "revenue",
                top_k=10,
                use_hyde=True,
                document_ids=["doc_123"],
            )

            # Force fast routing for low-latency
            results = await client.search.search(
                "What is X?",
                routing_tier="fast",
            )
            ```
        """
        if not query or not query.strip():
            raise QueryError("Query cannot be empty", query=query)

        return await self._search_internal(
            query=query,
            top_k=top_k,
            min_score=min_score,
            document_ids=document_ids,
            use_reranking=use_reranking,
            use_hyde=use_hyde,
            routing_tier=routing_tier,
        )

    async def get_context(
        self,
        query: str,
        document_ids: Optional[List[str]] = None,
        max_chunks: int = 5,
        routing_tier: Optional[str] = None,
    ) -> str:
        """Get formatted context string for LLM injection.

        This is the recommended method for RAG. Returns context with
        source citations ready to inject into LLM prompts.

        The output format depends on the `enable_citations` and `citation_format`
        configuration:
        - If citations disabled or format is NONE: plain text without markers
        - STABLE_ID: Includes [abc123] style citation IDs
        - INDEX: Includes [1], [2] style citation numbers
        - MARKDOWN: Includes full [Source: filename, p.X] markers

        Args:
            query: Natural language search query
            document_ids: Optional filter by document IDs (overrides focus)
            max_chunks: Maximum chunks to include in context
            routing_tier: Force a routing tier ("fast", "standard",
                "comprehensive"). If None, the router auto-selects.

        Returns:
            Formatted context string (with or without citations based on config)

        Example:
            ```python
            context = await client.search.get_context("key findings")
            # Use in LLM prompt:
            prompt = f"Based on the following context:\\n{context}\\n\\nAnswer: ..."
            ```
        """
        results = await self.search(
            query=query,
            top_k=max_chunks,
            document_ids=document_ids,
            routing_tier=routing_tier,
        )

        config = self._client.config

        # Check if citations are disabled or format is NONE
        if not config.enable_citations or config.citation_format == CitationFormat.NONE:
            return self._format_context_plain(results, max_chunks)

        # Format based on citation format
        if config.citation_format == CitationFormat.MARKDOWN:
            return self._format_context_markdown(results, max_chunks)
        elif config.citation_format == CitationFormat.INDEX:
            return self._format_context_indexed(results, max_chunks)
        else:
            # Default: STABLE_ID format (same as results.to_context)
            return results.to_context(max_chunks=max_chunks)

    def _format_context_plain(self, results: SearchResults, max_chunks: int) -> str:
        """Format context without citation markers."""
        if not results.results:
            return "No relevant information found."

        parts = []
        for result in results.results[:max_chunks]:
            parts.append(result.chunk.content)

        return "\n\n".join(parts)

    def _format_context_markdown(self, results: SearchResults, max_chunks: int) -> str:
        """Format context with full markdown source info."""
        if not results.results:
            return "No relevant information found."

        parts = []
        for result in results.results[:max_chunks]:
            source = result.document.filename
            page = f", p.{result.chunk.page_number}" if result.chunk.page_number else ""
            parts.append(f"[Source: {source}{page}]\n{result.chunk.content}")

        return "\n\n".join(parts)

    def _format_context_indexed(self, results: SearchResults, max_chunks: int) -> str:
        """Format context with indexed [1], [2] style citations."""
        if not results.results:
            return "No relevant information found."

        parts = []
        for idx, result in enumerate(results.results[:max_chunks], 1):
            source = result.document.filename
            page = f" (page {result.chunk.page_number})" if result.chunk.page_number else ""
            parts.append(f"[{idx}] {source}{page}\n{result.chunk.content}")

        return "\n\n".join(parts)

    async def search_with_citations(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> tuple[SearchResults, str]:
        """Search and return both results and formatted context.

        Convenience method that returns both the structured results
        and the formatted context string.

        Args:
            query: Natural language search query
            top_k: Maximum number of results
            document_ids: Optional filter by document IDs

        Returns:
            Tuple of (SearchResults, formatted_context)

        Example:
            ```python
            results, context = await client.search.search_with_citations(
                "quarterly performance"
            )
            # Use results for detailed inspection
            for r in results:
                print(f"[{r.citation_id}] {r.score:.2f}")
            # Use context for LLM
            print(context)
            ```
        """
        results = await self.search(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
        )
        context = results.to_context(max_chunks=top_k)
        return results, context

    # =========================================================================
    # Hybrid Search
    # =========================================================================

    async def hybrid_search(
        self,
        query: str,
        top_k: int = 20,
        alpha: Optional[float] = None,
        document_ids: Optional[List[str]] = None,
    ) -> SearchResults:
        """Perform hybrid dense + sparse search.

        Combines vector similarity with BM25 keyword matching using
        Reciprocal Rank Fusion (RRF).

        This method is thread-safe and does not modify shared configuration.

        Args:
            query: Natural language search query
            top_k: Number of results to return
            alpha: Weight between vector (1.0) and keyword (0.0).
                Default from config.
            document_ids: Optional filter by document IDs

        Returns:
            SearchResults with combined scores

        Raises:
            SearchError: If search operation fails

        Example:
            ```python
            # More emphasis on keywords
            results = await client.search.hybrid_search("API endpoint", alpha=0.3)

            # More emphasis on semantics (default)
            results = await client.search.hybrid_search("query", alpha=0.7)
            ```
        """
        return await self._search_internal(
            query=query,
            top_k=top_k,
            document_ids=document_ids,
            force_hybrid=True,
            hybrid_alpha=alpha,
        )

    async def _search_internal(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
        document_ids: Optional[List[str]] = None,
        use_reranking: Optional[bool] = None,
        use_hyde: Optional[bool] = None,
        force_hybrid: bool = False,
        hybrid_alpha: Optional[float] = None,
        routing_tier: Optional[str] = None,
    ) -> SearchResults:
        """Internal search method with all override parameters.

        This method accepts explicit overrides without modifying shared state,
        making it thread-safe for concurrent operations.
        """
        backend = await self._client._get_backend()
        session_id = self._client.session_id
        config = self._client.config

        # Determine effective document IDs (explicit > focused > all)
        effective_doc_ids = document_ids or self._client.focused_document_ids

        # Determine feature flags (explicit override > config)
        do_reranking = use_reranking if use_reranking is not None else config.enable_reranking
        do_hyde = use_hyde if use_hyde is not None else config.enable_hyde
        do_hybrid = force_hybrid or config.enable_hybrid_search
        effective_alpha = hybrid_alpha if hybrid_alpha is not None else config.hybrid_alpha

        logger.debug(
            f"Searching: query='{query[:50]}...', "
            f"top_k={top_k}, reranking={do_reranking}, hyde={do_hyde}, "
            f"hybrid={do_hybrid}, alpha={effective_alpha}"
        )
        start_time = datetime.now(timezone.utc)

        try:
            # Execute search with all parameters
            # Note: hybrid search and alpha are controlled by RAGConfig
            # internally in the search orchestrator, not passed as kwargs.
            search_kwargs = dict(
                query=query,
                session_id=session_id,
                top_k=top_k,
                min_score=min_score,
                document_ids=effective_doc_ids,
                use_hyde=do_hyde,
                use_reranking=do_reranking,
            )
            if routing_tier is not None:
                search_kwargs["routing_tier"] = routing_tier
            # async_search, not search: this coroutine runs on the caller's
            # event loop, where the synchronous path both blocks the loop and
            # cannot start the agentic pipeline (it needs a loop of its own).
            # Collected as the REST route does: a fallback (hybrid running
            # dense-only, HyDE without an LLM) is reported, not hidden.
            with _degraded.collecting() as degraded:
                raw_results = await backend.async_search(**search_kwargs)
        except Exception as e:
            raise SearchError(f"Search failed: {e}") from e

        search_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

        # Convert to SDK types
        results = []
        citations = []

        # Check if citations are enabled
        citations_enabled = config.enable_citations

        for raw_result in raw_results:
            chunk = DocumentChunk(
                chunk_id=str(raw_result.chunk.chunk_id),
                document_id=raw_result.chunk.document_id,
                content=raw_result.chunk.content,
                chunk_index=raw_result.chunk.chunk_index,
                page_number=raw_result.chunk.page_number,
                section_header=raw_result.chunk.metadata.get("section_header"),
                metadata=raw_result.chunk.metadata,
            )

            # Build Document from either local backend (full DocumentMetadata)
            # or remote backend (RemoteDocumentRef / flat fields).
            raw_doc = getattr(raw_result, "document", None)
            if raw_doc is not None:
                document = Document(
                    document_id=raw_doc.document_id,
                    filename=raw_doc.filename,
                    session_id=getattr(raw_doc, "session_id", ""),
                    status=DocumentStatus.READY,
                    file_type=getattr(raw_doc, "file_type", ""),
                    file_size=getattr(raw_doc, "file_size", 0),
                    chunk_count=getattr(raw_doc, "chunks_count", 0),
                    created_at=getattr(raw_doc, "upload_date", ""),
                )
            else:
                # Fallback for backends that put doc info on the result itself
                document = Document(
                    document_id=getattr(raw_result, "document_id", raw_result.chunk.document_id),
                    filename=getattr(raw_result, "filename", ""),
                    session_id="",
                    status=DocumentStatus.READY,
                )

            # Create citation only if enabled
            citation_id = None
            if citations_enabled:
                citation_id = raw_result.chunk.metadata.get("stable_id", f"cite_{chunk.chunk_id}")
                citation = Citation(
                    citation_id=citation_id,
                    chunk_id=chunk.chunk_id,
                    document_id=document.document_id,
                    filename=document.filename,
                    page_number=chunk.page_number,
                    content_preview=chunk.content[:200] if chunk.content else None,
                    score=raw_result.score,
                    highlights=raw_result.chunk.metadata.get("highlights"),
                )
                citations.append(citation)

            results.append(
                SearchResult(
                    chunk=chunk,
                    document=document,
                    score=raw_result.score,
                    citation_id=citation_id,
                )
            )

        return SearchResults(
            results=results,
            query=query,
            total_chunks_searched=len(raw_results),
            search_time_ms=search_time,
            citations=citations,
            degraded_stages=list(degraded),
        )

    # =========================================================================
    # Streaming Search
    # =========================================================================

    async def search_stream(
        self,
        query: str,
        top_k: int = 20,
        min_score: float = 0.0,
        document_ids: Optional[List[str]] = None,
    ) -> AsyncIterator[SearchResult]:
        """Yield search results one at a time from a completed search.

        Note: This is a convenience wrapper, not true streaming. The full
        search is executed first, then results are yielded individually.
        Useful when callers want to process results incrementally without
        buffering the entire list.

        Args:
            query: Natural language search query
            top_k: Maximum number of results to return
            min_score: Minimum similarity score threshold
            document_ids: Optional filter by document IDs

        Yields:
            SearchResult objects one at a time

        Example:
            ```python
            async for result in client.search.search_stream("revenue"):
                print(f"Found: {result.chunk.content[:50]}...")
                # Process immediately, don't wait for all results
            ```
        """
        results = await self.search(
            query=query,
            top_k=top_k,
            min_score=min_score,
            document_ids=document_ids,
        )

        for result in results:
            yield result

    # =========================================================================
    # Batch Search
    # =========================================================================

    async def search_batch(
        self,
        queries: List[str],
        top_k: int = 5,
        max_concurrent: int = 5,
        min_score: float = 0.0,
        document_ids: Optional[List[str]] = None,
    ) -> List[SearchResults]:
        """Search multiple queries with concurrency control.

        Executes multiple searches in parallel with configurable concurrency
        limit. Results are returned in the same order as input queries.

        Args:
            queries: List of search queries
            top_k: Results per query
            max_concurrent: Maximum concurrent searches (default: 5)
            min_score: Minimum similarity score threshold
            document_ids: Optional document filter applied to all queries

        Returns:
            List of SearchResults, one per query (order preserved)

        Raises:
            SearchError: If any search operation fails

        Example:
            ```python
            results = await client.search.search_batch([
                "revenue",
                "expenses",
                "profit margin",
            ])
            for query, result in zip(queries, results):
                print(f"{query}: {len(result)} results")
            ```
        """
        if not queries:
            return []

        semaphore = asyncio.Semaphore(max_concurrent)

        async def search_one(query: str) -> SearchResults:
            async with semaphore:
                return await self.search(
                    query=query,
                    top_k=top_k,
                    min_score=min_score,
                    document_ids=document_ids,
                )

        # Use gather to maintain order
        return await asyncio.gather(*[search_one(q) for q in queries])

    # =========================================================================
    # Multi-Query Search
    # =========================================================================

    async def multi_query_search(
        self,
        queries: List[str],
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
        deduplicate: bool = True,
    ) -> SearchResults:
        """Search with multiple queries and merge results.

        Useful for query expansion or when searching for multiple concepts.

        Args:
            queries: List of search queries
            top_k: Results per query
            document_ids: Optional filter by document IDs
            deduplicate: Remove duplicate chunks from results

        Returns:
            Merged SearchResults

        Example:
            ```python
            results = await client.search.multi_query_search([
                "quarterly revenue",
                "Q3 income",
                "financial performance",
            ])
            ```
        """
        all_results = []
        seen_chunks = set()

        # Execute all queries concurrently
        batch_results = await asyncio.gather(
            *[self.search(query=q, top_k=top_k, document_ids=document_ids) for q in queries]
        )

        for results in batch_results:
            for result in results:
                if deduplicate:
                    if result.chunk.chunk_id in seen_chunks:
                        continue
                    seen_chunks.add(result.chunk.chunk_id)
                all_results.append(result)

        # Sort by score
        all_results.sort(key=lambda r: r.score, reverse=True)

        # Take top results
        all_results = all_results[:top_k]

        # Only create citations list if citations are enabled
        citations = []
        if self._client.config.enable_citations:
            citations = [
                Citation(
                    citation_id=r.citation_id or f"cite_{r.chunk.chunk_id}",
                    chunk_id=r.chunk.chunk_id,
                    document_id=r.document.document_id,
                    filename=r.document.filename,
                    page_number=r.chunk.page_number,
                    score=r.score,
                )
                for r in all_results
            ]

        degraded = [s for r in batch_results for s in r.degraded_stages]
        return SearchResults(
            results=all_results,
            query=" | ".join(queries),
            total_chunks_searched=len(all_results),
            search_time_ms=0,  # Aggregate not tracked
            citations=citations,
            degraded_stages=list(dict.fromkeys(degraded)),
        )
