"""Agentic RAG Module.

Provides multi-hop retrieval with query refinement using LangGraph.

Example:
    ```python
    from lasagna.sdk.agentic import (
        AgenticRAG,
        AgenticRAGWrapper,
        create_rag_tools,
    )

    # Use via client
    results = await client.agentic_search("complex query")

    # Or directly
    agentic = AgenticRAGWrapper(client)
    results = await agentic.search("query", max_iterations=3)

    # Create LangChain tools
    tools = create_rag_tools(client)
    ```
"""

from typing import List, Optional, TYPE_CHECKING

from loguru import logger

from ..types import AgenticSearchResults, HopData, Citation, SearchResults
from ..exceptions import (
    AgenticError,
    MaxIterationsError,
    SearchError,
)

if TYPE_CHECKING:
    from ..client import KnowledgeClient

# Re-export from existing agentic module
from ...agentic.agentic_rag import (
    AgenticRAG,
    AgenticRAGResult,
    AgenticRAGMetrics,
)
from ...agentic.citation_manager import (
    CitationManager as BackendCitationManager,
    Citation as BackendCitation,
    get_citation_manager,
)
from ...agentic.document_awareness import (
    DocumentInfo,
    DocumentSnippet,
    get_document_awareness_context,
)


class AgenticRAGWrapper:
    """Wrapper for AgenticRAG that integrates with KnowledgeClient.

    Provides a clean interface for multi-hop retrieval.

    Example:
        ```python
        wrapper = AgenticRAGWrapper(client)
        results = await wrapper.search(
            "What were the key factors affecting Q3 performance?",
            max_iterations=3,
        )
        print(f"Found in {results.total_iterations} hops")
        print(results.final_context)
        ```
    """

    def __init__(self, client: "KnowledgeClient"):
        """Initialize agentic RAG wrapper.

        Args:
            client: Parent KnowledgeClient instance
        """
        self._client = client

    def _create_agentic_rag(self, max_iterations: int = 3) -> AgenticRAG:
        """Create a fresh AgenticRAG instance per call.

        Uses ``AgenticRAG.create()`` which properly provisions the LLM
        from the configured provider.  Not cached — ensures session config
        changes (quality-gate toggles, structured-graph flag, etc.) are
        always picked up and ``max_iterations`` is always honoured.

        SDK ``KnowledgeConfig`` agentic fields are applied as overrides
        on the backend session config so that ``client.set_config()``
        changes are respected.
        """
        session_config = None
        try:
            from ...core.config import rag_config

            session_config = rag_config.with_session(self._client.session_id)
        except Exception:
            pass

        # Bridge SDK KnowledgeConfig toggles into the backend session
        # config so the user's intent is honoured.
        #
        # Uses the centralized config bridge which only forwards
        # non-default fields (dirty-field tracking), preventing SDK
        # defaults from masking profile/TOML settings.
        if session_config is not None and hasattr(session_config, "_env_overrides"):
            from .._config_bridge import get_runtime_overrides

            overrides = get_runtime_overrides(self._client.config)
            if overrides:
                session_config._env_overrides.update(overrides)

        # Derive enable_final_search from session config (default=False)
        # rather than AgenticRAG.create()'s default (True).
        enable_final_search = False
        if session_config is not None:
            enable_final_search = getattr(session_config, "use_final_search", False)

        return AgenticRAG.create(
            max_iterations=max_iterations,
            enable_final_search=enable_final_search,
            session_config=session_config,
        )

    async def search(
        self,
        query: str,
        max_iterations: Optional[int] = None,
        document_ids: Optional[List[str]] = None,
    ) -> AgenticSearchResults:
        """Perform multi-hop agentic search.

        Uses a ReAct-style agent to iteratively search and refine queries.
        The agent decides when to stop based on coverage and relevance.

        Args:
            query: Initial search query
            max_iterations: Maximum search hops (default from config)
            document_ids: Optional filter by document IDs

        Returns:
            AgenticSearchResults with hop data and final context

        Example:
            ```python
            results = await wrapper.search(
                "Explain the relationship between revenue and costs",
                max_iterations=5,
            )
            for hop in results.hops:
                print(f"Hop {hop.iteration}: {hop.query}")
            ```
        """
        from datetime import datetime, timezone

        config = self._client.config
        session_id = self._client.session_id

        # Get max iterations
        iterations = max_iterations or config.max_agent_iterations

        # Create fresh AgenticRAG (no caching — always picks up config changes)
        agentic_rag = self._create_agentic_rag(max_iterations=iterations)

        start_time = datetime.now(timezone.utc)

        try:
            # Execute agentic retrieval
            result = await agentic_rag.retrieve(
                question=query,
                session_id=session_id,
                document_ids=document_ids,
            )

            search_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            # Convert backend HopData to SDK HopData.
            # Count unique chunks across hops by chunk_id to match
            # SDK type contract ("total unique chunks retrieved").
            hops = []
            seen_chunk_ids: set = set()
            for hop in result.hops:
                n_chunks = len(hop.retrieved_chunks)
                for chunk in hop.retrieved_chunks:
                    seen_chunk_ids.add(chunk.chunk_id)
                hops.append(
                    HopData(
                        iteration=hop.hop_index,
                        query=hop.query_used,
                        chunks_retrieved=n_chunks,
                        refined_query=None,
                        coverage_score=None,
                        hop_time_ms=hop.search_latency_ms + hop.refinement_latency_ms,
                    )
                )

            # Convert citation dicts to SDK Citation objects.
            # Citation dicts from refine_chunks_node have "index" but no
            # "citation_id".  Synthesise a stable id from document_id +
            # chunk_id + index so that get_citation() works.
            citations = []
            for cite in result.citations:
                if isinstance(cite, dict):
                    cid = str(cite.get("citation_id", ""))
                    if not cid:
                        cid = f"{cite.get('document_id', '')}_{cite.get('chunk_id', '')}_{cite.get('index', 0)}"
                    citations.append(
                        Citation(
                            citation_id=cid,
                            chunk_id=str(cite.get("chunk_id", "")),
                            document_id=str(cite.get("document_id", "")),
                            filename=str(cite.get("document_name", "")),
                            page_number=cite.get("page_number"),
                            content_preview=str(cite.get("content_preview", "")),
                            score=float(cite.get("score", 0.0)),
                            hop_index=int(cite.get("hop_index", 0)),
                        )
                    )
                else:
                    # Citation object
                    citations.append(
                        Citation(
                            citation_id=str(getattr(cite, "citation_id", "")),
                            chunk_id=str(getattr(cite, "chunk_id", "")),
                            document_id=str(getattr(cite, "document_id", "")),
                            filename=str(getattr(cite, "document_name", "")),
                            page_number=getattr(cite, "page_number", None),
                            content_preview=str(getattr(cite, "content_preview", "")),
                            score=float(getattr(cite, "score", 0.0)),
                            hop_index=int(getattr(cite, "hop_index", 0)),
                        )
                    )

            return AgenticSearchResults(
                final_context=result.formatted_context,
                citations=citations,
                hops=hops,
                total_iterations=result.search_count,
                total_chunks_retrieved=len(seen_chunk_ids),
                search_time_ms=search_time,
                refined_answer=result.agent_summary,
            )

        except Exception as e:
            search_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            logger.error(f"Agentic search failed: {e}")

            # Check for max iterations error
            if "max" in str(e).lower() and "iteration" in str(e).lower():
                raise MaxIterationsError(
                    f"Agentic search exceeded max iterations: {e}",
                    iteration=iterations,
                ) from e

            raise AgenticError(
                f"Agentic search failed: {e}",
                iteration=0,
            ) from e


def create_rag_tools(client: "KnowledgeClient") -> list:
    """Create LangChain tools for RAG operations.

    Creates tools that can be used with LangChain agents for
    knowledge base operations.

    Args:
        client: KnowledgeClient instance

    Returns:
        List of LangChain tools

    Example:
        ```python
        from lasagna.sdk import KnowledgeClient
        from lasagna.sdk.agentic import create_rag_tools

        client = KnowledgeClient()
        tools = create_rag_tools(client)

        # Use with LangChain agent
        from langchain.agents import create_agent
        agent = create_agent(llm, tools=tools)
        ```
    """
    try:
        from langchain_core.tools import tool
    except ImportError:
        raise ImportError(
            "langchain_core is required for create_rag_tools. "
            "Install it with: pip install langchain-core"
        )

    @tool
    async def search(query: str) -> str:
        """Search the knowledge base for relevant information.

        Args:
            query: Natural language search query

        Returns:
            Formatted context with source citations
        """
        return await client.get_context(query)

    @tool
    async def list_documents() -> str:
        """List all documents in the knowledge base.

        Returns:
            List of document names and IDs
        """
        docs = await client.list_documents()
        if not docs:
            return "No documents in knowledge base."
        lines = ["Documents:"]
        for doc in docs:
            lines.append(f"  - {doc.filename} (ID: {doc.document_id}, chunks: {doc.chunk_count})")
        return "\n".join(lines)

    @tool
    async def search_in_document(document_id: str, query: str) -> str:
        """Search for information within a specific document.

        Args:
            document_id: ID of document to search in
            query: Natural language search query

        Returns:
            Formatted context with source citations
        """
        return await client.get_context(query, document_ids=[document_id])

    return [search, list_documents, search_in_document]


__all__ = [
    # Wrappers
    "AgenticRAGWrapper",
    "create_rag_tools",
    # Types
    "AgenticSearchResults",
    "HopData",
    # Backend exports
    "AgenticRAG",
    "AgenticRAGResult",
    "AgenticRAGMetrics",
    "DocumentInfo",
    "DocumentSnippet",
    "get_document_awareness_context",
    "get_citation_manager",
]
