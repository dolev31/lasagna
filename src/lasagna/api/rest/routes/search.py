"""Search and citation endpoints.

- ``POST /api/v1/search``           — standard search
- ``POST /api/v1/search/citations`` — search with citation tracking
"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from starlette.concurrency import run_in_threadpool

from lasagna.api import _degraded
from lasagna.api._search_orchestrator import SearchScopeUnavailable
from lasagna.api.rag_backend import RAGBackend
from lasagna.core.quota import QuotaExceededError
from ..dependencies import get_backend, get_session_id, get_user_id
from ..models import (
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
    SearchWithCitationsResponse,
    ChunkResponse,
    ErrorResponse,
    CitationResponse,
    RoutingInfo,
)


def _missing_llm(exc: BaseException):
    """The no-LLM message when that is what failed, followed through wrapping."""
    from lasagna.providers import LLMProviderNotConfigured

    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, LLMProviderNotConfigured) or str(exc).startswith(
            "No LLMProvider configured"
        ):
            return str(exc)
        exc = exc.__cause__ or exc.__context__
    return None


router = APIRouter(prefix="/search", tags=["search"])


async def _within_deadline(call):
    """Await ``call`` for at most ``RAG_SEARCH_TIMEOUT`` seconds, else answer 504.

    A search that hung (an LLM stage waiting on a provider, a vector store
    mid-failover) held its request until the proxy or the client gave up,
    with nothing in the response to say why. The thread running it cannot be
    stopped and finishes on its own; the client is answered on time.
    """
    from lasagna.api.rest.limits import search_timeout

    deadline = search_timeout()
    try:
        return await asyncio.wait_for(call, timeout=deadline)
    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
            detail=(
                f"The search did not finish within {deadline:g} s (RAG_SEARCH_TIMEOUT). "
                f"Retry, raise RAG_SEARCH_TIMEOUT, or turn off the slow stages "
                f"(HyDE, agentic routing) for this session."
            ),
        ) from None


def _get_routing_info(backend: RAGBackend) -> RoutingInfo | None:
    """Read the last routing decision from the search orchestrator.

    The async search methods relay the routing decision from the executor
    thread back to the orchestrator instance (``_last_routing_decision``).
    Since there is no ``await`` between the async search return and this
    read, no other coroutine can overwrite it — safe in the event loop.
    """
    try:
        decision = getattr(backend._search_orchestrator, "_last_routing_decision", None)
        if decision is None:
            return None
        signals = decision.signals
        return RoutingInfo(
            tier=decision.tier.value,
            strategy=decision.strategy_used,
            confidence=decision.confidence,
            rationale=decision.rationale,
            query_type=signals.query_type if signals else None,
            estimated_complexity=signals.estimated_complexity if signals else None,
        )
    except Exception:
        return None


def _result_to_response(r) -> SearchResultResponse:
    """Convert internal SearchResult to API response."""
    return SearchResultResponse(
        chunk=ChunkResponse(
            chunk_id=r.chunk.chunk_id,
            content=r.chunk.content,
            document_id=r.chunk.document_id,
            chunk_index=r.chunk.chunk_index,
            page_number=r.chunk.page_number,
            metadata=r.chunk.metadata,
        ),
        score=r.score,
        document_id=r.chunk.document_id,
        filename=r.document.filename if r.document else "",
    )


_SEARCH_ERRORS = {
    400: {"model": ErrorResponse, "description": "The request was invalid for this session"},
    503: {
        "model": ErrorResponse,
        "description": "The session's focused documents could not be read, so nothing was searched",
    },
    504: {"model": ErrorResponse, "description": "The search ran past RAG_SEARCH_TIMEOUT"},
}


@router.post("", response_model=SearchResponse, responses=_SEARCH_ERRORS)
async def search(
    body: SearchRequest,
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Search with the session's retrieval pipeline.

    Stages, each as the session config sets them: hybrid dense and BM25
    search, cross-encoder reranking, MMR diversity, HyDE, adaptive-k
    retrieval, hierarchical auto-merge. A stage that failed and fell back is
    named in ``degraded_stages``.
    """
    try:
        with _degraded.collecting() as degraded:
            results = await _within_deadline(
                backend.async_search(
                    query=body.query,
                    session_id=session_id,
                    top_k=body.top_k,
                    min_score=body.min_score,
                    document_ids=body.document_ids,
                    use_hyde=body.use_hyde,
                    use_reranking=body.use_reranking,
                    routing_tier=body.routing_tier,
                )
            )
        routing_info = _get_routing_info(backend)
        return SearchResponse(
            results=[_result_to_response(r) for r in results],
            query=body.query,
            total_results=len(results),
            routing=routing_info,
            degraded_stages=degraded,
        )
    except HTTPException:
        raise
    except SearchScopeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        logger.error(f"Search failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


def _remember_citations(session_id: str, user_id: str, citations, results) -> None:
    """Keep a search's citations for ``GET /citations``, as MCP search does.

    The list read only what MCP search stored, so for a REST client it was
    always empty, even right after this endpoint returned citations.
    """
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        get_knowledge_config_manager(
            user_id=user_id, session_id=session_id
        ).set_last_search_results(
            citations=[c.model_dump() for c in citations],
            chunks=[
                {
                    "chunk_id": str(r.chunk.chunk_id),
                    "content": r.chunk.content,
                    "chunk_index": r.chunk.chunk_index,
                }
                for r in results
            ],
            session_id=session_id,
        )
    except Exception as exc:
        logger.warning(f"Search citations could not be kept for GET /citations: {exc}")


@router.post("/citations", response_model=SearchWithCitationsResponse, responses=_SEARCH_ERRORS)
async def search_with_citations(
    body: SearchRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Search with citation tracking and formatted LLM context.

    Returns search results, stable citations, and a pre-formatted
    context string ready for injection into LLM prompts.
    """
    try:
        with _degraded.collecting() as degraded:
            results, citations, context = await _within_deadline(
                backend.async_search_with_citations(
                    query=body.query,
                    session_id=session_id,
                    top_k=body.top_k,
                    document_ids=body.document_ids,
                    routing_tier=body.routing_tier,
                )
            )
        routing_info = _get_routing_info(backend)
        cited = [
            CitationResponse(
                citation_id=c.citation_id,
                document_id=c.document_id,
                document_name=c.document_name,
                chunk_id=c.chunk_id,
                chunk_index=c.chunk_index,
                page_number=c.page_number,
                score=c.score,
                content=c.content,
                highlight=(
                    c.highlights[0]
                    if getattr(c, "highlights", None)
                    else getattr(c, "highlight", None)
                ),
                highlights=getattr(c, "highlights", None),
                highlight_ranges=(
                    [list(r) for r in c.highlight_ranges]
                    if getattr(c, "highlight_ranges", None)
                    else None
                ),
            )
            for c in citations
        ]
        _remember_citations(session_id, user_id, cited, results)
        return SearchWithCitationsResponse(
            results=[_result_to_response(r) for r in results],
            citations=cited,
            context=context,
            query=body.query,
            total_results=len(results),
            routing=routing_info,
            degraded_stages=degraded,
        )
    except HTTPException:
        raise
    except SearchScopeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        logger.error(f"Search with citations failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/multi-query", response_model=SearchResponse, responses=_SEARCH_ERRORS)
async def multi_query_search(
    body: SearchRequest,
    n_variations: int = 3,
    session_id: str = Depends(get_session_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Multi-query expansion search.

    Generates ``n_variations`` alternative query phrasings, searches each,
    and fuses the results for higher recall.  Requires ``enable_query_expansion``
    to be set in the session config.

    Note: ``routing_tier`` from SearchRequest is ignored here — multi-query
    search uses RAG Fusion which applies its own retrieval strategy per sub-query.
    """
    if body.routing_tier:
        logger.debug("routing_tier ignored for multi-query endpoint")
    try:
        with _degraded.collecting() as degraded:
            results = await _within_deadline(
                run_in_threadpool(
                    backend.search_multi_query,
                    query=body.query,
                    session_id=session_id,
                    top_k=body.top_k,
                    n_variations=n_variations,
                    document_ids=body.document_ids,
                )
            )
        return SearchResponse(
            results=[_result_to_response(r) for r in results],
            query=body.query,
            total_results=len(results),
            degraded_stages=degraded,
        )
    except HTTPException:
        raise
    except SearchScopeUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        logger.error(f"Multi-query search failed: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/agentic", response_model=SearchWithCitationsResponse, responses=_SEARCH_ERRORS)
async def agentic_search(
    body: SearchRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Agentic multi-hop search with iterative retrieval.

    Uses an LLM agent to decompose complex queries, perform multiple
    retrieval rounds, and synthesise results with citations.
    Requires ``use_agentic_rag`` to be enabled in session config.

    Note: ``routing_tier`` from SearchRequest is ignored here — the agentic
    endpoint IS the comprehensive tier by definition.
    """
    if body.routing_tier:
        logger.debug("routing_tier ignored for agentic endpoint")
    try:
        result = await _within_deadline(
            backend.async_agentic_search(
                query=body.query,
                session_id=session_id,
                top_k=body.top_k,
                document_ids=body.document_ids,
            )
        )
        cited = [
            CitationResponse(
                citation_id=c.get("citation_id", ""),
                document_id=c.get("document_id", ""),
                document_name=c.get("document_name", ""),
                chunk_id=c.get("chunk_id", ""),
                chunk_index=c.get("chunk_index", 0),
                page_number=c.get("page_number"),
                score=c.get("score", 0.0),
                content=c.get("content", ""),
                highlight=c.get("highlight"),
            )
            for c in result.citations
        ]
        _remember_citations(session_id, user_id, cited, [])
        return SearchWithCitationsResponse(
            results=[],  # agentic search returns pre-formatted context
            citations=cited,
            context=result.formatted_context,
            query=body.query,
            total_results=len(result.citations),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except QuotaExceededError as exc:
        raise HTTPException(status_code=429, detail=str(exc))
    except Exception as exc:
        logger.error(f"Agentic search failed: {exc}")
        missing = _missing_llm(exc)
        if missing is not None:
            # A setup problem with a fix the caller can apply; a bare 500 hid
            # it in the server log.
            raise HTTPException(status_code=503, detail=missing)
        raise HTTPException(status_code=500, detail="Internal server error")
