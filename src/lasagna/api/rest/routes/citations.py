"""Citation management endpoints.

- ``GET /api/v1/citations``       — list session citations
- ``GET /api/v1/citations/{id}``  — get citation by ID
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from lasagna.api.rag_backend import RAGBackend
from ..dependencies import get_backend, get_session_id, get_user_id
from ..models import CitationResponse, CitationListResponse

router = APIRouter(prefix="/citations", tags=["citations"])


@router.get("", response_model=CitationListResponse)
async def list_citations(
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
    backend: RAGBackend = Depends(get_backend),
):
    """List all citations for the current session."""
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        raw_citations, _ = mgr.get_last_search_results(session_id=session_id)

        citations = [
            CitationResponse(
                citation_id=c.get("citation_id", ""),
                document_id=c.get("document_id", ""),
                document_name=c.get("document_name", ""),
                chunk_id=c.get("chunk_id", ""),
                chunk_index=c.get("chunk_index", 0),
                page_number=c.get("page_number"),
                score=c.get("score", 0.0),
                content=c.get("content") or c.get("content_preview", ""),
                highlight=c.get("highlight"),
                highlight_ranges=c.get("highlight_ranges"),
            )
            for c in raw_citations
        ]

        return CitationListResponse(
            citations=citations,
            session_id=session_id,
            total=len(citations),
        )
    except Exception as exc:
        logger.error(f"Failed to list citations: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/{citation_id}", response_model=CitationResponse)
async def get_citation(
    citation_id: str,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Get a specific citation by ID.

    Uses a two-tier lookup: first checks the in-memory session store
    (preserves ``content`` and ``highlight`` fields from the last search),
    then falls back to the persistent database (stores ``content_preview``,
    no ``highlight``).
    """
    try:
        # Tier 1: in-memory session store (recent search results)
        from lasagna.core.session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager(user_id=user_id, session_id=session_id)
        raw_citations, _ = mgr.get_last_search_results(session_id=session_id)

        for c in raw_citations:
            if c.get("citation_id") == citation_id:
                return CitationResponse(
                    citation_id=c.get("citation_id", ""),
                    document_id=c.get("document_id", ""),
                    document_name=c.get("document_name", ""),
                    chunk_id=c.get("chunk_id", ""),
                    chunk_index=c.get("chunk_index", 0),
                    page_number=c.get("page_number"),
                    score=c.get("score", 0.0),
                    content=c.get("content") or c.get("content_preview", ""),
                    highlight=c.get("highlight"),
                    highlight_ranges=c.get("highlight_ranges"),
                )

        # Tier 2: persistent database fallback (older citations)
        from lasagna.core.database import get_rag_database

        rag_db = get_rag_database()
        row = rag_db.get_citation(citation_id)
        if row and row.get("session_id") == session_id:
            return CitationResponse(
                citation_id=row.get("citation_id", ""),
                document_id=row.get("document_id", ""),
                document_name=row.get("document_name", ""),
                chunk_id=row.get("chunk_id", ""),
                chunk_index=row.get("chunk_index", 0),
                page_number=row.get("page_number"),
                score=row.get("score", 0.0),
                content=row.get("content_preview", ""),
                highlight=None,
            )

        raise HTTPException(status_code=404, detail=f"Citation '{citation_id}' not found")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to get citation: {exc}")
        raise HTTPException(status_code=500, detail="Internal server error")
