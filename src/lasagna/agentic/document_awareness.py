"""Document Awareness Module for Lasagna.

This module provides document snippet extraction and enrichment to help
the agent understand what documents are available and when to use RAG tools.

Features:
- Extract first N tokens from documents (fast, no LLM cost)
- Generate LLM-based summaries (better quality, costs tokens)
- Hybrid mode: first tokens for small docs, LLM for large docs
- Caching for performance

IMPORTANT: This module uses SQLite directly to get document metadata,
avoiding Milvus access to prevent database lock issues. Milvus Lite only
allows one process to have the database open at a time, and the MCP server
needs exclusive access for search operations.
"""

import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Any
from loguru import logger

from ..core.config import rag_config
from ..utils.monitoring import METRICS, log_json, trunc_text
from ..utils.model_utils import resolve_model_settings, get_default_model_settings
from ..providers import get_llm_provider


@dataclass
class DocumentInfo:
    """Lightweight document info from SQLite (no Milvus access needed)."""

    document_id: str
    session_id: str
    filename: str
    file_type: str
    file_size: int
    status: str
    chunks_count: int
    upload_date: Optional[datetime]
    file_path: Optional[str] = None
    preview_text: Optional[str] = None  # Pre-computed snippet from upload time


def _get_documents_from_sqlite(session_id: str) -> List[DocumentInfo]:
    """Get document list directly from SQLite without touching Milvus.

    This avoids Milvus lock issues since SQLite supports multiple readers.
    Now also fetches the pre-computed preview_text to avoid query-time PDF extraction.
    """
    try:
        from ..core.database import get_rag_database

        rag_db = get_rag_database()
        rows = rag_db.get_session_documents(session_id, status="ready")

        documents = []
        for row in rows:
            upload_date = None
            if row.get("upload_date"):
                try:
                    upload_date = (
                        datetime.fromisoformat(row["upload_date"])
                        if isinstance(row["upload_date"], str)
                        else row["upload_date"]
                    )
                except (ValueError, TypeError):
                    pass

            documents.append(
                DocumentInfo(
                    document_id=row["document_id"],
                    session_id=row["session_id"],
                    filename=row["filename"],
                    file_type=row["file_type"],
                    file_size=row.get("file_size") or 0,
                    status=row["status"],
                    chunks_count=row.get("chunks_count") or 0,
                    upload_date=upload_date,
                    file_path=row.get("file_path"),
                    preview_text=row.get("preview_text"),
                )
            )

        return documents
    except Exception as e:
        logger.error(f"Failed to get documents from SQLite: {e}")
        return []


@dataclass
class DocumentSnippet:
    """Enriched document snippet for agent awareness."""

    document_id: str
    filename: str
    file_type: str
    file_size_kb: int
    upload_date: Optional[datetime]
    snippet: str  # The extracted/generated snippet
    snippet_type: str  # "first_tokens" or "llm_summary"
    chunks_count: int


class SnippetCache:
    """LRU cache for document snippets."""

    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, tuple[DocumentSnippet, float]] = OrderedDict()

    def _make_key(self, document_id: str, session_id: str) -> str:
        """Create cache key from document and session ID."""
        return f"{session_id}:{document_id}"

    def get(self, document_id: str, session_id: str) -> Optional[DocumentSnippet]:
        """Get snippet from cache if not expired."""
        key = self._make_key(document_id, session_id)
        if key in self.cache:
            snippet, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                self.cache.move_to_end(key)
                return snippet
            else:
                del self.cache[key]
        return None

    def put(self, document_id: str, session_id: str, snippet: DocumentSnippet):
        """Store snippet in cache."""
        key = self._make_key(document_id, session_id)
        while len(self.cache) >= self.max_size:
            self.cache.popitem(last=False)
        self.cache[key] = (snippet, time.time())

    def invalidate(self, document_id: str, session_id: str):
        """Remove a specific snippet from cache."""
        key = self._make_key(document_id, session_id)
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        """Clear entire cache."""
        self.cache.clear()


# Global cache instance
_snippet_cache: Optional[SnippetCache] = None


def get_snippet_cache() -> SnippetCache:
    """Get or create the snippet cache singleton."""
    global _snippet_cache
    if _snippet_cache is None:
        _snippet_cache = SnippetCache(max_size=100, ttl=rag_config.snippet_cache_ttl)
    return _snippet_cache


def _extract_first_tokens(text: str, token_count: int = 150) -> str:
    """Extract approximately the first N tokens from text.

    Uses simple word-based tokenization for speed.
    Tokens are roughly equivalent to words.

    Args:
        text: Full document text
        token_count: Approximate number of tokens to extract

    Returns:
        Extracted text snippet
    """
    if not text:
        return ""

    # Simple word tokenization
    words = text.split()

    # Take first N words (rough token estimate)
    snippet_words = words[:token_count]
    snippet = " ".join(snippet_words)

    # Clean up: normalize whitespace
    snippet = " ".join(snippet.split())

    # Add ellipsis if truncated
    if len(words) > token_count:
        snippet += "..."

    return snippet


async def _generate_llm_summary(
    text: str,
    model: str = None,
    max_tokens: int = 100,
    prompt: str = None,
    document_id: Optional[str] = None,
    document_name: Optional[str] = None,
) -> str:
    """Generate an LLM-based summary of the document.

    Args:
        text: Full document text (will be truncated for LLM)
        model: LLM model to use (None = use MODEL_NAME env var or default)
        max_tokens: Maximum tokens in summary
        prompt: Custom prompt for summarization

    Returns:
        Generated summary
    """
    if model is None:
        from lasagna.utils.model_utils import get_default_model_name

        model = get_default_model_name()
    summary_start = time.perf_counter()
    model_settings = None
    try:
        model_settings = resolve_model_settings(
            model_ref=model,
            default_settings=get_default_model_settings(),
            max_tokens=max_tokens,
            context="summary_model",
        )
        llm = get_llm_provider().get_chat_model(model_settings)

        # Truncate text if too long (use first ~2000 tokens worth)
        text_for_summary = _extract_first_tokens(text, token_count=2000)

        prompt = prompt or rag_config.summary_prompt
        messages = [
            {
                "role": "system",
                "content": "You are a document summarizer. Be concise and focus on key information.",
            },
            {"role": "user", "content": f"{prompt}\n\n{text_for_summary}"},
        ]

        response = await llm.ainvoke(messages)
        summary = response.content if hasattr(response, "content") else str(response)

        latency_ms = (time.perf_counter() - summary_start) * 1000.0
        token_usage = {}
        if hasattr(response, "response_metadata") and isinstance(response.response_metadata, dict):
            token_usage = (
                response.response_metadata.get("token_usage")
                or response.response_metadata.get("usage")
                or {}
            )
        if (
            not token_usage
            and hasattr(response, "additional_kwargs")
            and isinstance(response.additional_kwargs, dict)
        ):
            token_usage = (
                response.additional_kwargs.get("token_usage")
                or response.additional_kwargs.get("usage")
                or {}
            )
        if (
            not token_usage
            and hasattr(response, "llm_output")
            and isinstance(response.llm_output, dict)
        ):
            token_usage = response.llm_output.get("token_usage") or {}

        tokens_in = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
        tokens_out = token_usage.get("completion_tokens") or token_usage.get("output_tokens")

        METRICS.counter("rag.summary.calls", 1)
        METRICS.histogram("rag.summary.latency_ms", latency_ms)
        if tokens_in is not None:
            METRICS.histogram("rag.summary.tokens_in", float(tokens_in))
        if tokens_out is not None:
            METRICS.histogram("rag.summary.tokens_out", float(tokens_out))

        log_json(
            "rag_summary_metrics",
            {
                "document_id": document_id,
                "document_name": document_name,
                "model_platform": model_settings.get("platform") if model_settings else None,
                "model_name": model_settings.get("model_name") if model_settings else model,
                "summary_latency_ms": round(latency_ms, 3),
                "summary_tokens_in": tokens_in,
                "summary_tokens_out": tokens_out,
                "summary_prompt": trunc_text(prompt or "", max_len=120),
                "summary_text_preview": trunc_text(summary.strip(), max_len=120),
                "summary_success": True,
            },
        )

        return summary.strip()

    except Exception as e:
        METRICS.counter("rag.summary.failures", 1)
        log_json(
            "rag_summary_metrics",
            {
                "document_id": document_id,
                "document_name": document_name,
                "model_platform": model_settings.get("platform") if model_settings else None,
                "model_name": model_settings.get("model_name") if model_settings else model,
                "summary_latency_ms": round((time.perf_counter() - summary_start) * 1000.0, 3),
                "summary_success": False,
                "error": str(e),
            },
        )
        logger.error(f"Failed to generate LLM summary: {e}")
        # Fall back to first tokens
        return _extract_first_tokens(text, rag_config.first_tokens_count)


async def get_document_snippet(
    document: DocumentInfo,
    session_id: str,
    backend=None,  # Kept for backwards compatibility but no longer used
) -> DocumentSnippet:
    """Get or create an enriched snippet for a document.

    Args:
        document: DocumentInfo object from SQLite
        session_id: Session ID for caching
        backend: Deprecated, no longer used (kept for backwards compatibility)

    Returns:
        DocumentSnippet with extracted/generated content
    """
    cache = get_snippet_cache()

    # Check cache first
    if rag_config.cache_snippets:
        cached = cache.get(document.document_id, session_id)
        if cached:
            logger.debug(f"Cache hit for document snippet: {document.document_id}")
            return cached

    file_size_kb = document.file_size // 1024 if document.file_size else 0

    # Use pre-stored preview_text if available (set at upload time)
    # This avoids expensive query-time PDF extraction
    if document.preview_text:
        logger.debug(f"Using stored preview_text for document: {document.document_id}")
        snippet_text = document.preview_text
        snippet_type = "first_tokens"
    else:
        # Legacy path: extract text from file for old documents without preview_text
        # This should only happen for documents uploaded before the preview_text feature
        logger.debug(f"No stored preview_text, extracting from file: {document.document_id}")
        document_text = ""
        if document.file_path:
            try:
                import os

                if os.path.exists(document.file_path):
                    from lasagna.processing.document_processor import DocumentProcessor

                    document_text = DocumentProcessor.extract_text(document.file_path)
            except Exception as e:
                logger.warning(f"Failed to extract text for snippet: {e}")

        # Determine enrichment strategy
        strategy = rag_config.enrichment_strategy

        # In hybrid mode, use LLM for large documents
        if strategy == "hybrid" and file_size_kb > rag_config.hybrid_llm_threshold_kb:
            strategy = "llm_summary"
        elif strategy == "hybrid":
            strategy = "first_tokens"

        # Generate snippet based on strategy
        if strategy == "llm_summary" and document_text:
            snippet_text = await _generate_llm_summary(
                document_text,
                model=rag_config.summary_model,
                max_tokens=rag_config.summary_max_tokens,
                prompt=rag_config.summary_prompt,
                document_id=document.document_id,
                document_name=document.filename,
            )
            snippet_type = "llm_summary"
        else:
            snippet_text = _extract_first_tokens(document_text, rag_config.first_tokens_count)
            snippet_type = "first_tokens"

    # Create snippet object
    snippet = DocumentSnippet(
        document_id=document.document_id,
        filename=document.filename,
        file_type=document.file_type,
        file_size_kb=file_size_kb,
        upload_date=document.upload_date,
        snippet=snippet_text,
        snippet_type=snippet_type,
        chunks_count=document.chunks_count,
    )

    # Cache the result
    if rag_config.cache_snippets:
        cache.put(document.document_id, session_id, snippet)

    return snippet


async def get_document_awareness_context(
    session_id: str, rag_search_count: int = 0
) -> Optional[str]:
    """Generate document awareness context for agent prompts.

    This creates a formatted section that tells the agent about available
    documents so it knows when to use RAG tools.

    The context adapts based on how many searches have been done this turn:
    - 0 searches: Guide agent to search the knowledge base
    - 1+ searches: Remind agent that search was done, encourage answering
    - 2+ searches: Strongly encourage synthesizing an answer from existing results

    NOTE: This function uses SQLite directly to get document metadata,
    avoiding Milvus to prevent database lock issues with the MCP server.

    Args:
        session_id: Session ID to get documents for
        rag_search_count: Number of RAG searches performed in current user turn

    Returns:
        Formatted document context string, or None if no documents
    """
    if not rag_config.document_awareness_enabled:
        logger.debug("Document awareness is disabled")
        return None

    # Get documents for session using SQLite directly
    # We avoid using the full RAG backend here to prevent Milvus lock issues
    # (Milvus Lite only allows one process to have the DB open at a time)
    try:
        documents = _get_documents_from_sqlite(session_id)
    except Exception as e:
        logger.error(f"Failed to get documents for awareness context: {e}")
        return None

    if not documents:
        logger.debug(f"No documents in session {session_id} for awareness context")
        return None

    # Limit number of documents
    documents = documents[: rag_config.max_documents_in_context]

    # Generate snippets for each document
    snippets: List[DocumentSnippet] = []
    for doc in documents:
        try:
            snippet = await get_document_snippet(doc, session_id)
            snippets.append(snippet)
        except Exception as e:
            logger.warning(f"Failed to get snippet for {doc.document_id}: {e}")

    if not snippets:
        return None

    # Generate different context based on search count this turn
    if rag_search_count == 0:
        # First call - guide agent to search
        return _generate_search_guidance(snippets, session_id)
    elif rag_search_count == 1:
        # One search done - remind to answer, but allow follow-up search if needed
        return _generate_answer_reminder(snippets, session_id, rag_search_count)
    else:
        # Multiple searches done - strongly encourage answering
        return _generate_answer_strongly(session_id, rag_search_count)


def _generate_search_guidance(snippets: List[DocumentSnippet], session_id: str) -> str:
    """Generate initial search guidance for first call."""
    context_lines = [
        "## Knowledge Base Available",
        "",
        "The user has uploaded the following documents that you can search:",
        "",
    ]

    for i, snippet in enumerate(snippets, 1):
        doc_line = f"**{i}. {snippet.filename}**"
        if rag_config.include_file_type:
            doc_line += f" ({snippet.file_type.upper()})"
        if rag_config.include_upload_date and snippet.upload_date:
            doc_line += f" - uploaded {snippet.upload_date.strftime('%Y-%m-%d')}"

        context_lines.append(doc_line)

        # Add snippet preview
        if snippet.snippet:
            preview = (
                snippet.snippet[:300] + "..." if len(snippet.snippet) > 300 else snippet.snippet
            )
            context_lines.append(f"   Preview: {preview}")

        context_lines.append(
            f"   (Document ID: `{snippet.document_id}`, {snippet.chunks_count} chunks)"
        )
        context_lines.append("")

    # Add usage guidance
    context_lines.extend(
        [
            "---",
            "",
            "## How to Use the Knowledge Base",
            "",
            "When the user asks questions that could be answered from these documents:",
            "",
            "1. **Search first** using the knowledge search tool:",
            "```python",
            f'results = await search(query="<your search query>", session_id="{session_id}")',
            "print(results)",
            "```",
            "",
            "2. **After receiving search results**, provide a helpful answer based on what you found.",
            "   - Include relevant citations from the search results",
            "   - If the search didn't find relevant information, say so",
            "",
            f"Session ID: `{session_id}`",
        ]
    )

    return "\n".join(context_lines)


def _generate_answer_reminder(
    snippets: List[DocumentSnippet], session_id: str, search_count: int
) -> str:
    """Generate reminder to answer after one search, but allow follow-up if needed."""
    context_lines = [
        "## Knowledge Base Search Completed",
        "",
        f"You have already searched the knowledge base {search_count} time(s) for this question.",
        "The search results are in the conversation above.",
        "",
        "**Now you should:**",
        "1. **Provide your answer** based on the search results you received",
        "2. Include citations like [1], [2] referring to the sources",
        "3. If the information was not found, say so clearly",
        "",
        "**Only search again if:**",
        "- You need to search for a DIFFERENT topic not covered by your first search",
        "- The user explicitly asks for more information",
        "",
        f"Session ID: `{session_id}`",
    ]

    return "\n".join(context_lines)


def _generate_answer_strongly(session_id: str, search_count: int) -> str:
    """Generate strong encouragement to answer after multiple searches."""
    context_lines = [
        "## IMPORTANT: Time to Answer",
        "",
        f"You have searched the knowledge base {search_count} times for this question.",
        "",
        "**STOP searching and provide your answer now.**",
        "",
        "Based on ALL the search results you've received:",
        "1. Synthesize the information into a clear answer",
        "2. Include citations [1], [2], etc. for your sources",
        "3. If you couldn't find the answer after multiple searches, tell the user",
        "",
        "Do NOT search again unless the user asks a completely new question.",
    ]

    return "\n".join(context_lines)


def format_documents_for_prompt(documents: List[Any], session_id: str) -> str:
    """Format document list for simple prompt injection (no async).

    This is a simpler, synchronous version for cases where async isn't available.

    Args:
        documents: List of document objects
        session_id: Session ID

    Returns:
        Formatted document context string
    """
    if not documents:
        return ""

    lines = [
        "## Available Knowledge Base Documents",
        "",
        "The user has uploaded these documents. Use `search(query, session_id)` to search them:",
        "",
    ]

    for i, doc in enumerate(documents[: rag_config.max_documents_in_context], 1):
        filename = doc.filename if hasattr(doc, "filename") else str(doc)
        file_type = doc.file_type if hasattr(doc, "file_type") else "unknown"
        chunks = doc.chunks_count if hasattr(doc, "chunks_count") else 0
        doc_id = doc.document_id if hasattr(doc, "document_id") else "unknown"

        lines.append(f"{i}. **{filename}** ({file_type.upper()}) - {chunks} chunks [ID: {doc_id}]")

    lines.extend(
        [
            "",
            f"Session ID for searches: `{session_id}`",
        ]
    )

    return "\n".join(lines)


def invalidate_document_cache(document_id: str, session_id: str):
    """Invalidate cached snippet for a document.

    Call this when a document is updated or deleted.
    """
    cache = get_snippet_cache()
    cache.invalidate(document_id, session_id)


def clear_session_cache(session_id: str):
    """Clear all cached snippets for a session.

    Call this when clearing a session's documents.
    """
    cache = get_snippet_cache()
    # Remove all entries for this session
    keys_to_remove = [k for k in cache.cache.keys() if k.startswith(f"{session_id}:")]
    for key in keys_to_remove:
        del cache.cache[key]
