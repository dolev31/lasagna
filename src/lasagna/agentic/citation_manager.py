"""Citation Manager for RAG System.

This module provides a robust citation system with:
- Collision-resistant citation IDs with session scoping
- Thread-safe citation tracking with session isolation
- Citation verification and validation pipeline
- Score normalization for consistent confidence display
- Database lifecycle management with automatic cleanup
- Comprehensive metrics and observability
"""

import hashlib
import re
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Set
from loguru import logger

from ..core.config import rag_config
from ..utils.normalization import (
    normalize_score_sigmoid,
    normalize_scores_minmax,
)

# Length of hex hash prefix used in citation IDs (12 chars = 281 trillion namespace)
_HASH_PREFIX_LENGTH = 12


# =============================================================================
# HIGHLIGHT EXTRACTION UTILITIES
# =============================================================================

# Common English stopwords to filter out from keyword matching
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "he",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "the",
        "this",
        "but",
        "they",
        "have",
        "had",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "how",
        "all",
        "each",
        "every",
        "both",
        "few",
        "more",
        "most",
        "other",
        "some",
        "such",
        "than",
        "too",
        "very",
        "can",
        "just",
        "should",
        "now",
        "also",
        "into",
        "only",
        "over",
        "after",
        "before",
        "between",
        "under",
        "again",
        "then",
        "once",
        "here",
        "there",
        "any",
        "about",
        "up",
        "down",
        "out",
        "if",
        "or",
        "because",
        "until",
        "while",
        "so",
        "do",
        "does",
        "did",
        "your",
        "you",
        "i",
        "me",
        "my",
        "we",
        "our",
        "their",
        "them",
        "his",
        "her",
        "she",
        "him",
        "us",
        "been",
        "being",
        "would",
        "could",
        "might",
    ]
)


def extract_keywords(query: str, min_length: int = 3) -> List[str]:
    """Extract meaningful keywords from a query.

    Filters out stopwords and short words to get the most relevant terms.

    Args:
        query: The search query
        min_length: Minimum keyword length to include

    Returns:
        List of lowercase keywords
    """
    # Split on non-alphanumeric characters
    words = re.split(r"[^\w]+", query.lower())

    # Filter stopwords and short words
    keywords = [w for w in words if len(w) >= min_length and w not in _STOPWORDS]

    return keywords


def find_keyword_spans(
    content: str, keywords: List[str], max_highlights: int = 5, case_insensitive: bool = True
) -> List[Tuple[int, int]]:
    """Find character spans where keywords appear in content.

    Args:
        content: The text content to search in
        keywords: List of keywords to find
        max_highlights: Maximum number of highlight spans to return
        case_insensitive: Whether to match case-insensitively

    Returns:
        List of (start, end) character position tuples, sorted by position
    """
    if not keywords:
        return []

    spans = []
    search_content = content.lower() if case_insensitive else content

    # Build single regex pattern with alternation for all keywords (optimization)
    # This avoids compiling a new pattern for each keyword
    escaped_keywords = [re.escape(kw.lower() if case_insensitive else kw) for kw in keywords]
    pattern = re.compile(r"\b(" + "|".join(escaped_keywords) + r")\b")

    # Find all matches with single pass
    for match in pattern.finditer(search_content):
        spans.append((match.start(), match.end()))

        # Early exit if we have enough highlights
        if len(spans) >= max_highlights * 2:
            break

    if not spans:
        return []

    # Sort by position (already sorted from single regex, but ensure consistency)
    spans.sort(key=lambda x: x[0])

    # Merge overlapping spans
    merged = []
    current_start, current_end = spans[0]

    for start, end in spans[1:]:
        if start <= current_end + 1:  # Merge if adjacent or overlapping
            current_end = max(current_end, end)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = start, end
    merged.append((current_start, current_end))

    # Return up to max_highlights
    return merged[:max_highlights]


def extract_highlight_ranges(
    content: str, query: str, max_highlights: int = 5, min_keyword_length: int = 3
) -> List[Tuple[int, int]]:
    """Extract highlight ranges from content based on query keywords.

    This is the main entry point for backend highlight extraction.
    It extracts keywords from the query and finds where they appear
    in the content.

    Args:
        content: The citation content text
        query: The user's search query
        max_highlights: Maximum number of highlight spans
        min_keyword_length: Minimum keyword length to consider

    Returns:
        List of (start, end) tuples for highlighting
    """
    if not content or not query:
        return []

    # Extract keywords from query
    keywords = extract_keywords(query, min_keyword_length)

    if not keywords:
        return []

    # Find spans in content
    return find_keyword_spans(content, keywords, max_highlights)


# =============================================================================
# CITATION DATA CLASS
# =============================================================================


@dataclass
class Citation:
    """A citation reference with stable ID and normalized score."""

    citation_id: str  # Unique ID: {session_prefix}_{content_hash}_{sequence}
    index: int  # Display index (1, 2, 3...)
    chunk_id: str
    document_id: str
    document_name: str
    content: str
    content_preview: str
    score: float  # Raw score (may be negative for cross-encoder)
    normalized_score: float  # Normalized to [0, 1] for display
    chunk_index: int
    page_number: Optional[int] = None
    section_header: Optional[str] = None
    stable_id: Optional[str] = None  # Content-based hash for deduplication
    metadata: Dict[str, Any] = field(default_factory=dict)
    query_id: Optional[str] = None  # Links citation to specific query
    hop_index: Optional[int] = None  # Which hop this citation came from
    highlight_ranges: Optional[List[Tuple[int, int]]] = None  # Character spans to highlight
    highlights: Optional[List[str]] = None  # Backend BM25 highlight fragments

    @staticmethod
    def generate_stable_id(content: str, length: int = _HASH_PREFIX_LENGTH) -> str:
        """Generate a stable content-based ID for deduplication.

        Args:
            content: Content to hash
            length: Length of hash (longer = fewer collisions)

        Returns:
            Stable content hash (e.g., "a1b2c3d4e5f6")
        """
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return content_hash[:length]

    @staticmethod
    def generate_citation_id(
        session_id: str, content: str, sequence: int, query_id: Optional[str] = None
    ) -> str:
        """Generate a collision-resistant citation ID.

        Format: {session_prefix}_{query_prefix}_{content_hash}_{sequence}
        This ensures uniqueness even across sessions with identical content.

        Args:
            session_id: Session identifier
            content: Content to hash
            sequence: Sequence number within the query
            query_id: Optional query identifier for additional scoping

        Returns:
            Unique citation ID (e.g., "s3f2_q1a2_c5d6_001")
        """
        session_prefix = hashlib.sha256(session_id.encode()).hexdigest()[:_HASH_PREFIX_LENGTH]
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:_HASH_PREFIX_LENGTH]

        if query_id:
            query_prefix = hashlib.sha256(query_id.encode()).hexdigest()[:_HASH_PREFIX_LENGTH]
            return f"{session_prefix}_{query_prefix}_{content_hash}_{sequence:03d}"

        return f"{session_prefix}_{content_hash}_{sequence:03d}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        result = {
            "citation_id": self.citation_id,
            "index": self.index,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "content_preview": self.content_preview,
            # The score the search reported, as REST and the SDK return it.
            # This was the min-max over the citations of one search, so the
            # first always read 1.0 and the last 0.0, whatever their relevance.
            "score": round(self.score, 4),
            "chunk_index": self.chunk_index,
            "page_number": self.page_number,
            "section_header": self.section_header,
            "hop_index": self.hop_index,
        }
        if self.metadata.get("line_start"):
            result["line_start"] = self.metadata.get("line_start")
            result["line_end"] = self.metadata.get("line_end")
        # Include highlight ranges for frontend text highlighting
        if self.highlight_ranges:
            result["highlight_ranges"] = self.highlight_ranges
        if self.highlights:
            result["highlights"] = self.highlights
        return result


# =============================================================================
# CITATION DATABASE WITH LIFECYCLE MANAGEMENT
# =============================================================================


class CitationDatabase:
    """SQLite database for tracking citations with lifecycle management.

    This class now delegates to the unified RAGDatabase for all operations,
    maintaining backward compatibility with existing code while using
    the consolidated rag.db database.

    Features:
    - Automatic cleanup of old citations
    - Session-scoped citation limits
    - Efficient indexing for fast lookups
    - Unified database with document metadata
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        max_age_hours: Optional[int] = None,
        max_per_session: Optional[int] = None,
        cleanup_on_init: bool = True,
    ):
        """Initialize citation database.

        Args:
            db_path: Path to SQLite database (ignored - uses unified rag.db)
            max_age_hours: Maximum age of citations before cleanup
            max_per_session: Maximum citations per session
            cleanup_on_init: Whether to run cleanup on initialization
        """
        # Import here to avoid circular imports
        from ..core.database import get_rag_database

        self._rag_db = get_rag_database()
        self.db_path = getattr(self._rag_db, "db_path", None) or getattr(
            self._rag_db, "_dsn", "unknown"
        )
        # Defaults come from citation_db_max_age_hours / citation_db_max_per_session,
        # which were documented and read by nothing.
        from ..core.config import rag_config

        self.max_age_hours = (
            max_age_hours
            if max_age_hours is not None
            else int(getattr(rag_config, "citation_db_max_age_hours", 168))
        )
        self.max_per_session = (
            max_per_session
            if max_per_session is not None
            else int(getattr(rag_config, "citation_db_max_per_session", 1000))
        )
        self._lock = threading.Lock()
        self._last_cleanup = 0
        self._cleanup_interval = 3600  # 1 hour between cleanups

        if cleanup_on_init:
            self._maybe_cleanup()

        logger.debug(f"Citation database initialized (unified): {self.db_path}")

    def _maybe_cleanup(self):
        """Run cleanup if enough time has passed since last cleanup."""
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return

        self._last_cleanup = now
        self.cleanup_expired()

    def store_citations(
        self,
        citations: List[Citation],
        session_id: str,
        query_id: Optional[str] = None,
        ttl_hours: Optional[int] = None,
    ) -> int:
        """Store citations in database.

        Args:
            citations: List of citations to store
            session_id: Session ID
            query_id: Optional query ID for grouping
            ttl_hours: Time-to-live in hours (default: max_age_hours)

        Returns:
            Number of citations stored
        """
        if not rag_config.track_citations_db or not citations:
            return 0

        ttl = ttl_hours or self.max_age_hours

        # Convert Citation objects to dicts for RAGDatabase
        citation_dicts = []
        for citation in citations:
            content_hash = Citation.generate_stable_id(citation.content)
            citation_dicts.append(
                {
                    "citation_id": citation.citation_id,
                    "query_id": query_id or citation.query_id,
                    "chunk_id": citation.chunk_id,
                    "document_id": citation.document_id,
                    "document_name": citation.document_name,
                    "content_preview": citation.content_preview,
                    "content_hash": content_hash,
                    "chunk_index": citation.chunk_index,
                    "page_number": citation.page_number,
                    "section_header": citation.section_header,
                    "score": citation.score,
                    "normalized_score": citation.normalized_score,
                    "hop_index": citation.hop_index,
                }
            )

        stored_count = self._rag_db.store_citations(citation_dicts, session_id, query_id, ttl)

        # Enforce session limits
        self._enforce_session_limit(session_id)

        logger.debug(f"Stored {stored_count} citations for session {session_id}")
        return stored_count

    def get_citation(self, citation_id: str) -> Optional[Dict[str, Any]]:
        """Look up a citation by ID.

        Args:
            citation_id: Citation ID to look up

        Returns:
            Citation data or None if not found
        """
        return self._rag_db.get_citation(citation_id)

    def get_citation_by_content_hash(
        self, content_hash: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        """Look up a citation by content hash within a session.

        Useful for deduplication.

        Args:
            content_hash: Content hash to look up
            session_id: Session to search in

        Returns:
            Citation data or None if not found
        """
        return self._rag_db.get_citation_by_content_hash(content_hash, session_id)

    def get_session_citations(
        self, session_id: str, query_id: Optional[str] = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get citations for a session, optionally filtered by query.

        Args:
            session_id: Session ID
            query_id: Optional query ID to filter by
            limit: Maximum number of citations to return

        Returns:
            List of citation dictionaries
        """
        return self._rag_db.get_session_citations(session_id, query_id, limit)

    def clear_session(self, session_id: str) -> int:
        """Clear citations for a session.

        Args:
            session_id: Session ID to clear

        Returns:
            Number of citations deleted
        """
        deleted = self._rag_db.clear_session_citations(session_id)
        logger.info(f"Cleared {deleted} citations for session {session_id}")
        return deleted

    def cleanup_expired(self) -> int:
        """Remove expired citations.

        Returns:
            Number of citations deleted
        """
        return self._rag_db.cleanup_expired_citations()

    def cleanup_old_citations(self, max_age_hours: Optional[int] = None) -> int:
        """Remove citations older than max_age_hours.

        Args:
            max_age_hours: Maximum age in hours (default: self.max_age_hours)

        Returns:
            Number of citations deleted
        """
        age = max_age_hours or self.max_age_hours
        deleted = self._rag_db.cleanup_citations_by_age(age)
        if deleted > 0:
            logger.info(f"Cleaned up {deleted} old citations (age > {age}h)")
        return deleted

    def _enforce_session_limit(self, session_id: str):
        """Keep only most recent N citations per session.

        Args:
            session_id: Session ID to enforce limit on
        """
        deleted = self._rag_db.enforce_citation_limit(session_id, self.max_per_session)
        if deleted > 0:
            logger.debug(f"Enforced session limit: deleted {deleted} excess citations")

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring.

        Returns:
            Dictionary with database stats
        """
        stats = self._rag_db.get_stats()
        citation_stats = stats.get("citations", {})

        # Calculate additional metrics
        agg = self._rag_db.get_citation_aggregate_stats()
        avg_score = agg.get("avg_score", 0.0)
        missing_page = agg.get("missing_page", 0)
        total = citation_stats.get("total", 0)

        return {
            "total_citations": total,
            "active_sessions": citation_stats.get("sessions", 0),
            "expired_pending_cleanup": citation_stats.get("expired_pending_cleanup", 0),
            "average_score": round(avg_score, 3),
            "missing_page_count": missing_page,
            "missing_page_rate": round(missing_page / total, 3) if total > 0 else 0.0,
            "db_path": self.db_path,
        }


# =============================================================================
# CITATION VERIFICATION
# =============================================================================


@dataclass
class CitationVerificationResult:
    """Result of citation verification."""

    citation_ref: str
    is_valid: bool
    citation: Optional[Citation] = None
    error: Optional[str] = None
    corrected_ref: Optional[str] = None  # If we can suggest a fix


class CitationVerifier:
    """Verifies and optionally fixes citation references in text."""

    # Pattern matches [1], [2], etc. - the standard numeric format
    CITATION_PATTERN = re.compile(r"\[(\d+)\]")

    # Pattern also matches hash-based citations for backward compatibility
    HASH_CITATION_PATTERN = re.compile(r"\[([a-f0-9]{6,16})\]", re.IGNORECASE)

    def __init__(self, citations: List[Citation]):
        """Initialize verifier with available citations.

        Args:
            citations: List of valid citations for this context
        """
        self.citations = citations
        self.max_index = len(citations)
        self._index_map = {c.index: c for c in citations}
        self._id_map = {c.citation_id: c for c in citations}
        self._stable_id_map = {c.stable_id: c for c in citations if c.stable_id}

    def verify_reference(self, ref: str) -> CitationVerificationResult:
        """Verify a single citation reference.

        Args:
            ref: Citation reference (e.g., "1", "2", or hash)

        Returns:
            Verification result
        """
        # Try numeric index first
        try:
            idx = int(ref)
            if 1 <= idx <= self.max_index:
                return CitationVerificationResult(
                    citation_ref=ref, is_valid=True, citation=self._index_map.get(idx)
                )
            else:
                # Out of range - suggest closest valid
                closest = min(max(1, idx), self.max_index)
                return CitationVerificationResult(
                    citation_ref=ref,
                    is_valid=False,
                    error=f"Index {idx} out of range (max: {self.max_index})",
                    corrected_ref=str(closest) if self.max_index > 0 else None,
                )
        except ValueError:
            pass

        # Try citation_id
        if ref in self._id_map:
            return CitationVerificationResult(
                citation_ref=ref, is_valid=True, citation=self._id_map[ref]
            )

        # Try stable_id
        if ref in self._stable_id_map:
            citation = self._stable_id_map[ref]
            return CitationVerificationResult(
                citation_ref=ref,
                is_valid=True,
                citation=citation,
                corrected_ref=str(citation.index),  # Suggest numeric format
            )

        return CitationVerificationResult(
            citation_ref=ref, is_valid=False, error="Citation not found"
        )

    def verify_text(self, text: str) -> Dict[str, Any]:
        """Verify all citations in a text.

        Args:
            text: Text containing citation markers

        Returns:
            Verification report
        """
        results = []
        valid_count = 0
        invalid_count = 0

        # Find all numeric citations
        for match in self.CITATION_PATTERN.finditer(text):
            ref = match.group(1)
            result = self.verify_reference(ref)
            results.append(result)
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1

        # Find hash-based citations
        for match in self.HASH_CITATION_PATTERN.finditer(text):
            ref = match.group(1)
            # Skip if it looks like it was already matched as numeric
            if ref.isdigit():
                continue
            result = self.verify_reference(ref)
            results.append(result)
            if result.is_valid:
                valid_count += 1
            else:
                invalid_count += 1

        return {
            "total_citations": len(results),
            "valid_count": valid_count,
            "invalid_count": invalid_count,
            "all_valid": invalid_count == 0,
            "results": results,
            "max_valid_index": self.max_index,
        }

    def fix_text(
        self, text: str, remove_invalid: bool = True, mark_invalid: bool = False
    ) -> Tuple[str, List[CitationVerificationResult]]:
        """Fix invalid citations in text.

        Args:
            text: Text with citation markers
            remove_invalid: Remove invalid citations entirely
            mark_invalid: Mark invalid citations with [?] instead of removing

        Returns:
            Tuple of (fixed_text, list of changes made)
        """
        fixed_text = text
        changes = []

        # Process in reverse order to maintain positions
        matches = list(self.CITATION_PATTERN.finditer(text))
        for match in reversed(matches):
            ref = match.group(1)
            result = self.verify_reference(ref)

            if not result.is_valid:
                changes.append(result)
                if result.corrected_ref:
                    # Replace with corrected reference
                    fixed_text = (
                        fixed_text[: match.start()]
                        + f"[{result.corrected_ref}]"
                        + fixed_text[match.end() :]
                    )
                elif mark_invalid:
                    fixed_text = (
                        fixed_text[: match.start()] + f"[{ref}?]" + fixed_text[match.end() :]
                    )
                elif remove_invalid:
                    fixed_text = fixed_text[: match.start()] + fixed_text[match.end() :]

        return fixed_text, changes


# =============================================================================
# THREAD-SAFE CITATION MANAGER
# =============================================================================


class CitationManager:
    """Thread-safe citation manager with session isolation.

    Provides:
    - Session-scoped citation tracking
    - Score normalization
    - Citation verification
    - Deduplication via content hashing
    - Comprehensive metrics
    """

    def __init__(self):
        """Initialize citation manager."""
        self.db = CitationDatabase() if rag_config.track_citations_db else None
        self._lock = threading.RLock()
        self._session_citations: Dict[str, Dict[str, Citation]] = {}
        self._session_sequences: Dict[str, int] = {}
        self._metrics = CitationMetrics()

    @contextmanager
    def session_scope(self, session_id: str):
        """Context manager for session-isolated citation operations.

        Args:
            session_id: Session identifier

        Yields:
            Session-specific citation cache
        """
        with self._lock:
            if session_id not in self._session_citations:
                self._session_citations[session_id] = {}
                self._session_sequences[session_id] = 0

        try:
            yield self._session_citations[session_id]
        finally:
            pass  # Could add cleanup logic

    def _get_next_sequence(self, session_id: str) -> int:
        """Get next sequence number for a session.

        Args:
            session_id: Session identifier

        Returns:
            Next sequence number
        """
        with self._lock:
            if session_id not in self._session_sequences:
                self._session_sequences[session_id] = 0
            self._session_sequences[session_id] += 1
            return self._session_sequences[session_id]

    def reset_session_sequence(self, session_id: str):
        """Reset sequence counter for a new query in the same session.

        Args:
            session_id: Session identifier
        """
        with self._lock:
            self._session_sequences[session_id] = 0
            if session_id in self._session_citations:
                self._session_citations[session_id].clear()

    def create_citations(
        self,
        search_results: List[Any],
        session_id: str,
        query_id: Optional[str] = None,
        start_index: int = 1,
        hop_index: Optional[int] = None,
        normalize_scores: bool = True,
        deduplicate: bool = True,
        query: Optional[str] = None,
    ) -> List[Citation]:
        """Create citations from search results with score normalization.

        Args:
            search_results: List of search results (various formats supported)
            session_id: Session ID
            query_id: Optional query ID for grouping
            start_index: Starting index for citation numbering (for global indexing)
            hop_index: Which retrieval hop these citations came from
            normalize_scores: Whether to normalize scores to [0, 1]
            deduplicate: Whether to deduplicate by content hash
            query: Optional search query for highlight extraction

        Returns:
            List of Citation objects
        """
        if not search_results:
            return []

        # Generate query_id if not provided
        if not query_id:
            query_id = f"q_{uuid.uuid4().hex[:8]}"

        citations = []
        seen_content_hashes: Set[str] = set()

        # Extract all scores first for batch normalization
        raw_scores = []
        for result in search_results:
            raw_scores.append(self._extract_score(result))

        # Normalize scores
        if normalize_scores and raw_scores:
            normalized_scores = normalize_scores_minmax(raw_scores)
        else:
            normalized_scores = [normalize_score_sigmoid(s) for s in raw_scores]

        with self.session_scope(session_id) as session_cache:
            current_index = start_index

            for i, result in enumerate(search_results):
                content = self._extract_content(result)
                content_hash = Citation.generate_stable_id(content)

                # Deduplicate
                if deduplicate and content_hash in seen_content_hashes:
                    self._metrics.record_dedup()
                    continue
                seen_content_hashes.add(content_hash)

                # Extract all metadata
                chunk_id = self._extract_chunk_id(result)
                document_id = self._extract_document_id(result)
                document_name = self._extract_document_name(result)
                raw_score = raw_scores[i]
                normalized_score = normalized_scores[i]
                chunk_index = self._extract_chunk_index(result)
                page_number = self._extract_page_number(result)
                section_header = self._extract_section_header(result)
                line_start, line_end = self._extract_line_range(result)

                # Generate unique citation ID
                sequence = self._get_next_sequence(session_id)
                citation_id = Citation.generate_citation_id(session_id, content, sequence, query_id)

                # Extract backend highlights (from OpenSearch BM25 etc.)
                backend_highlights = None
                if isinstance(result, dict):
                    backend_highlights = result.get("highlights") or (
                        result.get("entity", {}) or {}
                    ).get("highlights")
                elif hasattr(result, "highlights"):
                    backend_highlights = result.highlights

                # Extract highlight ranges if query is provided
                highlight_ranges = None
                if query and content:
                    highlight_ranges = extract_highlight_ranges(
                        content, query, max_highlights=5, min_keyword_length=3
                    )

                citation = Citation(
                    citation_id=citation_id,
                    index=current_index,
                    chunk_id=chunk_id,
                    document_id=document_id,
                    document_name=document_name,
                    content=content,
                    content_preview=content[:200] if content else "",
                    score=raw_score,
                    normalized_score=normalized_score,
                    chunk_index=chunk_index,
                    page_number=page_number,
                    section_header=section_header,
                    stable_id=content_hash,
                    metadata={
                        "line_start": line_start,
                        "line_end": line_end,
                    },
                    query_id=query_id,
                    hop_index=hop_index,
                    highlight_ranges=highlight_ranges if highlight_ranges else None,
                    highlights=backend_highlights if backend_highlights else None,
                )

                citations.append(citation)
                session_cache[citation_id] = citation
                session_cache[str(current_index)] = citation
                current_index += 1

                # Track metrics
                self._metrics.record_citation(
                    has_page=page_number is not None, score=normalized_score
                )

        # Store in database
        if self.db:
            self.db.store_citations(citations, session_id, query_id)

        logger.debug(
            f"Created {len(citations)} citations for session {session_id} "
            f"(start_index={start_index}, hop={hop_index})"
        )
        return citations

    def merge_citations(
        self,
        existing_citations: List[Citation],
        new_citations: List[Citation],
        session_id: str,
        deduplicate: bool = True,
    ) -> List[Citation]:
        """Merge new citations with existing ones, maintaining index continuity.

        Args:
            existing_citations: Already created citations
            new_citations: New citations to add
            session_id: Session ID
            deduplicate: Whether to skip duplicates

        Returns:
            Merged list of citations with updated indices
        """
        if not new_citations:
            return existing_citations

        merged = list(existing_citations)
        seen_hashes = {c.stable_id for c in existing_citations if c.stable_id}
        next_index = len(existing_citations) + 1

        with self.session_scope(session_id) as session_cache:
            for citation in new_citations:
                # Check for duplicates
                if deduplicate and citation.stable_id in seen_hashes:
                    self._metrics.record_dedup()
                    continue

                # Update index
                citation.index = next_index
                next_index += 1

                # Add to caches
                merged.append(citation)
                seen_hashes.add(citation.stable_id)
                session_cache[citation.citation_id] = citation
                session_cache[str(citation.index)] = citation

        return merged

    def get_citation(
        self, citation_ref: str, session_id: Optional[str] = None
    ) -> Optional[Citation]:
        """Get a citation by reference (index or ID).

        Args:
            citation_ref: Citation reference (number or ID)
            session_id: Optional session to search in first

        Returns:
            Citation if found, None otherwise
        """
        # Check session cache first
        if session_id and session_id in self._session_citations:
            session_cache = self._session_citations[session_id]
            if citation_ref in session_cache:
                return session_cache[citation_ref]

        # Check all sessions
        with self._lock:
            for cache in self._session_citations.values():
                if citation_ref in cache:
                    return cache[citation_ref]

        # Check database
        if self.db and rag_config.verify_citations:
            db_citation = self.db.get_citation(citation_ref)
            if db_citation:
                return Citation(
                    citation_id=db_citation["citation_id"],
                    index=0,
                    chunk_id=db_citation.get("chunk_id", 0),
                    document_id=db_citation["document_id"],
                    document_name=db_citation.get("document_name", ""),
                    content="",
                    content_preview=db_citation.get("content_preview", ""),
                    score=db_citation.get("score", 0.0),
                    normalized_score=db_citation.get("normalized_score", 0.5),
                    chunk_index=db_citation.get("chunk_index", 0),
                    page_number=db_citation.get("page_number"),
                    hop_index=db_citation.get("hop_index"),
                )

        return None

    def create_verifier(self, session_id: str) -> CitationVerifier:
        """Create a verifier for the current session's citations.

        Args:
            session_id: Session identifier

        Returns:
            CitationVerifier instance
        """
        citations = []
        if session_id in self._session_citations:
            # Get citations sorted by index
            seen_indices = set()
            for citation in self._session_citations[session_id].values():
                if citation.index not in seen_indices:
                    citations.append(citation)
                    seen_indices.add(citation.index)
            citations.sort(key=lambda c: c.index)

        return CitationVerifier(citations)

    def verify_and_fix_response(
        self,
        response_text: str,
        session_id: str,
        remove_invalid: bool = True,
    ) -> Tuple[str, Dict[str, Any]]:
        """Verify citations in response and optionally fix invalid ones.

        Args:
            response_text: LLM response with citation markers
            session_id: Session identifier
            remove_invalid: Whether to remove invalid citations

        Returns:
            Tuple of (fixed_text, verification_report)
        """
        verifier = self.create_verifier(session_id)
        verification = verifier.verify_text(response_text)

        if verification["invalid_count"] > 0:
            fixed_text, changes = verifier.fix_text(response_text, remove_invalid=remove_invalid)
            verification["fixed_text"] = fixed_text
            verification["changes_made"] = len(changes)
            self._metrics.record_verification(
                valid=verification["valid_count"], invalid=verification["invalid_count"]
            )
            return fixed_text, verification

        self._metrics.record_verification(valid=verification["valid_count"], invalid=0)
        return response_text, verification

    def format_context_with_citations(
        self,
        citations: List[Citation],
        include_content: bool = True,
        include_instructions: bool = True,
    ) -> str:
        """Format context for LLM with citation instructions.

        Args:
            citations: List of citations
            include_content: Whether to include full content
            include_instructions: Whether to include citation instructions

        Returns:
            Formatted context string
        """
        if not citations:
            return ""

        # Build source reference list
        sources_list = []
        for c in citations:
            source_line = f"[{c.index}] {c.document_name}"
            if c.page_number:
                source_line += f" (page {c.page_number})"
            elif c.metadata.get("line_start"):
                line_start = c.metadata.get("line_start")
                line_end = c.metadata.get("line_end")
                if line_end and line_end != line_start:
                    source_line += f" (lines {line_start}-{line_end})"
                else:
                    source_line += f" (line {line_start})"
            if c.section_header:
                source_line += f" - {c.section_header}"
            sources_list.append(source_line)

        sources = "\n".join(sources_list)

        # Build content sections
        content_sections = []
        if include_content:
            for c in citations:
                section = f"[Source {c.index}]\n{c.content}"
                content_sections.append(section)

        content = "\n\n".join(content_sections)

        if include_instructions:
            return f"""<knowledge_base>
The following information was retrieved from the user's uploaded documents.
Use this information to help answer the user's question.

IMPORTANT CITATION INSTRUCTIONS:
- Cite sources INLINE immediately after each fact or claim you reference
- Use citation markers like [1], [2], etc. right after the relevant sentence
- Do NOT group all citations at the end - place each citation where you use that information
- Example: "The cost was $1,370 [1]. The workshops included three sessions [2]."
- Only use citation numbers that exist (max: [{len(citations)}])

SOURCES:
{sources}

RETRIEVED CONTENT:
{content}
</knowledge_base>"""
        else:
            return f"""<knowledge_base>
SOURCES:
{sources}

RETRIEVED CONTENT:
{content}
</knowledge_base>"""

    def get_metrics(self) -> Dict[str, Any]:
        """Get citation manager metrics.

        Returns:
            Dictionary of metrics
        """
        metrics = self._metrics.to_dict()
        if self.db:
            metrics["database"] = self.db.get_stats()
        return metrics

    def clear_session(self, session_id: str):
        """Clear all citations for a session.

        Args:
            session_id: Session to clear
        """
        with self._lock:
            if session_id in self._session_citations:
                del self._session_citations[session_id]
            if session_id in self._session_sequences:
                del self._session_sequences[session_id]

        if self.db:
            self.db.clear_session(session_id)

    # =========================================================================
    # Helper methods for extracting data from various result formats
    # =========================================================================

    def _extract_content(self, result: Any) -> str:
        if isinstance(result, dict):
            if "content" in result:
                return result["content"]
            if "entity" in result and "content" in result["entity"]:
                return result["entity"]["content"]
        if hasattr(result, "content"):
            return result.content
        if hasattr(result, "chunk") and hasattr(result.chunk, "content"):
            return result.chunk.content
        return str(result)

    def _extract_chunk_id(self, result: Any) -> str:
        if isinstance(result, dict):
            val = result.get("id", result.get("chunk_id", ""))
            return str(val)
        if hasattr(result, "chunk_id"):
            return str(result.chunk_id)
        if hasattr(result, "chunk") and hasattr(result.chunk, "chunk_id"):
            return str(result.chunk.chunk_id)
        return ""

    def _extract_document_id(self, result: Any) -> str:
        if isinstance(result, dict):
            if "document_id" in result:
                return result["document_id"]
            if "entity" in result:
                return result["entity"].get("document_id", "")
        if hasattr(result, "document_id"):
            return result.document_id
        if hasattr(result, "chunk") and hasattr(result.chunk, "document_id"):
            return result.chunk.document_id
        if hasattr(result, "document") and hasattr(result.document, "document_id"):
            return result.document.document_id
        return ""

    def _extract_document_name(self, result: Any) -> str:
        if isinstance(result, dict):
            if "document_name" in result:
                return result["document_name"]
            if "filename" in result:
                return result["filename"]
        if hasattr(result, "document") and hasattr(result.document, "filename"):
            return result.document.filename
        return ""

    def _extract_score(self, result: Any) -> float:
        if isinstance(result, dict):
            return float(result.get("score", result.get("distance", 0.0)))
        if hasattr(result, "score"):
            return float(result.score)
        if hasattr(result, "rerank_score"):
            return float(result.rerank_score)
        return 0.0

    def _extract_chunk_index(self, result: Any) -> int:
        if isinstance(result, dict):
            if "chunk_index" in result:
                return result["chunk_index"]
            if "metadata" in result:
                return result["metadata"].get("chunk_index", 0)
        if hasattr(result, "chunk") and hasattr(result.chunk, "chunk_index"):
            return result.chunk.chunk_index
        return 0

    def _extract_page_number(self, result: Any) -> Optional[int]:
        """Extract page number with multiple fallback strategies.

        Tries in order:
        1. Direct page_number attribute/key
        2. Metadata fields (page_number, page, page_num, pageNumber)
        3. Content patterns (e.g., "Page 42", "[p. 5]")
        4. Document name patterns (e.g., "doc_page_5.pdf")
        """
        # Common metadata keys for page number
        PAGE_KEYS = ["page_number", "page", "page_num", "pageNumber", "pg"]

        def _try_parse_int(value: Any) -> Optional[int]:
            """Safely parse a value to int."""
            if value is None:
                return None
            if isinstance(value, int):
                return value
            if isinstance(value, float):
                return int(value)
            if isinstance(value, str):
                try:
                    return int(value.strip())
                except ValueError:
                    # Try extracting number from string like "page 5"
                    import re

                    match = re.search(r"\d+", value)
                    if match:
                        return int(match.group())
            return None

        # Strategy 1: Dict with direct keys
        if isinstance(result, dict):
            for key in PAGE_KEYS:
                if key in result:
                    page = _try_parse_int(result[key])
                    if page is not None:
                        return page
            # Check nested metadata
            if "metadata" in result and isinstance(result["metadata"], dict):
                for key in PAGE_KEYS:
                    if key in result["metadata"]:
                        page = _try_parse_int(result["metadata"][key])
                        if page is not None:
                            return page
            # Check entity metadata
            if "entity" in result and isinstance(result["entity"], dict):
                entity = result["entity"]
                for key in PAGE_KEYS:
                    if key in entity:
                        page = _try_parse_int(entity[key])
                        if page is not None:
                            return page
                if "metadata" in entity and isinstance(entity["metadata"], dict):
                    for key in PAGE_KEYS:
                        if key in entity["metadata"]:
                            page = _try_parse_int(entity["metadata"][key])
                            if page is not None:
                                return page

        # Strategy 2: Object with chunk attribute
        if hasattr(result, "chunk"):
            chunk = result.chunk
            # Direct attribute
            if hasattr(chunk, "page_number"):
                page = _try_parse_int(chunk.page_number)
                if page is not None:
                    return page
            # Chunk metadata
            if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                for key in PAGE_KEYS:
                    if key in chunk.metadata:
                        page = _try_parse_int(chunk.metadata[key])
                        if page is not None:
                            return page

        # Strategy 3: Direct object metadata
        if hasattr(result, "metadata") and isinstance(result.metadata, dict):
            for key in PAGE_KEYS:
                if key in result.metadata:
                    page = _try_parse_int(result.metadata[key])
                    if page is not None:
                        return page

        # Strategy 4: Try to extract from content (common patterns)
        content = self._extract_content(result)
        if content:
            import re

            # Pattern: "Page N" or "page N" at start of content
            page_pattern = re.match(r"^(?:page|Page|PAGE)\s+(\d+)", content)
            if page_pattern:
                return int(page_pattern.group(1))
            # Pattern: "[p. N]" or "(p. N)"
            ref_pattern = re.search(r"[\[\(]p\.?\s*(\d+)[\]\)]", content[:100])
            if ref_pattern:
                return int(ref_pattern.group(1))

        return None

    def _extract_section_header(self, result: Any) -> Optional[str]:
        if isinstance(result, dict):
            if "section_header" in result:
                return result["section_header"]
            if "metadata" in result:
                return result["metadata"].get("section_header")
        if hasattr(result, "metadata"):
            return result.metadata.get("section_header")
        return None

    def _extract_line_range(self, result: Any) -> Tuple[Optional[int], Optional[int]]:
        if isinstance(result, dict):
            if "metadata" in result:
                return (
                    result["metadata"].get("line_start"),
                    result["metadata"].get("line_end"),
                )
        if hasattr(result, "chunk") and hasattr(result.chunk, "metadata"):
            return (
                result.chunk.metadata.get("line_start"),
                result.chunk.metadata.get("line_end"),
            )
        if hasattr(result, "metadata"):
            return (
                result.metadata.get("line_start"),
                result.metadata.get("line_end"),
            )
        return (None, None)


# =============================================================================
# CITATION METRICS
# =============================================================================


class CitationMetrics:
    """Tracks citation-related metrics for observability."""

    def __init__(self):
        self._lock = threading.Lock()
        self.citations_created = 0
        self.citations_with_page = 0
        self.citations_without_page = 0
        self.verifications_valid = 0
        self.verifications_invalid = 0
        self.deduplications = 0
        self.total_score_sum = 0.0
        self.score_count = 0

    def record_citation(self, has_page: bool, score: float):
        """Record a citation creation."""
        with self._lock:
            self.citations_created += 1
            if has_page:
                self.citations_with_page += 1
            else:
                self.citations_without_page += 1
            self.total_score_sum += score
            self.score_count += 1

    def record_verification(self, valid: int, invalid: int):
        """Record verification results."""
        with self._lock:
            self.verifications_valid += valid
            self.verifications_invalid += invalid

    def record_dedup(self):
        """Record a deduplication event."""
        with self._lock:
            self.deduplications += 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        with self._lock:
            avg_score = self.total_score_sum / self.score_count if self.score_count > 0 else 0.0
            missing_page_rate = (
                self.citations_without_page / self.citations_created
                if self.citations_created > 0
                else 0.0
            )
            verification_accuracy = (
                self.verifications_valid / (self.verifications_valid + self.verifications_invalid)
                if (self.verifications_valid + self.verifications_invalid) > 0
                else 1.0
            )

            return {
                "citations_created": self.citations_created,
                "citations_with_page": self.citations_with_page,
                "citations_without_page": self.citations_without_page,
                "missing_page_rate": round(missing_page_rate, 3),
                "average_score": round(avg_score, 3),
                "verifications_valid": self.verifications_valid,
                "verifications_invalid": self.verifications_invalid,
                "verification_accuracy": round(verification_accuracy, 3),
                "deduplications": self.deduplications,
            }


# =============================================================================
# MODULE-LEVEL SINGLETON
# =============================================================================

_citation_manager: Optional[CitationManager] = None
_citation_manager_lock = threading.Lock()


def get_citation_manager() -> CitationManager:
    """Get or create the citation manager singleton (thread-safe).

    Returns:
        CitationManager instance
    """
    global _citation_manager
    if _citation_manager is None:
        with _citation_manager_lock:
            if _citation_manager is None:
                _citation_manager = CitationManager()
    return _citation_manager


def reset_citation_manager():
    """Reset the citation manager singleton."""
    global _citation_manager
    with _citation_manager_lock:
        _citation_manager = None
