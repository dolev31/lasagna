"""Citation Manager - Citation tracking and verification.

Provides citation retrieval, verification, and management operations.
Wraps the underlying CitationManager.

Example:
    ```python
    # Get citation details
    citation = await client.citations.get("abc123")

    # Verify citation still valid
    is_valid = await client.citations.verify("abc123")

    # List recent citations
    citations = await client.citations.list_recent()
    ```
"""

from typing import List, Optional, TYPE_CHECKING

from loguru import logger

from ..types import Citation
from ..exceptions import (
    CitationError,
    CitationNotFoundError,
    ConfigurationError,
)

if TYPE_CHECKING:
    from ..client import KnowledgeClient


class CitationManager:
    """Manages citation tracking and verification.

    Citations are automatically created during search operations.
    This manager provides access to citation details and verification.

    Example:
        ```python
        # Get citation by ID
        citation = await client.citations.get("abc123")
        if citation:
            print(f"Source: {citation.filename}, Page: {citation.page_number}")

        # Verify citation exists
        valid = await client.citations.verify("abc123")

        # Get all citations for a session
        citations = await client.citations.list_recent()
        ```
    """

    def __init__(self, client: "KnowledgeClient"):
        """Initialize citation manager.

        Args:
            client: Parent KnowledgeClient instance
        """
        self._client = client

    # =========================================================================
    # Citation State
    # =========================================================================

    @property
    def enabled(self) -> bool:
        """Check if citations are enabled in the current configuration.

        Returns:
            True if citations are enabled, False otherwise.

        Example:
            ```python
            if client.citations.enabled:
                citation = await client.citations.get("abc123")
            else:
                print("Citations are disabled")
            ```
        """
        return self._client.config.enable_citations

    def _check_enabled(self, operation: str = "Citation operation") -> None:
        """Check if citations are enabled, raise error if not.

        Args:
            operation: Name of the operation for error message

        Raises:
            ConfigurationError: If citations are disabled
        """
        if not self.enabled:
            raise ConfigurationError(
                f"{operation} failed: Citations are disabled. "
                "Set enable_citations=True in KnowledgeConfig to enable."
            )

    # =========================================================================
    # Citation Retrieval
    # =========================================================================

    async def get(self, citation_id: str) -> Citation:
        """Get citation details by ID.

        Args:
            citation_id: Citation ID to retrieve

        Returns:
            Citation object

        Raises:
            ConfigurationError: If citations are disabled
            CitationNotFoundError: If citation does not exist
            CitationError: If retrieval fails

        Example:
            ```python
            try:
                citation = await client.citations.get("abc123")
                print(f"From: {citation.filename}")
                print(f"Page: {citation.page_number}")
                print(f"Preview: {citation.content_preview}")
            except CitationNotFoundError:
                print("Citation not found")
            except ConfigurationError:
                print("Citations are disabled")
            ```
        """
        self._check_enabled("Get citation")

        try:
            from ...agentic.citation_manager import get_citation_manager

            manager = get_citation_manager()
            raw_citation = manager.get_citation(
                citation_id=citation_id,
                session_id=self._client.session_id,
            )

            if raw_citation is None:
                raise CitationNotFoundError(
                    f"Citation not found: {citation_id}",
                    citation_id=citation_id,
                )

            return Citation(
                citation_id=raw_citation.citation_id,
                chunk_id=str(raw_citation.chunk_id),
                document_id=raw_citation.document_id,
                filename=raw_citation.document_name,
                page_number=raw_citation.page_number,
                content_preview=raw_citation.content_preview,
                score=raw_citation.score,
                hop_index=getattr(raw_citation, "hop_index", 0),
            )

        except CitationNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Could not get citation {citation_id}: {e}")
            raise CitationError(
                f"Could not get citation {citation_id}: {e}",
                citation_id=citation_id,
            ) from e

    async def verify(self, citation_id: str) -> bool:
        """Verify that a citation still exists and is valid.

        Checks that the cited chunk still exists in the database.

        Args:
            citation_id: Citation ID to verify

        Returns:
            True if citation is valid, False if citation not found or invalid

        Raises:
            ConfigurationError: If citations are disabled
            CitationError: If verification operation fails

        Example:
            ```python
            try:
                if await client.citations.verify("abc123"):
                    print("Citation is valid")
                else:
                    print("Citation no longer valid")
            except CitationError as e:
                print(f"Verification failed: {e}")
            ```
        """
        self._check_enabled("Verify citation")

        try:
            from ...agentic.citation_manager import get_citation_manager

            manager = get_citation_manager()
            return manager.verify_citation(
                citation_id=citation_id,
                session_id=self._client.session_id,
            )

        except Exception as e:
            logger.error(f"Could not verify citation {citation_id}: {e}")
            raise CitationError(
                f"Could not verify citation {citation_id}: {e}",
                citation_id=citation_id,
            ) from e

    # =========================================================================
    # Citation Listing
    # =========================================================================

    async def list_recent(
        self,
        limit: int = 100,
        query_id: Optional[str] = None,
    ) -> List[Citation]:
        """List recent citations for the session.

        Args:
            limit: Maximum citations to return
            query_id: Optional filter by query ID

        Returns:
            List of Citation objects (empty if citations disabled)

        Raises:
            CitationError: If listing operation fails

        Example:
            ```python
            citations = await client.citations.list_recent(limit=20)
            for c in citations:
                print(f"[{c.citation_id}] {c.filename}: {c.score:.2f}")
            ```
        """
        # Return empty list if citations disabled (non-throwing behavior for listing)
        if not self.enabled:
            return []

        try:
            from ...agentic.citation_manager import get_citation_manager

            manager = get_citation_manager()
            raw_citations = manager.list_citations(
                session_id=self._client.session_id,
                limit=limit,
                query_id=query_id,
            )

            citations = []
            for raw in raw_citations:
                citations.append(
                    Citation(
                        citation_id=raw.citation_id,
                        chunk_id=str(raw.chunk_id),
                        document_id=raw.document_id,
                        filename=raw.document_name,
                        page_number=raw.page_number,
                        content_preview=raw.content_preview,
                        score=raw.score,
                        hop_index=getattr(raw, "hop_index", 0),
                    )
                )

            return citations

        except Exception as e:
            logger.error(f"Could not list citations: {e}")
            raise CitationError(f"Could not list citations: {e}") from e

    async def list_for_document(self, document_id: str) -> List[Citation]:
        """List all citations for a specific document.

        Args:
            document_id: Document ID to get citations for

        Returns:
            List of Citation objects (empty if citations disabled)

        Example:
            ```python
            citations = await client.citations.list_for_document("doc_abc123")
            print(f"Found {len(citations)} citations for document")
            ```
        """
        # Return empty list if citations disabled
        if not self.enabled:
            return []

        all_citations = await self.list_recent(limit=1000)
        return [c for c in all_citations if c.document_id == document_id]

    # =========================================================================
    # Citation Management
    # =========================================================================

    async def clear(self, confirm: bool = False) -> int:
        """Clear all citations for the session.

        Args:
            confirm: Must be True to proceed (safety check)

        Returns:
            Number of citations cleared (0 if citations disabled)

        Raises:
            ValueError: If confirm is not True

        Example:
            ```python
            # This will fail
            await client.citations.clear()

            # This will work
            count = await client.citations.clear(confirm=True)
            ```
        """
        # Return 0 if citations disabled (nothing to clear)
        if not self.enabled:
            return 0

        if not confirm:
            raise ValueError(
                "clear requires confirm=True. This will delete all citations for the session."
            )

        try:
            from ...agentic.citation_manager import get_citation_manager

            manager = get_citation_manager()
            # clear_session clears in-memory state and database
            manager.clear_session(self._client.session_id)
            # Note: clear_session doesn't return a count, so we return 0
            logger.info(f"Cleared citations for session {self._client.session_id}")
            return 0

        except Exception as e:
            logger.error(f"Could not clear citations: {e}")
            raise CitationError(f"Could not clear citations: {e}") from e

    async def cleanup_expired(self) -> int:
        """Remove expired citations.

        Citations older than the configured max age are removed.

        Returns:
            Number of citations cleaned up (0 if citations disabled)

        Raises:
            CitationError: If cleanup operation fails

        Example:
            ```python
            count = await client.citations.cleanup_expired()
            print(f"Removed {count} expired citations")
            ```
        """
        # Return 0 if citations disabled (nothing to cleanup)
        if not self.enabled:
            return 0

        try:
            from ...agentic.citation_manager import get_citation_manager

            manager = get_citation_manager()
            count = manager.cleanup_expired()
            logger.info(f"Cleaned up {count} expired citations")
            return count

        except Exception as e:
            logger.error(f"Could not cleanup citations: {e}")
            raise CitationError(f"Could not cleanup citations: {e}") from e
