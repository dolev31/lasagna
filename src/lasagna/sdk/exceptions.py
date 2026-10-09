"""Knowledge SDK Exceptions.

Provides a comprehensive exception hierarchy for the Knowledge SDK,
enabling proper error handling and distinguishing between different
failure modes.

Example:
    ```python
    from lasagna.sdk import KnowledgeClient
    from lasagna.sdk.exceptions import (
        DocumentNotFoundError,
        SearchError,
        BackendError,
    )

    client = KnowledgeClient()

    try:
        await client.documents.delete("nonexistent")
    except DocumentNotFoundError as e:
        print(f"Document not found: {e}")
    except DocumentError as e:
        print(f"Document operation failed: {e}")
    except KnowledgeError as e:
        print(f"SDK error: {e}")
    ```
"""


# =============================================================================
# BASE EXCEPTION
# =============================================================================


class KnowledgeError(Exception):
    """Base exception for all Knowledge SDK errors.

    All SDK-specific exceptions inherit from this class,
    enabling catch-all error handling.

    Example:
        ```python
        try:
            await client.search("query")
        except KnowledgeError as e:
            logger.error(f"Knowledge SDK error: {e}")
        ```
    """

    pass


# =============================================================================
# DOCUMENT EXCEPTIONS
# =============================================================================


class DocumentError(KnowledgeError):
    """Document operation failed.

    Base class for all document-related errors including upload,
    delete, reindex, and retrieval operations.
    """

    pass


class DocumentNotFoundError(DocumentError):
    """Document not found.

    Raised when attempting to access, delete, or modify a document
    that does not exist in the knowledge base.

    Attributes:
        document_id: The ID of the document that was not found
    """

    def __init__(self, message: str, document_id: str = None):
        super().__init__(message)
        self.document_id = document_id


class UploadError(DocumentError):
    """Document upload failed.

    Raised when document upload or processing fails, including
    text extraction, chunking, or indexing errors.

    Attributes:
        file_path: Path to the file that failed to upload
        stage: Processing stage where failure occurred
    """

    def __init__(self, message: str, file_path: str = None, stage: str = None):
        super().__init__(message)
        self.file_path = file_path
        self.stage = stage


class ChunkingError(DocumentError):
    """Document chunking failed.

    Raised when document chunking fails due to invalid content,
    encoding issues, or chunker configuration problems.
    """

    pass


class ExtractionError(DocumentError):
    """Text extraction failed.

    Raised when text extraction from a document fails,
    including PDF parsing, OCR, or format conversion errors.
    """

    pass


# =============================================================================
# SEARCH EXCEPTIONS
# =============================================================================


class SearchError(KnowledgeError):
    """Search operation failed.

    Base class for all search-related errors including query
    processing, embedding generation, and retrieval errors.
    """

    pass


class QueryError(SearchError):
    """Query processing failed.

    Raised when query processing fails, including validation,
    enhancement, or decomposition errors.

    Attributes:
        query: The query that caused the error
    """

    def __init__(self, message: str, query: str = None):
        super().__init__(message)
        self.query = query


class EmbeddingError(SearchError):
    """Embedding generation failed.

    Raised when embedding generation fails, typically due to
    model loading issues or input validation errors.
    """

    pass


class RetrievalError(SearchError):
    """Retrieval operation failed.

    Raised when vector search or hybrid search operations fail.
    """

    pass


# =============================================================================
# CITATION EXCEPTIONS
# =============================================================================


class CitationError(KnowledgeError):
    """Citation operation failed.

    Raised when citation tracking, retrieval, or verification fails.

    Attributes:
        citation_id: The citation ID involved in the error
    """

    def __init__(self, message: str, citation_id: str = None):
        super().__init__(message)
        self.citation_id = citation_id


class CitationNotFoundError(CitationError):
    """Citation not found.

    Raised when attempting to access a citation that does not exist
    or has expired.
    """

    pass


# =============================================================================
# CONFIGURATION EXCEPTIONS
# =============================================================================


class ConfigurationError(KnowledgeError):
    """Configuration operation failed.

    Raised when configuration validation, loading, or saving fails.

    Attributes:
        config_key: The configuration key that caused the error
    """

    def __init__(self, message: str, config_key: str = None):
        super().__init__(message)
        self.config_key = config_key


class ProfileNotFoundError(ConfigurationError):
    """Configuration profile not found.

    Raised when attempting to load or apply a configuration profile
    that does not exist.

    Attributes:
        profile_name: The name of the profile that was not found
    """

    def __init__(self, message: str, profile_name: str = None):
        super().__init__(message)
        self.profile_name = profile_name


# =============================================================================
# BACKEND EXCEPTIONS
# =============================================================================


class BackendError(KnowledgeError):
    """Backend communication failed.

    Raised when communication with the vector database or other
    backend services fails.

    Attributes:
        backend_name: Name of the backend that failed
        operation: The operation that failed
    """

    def __init__(self, message: str, backend_name: str = None, operation: str = None):
        super().__init__(message)
        self.backend_name = backend_name
        self.operation = operation


class BackendConnectionError(BackendError):
    """Backend connection failed.

    Raised when establishing connection to a backend service fails.
    """

    pass


class BackendTimeoutError(BackendError):
    """Backend operation timed out.

    Raised when a backend operation exceeds its timeout limit.
    """

    pass


# =============================================================================
# AGENTIC EXCEPTIONS
# =============================================================================


class AgenticError(KnowledgeError):
    """Agentic RAG operation failed.

    Raised when multi-hop retrieval or agent execution fails.

    Attributes:
        iteration: The iteration number where failure occurred
    """

    def __init__(self, message: str, iteration: int = None):
        super().__init__(message)
        self.iteration = iteration


class MaxIterationsError(AgenticError):
    """Maximum iterations exceeded.

    Raised when agentic search exceeds the maximum allowed iterations
    without reaching a satisfactory result.
    """

    pass


# =============================================================================
# EVALUATION EXCEPTIONS
# =============================================================================


class EvaluationError(KnowledgeError):
    """Evaluation operation failed.

    Raised when RAG evaluation operations fail.
    """

    pass


class JudgeError(EvaluationError):
    """LLM judge evaluation failed.

    Raised when LLM-as-judge evaluation fails.
    """

    pass


# =============================================================================
# INTEGRATION EXCEPTIONS
# =============================================================================


class IntegrationError(KnowledgeError):
    """Framework integration error.

    Raised when integration with external frameworks (LangChain, MCP)
    fails.

    Attributes:
        framework: The framework that caused the error
    """

    def __init__(self, message: str, framework: str = None):
        super().__init__(message)
        self.framework = framework


# =============================================================================
# PUBLIC API
# =============================================================================

__all__ = [
    # Base
    "KnowledgeError",
    # Document
    "DocumentError",
    "DocumentNotFoundError",
    "UploadError",
    "ChunkingError",
    "ExtractionError",
    # Search
    "SearchError",
    "QueryError",
    "EmbeddingError",
    "RetrievalError",
    # Citation
    "CitationError",
    "CitationNotFoundError",
    # Configuration
    "ConfigurationError",
    "ProfileNotFoundError",
    # Backend
    "BackendError",
    "BackendConnectionError",
    "BackendTimeoutError",
    # Agentic
    "AgenticError",
    "MaxIterationsError",
    # Evaluation
    "EvaluationError",
    "JudgeError",
    # Integration
    "IntegrationError",
]
