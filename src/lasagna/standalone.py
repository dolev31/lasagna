"""Standalone RAG module for external agents.

This module provides a simple interface for external agents to use the RAG
knowledge base without running inside a host application. It configures providers automatically
and exposes a clean API.

IMPORTANT: For standalone use, always use `create_rag_backend()` or
`create_agentic_rag()` from this module. Do NOT import directly from
`lasagna` as that triggers host-application detection.

Example Usage:

    from lasagna.standalone import create_rag_backend
    from pathlib import Path

    # Create a RAG backend with custom paths
    backend = create_rag_backend(
        db_dir=Path("/path/to/databases"),
        upload_dir=Path("/path/to/uploads"),
        llm_provider="openai",  # or "anthropic", "azure", "google"
    )

    # Upload a document
    doc_info = backend.upload_document("/path/to/document.pdf", session_id="user123")
    print(f"Uploaded: {doc_info}")

    # Search
    results = backend.search("What is machine learning?", top_k=5)
    for result in results:
        print(f"Score: {result['score']:.3f} - {result['content'][:100]}...")

    # Use agentic RAG for better results
    agentic = create_agentic_rag(
        db_dir=Path("/path/to/databases"),
        upload_dir=Path("/path/to/uploads"),
    )
    enhanced_results = agentic.search("Explain neural networks", session_id="user123")

Advanced: Direct Provider Configuration:

    from lasagna.providers import (
        configure_providers,
        SimpleConfigProvider,
        SimpleLLMProvider,
        InMemorySessionProvider,
        reset_providers,
    )
    from pathlib import Path

    # IMPORTANT: Configure providers BEFORE importing RAGBackend
    reset_providers()  # Clear any auto-detected providers
    configure_providers(
        config=SimpleConfigProvider(
            db_dir=Path("/data/rag/dbs"),
            upload_dir=Path("/data/rag/uploads"),
        ),
        llm=SimpleLLMProvider(default_provider="anthropic"),
        session=InMemorySessionProvider(),
    )

    # Now import and create backend
    from lasagna.api.rag_backend import RAGBackend
    backend = RAGBackend()
"""

from pathlib import Path
from typing import Optional, Dict, Any, TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from lasagna.api.rag_backend import RAGBackend


def create_rag_backend(
    db_dir: Optional[Path] = None,
    upload_dir: Optional[Path] = None,
    llm_provider: str = "openai",
    rag_settings: Optional[Dict[str, Any]] = None,
) -> "RAGBackend":
    """Create a standalone RAG backend for external agents.

    This function configures providers and creates a RAGBackend instance
    ready for use outside of Lasagna.

    Args:
        db_dir: Directory for databases (Milvus, SQLite). Defaults to ./rag_data/dbs
        upload_dir: Directory for uploaded files. Defaults to ./rag_data/uploads
        llm_provider: LLM provider name ("openai", "anthropic", "azure", "google")
        rag_settings: Optional RAG configuration overrides (chunk_size, etc.)

    Returns:
        Configured RAGBackend instance

    Example:
        backend = create_rag_backend(
            db_dir=Path("/tmp/rag/dbs"),
            upload_dir=Path("/tmp/rag/uploads"),
            llm_provider="openai",
            rag_settings={"chunk_size": 1000, "enable_hybrid_search": True},
        )

    Note:
        Requires appropriate API keys set as environment variables:
        - OPENAI_API_KEY for OpenAI
        - ANTHROPIC_API_KEY for Anthropic
        - AZURE_OPENAI_API_KEY for Azure
        - GOOGLE_API_KEY for Google
    """
    from lasagna.providers import (
        configure_providers,
        SimpleConfigProvider,
        SimpleLLMProvider,
        InMemorySessionProvider,
        reset_providers,
    )
    from lasagna.api.rag_backend import RAGBackend

    # Validate llm_provider
    valid_providers = {"openai", "anthropic", "azure", "google"}
    if llm_provider not in valid_providers:
        raise ValueError(
            f"Invalid llm_provider: '{llm_provider}'. "
            f"Must be one of: {', '.join(sorted(valid_providers))}"
        )

    # Reset any existing providers to ensure clean state
    reset_providers()

    # Set up default paths
    base_dir = Path.cwd() / "rag_data"
    if db_dir is None:
        db_dir = base_dir / "dbs"
    if upload_dir is None:
        upload_dir = base_dir / "uploads"

    # Create directories
    db_dir.mkdir(parents=True, exist_ok=True)
    upload_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Initializing standalone RAG: db_dir={db_dir}, upload_dir={upload_dir}")

    # Configure providers
    configure_providers(
        config=SimpleConfigProvider(
            db_dir=db_dir,
            upload_dir=upload_dir,
            rag_settings=rag_settings or {},
        ),
        llm=SimpleLLMProvider(default_provider=llm_provider),
        session=InMemorySessionProvider(),
    )

    # Create and return backend
    return RAGBackend()


def create_agentic_rag(
    db_dir: Optional[Path] = None,
    upload_dir: Optional[Path] = None,
    llm_provider: str = "openai",
    rag_settings: Optional[Dict[str, Any]] = None,
):
    """Create a standalone Agentic RAG instance.

    This provides the full agentic RAG experience with query refinement,
    HyDE, and iterative search.

    Args:
        db_dir: Directory for databases
        upload_dir: Directory for uploaded files
        llm_provider: LLM provider name
        rag_settings: Optional RAG configuration overrides

    Returns:
        Configured AgenticRAG instance

    Example:
        agentic = create_agentic_rag(
            llm_provider="anthropic",
            rag_settings={"max_agent_iterations": 5},
        )

        # Upload documents
        agentic.backend.add_document("/path/to/doc.pdf")

        # Search with agentic enhancements
        results = agentic.search(
            "What are the key findings?",
            session_id="my_session",
        )
    """
    from lasagna.agentic.agentic_rag import AgenticRAG

    backend = create_rag_backend(
        db_dir=db_dir,
        upload_dir=upload_dir,
        llm_provider=llm_provider,
        rag_settings=rag_settings,
    )

    return AgenticRAG(backend)


def reset_standalone():
    """Reset all providers to uninitialized state.

    Call this to reconfigure the RAG system with different settings.
    """
    from lasagna.providers import reset_providers

    reset_providers()
    logger.info("Standalone RAG providers reset")


__all__ = [
    "create_rag_backend",
    "create_agentic_rag",
    "reset_standalone",
]
