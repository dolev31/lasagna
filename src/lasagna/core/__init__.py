"""Core layer - Configuration, schemas, and session management.

This layer contains the foundational components:
- RAGConfig: System configuration with SOTA defaults
- Document/DocumentChunk: Data models
- Session management: Per-session RAG settings
"""

from .config import RAGConfig, get_rag_config, reset_rag_config
from .schema import (
    Document,
    DocumentChunk,
    DocumentStatus,
    DocumentMetadata,
    RecordedDocumentChunk,
    SearchResult,
    create_rag_document_schema,
    CREATE_DOCUMENTS_TABLE,
    CREATE_DOCUMENTS_INDEX,
    CREATE_DOCUMENTS_STATUS_INDEX,
)
from .session_config import (
    KnowledgeConfigManager,
    KnowledgeConfigState,
    RAGProfile,
    get_knowledge_config_manager,
)
from .base_components import (
    BaseEmbedder,
    BaseChunker,
    BaseReranker,
    BaseRetriever,
    BaseDocumentProcessor,
    BaseQueryEnhancer,
    BaseAgenticPipeline,
    AgenticPipelineInput,
    Chunk,
    ScoredDocument,
    ProcessedDocument,
)
from .registry import ComponentRegistry, registry, COMPONENT_TYPES
from .pipeline import KnowledgePipeline, PipelineResult

__all__ = [
    # Configuration
    "RAGConfig",
    "get_rag_config",
    "reset_rag_config",
    # Schema
    "Document",
    "DocumentChunk",
    "DocumentStatus",
    "DocumentMetadata",
    "RecordedDocumentChunk",
    "SearchResult",
    "create_rag_document_schema",
    "CREATE_DOCUMENTS_TABLE",
    "CREATE_DOCUMENTS_INDEX",
    "CREATE_DOCUMENTS_STATUS_INDEX",
    # Session
    "KnowledgeConfigManager",
    "KnowledgeConfigState",
    "RAGProfile",
    "get_knowledge_config_manager",
    # Base Components
    "BaseEmbedder",
    "BaseChunker",
    "BaseReranker",
    "BaseRetriever",
    "BaseDocumentProcessor",
    "BaseQueryEnhancer",
    "BaseAgenticPipeline",
    "AgenticPipelineInput",
    "Chunk",
    "ScoredDocument",
    "ProcessedDocument",
    # Registry
    "ComponentRegistry",
    "registry",
    "COMPONENT_TYPES",
    # Pipeline
    "KnowledgePipeline",
    "PipelineResult",
]
