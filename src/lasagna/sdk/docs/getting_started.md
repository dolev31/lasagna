# Getting Started

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Core Concepts](#core-concepts)

---

## Quick Start

```python
from lasagna.sdk import KnowledgeClient

async def main():
    # Create client
    client = KnowledgeClient()

    # Upload a document
    result = await client.upload("./quarterly_report.pdf")
    # Processing is asynchronous: chunk_count is 0 until it finishes.
    await client.documents.wait_for_processing(result.document_id)
    print(f"Uploaded: {result.document_id}")

    # Search for information
    results = await client.search_documents("What is the quarterly revenue?")
    for r in results:
        print(f"[{r.score:.2f}] {r.chunk.content[:100]}...")

    # Get context for LLM injection
    context = await client.get_context("key findings and recommendations")

    # Use in your LLM prompt
    prompt = f"""Based on the following context:
{context}

Answer the question: What are the key findings?"""

    # Cleanup
    await client.close()

# Run
import asyncio
asyncio.run(main())
```

---

## Installation

```bash
pip install lasagna-rag
```

**Prerequisites:** Set one LLM API key in your environment. The SDK picks the provider from whichever key it finds (the platforms and their default models are in `providers/llm_provider.py`, `SUPPORTED_PLATFORMS`):

```bash
# Any ONE of these is sufficient:
export OPENAI_API_KEY=sk-...       # or ANTHROPIC_API_KEY, GROQ_API_KEY,
                                    # MISTRAL_API_KEY, TOGETHER_API_KEY,
                                    # DEEPSEEK_API_KEY, GOOGLE_API_KEY, etc.
```

The LangChain integration and the MCP server come with the core package. Other vector backends are optional extras:

```bash
pip install "lasagna-rag[vector-backends]"  # Pinecone, Qdrant, Weaviate
pip install "lasagna-rag[pgvector]"         # PostgreSQL pgvector
pip install "lasagna-rag[opensearch]"       # OpenSearch
```

Documents, the metadata database and the vector store are kept under `~/.lasagna`. Set `LASAGNA_DATA_DIR` (or `RAG_DATA_DIR`) to keep them elsewhere; see [where data lives](../../../../docs/configuration.md#where-data-lives).

---

## Core Concepts

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     KnowledgeClient                         │
│  - upload()          - search_documents()                   │
│  - get_context()     - agentic_search()                     │
└─────────────────────────────────────────────────────────────┘
        │                    │                    │
        ▼                    ▼                    ▼
┌───────────────┐  ┌─────────────────┐  ┌─────────────────────┐
│DocumentManager│  │  SearchManager  │  │  CitationManager    │
│ - upload()    │  │ - search()      │  │ - get()             │
│ - list()      │  │ - get_context() │  │ - verify()          │
│ - delete()    │  │ - hybrid_search │  │ - list_recent()     │
└───────────────┘  └─────────────────┘  └─────────────────────┘
        │                    │
        ▼                    ▼
┌─────────────────────────────────────────────────────────────┐
│                      RAGBackend                             │
│  Processing → Embedding → Vector Store → Retrieval          │
└─────────────────────────────────────────────────────────────┘
```

### Key Types

| Type | Description |
|------|-------------|
| `KnowledgeClient` | Main entry point for all operations |
| `KnowledgeConfig` | Configuration dataclass |
| `Document` | Document metadata |
| `DocumentChunk` | A chunk of document content |
| `SearchResult` | Single search result with score |
| `SearchResults` | Collection of results with helpers |
| `Citation` | Source citation for tracking |
| `UploadResult` | Result of upload operation |

### Enums for Configuration

```python
from lasagna.sdk import (
    ChunkingStrategy,    # FIXED, SEMANTIC, RECURSIVE, SENTENCE_WINDOW, STRUCTURE_AWARE, HIERARCHICAL
    EmbeddingProvider,   # HUGGINGFACE, OPENAI
    VectorBackend,       # MILVUS, PINECONE, QDRANT, WEAVIATE, PGVECTOR, OPENSEARCH, MEMORY
    HyDEStrategy,        # OFF, FIRST_HOP, ADAPTIVE, ALWAYS
    DocumentStatus,      # PENDING, PROCESSING, INDEXING, READY, FAILED, SOFT_DELETED
)
```

---

## See Also

- [User Guide](user_guide.md)
- [Configuration](configuration.md)
