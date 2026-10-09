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
pip install lasagna-rag[rag]
```

**Prerequisites:** Set one LLM API key in your environment. The SDK auto-detects from 15+ supported platforms:

```bash
# Any ONE of these is sufficient:
export OPENAI_API_KEY=sk-...       # or ANTHROPIC_API_KEY, GROQ_API_KEY,
                                    # MISTRAL_API_KEY, TOGETHER_API_KEY,
                                    # DEEPSEEK_API_KEY, GOOGLE_API_KEY, etc.
```

For optional integrations:

```bash
# LangChain integration
pip install langchain-core

# MCP server
pip install mcp

# Additional vector backends
pip install pinecone-client  # Pinecone
pip install qdrant-client    # Qdrant
pip install weaviate-client  # Weaviate
```

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
    ChunkingStrategy,    # FIXED, SEMANTIC, RECURSIVE, SENTENCE_WINDOW, HIERARCHICAL
    EmbeddingProvider,   # HUGGINGFACE, OPENAI
    VectorBackend,       # MILVUS, PINECONE, QDRANT, WEAVIATE
    HyDEStrategy,        # OFF, FIRST_HOP, ADAPTIVE, ALWAYS
    DocumentStatus,      # PENDING, PROCESSING, INDEXING, READY, FAILED
)
```

---

## See Also

- [User Guide](user_guide.md)
- [Configuration](configuration.md)
