# `sdk/` — Knowledge SDK

A clean, minimal API for building RAG (Retrieval-Augmented Generation) applications with semantic search, document management, and AI-powered retrieval. Works standalone with any agent or as part of Lasagna.

## Quick Start

```python
from lasagna.sdk import KnowledgeClient

async def main():
    client = KnowledgeClient()

    # Upload a document
    result = await client.upload("./quarterly_report.pdf")
    # upload() returns as soon as the document is queued, so chunk_count is
    # still 0 here. Wait if you need the final count.
    await client.documents.wait_for_processing(result.document_id)
    print(f"Uploaded: {result.document_id}")

    # Search for information
    results = await client.search_documents("What is the quarterly revenue?")
    for r in results:
        print(f"[{r.score:.2f}] {r.chunk.content[:100]}...")

    # Get context for LLM injection
    context = await client.get_context("key findings and recommendations")
    await client.close()

import asyncio
asyncio.run(main())
```

```bash
pip install lasagna-rag
```

## Start Here

| You are a... | Start with | Then read |
| --- | --- | --- |
| **Non-technical user** (CLI, MCP) | [Getting Started](docs/getting_started.md) | [Integrations](docs/integrations.md) |
| **Python SDK developer** | [Getting Started](docs/getting_started.md) | [User Guide](docs/user_guide.md), [Configuration](docs/configuration.md) |
| **REST API / remote client** | [Getting Started](docs/getting_started.md) | [User Guide](docs/user_guide.md), [Backends](docs/backends.md) |
| **Researcher / data scientist** | [Extending](docs/extending.md) | [Evaluation](docs/evaluation.md), [Features](docs/features.md) |
| **Enterprise operator (DevOps)** | [Deployment](docs/deployment.md) | [Backends](docs/backends.md), [Configuration](docs/configuration.md) |
| **Multi-tenant SaaS operator** | [Deployment](docs/deployment.md) | [Configuration](docs/configuration.md) |
| **UI / frontend developer** | [UI Integration](docs/ui/integration.md) | [UI Components](docs/ui/components.md) |
| **AI agent developer** | [Integrations](docs/integrations.md) | [Features](docs/features.md) |

## Documentation Index

| Document | Description | Audience |
| --- | --- | --- |
| [Getting Started](docs/getting_started.md) | Installation, quick start, architecture overview, key types | Everyone |
| [User Guide](docs/user_guide.md) | KnowledgeClient API, document management, sessions, search, error handling, API reference | Developers |
| [Configuration](docs/configuration.md) | Config options, presets, profiles, retry/resilience, advanced usage | Developers, Operators |
| [Features](docs/features.md) | Agentic RAG (multi-hop), processing pipeline, SOTA retrieval features | Developers, Researchers |
| [Integrations](docs/integrations.md) | LangChain retriever/loader, MCP server, MCPKnowledgeClient | Integration developers |
| [Backends](docs/backends.md) | Vector backends (Milvus, Pinecone, Qdrant, Weaviate), storage backends (S3, GCS) | Infrastructure |
| [Deployment](docs/deployment.md) | Docker Compose, multi-tenant SaaS, quotas, rate limiting, troubleshooting | DevOps, SRE |
| [Evaluation](docs/evaluation.md) | KnowledgeEvaluator, LLM-as-judge, ablation studies, benchmarking | Researchers |
| [Extending](docs/extending.md) | Component registry, swappable components, KnowledgePipeline, RAGAS, contributing | Researchers, ML engineers |
| [Custom Pipelines (BYOA)](docs/custom_pipelines.md) | Writing, registering, and deploying custom agentic retrieval pipelines | Advanced developers |
| [UI Integration](docs/ui/integration.md) | Frontend integration contract reference (3-tier system, SDK + REST examples) | Frontend developers |
| [UI Components](docs/ui/components.md) | Copy-paste TypeScript types, React hooks, and components | Frontend developers |

## Public API

| Entry Point | Description |
| --- | --- |
| `KnowledgeClient` | Main client for all operations (upload, search, get_context, agentic_search) |
| `KnowledgeClient.from_url()` | Remote client connecting to a deployed RAG server (only needs `httpx`) |
| `KnowledgeConfig` | Configuration dataclass with 100+ fields |
| `config_for_speed()` | Preset: pure vector search, ~50-100ms |
| `config_for_standard()` | Preset: SOTA features + lightweight model, ~200-400ms |
| `config_for_balanced()` | Preset: SOTA features + BGE-M3 + hierarchical, ~300-600ms |
| `config_for_max_quality()` | Preset: all features enabled, ~1-3s |
| `config_for_standalone()` | Preset: explicit paths for standalone usage |
| `config_for_production()` | Preset: PostgreSQL, Redis, multi-tenant |
| `Document`, `SearchResult`, `SearchResults`, `Citation`, `UploadResult` | Core data types |
| `ChunkingStrategy`, `VectorBackend`, `HyDEStrategy`, `DocumentStatus` | Configuration enums |
| `KnowledgeError` and subclasses | Exception hierarchy |

## Related Documentation

- [Knowledge Package Overview](../README.md) — top-level features, architecture, all integration paths
- [REST API](../api/README.md) — HTTP endpoints and API design
- [Agentic RAG](../agentic/README.md) — multi-hop retrieval system internals
- [Vector Backends](../vector_backends/README.md) — backend abstraction layer and filter expressions
- [Evaluation SDK](../knowledge_evaluation_sdk/README.md) — full evaluation framework
