# Lasagna: Production-Ready RAG for Any Application

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Vector Backends](https://img.shields.io/badge/vector%20backends-6-orange.svg)](#vector-backend-support-byovdb)
[![LLM Platforms](https://img.shields.io/badge/LLM%20platforms-15%2B-purple.svg)](#supported-llm-platforms-15)

A **state-of-the-art**, plug-and-play RAG engine that works standalone with any agent framework or as part of Lasagna. Ingest documents, search semantically, track citations, and run multi-hop agentic retrieval -- all with zero external dependencies on Lasagna.

---

## Key Features

- **BGE-M3 Embedding** -- dense + sparse + ColBERT in a single model, 8K token context, 100+ languages
- **ms-marco-MiniLM-L-6-v2 Reranker** -- +0.084 NDCG@10 on held-out test, cross-encoder precision
- **Docling Document Processing** -- IBM's AI layout analysis with TableFormer (97.9% accuracy), OCR, structure-aware chunking
- **Contextual Retrieval** -- Anthropic-style chunk enrichment, -67% retrieval failure rate
- **RAG Fusion** -- multi-query generation + Reciprocal Rank Fusion
- **Corrective RAG (CRAG)** -- quality grading with automatic web fallback
- **Self-RAG Quality Gates** -- ISSUP hallucination grading + ISUSE answer quality checking (opt-in)
- **HyDE (Hypothetical Document Embeddings)** -- adaptive query expansion for better recall
- **Query Decomposition** -- LLM-based breakdown of complex questions into focused sub-queries
- **Structured Graph Mode** -- explicit LangGraph StateGraph with 13 nodes for debuggable multi-hop retrieval
- **Adaptive-K** -- score-gap detection to return only relevant results
- **Lost-in-the-Middle Reordering** -- +15 percentage points on recall benchmarks
- **6 Vector Backends** -- Milvus, Pinecone, Qdrant, Weaviate, PgVector, OpenSearch (BYOVDB)
- **Multi-Tenant SaaS** -- partition isolation, per-tenant quotas, adaptive rate limiting
- **Component Registry** -- swap-and-measure any pipeline component for research
- **RAGAS Evaluation** -- built-in evaluation integration with ablation sweeps

---

## 60-Second Quickstart

```python
from lasagna.sdk import KnowledgeClient

async with KnowledgeClient() as client:
    await client.upload("report.pdf")
    results = await client.search_documents("quarterly revenue")
    context = await client.get_context("key findings")
```

Set any supported LLM API key (e.g. `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and the engine auto-configures. No YAML, no boilerplate.

---

## Choose Your Path

### For Non-Technical Users (CLI)

Use the **standalone CLI** for zero-code RAG management:

```bash
# Interactive setup wizard
python -m lasagna init

# Upload and index a document
python -m lasagna upload report.pdf

# Semantic search with Rich table output
python -m lasagna search "quarterly revenue"

# Start REST API + configurator UI
python -m lasagna serve

# Show system status
python -m lasagna status

# Apply a quality profile
python -m lasagna configure --profile max_quality

# Export / import configuration
python -m lasagna export --output config.json
python -m lasagna import config.json

# Run evaluation benchmark
python -m lasagna evaluate --document data.pdf --ground-truth gt.jsonl
```

### For Non-Technical Users (MCP)

Use the **MCP Server** to connect any MCP-compatible client (Claude Desktop, Cursor, or custom agents):

```bash
export OPENAI_API_KEY=sk-...
python -m lasagna.mcp --host 0.0.0.0 --port 8005
```

Connect your client to `http://localhost:8005/sse` -- upload documents, ask questions, and get cited answers through a conversational interface. No code required.

### For Developers

Use the **Python SDK** or **REST API** to integrate RAG into your application in minutes:

```python
from lasagna.sdk import KnowledgeClient, config_for_standard

client = KnowledgeClient(config=config_for_standard())

# Upload documents
result = await client.upload("contracts/2025-q4.pdf")

# Semantic search with citations
results = await client.search_documents("termination clause penalties")
for r in results:
    print(f"[{r.score:.2f}] {r.chunk.content[:120]}")

# Get LLM-ready context
context = await client.get_context("payment terms summary")
```

Four built-in profiles (`config_for_speed`, `config_for_standard`, `config_for_balanced`, `config_for_max_quality`) let you trade latency for quality with a single function call.

### For Researchers

Use the **Component Registry** to swap any pipeline component and benchmark against built-ins:

```python
from lasagna.retrieval.registry import component_registry

# Register a custom reranker
class MyColBERTReranker:
    def rerank(self, query, results, top_k=5, config=None):
        ...

component_registry.register_reranker("my_colbert", MyColBERTReranker)

# List all registered components
print(component_registry.list_rerankers())
# [{"name": "cross_encoder", ...}, {"name": "combined", ...}, {"name": "my_colbert", ...}]

# Create instances via factory
reranker = component_registry.create_reranker("my_colbert")

# Use via config: set custom_reranker in settings.lasagna.toml
# custom_reranker = "my_colbert"
```

Four component categories are supported: **rerankers**, **embedders**, **chunkers**, and **query enhancers**. Built-in components are auto-registered; custom ones integrate without modifying package source code.

Evaluate results with the built-in evaluation SDK:

```python
from lasagna.knowledge_evaluation_sdk import KnowledgeEvaluator, EvalConfig

config = EvalConfig(
    document_path="./data/report.pdf",
    ground_truth_path="./benchmarks/ground_truth.jsonl",
    judge_model_platform="openai",
    judge_model_name="gpt-4o",
)
evaluator = KnowledgeEvaluator(config)
metrics = await evaluator.run()
```

### For Enterprise Operators (DevOps/SRE)

Deploy, scale, and monitor the RAG service with Docker Compose or Kubernetes:

```bash
cd src/lasagna/deploy
VECTOR_BACKEND=pgvector make up
COMPOSE_PROFILES=celery,mcp,tracing VECTOR_BACKEND=pgvector make up
make health
```

Full deployment guide — Docker, Kubernetes, monitoring, security: [deploy/README.md](deploy/README.md)

### For Multi-Tenant SaaS Operators

Enable partition isolation, per-tenant quotas, and adaptive rate limiting:

```bash
RAG_TENANT_ISOLATION=partition RAG_QUOTA_ENABLED=true RAG_RATE_LIMITING_ENABLED=true
```

See [Multi-Tenancy](#multi-tenancy) below and [deploy/README.md](deploy/README.md) for production multi-tenant deployment.

### For Business Power Users (Configurator UI)

Fine-tune all 91 RAG parameters through a web interface — no code required:

```bash
RAG_CONFIGURATOR=true python -m lasagna serve
# Open http://localhost:8000/configurator
```

Configurator documentation: [configurator/README.md](configurator/README.md)

---

## Documentation Map

| Module | README | Deep Dives | Covers |
| ------ | ------ | ---------- | ------ |
| **sdk** | [sdk/](sdk/README.md) | [12 guides](sdk/docs/) | KnowledgeClient, config, remote client, UI components |
| **api** | [api/](api/README.md) | [3 docs](api/docs/) | REST endpoints, middleware, FastAPI |
| **agentic** | [agentic/](agentic/README.md) | [4 docs](agentic/docs/) | Multi-hop pipeline, nodes, state, CRAG |
| **core** | [core/](core/README.md) | [5 docs](core/docs/) | Config (160 properties), database, sessions, schema |
| **retrieval** | [retrieval/](retrieval/README.md) | — | Search pipeline, reranking, query routing |
| **processing** | [processing/](processing/README.md) | — | Docling, OCR, semantic chunking |
| **infrastructure** | [infrastructure/](infrastructure/README.md) | [5 docs](infrastructure/docs/) | Embedding, caching, LLM manager |
| **vector_backends** | [vector_backends/](vector_backends/README.md) | — | 6 backends, BYOVDB, filters, pooling |
| **storage** | [storage/](storage/README.md) | — | File storage (local, S3, fsspec) |
| **providers** | [providers/](providers/README.md) | — | LLM, config, session providers |
| **deploy** | [deploy/](deploy/README.md) | — | Docker, Kubernetes, monitoring, security |
| **configurator** | [configurator/](configurator/README.md) | [3 docs](configurator/docs/) | Web UI, profiles, validation |
| **mcp** | [mcp/](mcp/README.md) | — | MCP server, 33 tools |
| **evaluation** | [knowledge_evaluation_sdk/](knowledge_evaluation_sdk/README.md) | [3 docs](knowledge_evaluation_sdk/docs/) | Benchmarks, IR metrics, ablation |
| **utils** | [utils/](utils/README.md) | — | Monitoring, tracing, model utilities |

---

## Architecture

```text
Query / Document
       |
       v
+----------------------------------------------+
|              KnowledgeClient (SDK)            |
+----------------------------------------------+
       |                          |
       v                          v
+--------------+     +-------------------------+
|   Document   |     |    Search Pipeline      |
|   Pipeline   |     |                         |
|              |     |  Query Classification   |
|  Docling /   |     |         |               |
|  Legacy      |     |  [RAG Fusion: multi-    |
|  Extraction  |     |   query + RRF]          |
|     |        |     |         |               |
|  Semantic /  |     |  [HyDE Expansion]       |
|  Hierarchical|     |         |               |
|  Chunking    |     |  Dense + Sparse Search  |
|     |        |     |  (BGE-M3 hybrid)        |
|  [Contextual |     |         |               |
|   Enrichment]|     |  [BM25 Fusion]          |
|     |        |     |         |               |
|  Embedding   |     |  [Reranking: ms-marco]  |
|  (BGE-M3)    |     |         |               |
|     |        |     |  [Adaptive-K Filtering] |
|     v        |     |         |               |
|  Vector DB   |     |  [CRAG: quality grade   |
|  Upsert      |     |   + web fallback]       |
|              |     |         |               |
+--------------+     |  [Lost-in-Middle        |
                     |   Reordering]           |
                     |         |               |
                     |  [Auto-Merge            |
                     |   Hierarchical Chunks]  |
                     |         |               |
                     |  Citation Extraction    |
                     |         |               |
                     |      Results            |
                     +-------------------------+
                              |
                              v
                     +------------------+
                     |   Vector Backend |
                     |   (BYOVDB)       |
                     |                  |
                     |  Milvus | Qdrant |
                     |  Pinecone |      |
                     |  Weaviate |      |
                     |  PgVector |      |
                     |  OpenSearch      |
                     +------------------+
```

Brackets `[...]` indicate optional stages activated by profile selection. The `SPEED` profile skips all optional stages; `MAX_QUALITY` enables everything.

### Document Upload Lifecycle

When a document is uploaded, it flows through these stages:

```text
HTTP POST /api/v1/documents
  │
  ▼
routes/documents.py ─── File validation, size check, streaming to temp dir
  │
  ▼
_document_pipeline.py ─── Orchestrator (owns the full lifecycle)
  │
  ├── quota.py ─── Atomic quota reservation (prevents race conditions)
  │
  ├── storage backend ─── Persist file (local / S3 / fsspec)
  │
  ├── database.py ─── Insert metadata (status=PROCESSING)
  │
  ├── Dispatch decision:
  │   ├── Celery enabled → enqueue task, return status=PROCESSING
  │   └── Celery disabled → process synchronously (blocking)
  │
  ├── document_processor.py ─── Text extraction (PDF, DOCX, HTML, OCR)
  │
  ├── chunker/semantic.py ─── Semantic chunking
  │
  ├── embedding/manager.py ─── Batch encode → vectors
  │
  ├── _collection_manager.py ─── Resolve vector backend + tenant partition
  │
  ├── vector_backends/*.py ─── Insert into vector DB
  │
  └── database.py ─── Update status=READY, chunks_count
```

**Key responsibilities:**

| Component | File | Role |
| --------- | ---- | ---- |
| REST Layer | `api/rest/routes/documents.py` | HTTP handling, file streaming, size validation |
| Document Pipeline | `api/_document_pipeline.py` | **Main orchestrator** — upload, storage, processing, status |
| Quota Manager | `core/quota.py` | Atomic reservation/release, per-tenant limits |
| Celery Tasks | `infrastructure/tasks/document_tasks.py` | Async dispatch, retries (max 2, acks_late) |
| Document Processor | `processing/document_processor.py` | Text extraction per format |
| Semantic Chunker | `processing/chunker/semantic.py` | Intelligent text splitting |
| Embedding Manager | `infrastructure/embedding/manager.py` | Batch vector generation |
| Collection Manager | `api/_collection_manager.py` | Vector backend lifecycle, tenant partitioning |
| Database | `core/database.py` | Metadata persistence, status tracking |

**Processing paths:**

- **Synchronous** (Celery disabled / milvus-lite): Upload blocks until processing completes, returns final status immediately.
- **Asynchronous** (Celery enabled): Upload returns `status=PROCESSING` immediately. Celery worker processes in background. Client polls `GET /api/v1/documents/{id}/status` until `READY`.

**Status lifecycle:** `PROCESSING` → `READY` (success) or `ERROR` (failure, quota released).

A document becomes searchable only after vectors are inserted into the backend, status is updated to `READY`, and the search cache is invalidated.

---

## Profile Matrix

Choose a profile that matches your latency and quality requirements. Metrics from empirical evaluation on a 3,132-document multi-domain corpus (see [Evaluation Report](../../docs/RAG_EVALUATION_REPORT.md)):

| Profile | Retrieval Latency | NDCG@10 | Faithfulness | Key Features | Use Case |
| ------- | ----------------- | ------- | ------------ | ------------ | -------- |
| **SPEED** | <100ms | 0.468 | 0.41 | Dense-only, MiniLM, no agentic | Real-time autocomplete, chatbots |
| **STANDARD** | 2–3s | 0.496 | 0.41 | Hybrid + ms-marco reranker, no MMR | Most applications (recommended) |
| **QUALITY** | 10–20s | 0.50+ | 0.39–0.41 | Standard + HyDE + RAG Fusion | Research, compliance, legal |

> **Evidence-based changes from grid search (23 experiments, 180+ configs):**
>
> - MMR disabled across all profiles: +0.046 NDCG, zero latency cost
> - Reranker switched to `ms-marco-MiniLM-L-6-v2`: +0.084 NDCG on held-out test
> - Document awareness (100-token prefix) enabled: +0.062 faithfulness
> - Agentic RAG disabled by default: simpler pipelines outperform on multi-domain corpora
> - `top_k` reduced to 10: deeper retrieval hurts precision

```python
from lasagna.sdk import (
    config_for_speed,
    config_for_standard,
    config_for_max_quality,
)

# Pick one:
config = config_for_speed()             # <100ms, NDCG≈0.47, faithfulness≈0.41
config = config_for_standard()          # 2-3s, NDCG≈0.50, faithfulness≈0.41 (recommended)
config = config_for_max_quality()       # 10-20s, NDCG≈0.50+, faithfulness≈0.39-0.41
```

### Key Findings

| Finding | Impact | Evidence |
| ------- | ------ | -------- |
| Disable MMR diversity filtering | **+0.046 NDCG** (largest single feature gain) | Exp 03: MMR pushes relevant docs out of top-k |
| ms-marco-MiniLM-L6 reranker | **+0.084 NDCG on test** (most robust gain) | Exp 14/15: outperforms mxbai-rerank-large |
| 100-token doc awareness prefix | **+0.062 faithfulness** (+17.8%) | Exp 17: zero LLM cost, simple extraction |
| Disable agentic RAG (multi-domain) | **+0.056 faithfulness** | Exp 07: aggressive refinement causes context starvation |
| Features don't compose | no-MMR (+0.046) + reranker (+0.037) = only +0.007 combined | Exp 11: 8.4% composition efficiency |
| Never use BM25-only | **-0.108 NDCG on test** (-26%) | Exp 14: catastrophic regression |

---

## SOTA Techniques

### BGE-M3 Embedding

Multi-representation embedding model that produces dense, sparse, and ColBERT vectors in a single forward pass. Supports 8,192-token context and 100+ languages, eliminating the need for separate sparse indexing.

### ms-marco-MiniLM-L-6-v2 Reranker

Cross-encoder reranker providing **+0.084 NDCG@10 on held-out test data** — the single most impactful validated improvement. Counterintuitively outperforms both the 12-layer ms-marco variant and the larger mxbai-rerank-large-v1, likely due to better calibration for typical RAG query distributions. Applied after initial retrieval to re-score candidates with full query-document attention. Latency: ~12s on CPU, <500ms with GPU.

### Document Awareness (First-Tokens Enrichment)

Prepends the first 100 tokens of each source document (title, abstract, introduction) to its chunks at retrieval time. Provides **+0.062 faithfulness (+17.8%)** with zero LLM cost — the strongest generation quality gain found. Longer prefixes (300 tokens) and LLM-generated summaries perform at or below baseline due to content dilution.

### Docling Document Processing

IBM's open-source document understanding engine. Performs AI-powered layout analysis, TableFormer table extraction (97.9% accuracy), OCR for scanned pages, and structure-aware chunking that preserves headings, lists, and code blocks.

### Contextual Retrieval

Inspired by Anthropic's research. Each chunk is enriched with document-level context at ingestion time, reducing retrieval failure rates by 67%. Works by prepending a short context summary that situates the chunk within its source document.

### RAG Fusion

Generates multiple reformulations of the user query, runs parallel searches, and merges results using Reciprocal Rank Fusion (RRF). Provides **+0.007 NDCG and +0.022 HitRate** with 3 query variants. Use only in quality tier — adds ~2.6s latency. Never use `max_query_variations: 2` (hurts precision by -0.014 NDCG).

### Corrective RAG (CRAG)

Grades retrieved document quality before generation. If the top results score below a confidence threshold, the system automatically falls back to web search to supplement or replace low-quality retrievals. **Note**: Cannot be combined with RAG Fusion (known runtime crash).

### Self-RAG Quality Gates

Post-generation verification inspired by [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511). Two opt-in checks run after the LLM generates an answer:

- **ISSUP (hallucination check)** -- verifies every claim in the answer is grounded in source documents
- **ISUSE (answer quality check)** -- verifies the answer adequately addresses the original question

When checks fail and budget allows, the system rewrites the query (hallucination) or regenerates (quality) automatically. Configure via `enable_hallucination_check` and `enable_answer_quality_check` (requires structured graph mode). **Empirical note**: In live mode, both gates reduce available context and decrease faithfulness by -0.041 to -0.065 on multi-domain corpora. Use selectively for domain-specific high-precision retrieval.

### HyDE (Hypothetical Document Embeddings)

Generates a hypothetical answer document to use for embedding-based search instead of the raw query. Improves recall on short or ambiguous queries. Directional gain of **+0.017 NDCG** at ~1.9s latency cost. Four strategies: `off`, `always`, `first_hop`, and `adaptive` (default). Adaptive mode activates HyDE on short queries during the first or final hop, or when prior hops returned insufficient results.

### Query Decomposition

For complex multi-faceted questions, the LLM decomposes the original question into 2-4 focused sub-queries. Each sub-query is processed as a separate retrieval loop with results accumulated. The global retrieval budget is shared across all sub-queries to prevent runaway costs. Configure via `enable_query_decomposition`.

### Pluggable Agentic Pipelines

The agentic retrieval layer supports three execution modes, controlled by the `agentic_pipeline` config key:

- **`"legacy_react"`** (default) -- LangChain ReAct agent loop with tool-based retrieval.
- **`"default"`** -- Built-in LangGraph `StateGraph` with 13 explicit nodes and 6 conditional edges. Provides deterministic, debuggable retrieval with CRAG grading, quality gates, and query decomposition.
- **Custom pipelines** -- Any `BaseAgenticPipeline` subclass registered in the `agentic_pipeline` namespace of the component registry.

Individual graph nodes within the structured pipeline can also be overridden via `node_overrides` or the `graph_node` callable registry, without replacing the entire pipeline.

The legacy `use_structured_graph` boolean is still supported: `true` maps to `"default"`, `false` maps to `"legacy_react"`. See the [agentic RAG documentation](agentic/README.md) for architecture details.

### Adaptive-K

Analyzes score gaps between consecutive search results to dynamically determine the optimal number of results to return, rather than using a fixed top-K. Prevents injecting irrelevant context into the LLM prompt.

### Lost-in-the-Middle Reordering

Reorders retrieved passages so the most relevant ones appear at the beginning and end of the context window, counteracting the well-documented tendency of LLMs to under-attend to information in the middle. Measured improvement: +15 percentage points on recall benchmarks.

### Intelligent Query Routing (Adaptive-RAG)

Inspired by [Adaptive-RAG (NAACL 2024)](https://arxiv.org/abs/2403.14403), the router classifies each query and selects the optimal retrieval tier:

- **FAST** -- Vector-only (~50-100ms). Short factual/navigational queries.
- **STANDARD** -- Hybrid + reranking + HyDE (~200-400ms). Most queries (default).
- **COMPREHENSIVE** -- Agentic multi-hop (~1-5s). Complex multi-hop or cross-document queries. Optionally enhanced with structured graph mode, quality gates, and query decomposition.

Reduces average latency by 30-40% and costs by ~28% while improving accuracy on complex queries by ~8%. Configure via `enable_query_routing`, `routing_strategy`, and `routing_default_tier` in config, profiles, REST API, MCP, or SDK.

```python
# Auto-routing (default)
results = await client.search_documents("What is X?")  # -> FAST

# Force a tier
results = await client.search_documents("Compare X with Y across all docs", routing_tier="comprehensive")
```

### Component Registry

A research-oriented system for swapping any pipeline component at runtime. Register custom implementations, run ablation studies, and compare results across configurations. The registry supports seven component namespaces and a separate callable store:

| Namespace | Description |
| --------- | ----------- |
| `embedder` | Text-to-vector encoding |
| `chunker` | Document segmentation |
| `reranker` | Result reranking |
| `retriever` | Vector/hybrid retrieval |
| `document_processor` | Text extraction from files |
| `query_enhancer` | Query rewriting and expansion |
| `agentic_pipeline` | Pluggable agentic RAG pipelines (e.g. `"default"`, `"legacy_react"`, custom) |

Additionally, `graph_node` callables are stored in a separate registry (not in `COMPONENT_TYPES`) to avoid accidental instantiation. Use `registry.register_graph_node()` and `registry.get_graph_node()` to manage them.

### KnowledgePipeline

Encapsulates a full retrieval pipeline as a reproducible, serializable object. Combined with `sweep()` and `RAGASEvaluator`, enables systematic experimentation across datasets and configurations.

---

## Installation

```bash
# Core (CPU, Milvus Lite)
pip install lasagna-rag

# With specific vector backends
pip install lasagna-rag[qdrant]
pip install lasagna-rag[pinecone]
pip install lasagna-rag[weaviate]
pip install lasagna-rag[pgvector]
pip install lasagna-rag[opensearch]

# With Docling document processing (tables, OCR, layout analysis)
pip install lasagna-rag[docling]

# With RAGAS evaluation
pip install lasagna-rag[eval]

# Everything
pip install lasagna-rag[all]
```

---

## Integration Surfaces

### Python SDK

The primary interface for application developers.

```python
from lasagna.sdk import KnowledgeClient, KnowledgeConfig

config = KnowledgeConfig(
    enable_reranking=True,
    enable_hybrid_search=True,
    enable_hierarchical=True,
    chunk_size=800,
)
async with KnowledgeClient(config=config) as client:
    await client.upload("data/report.pdf")
    results = await client.search_documents("revenue growth")
    context = await client.get_context("executive summary")
```

**Remote client** — connect to a deployed RAG server with no heavy dependencies (only `httpx`):

```python
from lasagna.sdk import KnowledgeClient

# Connect to a remote server (lightweight — no torch, no sentence-transformers)
client = KnowledgeClient.from_url("https://rag.myapp.com", api_key="sk-...")
await client.upload("report.pdf")
context = await client.get_context("quarterly revenue")
await client.close()
```

LangChain integration is also available:

```python
from lasagna.sdk.integrations import KnowledgeRetriever

retriever = KnowledgeRetriever(client, top_k=5)
docs = await retriever.ainvoke("quarterly earnings")
```

### REST API

FastAPI-based HTTP API for language-agnostic access.

```bash
# Start the API server
python -m lasagna.api.rest.app --host 0.0.0.0 --port 8000

# Upload
curl -X POST http://localhost:8000/api/v1/documents \
  -F "file=@report.pdf"

# Search
curl http://localhost:8000/api/v1/search?q=revenue+growth

# Health
curl http://localhost:8000/api/v1/health
```

### MCP Server

Model Context Protocol server for AI agent integration.

```bash
# SSE transport (Claude Desktop, Cursor, web clients)
python -m lasagna.mcp --host 0.0.0.0 --port 8005

# stdio transport (local MCP clients)
python -m lasagna.mcp --transport stdio
```

Any MCP client connects to `http://localhost:8005/sse` and gets access to 33 tools across these categories:

| Category | Tools |
| -------- | ----- |
| **Search** | `search` (deprecated aliases: `search_knowledge`, `search_with_context`) |
| **Documents** | `upload_document`, `list_documents`, `delete_document`, `get_document_info`, `get_document_chunks`, `reindex_document`, `clear_session` |
| **Agent Support** | `get_document_context` (adaptive document awareness for agent prompts) |
| **Profiles** | `list_profiles`, `get_active_profile`, `set_active_profile`, `get_profile_stats` |
| **Configuration** | `get_rag_config`, `patch_rag_config`, `save_rag_profile`, `delete_rag_profile`, `reset_rag_config` |
| **Focus** | `set_focused_documents`, `get_focused_documents` |
| **System** | `get_system_status`, `rebuild_collection` |

### MCPKnowledgeClient (Programmatic MCP Access)

For agents and applications that need programmatic access to the MCP server:

```python
from lasagna.sdk import MCPKnowledgeClient

async with MCPKnowledgeClient(url="http://localhost:8005/sse") as client:
    # Search with inline citations
    response = await client.search("revenue growth", "session-1")

    # Extract structured citation data from the response
    metadata = MCPKnowledgeClient.parse_search_metadata(response)
    citations = metadata.get("citations", [])

    # Get document awareness context for agent prompts
    doc_context = await client.get_document_context("session-1", rag_search_count=0)
```

The `search` tool returns formatted context for LLMs with a `<search_metadata>` JSON block appended. Use `MCPKnowledgeClient.parse_search_metadata()` to extract citations and chunks without a separate API call.

---

## Docker Deployment

```bash
cd src/lasagna/deploy

# Start with your preferred vector backend
VECTOR_BACKEND=pgvector make up

# Enable optional services (background processing, MCP, tracing)
COMPOSE_PROFILES=celery,mcp VECTOR_BACKEND=pgvector make up

# Verify
make health
```

The deployment system uses a **base compose + backend fragment** pattern — pick any of 7 vector backends with a single environment variable. The Dockerfile pre-downloads BGE-M3 and mxbai-rerank-large-v1 models at build time.

For production deployment including Kubernetes, Makefile reference, monitoring, security checklist, and stress testing: **[deploy/README.md](deploy/README.md)**

---

## File Storage Backends

Pluggable document storage for uploaded files. Three built-in backends cover every deployment scenario:

| Backend | Dependency | Services | Best For |
| ------- | ---------- | -------- | -------- |
| `local` (default) | None | Local filesystem | Dev, single-machine, teams |
| `s3` | `boto3` | AWS S3, MinIO, SeaweedFS, Ceph, R2, B2, Wasabi, DO Spaces, 15+ | Production, multi-container, SaaS |
| `fsspec` | `fsspec` | GCS, Azure Blob, FTP, SFTP, HDFS, 30+ | Non-S3 cloud providers |

Configuration follows the same priority chain as vector backends: Environment variable > Session config > TOML config > Default.

```bash
# Local (default, zero config)
export RAG_STORAGE_BACKEND=local

# S3-compatible (AWS, MinIO, R2, etc.)
export RAG_STORAGE_BACKEND=s3
export RAG_STORAGE_S3_BUCKET=my-rag-documents
export RAG_STORAGE_S3_ENDPOINT_URL=http://minio:9000   # omit for AWS S3

# Universal via fsspec (GCS, Azure, etc.)
export RAG_STORAGE_BACKEND=fsspec
export RAG_STORAGE_FSSPEC_PROTOCOL=gcs
export RAG_STORAGE_FSSPEC_BASE_PATH=my-bucket/rag-documents
```

Install optional dependencies: `pip install lasagna-rag[storage-s3]` or `pip install lasagna-rag[storage-all]`.

Custom backends can be registered via `register_storage_backend()` or entry points (`lasagna_storage_` prefix).

---

## Vector Backend Support (BYOVDB)

| Backend | Status | Native BM25 | Docker Deploy | E2E Tests |
| ------- | ------ | ----------- | ------------- | --------- |
| Milvus/Zilliz | Primary | Yes | `backends/milvus.yml` | 32/32 |
| PgVector | Supported | No | `backends/pgvector.yml` | 32/32 |
| Qdrant | Supported | No | `backends/qdrant.yml` | 31/32 |
| Weaviate | Supported | Yes | `backends/weaviate.yml` | 31/32 |
| Pinecone | Supported | No | `backends/pinecone.yml` | Cloud-only |
| OpenSearch | Supported | Yes (Lucene) | `backends/opensearch.yml` | 32/32 |

Per-session backend switching allows different sessions to use different vector stores. Connection resolution follows a priority chain: Environment variable > Session config > TOML config > Built-in fallback.

To point at your own database:

```bash
export RAG_VECTOR_BACKEND=qdrant
export QDRANT_URL=http://your-qdrant-host:6333
python -m lasagna.mcp
```

---

## Supported LLM Platforms (15+)

Auto-detected from environment variables -- first key found wins:

| Tier | Platform | Env Var | Default Model |
| ---- | -------- | ------- | ------------- |
| 1 | OpenAI | `OPENAI_API_KEY` | gpt-4o-mini |
| 1 | Anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4 |
| 1 | Azure OpenAI | `AZURE_OPENAI_API_KEY` | gpt-4o-mini |
| 1 | Google Gemini | `GOOGLE_API_KEY` | gemini-2.0-flash |
| 1 | Groq | `GROQ_API_KEY` | llama-3.3-70b-versatile |
| 1 | WatsonX | `WATSONX_API_KEY` | llama-4-maverick |
| 1 | Mistral | `MISTRAL_API_KEY` | mistral-large-latest |
| 1 | Cohere | `COHERE_API_KEY` | command-r-plus |
| 1 | NVIDIA NIM | `NVIDIA_API_KEY` | llama-3.1-70b-instruct |
| 2 | OpenRouter | `OPENROUTER_API_KEY` | claude-3.5-sonnet |
| 2 | Together | `TOGETHER_API_KEY` | Llama-3.3-70B |
| 2 | DeepSeek | `DEEPSEEK_API_KEY` | deepseek-chat |
| 2 | Fireworks | `FIREWORKS_API_KEY` | llama-v3p1-70b |
| 2 | Perplexity | `PERPLEXITY_API_KEY` | sonar |
| 2 | Ollama | `OLLAMA_BASE_URL` | llama3.2 (local) |

---

## Configuration

All features are controlled via `settings.lasagna.toml` and overridable via `RAG_*` environment variables:

```toml
# Key feature toggles
enable_hybrid_search = true     # Dense + BM25 fusion
enable_reranking = true         # Cross-encoder reranking
enable_hyde = false              # HyDE query expansion
enable_mmr = true                # MMR diversity
enable_ocr = false               # OCR for scanned PDFs
enable_hierarchical = true       # Parent/child chunk relationships

# Docling (default processor; falls back to legacy if not installed)
docling_enabled = true
docling_mode = "auto"
docling_ocr_enabled = true
docling_table_structure = true
docling_max_tokens = 512
```

Session-scoped overrides are supported via `KnowledgeConfigManager` -- users can modify settings at runtime without altering the config file.

### Environment Variables (12-Factor)

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `RAG_VECTOR_BACKEND` | `milvus` | Vector backend name |
| `RAG_DB_DIR` | `~/.rag_knowledge/dbs` | Database directory |
| `RAG_DATABASE_BACKEND` | `sqlite` | `sqlite` or `postgresql` |
| `RAG_DATABASE_DSN` | *(empty)* | PostgreSQL connection string |
| `RAG_CACHE_BACKEND` | `memory` | `memory` or `redis` |
| `RAG_REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `RAG_TENANT_ISOLATION` | `none` | `none` or `partition` |
| `RAG_QUOTA_ENABLED` | `false` | Per-tenant resource quotas |
| `RAG_RATE_LIMITING_ENABLED` | `false` | Adaptive rate limiting |
| `RAG_ENABLE_PROMETHEUS` | `false` | Prometheus metrics |
| `RAG_ENABLE_TRACING` | `false` | OpenTelemetry tracing |
| `RAG_CONNECTION_POOL_SIZE` | `1` | Vector backend connection pool |
| `RAG_EMBEDDING_MODEL` | `BAAI/bge-m3` | Embedding model name |
| `RAG_EMBEDDING_BACKEND` | `auto` (SDK) / `sentence-transformers` (Docker) | Embedding backend: `auto` detects from model name (BGE-M3 → FlagEmbedding if installed, else sentence-transformers), or force `sentence-transformers` / `flag-embedding` |
| `RAG_RERANKER_MODEL` | `mixedbread-ai/mxbai-rerank-large-v1` | Cross-encoder reranker model |
| `RAG_ENABLE_RERANKING` | `true` | Cross-encoder reranking |
| `RAG_ENABLE_HYBRID_SEARCH` | `true` | Dense + BM25 fusion |
| `RAG_ENABLE_CONTEXT_REORDERING` | `true` | Lost-in-the-middle reordering |
| `RAG_ENABLE_RAG_FUSION` | `false` | Multi-query RAG fusion |
| `RAG_ENABLE_CRAG` | `false` | Corrective RAG |
| `RAG_ENABLE_CONTEXTUAL_RETRIEVAL` | `false` | Contextual retrieval |
| `RAG_EMBED_MCP` | `false` | Embed MCP inside API process |
| `RAG_CELERY_ENABLED` | `false` | Enable background processing |
| `RAG_DOCLING_ENABLED` | `true` | Docling document processing (falls back to legacy if not installed) |
| `RAG_STORAGE_BACKEND` | `local` | File storage backend: `local`, `s3`, `fsspec` |
| `RAG_STORAGE_S3_BUCKET` | *(empty)* | S3 bucket name (required for `s3` backend) |
| `RAG_STORAGE_S3_ENDPOINT_URL` | *(empty)* | S3-compatible endpoint (MinIO, R2, etc.) |
| `RAG_STORAGE_S3_REGION` | `us-east-1` | AWS region |
| `RAG_STORAGE_FSSPEC_PROTOCOL` | *(empty)* | fsspec protocol (e.g. `gcs`, `az`) |
| `RAG_STORAGE_FSSPEC_BASE_PATH` | *(empty)* | fsspec base path (e.g. `my-bucket/rag`) |
| `RAG_SETTINGS_FILE` | *(empty)* | Path to alternative TOML config file |

Priority: Session overrides > Environment variables > TOML config > Defaults

---

## Multi-Tenancy

| Feature | Default (Single-User) | Production |
| ------- | --------------------- | ---------- |
| Tenant isolation | Metadata filter | Native partitions |
| Resource quotas | Disabled | Per-tenant limits |
| Rate limiting | Disabled | Adaptive, priority-tiered |
| Session config | In-memory | JSON file / DB persistence |

Enable with:

```bash
RAG_TENANT_ISOLATION=partition RAG_QUOTA_ENABLED=true RAG_RATE_LIMITING_ENABLED=true
```

---

## Testing

```bash
# Unit tests
python -m pytest tests/ -v

# Deployment tests (static, no Docker needed)
python -m pytest tests/test_unified_deployment.py -v -k "not Live"

# Deployment E2E (requires Docker)
VECTOR_BACKEND=pgvector make -C src/lasagna/deploy up
python -m pytest tests/test_unified_deployment.py -v -k "Live"

# Integration tests
python -m pytest tests/integration/knowledge/

# Full suite
python -m pytest tests/
```

---

## Troubleshooting

| Issue | Solution |
| ----- | -------- |
| No LLM provider configured | Set an API key env var (e.g. `OPENAI_API_KEY`) or call `configure_providers()` |
| Import errors | Run `python -c "from lasagna import *"` to verify exports |
| Vector backend connection | Check `settings.lasagna.toml` backend settings and connection string |
| Embedding dimension mismatch | Use `backend.recreate_collection_for_config(config)` to rebuild |
| Docling slow first document | First run downloads AI models (~1 GB). Set `DOCLING_ARTIFACTS_PATH` to pre-download |
| Docling GPU not detected | Install PyTorch with CUDA. Set `docling_mode = "production"` to force GPU |

---

## Contributing

Contributions are welcome. To get started:

1. **Fork and clone** the repository.
2. **Install dev dependencies**: `pip install -e ".[dev]"`.
3. **Run the test suite** before making changes: `python -m pytest tests/ -v`.
4. **Create a feature branch** from `main`.
5. **Follow existing patterns** -- the codebase uses the facade pattern (`RAGBackend`), mixin composition (`KnowledgeConfigManager`), and provider abstraction for decoupling.
6. **Adding a new vector backend**: implement `VectorStoreAdapter`, add one YAML fragment, one Kustomize overlay -- zero changes to existing files.
7. **Adding a new pipeline component**: extend the appropriate base class (`BaseEmbedder`, `BaseChunker`, `BaseReranker`, `BaseRetriever`, `BaseQueryEnhancer`, `BaseDocumentProcessor`), register it in the component registry, and add tests.
8. **Submit a PR** with tests covering your changes.

---
