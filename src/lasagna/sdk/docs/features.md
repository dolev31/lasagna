# Features

## Table of Contents

1. [Agentic RAG (Multi-hop)](#agentic-rag-multi-hop)
2. [Processing Pipeline](#processing-pipeline)
3. [SOTA Retrieval Features](#sota-retrieval-features)

---

## Agentic RAG (Multi-hop)

Multi-hop retrieval with query refinement for complex questions:

```python
# Perform agentic search
results = await client.agentic_search(
    "What were the key factors affecting Q3 performance and how do they relate to market conditions?",
    max_iterations=5,
)

print(f"Completed in {results.total_iterations} hops")
print(f"Retrieved {results.total_chunks_retrieved} unique chunks")
print(f"Time: {results.search_time_ms:.0f}ms")

# Access hop data
for hop in results.hops:
    print(f"Hop {hop.iteration}:")
    print(f"  Query: {hop.query}")
    print(f"  Refined: {hop.refined_query}")
    print(f"  Coverage: {hop.coverage_score:.2f}")

# Get final context
print(results.final_context)

# Access citations
for citation in results.citations:
    print(f"[{citation.citation_id}] {citation.filename} p.{citation.page_number}")
```

### SDK-Configurable Features

The following agentic features can be set via `KnowledgeConfig`:

```python
from lasagna.sdk import KnowledgeConfig, HyDEStrategy

config = KnowledgeConfig(
    # Agentic RAG
    use_agentic_rag=True,
    max_agent_iterations=5,

    # Chunk refinement: "vanilla" (passthrough), "refined" (LLM), "auto" (heuristic gate)
    chunk_refinement_method="auto",
    refinement_model="gpt-4o-mini",

    # HyDE: OFF, ALWAYS, FIRST_HOP, ADAPTIVE (default)
    hyde_strategy=HyDEStrategy.ADAPTIVE,

    # CRAG quality grading
    enable_crag=True,
    crag_web_search_fallback=False,

    # Query decomposition for complex questions
    enable_query_decomposition=True,
)
```

**Pipeline selection** -- choose which agentic pipeline to run:

```python
config = KnowledgeConfig(
    agentic_pipeline="default",  # Built-in structured graph (13 nodes, CRAG, quality gates)
    # agentic_pipeline="legacy_react",  # Legacy ReAct agent (default)
    # agentic_pipeline="my_custom_rag",  # Any registered BaseAgenticPipeline
)
```

The `agentic_pipeline` field accepts any name registered in the `agentic_pipeline` namespace of the component registry. The legacy `use_structured_graph` boolean is still supported as a convenience: `true` maps to `"default"`, `false` maps to `"legacy_react"`. When both are set, `agentic_pipeline` takes precedence.

> For writing and registering custom pipelines, see the [BYOA Guide](custom_pipelines.md).

**Additional backend-configurable features** (also settable via `settings.lasagna.toml` or environment variables):

- `enable_hallucination_check` -- Self-RAG ISSUP grounding verification (structured graph only)
- `enable_answer_quality_check` -- Self-RAG ISUSE answer relevance check (structured graph only)
- `use_final_search` -- Final comprehensive search on original question
- `coverage_scorer_mode` -- Coverage scoring method (`keyword` or `llm`)

For full configuration details, see the [agentic RAG documentation](../../agentic/docs/config.md).

### Creating LangChain Tools

```python
from lasagna.sdk.agentic import create_rag_tools

# Create tools
tools = create_rag_tools(client)

# Available tools:
# - search_knowledge(query) - Search knowledge base
# - list_documents() - List all documents
# - search_in_document(document_id, query) - Search specific document

# Use with LangChain agent
from langchain.agents import create_agent

agent = create_agent(llm, tools=tools, system_prompt="You are a helpful assistant.")
result = await agent.ainvoke({"messages": [("user", "What is the revenue?")]})
```

### Adaptive Chunk Retrieval

By default, the system uses **Adaptive-k** (based on EMNLP 2025 research) to dynamically determine how many chunks to return instead of a fixed number. This prevents both information loss (too few chunks) and hallucination (too many irrelevant chunks).

**How it works**: After cross-encoder reranking, the system sorts relevance scores in descending order and finds the largest relative gap in the score distribution. Chunks above the gap are "clearly relevant"; chunks below are not. A configurable buffer is added past the gap as a safety margin.

This applies to both regular search (via `client.search_documents()`) and agentic search (via `client.agentic_search()`).

```python
from lasagna.sdk import KnowledgeConfig, ChunkRetrievalStrategy

# Adaptive (default) — dynamically determines chunk count per query
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.ADAPTIVE,
    chunk_retrieval_max=20,     # Maximum chunks (ceiling)
    chunk_retrieval_min=2,      # Minimum chunks (floor)
    score_gap_buffer=2,         # Safety margin past the score gap
    min_gap_ratio=0.15,         # Minimum gap to trigger cutoff
)

# Fixed — always return exactly N chunks (legacy behavior)
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.FIXED,
    rerank_top_k=5,
)

# Above threshold — return all chunks above a score threshold
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.ABOVE_THRESHOLD,
    chunk_retrieval_threshold=0.5,
    chunk_retrieval_max=20,
)
```

**Preset behavior**:

| Preset | `rerank_top_k` | `chunk_retrieval_max` | Strategy |
| ------------- | -------------- | --------------------- | -------- |
| `speed` | 5 | 10 | adaptive |
| `standard` | 10 | 15 | adaptive |
| `balanced` | 10 | 20 | adaptive |
| `max_quality` | 20 | 30 | adaptive |

### Hierarchical Chunking + Auto-Merge

Hierarchical chunking creates a **parent-child chunk structure** that combines fine-grained search accuracy with broad context for LLM responses. This is ported from LlamaIndex's `AutoMergingRetriever` (Apache 2.0).

**How it works:**

1. **Upload**: Documents are split into large **parent chunks** (default 2000 chars) and smaller **child chunks** (default 500 chars). Only children are embedded and stored in the vector database. Parents are stored in SQLite.

2. **Search**: The retrieval pipeline searches child chunks as usual (vector search + hybrid + reranking + MMR). This gives **fine-grained matching accuracy**.

3. **Auto-Merge**: After retrieval, if enough child chunks from the same parent are in the results (ratio >= `auto_merge_threshold`), they are **replaced with the parent's full content**. This gives the LLM **richer context** without losing relevance.

**Why this matters**: Fixed-size chunks often split important information across boundaries. A question about a topic might match 3 small chunks from the same section. Instead of feeding 3 small fragments to the LLM, auto-merge replaces them with the full parent section — giving the LLM the complete picture.

```python
from lasagna.sdk import KnowledgeConfig

# Enable hierarchical chunking with auto-merge (default in balanced/max_quality)
config = KnowledgeConfig(
    enable_hierarchical=True,
    parent_chunk_size=2000,      # Large context windows
    child_chunk_size=500,        # Fine-grained search targets
    auto_merge_threshold=0.5,    # Merge when ≥50% of siblings are retrieved
)

client = KnowledgeClient(config=config)

# Upload — creates parent-child chunk hierarchy automatically
await client.upload("./long_report.pdf")

# Search — auto-merge happens transparently
results = await client.search_documents("quarterly performance analysis")

# Results may contain auto-merged chunks with richer content
for r in results:
    if r.chunk.metadata.get("auto_merged"):
        print(f"  [MERGED from {r.chunk.metadata['merged_child_count']} children]")
    print(f"  Score: {r.score:.2f} | {r.chunk.content[:100]}...")
```

**Configuration options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_hierarchical` | bool | True | Enable parent-child chunking |
| `parent_chunk_size` | int | 2000 | Characters per parent chunk |
| `child_chunk_size` | int | 500 | Characters per child chunk |
| `auto_merge_threshold` | float | 0.5 | Ratio of retrieved children / total siblings to trigger merge |

**Preset behavior:**

| Preset | Hierarchical | Auto-Merge Threshold | Effect |
|--------|-------------|---------------------|--------|
| `speed` | Disabled | N/A | Pure flat chunks for minimum latency |
| `standard` | Disabled | N/A | Flat chunks with lightweight model |
| `balanced` | Enabled | 0.5 | Parent-child with standard merging |
| `max_quality` | Enabled | 0.4 | More aggressive merging for max recall |

**Backward compatibility**: Documents uploaded before enabling hierarchical chunking work normally — they have no parent references and pass through auto-merge unchanged. Re-upload documents to benefit from hierarchical chunking.

---

## Processing Pipeline

### Document Extraction

```python
from lasagna.sdk.processing import (
    DocumentExtractor,
    SemanticChunker,
    OCRProcessor,
)

# Extract text from document
extractor = DocumentExtractor()
text, metadata = extractor.extract("./document.pdf")
print(f"Extracted {len(text)} characters")

# For scanned documents
from lasagna.sdk.processing import is_scanned_pdf

if is_scanned_pdf("./scanned.pdf"):
    ocr = OCRProcessor()
    text = ocr.process("./scanned.pdf")
```

### Chunking

```python
from lasagna.sdk.processing import SemanticChunker, FixedChunker

# Semantic chunking (recommended)
chunker = SemanticChunker(
    chunk_size=800,
    overlap=150,
    similarity_threshold=0.75,
)
chunks = chunker.chunk(text)

# Fixed-size chunking
chunker = FixedChunker(chunk_size=500, overlap=50)
chunks = chunker.chunk(text)

for chunk_text, metadata in chunks:
    print(f"Chunk {metadata['chunk_index']}: {len(chunk_text)} chars")
```

### Embeddings

```python
from lasagna.sdk.retrieval import EmbeddingModel

# Create embedding model
model = EmbeddingModel()

# Generate embeddings
embeddings = model.encode(["Hello world", "Semantic search"])
print(f"Shape: {embeddings.shape}")  # (2, 1024)
```

### Reranking

```python
from lasagna.sdk.retrieval import Reranker

reranker = Reranker()

# Rerank results
reranked = reranker.rerank(
    query="revenue growth",
    results=search_results,
    top_k=5,
)
```

---

## SOTA Retrieval Features

The Knowledge SDK implements a comprehensive set of state-of-the-art retrieval techniques. Each feature can be toggled independently via configuration, and the built-in presets combine them into battle-tested configurations.

### BGE-M3 Embedding

The default embedding model is **BAAI/bge-m3**, a SOTA multi-function embedding model that supports dense, sparse, and ColBERT retrieval in a single model. It supports 100+ languages and an 8K-token context window.

**Auto-detection**: The SDK auto-detects the best available embedding backend at startup. When the `FlagEmbedding` library is installed, it uses the native `BGEM3FlagModel` for optimal performance (including native sparse vector generation). Otherwise, it falls back to `sentence-transformers`, which provides dense embeddings only.

```python
from lasagna.sdk import KnowledgeConfig

# Default: BGE-M3 with auto-detected backend
config = KnowledgeConfig(
    embedding_model="BAAI/bge-m3",       # Default
    embedding_dim=1024,                    # BGE-M3 dimension
    embedding_backend="auto",             # Auto-detect best backend
)

# Force a specific backend
config = KnowledgeConfig(
    embedding_backend="flag-embedding",   # Use FlagEmbedding (native BGE-M3)
)

# Or use sentence-transformers explicitly
config = KnowledgeConfig(
    embedding_backend="sentence-transformers",
)

# Fallback model (used if BGE-M3 fails to load)
config = KnowledgeConfig(
    embedding_model="BAAI/bge-m3",
    embedding_model_fallback="BAAI/bge-large-en-v1.5",  # 1024d fallback
)
```

**Configuration options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `embedding_model` | str | `BAAI/bge-m3` | HuggingFace model name |
| `embedding_model_fallback` | str | `BAAI/bge-large-en-v1.5` | Fallback if primary fails |
| `embedding_backend` | str | `auto` | `auto`, `sentence-transformers`, or `flag-embedding` |
| `embedding_dim` | int | 1024 | Vector dimension (must match model) |
| `normalize_embeddings` | bool | True | L2-normalize vectors for cosine similarity |
| `use_gpu` | bool | True | Use GPU (CUDA/MPS) if available |

**Environment variable**: `RAG_EMBEDDING_MODEL=BAAI/bge-m3`

### mxbai Reranker

The SDK uses the **mixedbread-ai/mxbai-rerank-large-v1** cross-encoder reranker, which achieves **BEIR NDCG@10 = 57.49** -- among the best open-source rerankers available. It is loaded via `sentence-transformers.CrossEncoder` and runs on GPU (CUDA/MPS) when available.

The reranking pipeline applies cross-encoder scoring to refine initial retrieval results, significantly improving precision compared to embedding-only retrieval.

```python
from lasagna.sdk import KnowledgeConfig

config = KnowledgeConfig(
    enable_reranking=True,                                    # Enable cross-encoder reranking
    reranker_model="mixedbread-ai/mxbai-rerank-large-v1",   # BEIR NDCG@10 = 57.49
    rerank_top_k=5,                                           # Final results after reranking
    reranker_batch_size=16,                                   # Batch size for scoring
)
```

**Pipeline**: Dense retrieval (top_k=20) -> Cross-encoder reranking -> Adaptive-K selection -> MMR diversity -> Final results

**Configuration options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_reranking` | bool | True | Enable cross-encoder reranking |
| `reranker_model` | str | `mixedbread-ai/mxbai-rerank-large-v1` | Reranker model |
| `rerank_top_k` | int | 5 | Max results after reranking |
| `reranker_batch_size` | int | 16 | Scoring batch size |

### Contextual Retrieval

Implements **Anthropic-style contextual retrieval** ([reference](https://www.anthropic.com/news/contextual-retrieval)). At index time, each chunk is enriched with LLM-generated context that describes how it fits within the broader document. This reduces retrieval failure rates by up to 67%.

When enabled, the upload pipeline calls an LLM for each chunk to generate a short contextual preamble (e.g., "This chunk describes Q3 revenue figures from the 2024 annual report"). The preamble is prepended to the chunk before embedding.

```python
config = KnowledgeConfig(
    enable_contextual_retrieval=True,   # Enrich chunks at index time
)
```

**Trade-offs:**

- Significantly improves retrieval recall and precision
- Increases indexing time and cost (one LLM call per chunk)
- No impact on search latency (enrichment happens at upload time)

**Environment variable**: `RAG_ENABLE_CONTEXTUAL_RETRIEVAL=true`

### RAG Fusion

Implements **RAG Fusion** ([arXiv:2402.03367](https://arxiv.org/abs/2402.03367)): generates multiple query variations using an LLM and merges the results using **Reciprocal Rank Fusion (RRF)**. This improves comprehensiveness by 30-40% and accuracy by 8-10%.

When enabled, the search pipeline generates N alternative phrasings of the query (each approaching the topic from a different angle), runs separate searches for each, and fuses the result lists using RRF scoring.

```python
config = KnowledgeConfig(
    enable_rag_fusion=True,       # Enable multi-query + RRF
    rag_fusion_n_queries=3,       # Number of query variations to generate
)
```

**Configuration options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_rag_fusion` | bool | False | Enable RAG Fusion |
| `rag_fusion_n_queries` | int | 3 | Number of alternative queries |
| `rrf_k` | int | 60 | RRF constant (higher = more weight to lower ranks) |

**Environment variable**: `RAG_ENABLE_RAG_FUSION=true`

### Adaptive-K (Score-Gap Detection)

Based on EMNLP 2025 research, **Adaptive-K** dynamically determines how many chunks to return instead of using a fixed number. This prevents both information loss (too few chunks) and hallucination from irrelevant context (too many chunks).

**How it works**: After cross-encoder reranking, normalized relevance scores are sorted in descending order. The system identifies the largest relative gap in the score distribution. Chunks above the gap are "clearly relevant"; chunks below are not. A configurable buffer is added past the gap as a safety margin.

Adaptive-K is **enabled by default** in all presets.

```python
from lasagna.sdk import KnowledgeConfig, ChunkRetrievalStrategy

# Adaptive (default) -- dynamically determines chunk count per query
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.ADAPTIVE,
    chunk_retrieval_max=20,      # Maximum chunks (ceiling)
    chunk_retrieval_min=2,       # Minimum chunks (floor)
    score_gap_buffer=2,          # Safety margin past the score gap
    min_gap_ratio=0.15,          # Minimum relative gap to trigger cutoff
)

# Fixed -- always return exactly N chunks (legacy behavior)
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.FIXED,
    rerank_top_k=5,
)

# Above threshold -- return all chunks above a score threshold
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.ABOVE_THRESHOLD,
    chunk_retrieval_threshold=0.5,
    chunk_retrieval_max=20,
)
```

### Lost-in-the-Middle (Context Reordering)

Addresses the **"Lost in the Middle"** problem ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)) where LLMs tend to ignore information placed in the middle of long contexts. When enabled, retrieved chunks are reordered so that the most relevant results appear at the **start and end** of the context window, with less relevant results in the middle.

This is **enabled by default**.

```python
config = KnowledgeConfig(
    enable_context_reordering=True,   # Place best results at start/end
)
```

**Environment variable**: `RAG_ENABLE_CONTEXT_REORDERING=true`

### Corrective RAG (CRAG)

Implements **Corrective RAG** ([Yan et al., 2024](https://arxiv.org/abs/2401.15884)): after retrieval, an LLM grades the quality of retrieved chunks. If the retrieval quality is judged to be poor, the query is refined. Optionally, the system can fall back to web search to supplement results.

```python
config = KnowledgeConfig(
    enable_crag=True,                    # Enable quality grading
    crag_web_search_fallback=False,      # Optional: fall back to web search
)
```

**Configuration options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_crag` | bool | False | Enable Corrective RAG |
| `crag_web_search_fallback` | bool | False | Fall back to web search if quality is low |

**Environment variables**: `RAG_ENABLE_CRAG=true`, `RAG_CRAG_WEB_SEARCH_FALLBACK=true`

### Feature Summary

| Feature | Config Key | Default | Latency Impact |
|---------|-----------|---------|----------------|
| BGE-M3 Embedding | `embedding_model` | `BAAI/bge-m3` | Baseline |
| mxbai Reranker | `enable_reranking` | True | +50-100ms |
| Contextual Retrieval | `enable_contextual_retrieval` | False | Index-time only |
| RAG Fusion | `enable_rag_fusion` | False | +200-500ms (LLM call) |
| Adaptive-K | `chunk_retrieval_strategy` | ADAPTIVE | Negligible |
| Lost-in-the-Middle | `enable_context_reordering` | True | Negligible |
| Corrective RAG | `enable_crag` | False | +300-800ms (LLM call) |
| HyDE | `enable_hyde` | True | +200-500ms (LLM call) |
| Hybrid Search (BM25) | `enable_hybrid_search` | True | +10-30ms |
| MMR Diversity | `enable_mmr` | True | +20-50ms |

---

## See Also

- [User Guide](user_guide.md)
- [Configuration](configuration.md)
- [Extending](extending.md)
