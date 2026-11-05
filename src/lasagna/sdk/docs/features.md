# Features

## Table of Contents

1. [Agentic RAG (Multi-hop)](#agentic-rag-multi-hop)
2. [Processing Pipeline](#processing-pipeline)
3. [Retrieval Features](#retrieval-features)

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
    print(f"  Chunks: {hop.chunks_retrieved}")
    print(f"  Time: {hop.hop_time_ms:.0f}ms")

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
    refinement_model="gpt-6-luna",

    # HyDE: OFF, ALWAYS, FIRST_HOP, ADAPTIVE (default)
    hyde_strategy=HyDEStrategy.ADAPTIVE,

    # CRAG quality grading
    enable_crag=True,
    crag_web_search_fallback=False,

    # Query decomposition for complex questions
    enable_query_decomposition=True,
)
```

**Pipeline selection**: choose which agentic pipeline runs.


```python
config = KnowledgeConfig(
    agentic_pipeline="default",  # Built-in structured graph (13 nodes, CRAG, quality gates)
    # agentic_pipeline="legacy_react",  # ReAct agent; what runs when nothing selects a pipeline
    # agentic_pipeline="self_ask",  # Also ircot, flare, adaptive, self_rag, search_o1 (published loops)
    # agentic_pipeline="my_custom_rag",  # Any registered BaseAgenticPipeline
)
```

The `agentic_pipeline` field accepts any name registered in the `agentic_pipeline` namespace of the component registry (`lasagna pipelines list` shows them). When it is unset, the `use_structured_graph` boolean picks one of the two built-ins: `true` means `"default"`, `false` means `"legacy_react"`. When both are set, `agentic_pipeline` wins.

> For writing and registering custom pipelines, see the [BYOA Guide](custom_pipelines.md).

**More agentic settings** (in `KnowledgeConfig`, in `settings.lasagna.toml`, or as `LASAGNA_<NAME>` / `RAG_<NAME>` environment variables, resolved as [Where a setting comes from](../../../../docs/configuration.md#where-a-setting-comes-from) describes):

- `enable_hallucination_check`: Self-RAG ISSUP grounding check (structured graph only)
- `enable_answer_quality_check`: Self-RAG ISUSE answer relevance check (structured graph only)
- `use_final_search`: one more search on the original question after refinement, for complete citations (not a `KnowledgeConfig` field; default `false`)
- `coverage_scorer_mode`: how coverage is scored, `keyword` or `llm`

For full configuration details, see the [agentic RAG documentation](../../agentic/docs/config.md).

### Creating LangChain Tools

```python
from lasagna.sdk.agentic import create_rag_tools

# Create tools
tools = create_rag_tools(client)

# Available tools:
# - search(query) - Search knowledge base
# - list_documents() - List all documents
# - search_in_document(document_id, query) - Search specific document

# Use with LangChain agent
from langchain.agents import create_agent

agent = create_agent(llm, tools=tools, system_prompt="You are a helpful assistant.")
result = await agent.ainvoke({"messages": [("user", "What is the revenue?")]})
```

### Adaptive Chunk Retrieval

By default, the number of chunks returned is chosen per query with **Adaptive-k** (the score-gap method the code cites as EMNLP 2025) instead of a fixed number. The aim is to avoid both missing evidence (too few chunks) and padding the context with irrelevant ones (too many).

**How it works**: after cross-encoder reranking, the normalized relevance scores are sorted in descending order and the largest relative gap is found. Chunks above the gap are kept; chunks below it are dropped, except for a configurable buffer of extra chunks past the gap.

This applies to both regular search (via `client.search_documents()`) and agentic search (via `client.agentic_search()`).

```python
from lasagna.sdk import KnowledgeConfig, ChunkRetrievalStrategy

# Adaptive (default): chunk count chosen per query
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.ADAPTIVE,
    chunk_retrieval_max=20,     # Maximum chunks (ceiling)
    chunk_retrieval_min=2,      # Minimum chunks (floor)
    score_gap_buffer=2,         # Safety margin past the score gap
    min_gap_ratio=0.15,         # Minimum gap to trigger cutoff
)

# Fixed: always return the same number of chunks
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.FIXED,
    rerank_top_k=5,
)

# Above threshold: return all chunks above a score threshold
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.ABOVE_THRESHOLD,
    chunk_retrieval_threshold=0.5,
    chunk_retrieval_max=20,
)
```

**Preset behavior**:

| Preset | `rerank_top_k` | `chunk_retrieval_max` | Strategy |
| ------------- | -------------- | --------------------- | -------- |
| `speed` | 5 | 10 | fixed |
| `standard` | 5 | 15 | adaptive |
| `balanced` | 5 | 20 | adaptive |
| `max_quality` | 5 | 30 | adaptive |

### Hierarchical Chunking + Auto-Merge

Hierarchical chunking creates a **parent-child chunk structure**: small chunks are what search matches, and their larger parents are what the LLM can receive. The merge step is a port of LlamaIndex's `AutoMergingRetriever` algorithm (Apache 2.0; `api/_auto_merger.py`).

**How it works:**

1. **Upload**: Documents are split into large **parent chunks** (default 2000 chars) and smaller **child chunks** (default 500 chars). Only children are embedded and stored in the vector database. Parents are stored in the metadata database (SQLite, or PostgreSQL when `database_backend` is `postgresql`).

2. **Search**: the retrieval pipeline searches child chunks as usual (vector search, hybrid, reranking, MMR), so matching works on small pieces of text.

3. **Auto-Merge**: after retrieval, if enough child chunks from the same parent are in the results (ratio >= `auto_merge_threshold`), they are **replaced with the parent's full content**, scored as the best of those children.

**Why this matters**: fixed-size chunks often split related information across boundaries. A question about a topic might match 3 small chunks from the same section. Instead of passing 3 fragments to the LLM, auto-merge passes the parent section they came from.

```python
from lasagna.sdk import KnowledgeClient, KnowledgeConfig

# Enable hierarchical chunking with auto-merge (on in balanced and max_quality)
config = KnowledgeConfig(
    enable_hierarchical=True,
    parent_chunk_size=2000,      # Large context windows
    child_chunk_size=500,        # Fine-grained search targets
    auto_merge_threshold=0.5,    # Merge when ≥50% of siblings are retrieved
)

client = KnowledgeClient(config=config)

# Upload: creates the parent-child chunk hierarchy
await client.upload("./long_report.pdf")

# Search: auto-merge runs as part of the search
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
| `speed` | Disabled | N/A | Flat chunks |
| `standard` | Disabled | N/A | Flat chunks |
| `balanced` | Enabled | 0.5 | Parent-child with standard merging |
| `max_quality` | Enabled | 0.4 | More aggressive merging for max recall |

**Documents indexed without it**: documents uploaded before hierarchical chunking was turned on have no parent references and pass through auto-merge unchanged. Re-upload them (or call `client.documents.reindex(document_id)`) to give them parents.

---

## Processing Pipeline

### Document Extraction

```python
from lasagna.sdk.processing import (
    DocumentExtractor,
    SemanticChunker,
    OCRProcessor,
)

# Extract text from document (returns a string)
extractor = DocumentExtractor()
text = extractor.extract_text("./document.pdf")
print(f"Extracted {len(text)} characters")

# For scanned documents
from lasagna.sdk.processing import is_scanned_pdf

if is_scanned_pdf("./scanned.pdf"):
    ocr = OCRProcessor()
    text = ocr.process_pdf("./scanned.pdf")  # returns "" unless enable_ocr is on
```

### Chunking

```python
from lasagna.sdk.processing import SemanticChunker, FixedChunker

# Semantic chunking (recommended): needs the embedding model
from lasagna.sdk.retrieval import EmbeddingModel

chunker = SemanticChunker(embedding_manager=EmbeddingModel())
chunks = chunker.chunk_semantic(text, threshold=0.75, max_size=800)
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index}: {len(chunk.content)} chars")

# Or split by size (chunk_fixed does not use the model)
chunks = chunker.chunk_fixed(text, chunk_size=800, overlap=150)

# FixedChunker returns (text, metadata) tuples
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

# Generate embeddings (one numpy vector per text)
embeddings = model.encode_documents_batch(["Hello world", "Semantic search"])
print(f"{len(embeddings)} vectors of {embeddings[0].shape[0]} dims")  # 2 vectors of 1024 dims

# Queries have their own method
query_vector = model.encode_query("What is semantic search?")
```

### Reranking

```python
from lasagna.sdk.retrieval import Reranker

reranker = Reranker()

# Rerank results: a list of dicts with "content" (plus optional "chunk_id",
# "document_id" and "score"); returns RerankedResult objects
search_results = [
    {"chunk_id": "c1", "content": "Revenue grew 12% year over year.", "score": 0.61},
    {"chunk_id": "c2", "content": "The office moved to Boston.", "score": 0.58},
]
reranked = reranker.rerank(
    query="revenue growth",
    results=search_results,
    top_k=5,
)
for r in reranked:
    print(f"[{r.normalized_score:.2f}] {r.content}")
```

---

## Retrieval Features

Each feature below is a setting you can turn on or off on its own, and the built-in presets combine them (see [Configuration Profiles](configuration.md#configuration-profiles)). To find out whether one helps on your data, compare it in a sweep ([Ablation Studies](extending.md#ablation-studies)) rather than relying on published numbers.

### BGE-M3 Embedding

The default embedding model is **BAAI/bge-m3**, one model that produces dense, sparse and ColBERT (multi-vector) representations. It covers 100+ languages and inputs of up to 8K tokens.

**Backend selection**: with `embedding_backend="auto"`, a BGE-M3 model loads through `FlagEmbedding`'s `BGEM3FlagModel` when that library is installed (`pip install "lasagna-rag[flag-embedding]"`), which also gives sparse vectors. Without it, and for every other model, it loads through `sentence-transformers`, which gives dense vectors only (`infrastructure/embedding/manager.py`).

Each embedding model has its own vector collection, `rag_documents_v2__<model>__<dim>`, so switching models does not mix vectors of different sizes; documents indexed under one model must be uploaded again to be searched with another.

```python
from lasagna.sdk import KnowledgeConfig

# Default: BGE-M3 with auto-detected backend
config = KnowledgeConfig(
    embedding_model="BAAI/bge-m3",       # Default
    embedding_dim=1024,                    # BGE-M3 dimension
    embedding_backend="auto",             # FlagEmbedding for BGE-M3 when installed
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

**Environment variable**: `LASAGNA_EMBEDDING_MODEL=BAAI/bge-m3` (`RAG_EMBEDDING_MODEL` is also read; `LASAGNA_` wins when both are set)

### mxbai Reranker

The default reranker is the **mixedbread-ai/mxbai-rerank-large-v1** cross-encoder (its authors report BEIR NDCG@10 = 57.49). It is loaded with `sentence_transformers.CrossEncoder` and runs on the GPU (CUDA or MPS) when one is available.

The cross-encoder scores each query and candidate chunk together and reorders the candidates that first-stage retrieval returned.

```python
from lasagna.sdk import KnowledgeConfig

config = KnowledgeConfig(
    enable_reranking=True,                                    # Enable cross-encoder reranking
    reranker_model="mixedbread-ai/mxbai-rerank-large-v1",   # the default
    rerank_top_k=5,                                           # Final results after reranking
    reranker_batch_size=16,                                   # Batch size for scoring
)
```

**Pipeline**: retrieval (`default_top_k`=20 candidates) -> cross-encoder reranking -> Adaptive-k selection -> MMR diversity -> final results

**Configuration options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_reranking` | bool | True | Enable cross-encoder reranking |
| `reranker_model` | str | `mixedbread-ai/mxbai-rerank-large-v1` | Reranker model |
| `rerank_top_k` | int | 5 | Max results after reranking |
| `reranker_batch_size` | int | 16 | Scoring batch size |

### Contextual Retrieval

Implements **contextual retrieval** as Anthropic describes it ([reference](https://www.anthropic.com/news/contextual-retrieval)). At index time, an LLM writes a short note for each chunk on how it fits within the whole document.

When enabled, the upload pipeline calls an LLM for each chunk to generate a short contextual preamble (e.g., "This chunk describes Q3 revenue figures from the 2024 annual report"). The preamble is prepended to the chunk before embedding.

```python
config = KnowledgeConfig(
    enable_contextual_retrieval=True,   # Enrich chunks at index time
)
```

**Trade-offs:**

- Can improve recall and precision; measure it on your corpus
- Increases indexing time and cost (one LLM call per chunk)
- No effect on search latency (enrichment happens at upload time)

**Environment variable**: `LASAGNA_ENABLE_CONTEXTUAL_RETRIEVAL=true` (or `RAG_ENABLE_CONTEXTUAL_RETRIEVAL`)

### RAG Fusion

Implements **RAG Fusion** ([arXiv:2402.03367](https://arxiv.org/abs/2402.03367)): an LLM writes several variations of the query, each is searched, and the result lists are merged with **Reciprocal Rank Fusion (RRF)**.

When enabled, the search pipeline generates `rag_fusion_n_queries` alternative phrasings of the query, runs a search for each, and fuses the result lists with RRF. It is skipped for searches limited to specific documents.

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

**Environment variable**: `LASAGNA_ENABLE_RAG_FUSION=true` (or `RAG_ENABLE_RAG_FUSION`)

### Adaptive-K (Score-Gap Detection)

**Adaptive-k** picks how many chunks to return for each query instead of using a fixed number (the method and its settings are described under [Adaptive Chunk Retrieval](#adaptive-chunk-retrieval)).

**How it works**: after cross-encoder reranking, the normalized relevance scores are sorted in descending order and the largest relative gap is found. Chunks above the gap are kept, plus a configurable buffer past it.

Adaptive-k is the `KnowledgeConfig` default and is on in every preset except `speed`, which returns a fixed number of chunks.

```python
from lasagna.sdk import KnowledgeConfig, ChunkRetrievalStrategy

# Adaptive (default): chunk count chosen per query
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.ADAPTIVE,
    chunk_retrieval_max=20,      # Maximum chunks (ceiling)
    chunk_retrieval_min=2,       # Minimum chunks (floor)
    score_gap_buffer=2,          # Safety margin past the score gap
    min_gap_ratio=0.15,          # Minimum relative gap to trigger cutoff
)

# Fixed: always return the same number of chunks
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.FIXED,
    rerank_top_k=5,
)

# Above threshold: return all chunks above a score threshold
config = KnowledgeConfig(
    chunk_retrieval_strategy=ChunkRetrievalStrategy.ABOVE_THRESHOLD,
    chunk_retrieval_threshold=0.5,
    chunk_retrieval_max=20,
)
```

### Lost-in-the-Middle (Context Reordering)

Addresses the **"Lost in the Middle"** problem ([Liu et al., 2023](https://arxiv.org/abs/2307.03172)) where LLMs tend to ignore information placed in the middle of long contexts. When enabled, retrieved chunks are reordered so that the most relevant results appear at the **start and end** of the context window, with less relevant results in the middle.

This is **on by default** (`enable_context_reordering=True`).

```python
config = KnowledgeConfig(
    enable_context_reordering=True,   # Place best results at start/end
)
```

**Environment variable**: `LASAGNA_ENABLE_CONTEXT_REORDERING=true` (or `RAG_ENABLE_CONTEXT_REORDERING`)

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

**Environment variables**: `LASAGNA_ENABLE_CRAG=true`, `LASAGNA_CRAG_WEB_SEARCH_FALLBACK=true` (the `RAG_` names are also read). The web search fallback needs `pip install "lasagna-rag[web]"`.

### Feature Summary

| Feature | Config Key | Default | Extra Work |
|---------|-----------|---------|------------|
| BGE-M3 Embedding | `embedding_model` | `BAAI/bge-m3` | Baseline |
| mxbai Reranker | `enable_reranking` | True | A cross-encoder pass over the candidates |
| Contextual Retrieval | `enable_contextual_retrieval` | False | An LLM call per chunk, at index time only |
| RAG Fusion | `enable_rag_fusion` | False | An LLM call plus one search per query variant |
| Adaptive-k | `chunk_retrieval_strategy` | ADAPTIVE | Arithmetic on the reranked scores |
| Lost-in-the-Middle | `enable_context_reordering` | True | Reordering only |
| Corrective RAG | `enable_crag` | False | LLM grading of the retrieved chunks |
| HyDE | `enable_hyde` | False | An LLM call per query |
| Hybrid Search (BM25) | `enable_hybrid_search` | True | A BM25 search fused with the vector search |
| MMR Diversity | `enable_mmr` | True | Similarity comparisons among the candidates |

---

## See Also

- [User Guide](user_guide.md)
- [Configuration](configuration.md)
- [Extending](extending.md)
