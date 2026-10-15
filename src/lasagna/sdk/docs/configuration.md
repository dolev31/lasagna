# Configuration

## Table of Contents

1. [Configuration](#configuration)
2. [Retry and Resilience](#retry-and-resilience)
3. [Advanced Usage](#advanced-usage)
4. [Configuration Profiles](#configuration-profiles)

---

## Configuration

### KnowledgeConfig Overview

```python
from lasagna.sdk import KnowledgeConfig, ChunkingStrategy

config = KnowledgeConfig(
    # Feature Toggles
    enable_hybrid_search=True,      # Vector + BM25
    enable_reranking=True,          # Cross-encoder reranking
    enable_mmr=True,                # Diversity via MMR
    enable_hyde=False,              # Hypothetical doc embeddings
    enable_ocr=False,               # OCR for scanned docs

    # Chunking
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=800,
    chunk_overlap=150,

    # Search
    default_top_k=50,
    rerank_top_k=10,
    similarity_threshold=0.3,
    hybrid_alpha=0.7,

    # Embedding
    embedding_model="BAAI/bge-large-en-v1.5",
    embedding_dim=1024,

    # Agentic RAG
    use_agentic_rag=True,
    max_agent_iterations=3,
)

client = KnowledgeClient(config=config)
```

### Configuration Presets

```python
from lasagna.sdk import (
    config_for_speed,
    config_for_standard,
    config_for_balanced,
    config_for_max_quality,
    config_for_standalone,
)

# Speed - Pure vector search, no expensive processing
client = KnowledgeClient(config=config_for_speed())
# - Disables reranking, HyDE, MMR, agentic RAG
# - Uses MiniLM-L6-v2 (384d, 80MB)
# - Fixed chunking, ~50-100ms latency

# Standard (Recommended Starter) - SOTA features, lightweight model
client = KnowledgeClient(config=config_for_standard())
# - Reranking, hybrid search, MMR, agentic RAG enabled
# - Uses MiniLM-L6-v2 (384d, 80MB) — no large downloads needed
# - Semantic chunking, ~200-400ms latency

# Balanced - SOTA features, best embedding model
client = KnowledgeClient(config=config_for_balanced())
# - Reranking, hybrid search, MMR, agentic RAG enabled
# - Uses BGE-large-en-v1.5 (1024d, 1.3GB download)
# - Hierarchical chunking + auto-merge (threshold 0.5)
# - Semantic chunking, ~300-600ms latency

# Max Quality - All features for maximum recall
client = KnowledgeClient(config=config_for_max_quality())
# - All features: HyDE, query expansion, reranking, MMR
# - Uses BGE-large-en-v1.5 (1024d, 1.3GB)
# - Hierarchical chunking + aggressive auto-merge (threshold 0.4)
# - Larger candidate pool (100), adaptive up to 30 results, ~1-3s latency

# Standalone - custom paths (default: ~/.rag_knowledge/)
config = config_for_standalone(
    db_path="./my_data/vectors.db",
    upload_dir="./my_data/uploads",
)
client = KnowledgeClient(config=config)
```

### Preset Comparison

| Preset | Embedding Model | Reranking | Hybrid | MMR | Agentic | HyDE | Hierarchical | Latency |
|--------|----------------|-----------|--------|-----|---------|------|-------------|---------|
| `speed` | MiniLM-L6 (384d) | No | No | No | No | No | No | ~50-100ms |
| `standard` | MiniLM-L6 (384d) | Yes | Yes | Yes | Yes | No | No | ~200-400ms |
| `balanced` | BGE-large (1024d) | Yes | Yes | Yes | Yes | No | Yes (0.5) | ~300-600ms |
| `max_quality` | BGE-large (1024d) | Yes | Yes | Yes | Yes | Yes | Yes (0.4) | ~1-3s |

### Dynamic Configuration

```python
# Update configuration
await client.set_config(
    enable_hyde=True,
    chunk_size=1000,
)

# Use preset
client.use_preset("standard")     # SOTA + lightweight model
client.use_preset("balanced")      # SOTA + best model
client.use_preset("max_quality")   # All features enabled

# Get current config
config_dict = client.get_config_dict()
print(config_dict)
```

### Config Bridge Architecture

SDK configuration is automatically bridged to the backend via a two-tier system:

**Init-time fields** (e.g., `vector_backend`, `database_backend`, `cache_backend`, `celery_enabled`) are applied to the global backend config **before** the backend singleton is created. These must be set at `KnowledgeClient` creation time — changing them after initialization logs a warning.

**Runtime fields** (e.g., `enable_hyde`, `enable_reranking`, `chunk_size`, `embedding_model`) are applied per-session via the session config provider. They can be changed via `set_config()` or `use_preset()` and take effect on the next operation. If called before the backend is initialized, overrides are applied automatically when the backend is first created.

**Dirty-field tracking**: Only fields whose value differs from the SDK dataclass default are forwarded. A default `KnowledgeConfig()` writes zero overrides, so profile/TOML settings take full effect.

```python
# Init-time fields must be set at creation
client = KnowledgeClient(
    config=KnowledgeConfig(database_backend="postgresql", database_dsn="...")
)

# Runtime fields can change anytime
await client.set_config(enable_hyde=True)   # Applied immediately
await client.set_config(enable_hyde=False)  # Reverts (removes override)

# Switching sessions re-applies overrides
client.session_id = "new_session"  # New session gets current config
```

**Remote mode** (`from_url()`): The config bridge only affects local SDK behavior (citation format, presets). Search pipeline config is managed by the remote server — `set_config()` logs a warning in this mode.

### Profile Management

Profiles persist across sessions. Save custom configurations and switch between them. Profile operations live on `client.config_manager`.

```python
config_manager = client.config_manager

# List all profiles (built-in + custom)
profiles = await config_manager.list_profiles()
for p in profiles:
    print(f"{p['name']}: {p['description']}")
# Output:
#   default: Default configuration from settings.lasagna.toml
#   speed: Fast, pure vector search - no reranking/MMR/HyDE
#   standard: SOTA features with BGE-M3 embedding model (recommended starter)
#   balanced: SOTA features with BGE-M3, hierarchical chunking, structured graph
#   max_quality: All features enabled - HyDE, query expansion, max recall

# Get active profile
active = await config_manager.get_active_profile()
print(f"Active: {active}")  # "default"

# Save current config as a custom profile
await config_manager.save_profile("my_research_profile")

# Save a specific config as a custom profile
await config_manager.save_profile(
    "deep_analysis",
    config={
        "enable_hyde": True,
        "max_agent_iterations": 5,
        "default_top_k": 80,
    },
    set_active=True,  # Activate immediately
)

# Switch to a saved profile
await config_manager.set_profile("balanced")

# Delete a custom profile (built-in profiles cannot be deleted)
await config_manager.delete_profile("deep_analysis")

# Reset the client's config to the standard preset. The scope argument
# ("session" or "all") is accepted, but both currently do the same thing.
await config_manager.reset()
```

### All Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| **Storage** | | | |
| `db_path` | str | auto | Path to vector database |
| `upload_dir` | str | auto | Upload directory |
| **Features** | | | |
| `enable_hybrid_search` | bool | True | Enable hybrid search (vector + BM25) |
| `enable_reranking` | bool | True | Enable cross-encoder reranking |
| `enable_mmr` | bool | True | Enable MMR diversity |
| `enable_hyde` | bool | False | Enable HyDE query expansion |
| `enable_ocr` | bool | False | Enable OCR for scanned PDFs |
| **Chunking** | | | |
| `chunking_strategy` | enum | SEMANTIC | Chunking strategy |
| `chunk_size` | int | 800 | Target chunk size |
| `chunk_overlap` | int | 150 | Overlap between chunks |
| `semantic_threshold` | float | 0.75 | Semantic similarity threshold |
| `enable_hierarchical` | bool | True | Enable parent-child chunking with auto-merge |
| `parent_chunk_size` | int | 2000 | Characters per parent chunk |
| `child_chunk_size` | int | 500 | Characters per child chunk |
| `auto_merge_threshold` | float | 0.5 | Ratio of retrieved children / total siblings to trigger merge |
| **Search** | | | |
| `default_top_k` | int | 50 | Candidates to retrieve before reranking |
| `rerank_top_k` | int | 10 | Max results after reranking (ceiling for adaptive mode) |
| `similarity_threshold` | float | 0.3 | Minimum score threshold |
| `hybrid_alpha` | float | 0.7 | Vector vs keyword weight (0=keyword, 1=vector) |
| `mmr_lambda` | float | 0.5 | MMR relevance vs diversity (0=diverse, 1=relevant) |
| **Adaptive Chunk Retrieval** | | | |
| `chunk_retrieval_strategy` | enum | ADAPTIVE | How many chunks to return: FIXED, ABOVE_THRESHOLD, ADAPTIVE, AGENT_DECIDES |
| `chunk_retrieval_max` | int | 20 | Maximum chunks for adaptive strategy (ceiling) |
| `chunk_retrieval_min` | int | 2 | Minimum chunks for adaptive strategy (floor) |
| `chunk_retrieval_threshold` | float | 0.5 | Score threshold for ABOVE_THRESHOLD strategy |
| `score_gap_buffer` | int | 2 | Extra chunks past the score gap as safety margin |
| `min_gap_ratio` | float | 0.15 | Minimum relative gap to trigger cutoff |
| **Embedding** | | | |
| `embedding_model` | str | BAAI/bge-large-en-v1.5 | Embedding model name |
| `embedding_dim` | int | 1024 | Vector dimension (must match model) |
| `use_gpu` | bool | True | Use GPU if available |
| **Agentic RAG** | | | |
| `use_agentic_rag` | bool | True | Enable agentic multi-hop search |
| `agentic_pipeline` | str | legacy_react | Pipeline selection: `legacy_react`, `default` (structured graph), or any registered name |
| `max_agent_iterations` | int | 3 | Max search hops |
| `chunk_refinement_method` | str | auto | Chunk refinement (vanilla, refined, auto) |
| `refinement_model` | str | gpt-6-luna | Model for chunk refinement |
| `agent_temperature` | float | 0.1 | Temperature for agent LLM calls |
| `enable_crag` | bool | False | Enable Corrective RAG quality grading |
| `crag_web_search_fallback` | bool | False | Allow web search fallback on poor CRAG grade |
| `enable_query_decomposition` | bool | False | Break complex queries into sub-questions |
| `hyde_strategy` | HyDEStrategy | ADAPTIVE | HyDE strategy (OFF, ALWAYS, FIRST_HOP, ADAPTIVE) |

---

## Retry and Resilience

The SDK provides retry utilities for handling transient failures with exponential backoff.

### Using Retry Decorators

```python
from lasagna.sdk import (
    with_retry,
    with_retry_sync,
    BackendError,
    BackendConnectionError,
)

# Async retry decorator
@with_retry(
    max_attempts=3,
    delay=1.0,           # Initial delay in seconds
    backoff=2.0,         # Multiplier for each retry
    max_delay=30.0,      # Maximum delay cap
    exceptions=(BackendError, ConnectionError),
)
async def search_with_retry(query: str):
    return await client.search_documents(query)

# Sync retry decorator
@with_retry_sync(max_attempts=3, delay=0.5)
def load_config():
    return load_external_config()
```

### RetryConfig Presets

```python
from lasagna.sdk import (
    RetryConfig,
    RETRY_BACKEND,   # For backend operations (3 attempts, 0.5s delay)
    RETRY_SEARCH,    # For search operations (2 attempts, 0.2s delay)
)

# Use preset
@RETRY_BACKEND.decorator()
async def backend_operation():
    ...

# Custom config
my_retry = RetryConfig(
    max_attempts=5,
    delay=2.0,
    backoff=1.5,
    max_delay=60.0,
    exceptions=(BackendError,),
)

@my_retry.decorator()
async def critical_operation():
    ...
```

### Retry Callbacks

```python
def on_retry(exception, attempt):
    logger.warning(f"Retry {attempt}: {exception}")
    send_alert(f"Operation retrying: {exception}")

@with_retry(
    max_attempts=3,
    exceptions=(BackendError,),
    on_retry=on_retry,
)
async def monitored_operation():
    return await client.search_documents("query")
```

---

## Advanced Usage

### Session Management

```python
# Create client with session ID
client = KnowledgeClient(session_id="user_123")

# Change session
client.session_id = "user_456"

# Documents are isolated per session
docs = await client.list_documents()  # Only shows user_456's docs
```

### Concurrent Operations

```python
import asyncio

# Upload multiple documents concurrently
async def upload_many(files):
    tasks = [client.upload(f) for f in files]
    results = await asyncio.gather(*tasks)
    return results

results = await upload_many(["doc1.pdf", "doc2.pdf", "doc3.pdf"])
```

### Custom Processing Pipeline

```python
from lasagna.sdk.processing import (
    DocumentExtractor,
    SemanticChunker,
)
from lasagna.sdk.retrieval import EmbeddingModel

async def custom_ingest(file_path: str):
    # Extract (returns the text as a string)
    extractor = DocumentExtractor()
    text = extractor.extract_text(file_path)

    # Custom preprocessing
    text = text.replace("\n\n\n", "\n\n")  # Clean extra newlines

    # Chunk: semantic splitting needs the embedding model
    model = EmbeddingModel()
    chunker = SemanticChunker(embedding_manager=model)
    chunks = chunker.chunk_semantic(text, max_size=1000)
    # Or fixed-size: chunker.chunk_fixed(text, chunk_size=1000, overlap=100)

    # Generate embeddings
    texts = [c.content for c in chunks]
    embeddings = model.encode_documents_batch(texts)

    # Store in backend...
    return len(chunks)
```

### Monitoring and Metrics

`METRICS` records counters and histograms but has no read API. Turn on Prometheus to collect them. A REST server started with `enable_prometheus` on serves them at `/metrics`. Search-cache statistics can be read in process.

```python
from lasagna.sdk.utils import METRICS, get_search_cache
from lasagna.utils import configure_metrics

# Record metrics in Prometheus (needs prometheus-client, from pip install "lasagna-rag[production]")
configure_metrics(enable_prometheus=True)
print(METRICS.is_prometheus_enabled)

# Search cache hit rate and size
stats = get_search_cache().stats()
print(stats["hits"], stats["misses"], stats["hit_rate"])
```

---

## Configuration Profiles

The SDK ships with four built-in profiles that combine SOTA features into pre-tuned configurations. Use them as starting points and customize from there.

### Profile Overview

| Profile | Embedding | Reranking | Hybrid | HyDE | MMR | Agentic | Hierarchical | CRAG | RAG Fusion | Latency |
|---------|-----------|-----------|--------|------|-----|---------|--------------|------|------------|---------|
| `speed` | MiniLM-L6 (384d) | No | No | No | No | No | No | No | No | ~50-100ms |
| `standard` | MiniLM-L6 (384d) | Yes | Yes | No | Yes | Yes | No | No | No | ~200-400ms |
| `balanced` | BGE-large (1024d) | Yes | Yes | No | Yes | Yes | Yes (0.5) | No | No | ~300-600ms |
| `max_quality` | BGE-large (1024d) | Yes | Yes | Yes | Yes | Yes | Yes (0.4) | No | No | ~1-3s |

### Speed Profile

Optimized for minimum latency. Pure vector search with no expensive processing. Ideal for real-time autocomplete, chatbot suggestions, or high-throughput pipelines.

```python
from lasagna.sdk import config_for_speed, KnowledgeClient

config = config_for_speed()
client = KnowledgeClient(config=config)

# Features:
# - MiniLM-L6-v2 embedding (384d, 80MB, fast)
# - Fixed chunking (no semantic splitting)
# - No reranking, HyDE, MMR, or agentic RAG
# - ~50-100ms search latency
```

### Standard Profile

**Recommended starter profile.** Enables SOTA retrieval features with a lightweight embedding model. Good balance between quality and resource usage -- no large model downloads required.

```python
from lasagna.sdk import config_for_standard, KnowledgeClient

config = config_for_standard()
client = KnowledgeClient(config=config)

# Features:
# - MiniLM-L6-v2 embedding (384d, 80MB)
# - Semantic chunking
# - Cross-encoder reranking (mxbai-rerank-large-v1)
# - Hybrid search (vector + BM25)
# - MMR diversity
# - Agentic multi-hop RAG
# - ~200-400ms search latency
```

### Balanced Profile

SOTA features with the best embedding model. Enables hierarchical chunking with auto-merge for richer context. Recommended for production workloads with moderate latency tolerance.

```python
from lasagna.sdk import config_for_balanced, KnowledgeClient

config = config_for_balanced()
client = KnowledgeClient(config=config)

# Features:
# - BGE-large-en-v1.5 embedding (1024d, 1.3GB download)
# - Semantic chunking + hierarchical parent-child auto-merge
# - Cross-encoder reranking
# - Hybrid search + MMR + Agentic RAG
# - Auto-merge threshold: 0.5
# - ~300-600ms search latency
```

### Max Quality Profile

All features enabled for maximum recall and answer quality. Use when accuracy is paramount and latency of 1-3 seconds per query is acceptable.

```python
from lasagna.sdk import config_for_max_quality, KnowledgeClient

config = config_for_max_quality()
client = KnowledgeClient(config=config)

# Features:
# - BGE-large-en-v1.5 embedding (1024d, 1.3GB)
# - All SOTA features: HyDE, query expansion, reranking, MMR
# - Hierarchical chunking with aggressive auto-merge (threshold 0.4)
# - Larger candidate pool (100), adaptive up to 30 results
# - ~1-3s search latency
```

### Extending Profiles with SOTA Toggles

You can start from any profile and layer on additional SOTA features:

```python
from lasagna.sdk import config_for_balanced, KnowledgeClient

config = config_for_balanced()

# Add Contextual Retrieval for higher recall (adds index-time cost)
config.enable_contextual_retrieval = True

# Add RAG Fusion for improved comprehensiveness
config.enable_rag_fusion = True
config.rag_fusion_n_queries = 3

# Add Corrective RAG for quality grading + web fallback
config.enable_crag = True
config.crag_web_search_fallback = True

client = KnowledgeClient(config=config)
```

### Saving Custom Profiles

```python
config_manager = client.config_manager

# Save your custom configuration as a reusable profile
await config_manager.save_profile(
    "my_production_profile",
    config={
        "enable_reranking": True,
        "enable_rag_fusion": True,
        "enable_crag": True,
        "enable_contextual_retrieval": True,
        "max_agent_iterations": 5,
    },
    set_active=True,
)

# Switch between profiles
await config_manager.set_profile("my_production_profile")
await config_manager.set_profile("speed")  # Switch back

# List all available profiles
profiles = await config_manager.list_profiles()
```

---

## See Also

- [User Guide](user_guide.md)
- [Features](features.md)
- [Deployment](deployment.md)
