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
from lasagna.sdk import ChunkingStrategy, KnowledgeClient, KnowledgeConfig

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
    chunk_overlap=0,

    # Search
    default_top_k=20,
    rerank_top_k=5,
    similarity_threshold=0.3,
    hybrid_alpha=0.7,

    # Embedding
    embedding_model="BAAI/bge-m3",
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
    KnowledgeClient,
    config_for_speed,
    config_for_standard,
    config_for_balanced,
    config_for_max_quality,
    config_for_standalone,
)

# Speed: dense vector search only
client = KnowledgeClient(config=config_for_speed())
# - No reranking, hybrid search, MMR, HyDE or agentic RAG
# - all-MiniLM-L6-v2 (384d)
# - Structure-aware chunking, fixed-k chunk selection (no adaptive cutoff)

# Standard: hybrid search, reranking, MMR and agentic RAG
client = KnowledgeClient(config=config_for_standard())
# - BGE-M3 (1024d), mxbai-rerank-large-v1
# - Structure-aware chunking, adaptive chunk count up to 15

# Balanced: standard plus hierarchical chunking
client = KnowledgeClient(config=config_for_balanced())
# - BGE-M3 (1024d), mxbai-rerank-large-v1
# - Hierarchical chunking with auto-merge (threshold 0.5)
# - Hallucination check, adaptive query routing, adaptive chunk count up to 20

# Max quality: every retrieval feature on
client = KnowledgeClient(config=config_for_max_quality())
# - Adds HyDE, query expansion, RAG fusion, CRAG and contextual retrieval
# - BGE-M3 (1024d), hierarchical chunking with auto-merge (threshold 0.4)
# - Always the comprehensive (agentic) tier, 5 agent iterations, adaptive chunk count up to 30

# Standalone: explicit paths (default: ./knowledge_data/ in the working directory)
config = config_for_standalone(
    db_path="./my_data/vectors.db",
    upload_dir="./my_data/uploads",
)
client = KnowledgeClient(config=config)
```

### Preset Comparison

| Preset | Embedding Model | Reranking | Hybrid | MMR | Agentic | HyDE | Hierarchical | CRAG | RAG Fusion |
|--------|----------------|-----------|--------|-----|---------|------|-------------|------|------------|
| `speed` | all-MiniLM-L6-v2 (384d) | No | No | No | No | No | No | No | No |
| `standard` | BGE-M3 (1024d) | Yes | Yes | Yes | Yes | No | No | No | No |
| `balanced` | BGE-M3 (1024d) | Yes | Yes | Yes | Yes | No | Yes (0.5) | No | No |
| `max_quality` | BGE-M3 (1024d) | Yes | Yes | Yes | Yes | Yes | Yes (0.4) | Yes | Yes |

Every preset except `speed` can call an LLM during search: `standard` and `balanced` send a query the router judges complex to agentic retrieval, and `max_quality` sends every query there. They need an API key; see [Configuration](../../../../docs/configuration.md#profiles).

### Dynamic Configuration

```python
# Update configuration
await client.set_config(
    enable_hyde=True,
    chunk_size=1000,
)

# Use preset
client.use_preset("standard")      # hybrid search, reranking, MMR, agentic RAG
client.use_preset("balanced")      # standard plus hierarchical chunking
client.use_preset("max_quality")   # every retrieval feature on

# Get current config
config_dict = client.get_config_dict()
print(config_dict)
```

### How SDK Settings Reach the Backend

A setting is looked up in this order, and the first place that has it wins: per-query overrides, then session overrides, then `LASAGNA_<NAME>` or `RAG_<NAME>` environment variables, then the settings TOML (`settings.lasagna.toml`, or the file `LASAGNA_SETTINGS_FILE` names), then built-in defaults. The full table is in [Where a setting comes from](../../../../docs/configuration.md#where-a-setting-comes-from).

`KnowledgeConfig` fields enter that order in one of two places (`sdk/_config_bridge.py`):

**Init-time fields** (for example `vector_backend`, `database_backend`, `cache_backend`, `celery_enabled`, `storage_backend`) are written where environment variables go, before the backend is created, so they apply to the whole process. Set them when you create the `KnowledgeClient`; changing them afterward logs a warning and has no effect until a new client creates the backend. The embedding fields (`embedding_model`, `embedding_dim`, `embedding_provider`) are written there too, and are also runtime fields: a later `set_config()` of one applies to the session at once.

**Runtime fields** (for example `enable_hyde`, `enable_reranking`, `chunk_size`) are written as session overrides for the client's session. `set_config()` and `use_preset()` change them, and the next operation uses the new values. If you call them before the backend exists, the overrides are written when it is created.

**The session's settings are shared.** They are kept in the session store that the CLI, the REST API and the MCP server read too, so a client's settings are the session's settings on every surface: two clients with different settings on one session take turns overwriting each other, and the last one to apply its settings decides. Use one session per configuration. A client writes nothing when the session already has its settings.

**A session's documents are protected, as on every surface.** Settings that would search another model's collection than the one holding the session's documents (for example `config=config_for_speed()` on a session uploaded with BGE-M3) are refused with `DocumentsLeftBehind` at the client's first operation, or by `set_config()`, and nothing is written; every operation of that client raises it until it is resolved. Use another session for those settings, or:

```python
await client.set_config(reindex=True)  # reindex every document with these settings first
await client.set_config(force=True)    # apply them anyway; reindex the documents later
```

`reindex=True` keeps the session as it is if one document fails, like `lasagna configure --reindex`; `force=True` is `lasagna configure --force`, and the documents are found again once each is reindexed (`client.documents.reindex(document_id)`).

**Only chosen fields are sent.** A field is forwarded when you set it (by keyword or by assignment) or when its value differs from the dataclass default. A plain `KnowledgeConfig()` writes no overrides, so the profile, environment and settings file decide.

```python
# Init-time fields must be set at creation
client = KnowledgeClient(
    config=KnowledgeConfig(database_backend="postgresql", database_dsn="...")
)

# Runtime fields can change anytime
await client.set_config(enable_hyde=True)   # Applied immediately
await client.set_config(enable_hyde=False)  # Still a session override, set to False

# Switching sessions re-applies overrides
client.session_id = "new_session"  # New session gets current config
```

**Remote mode** (`from_url()`): the config bridge only affects local SDK behavior (citation format, presets). The remote server owns the search pipeline configuration, and `set_config()` logs a warning in this mode.

### Profile Management

Saved profiles are shared by every session and stored in the metadata database, so they outlive the process. Save custom configurations and switch between them. Profile operations live on `client.config_manager`.

```python
config_manager = client.config_manager

# List all profiles (built-in + custom)
profiles = await config_manager.list_profiles()
for p in profiles:
    print(f"{p['name']}: {p['description']}")
# Output:
#   default: Default configuration from settings.lasagna.toml
#   speed: Vector search only, with all-MiniLM-L6-v2: no reranking, MMR or HyDE
#   standard: Hybrid search, reranking and agentic retrieval with BGE-M3 (recommended start)
#   balanced: Standard plus hierarchical chunking and the structured agentic graph, with BGE-M3
#   max_quality: Every retrieval feature on: HyDE, query expansion, RAG fusion and CRAG

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
# ("session" or "all") is accepted, and both do the same thing.
await config_manager.reset()
```

### Common Configuration Options

These are the `KnowledgeConfig` defaults (`sdk/config.py`). Every backend setting, with its type, default and environment variable, is in the [settings reference](../../../../docs/reference/settings.md).

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
| `chunk_overlap` | int | 0 | Overlap between chunks |
| `semantic_threshold` | float | 0.75 | Semantic similarity threshold |
| `enable_hierarchical` | bool | True | Enable parent-child chunking with auto-merge |
| `parent_chunk_size` | int | 2000 | Characters per parent chunk |
| `child_chunk_size` | int | 500 | Characters per child chunk |
| `auto_merge_threshold` | float | 0.5 | Ratio of retrieved children / total siblings to trigger merge |
| **Search** | | | |
| `default_top_k` | int | 20 | Candidates to retrieve before reranking |
| `rerank_top_k` | int | 5 | Max results after reranking (ceiling for adaptive mode) |
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
| `embedding_model` | str | BAAI/bge-m3 | Embedding model name |
| `embedding_dim` | int | 1024 | Vector dimension (must match model) |
| `use_gpu` | bool | True | Use GPU if available |
| **Agentic RAG** | | | |
| `use_agentic_rag` | bool | True | Enable agentic multi-hop search |
| `agentic_pipeline` | str | unset | Pipeline selection: `legacy_react`, `default` (structured graph), or any registered name. Unset, it is `default` when `use_structured_graph` is true and `legacy_react` otherwise |
| `max_agent_iterations` | int | 3 | Max search hops |
| `chunk_refinement_method` | str | auto | Chunk refinement (vanilla, refined, auto) |
| `refinement_model` | str | `MODEL_NAME`, else gpt-6-luna | Model for chunk refinement |
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

The SDK has four presets, `config_for_speed()`, `config_for_standard()`, `config_for_balanced()` and `config_for_max_quality()`, which set the same features as the backend profiles of the same names (`core/session_config/_profiles.py`). Use them as starting points and change single fields from there. [Profiles](../../../../docs/configuration.md#profiles) lists what each one downloads and whether it needs an LLM key.

### Profile Overview

| Profile | Embedding | Reranking | Hybrid | HyDE | MMR | Agentic | Hierarchical | CRAG | RAG Fusion |
|---------|-----------|-----------|--------|------|-----|---------|--------------|------|------------|
| `speed` | all-MiniLM-L6-v2 (384d) | No | No | No | No | No | No | No | No |
| `standard` | BGE-M3 (1024d) | Yes | Yes | No | Yes | Yes | No | No | No |
| `balanced` | BGE-M3 (1024d) | Yes | Yes | No | Yes | Yes | Yes (0.5) | No | No |
| `max_quality` | BGE-M3 (1024d) | Yes | Yes | Yes | Yes | Yes | Yes (0.4) | Yes | Yes |

### Speed Profile

Dense vector search with nothing else: no reranking, hybrid search, MMR, HyDE, query routing or agentic loop, and no LLM call during search. It is the only preset that runs without an API key.

```python
from lasagna.sdk import config_for_speed, KnowledgeClient

config = config_for_speed()
client = KnowledgeClient(config=config)

# Features:
# - all-MiniLM-L6-v2 embedding (384d)
# - Structure-aware chunking, no hierarchical chunks
# - Fixed-k chunk selection
# - Docling off, context reordering off
```

### Standard Profile

The recommended starting point: hybrid search, cross-encoder reranking, MMR and agentic retrieval, with BGE-M3.

```python
from lasagna.sdk import config_for_standard, KnowledgeClient

config = config_for_standard()
client = KnowledgeClient(config=config)

# Features:
# - BGE-M3 embedding (1024d)
# - Structure-aware chunking
# - Cross-encoder reranking (mxbai-rerank-large-v1)
# - Hybrid search (vector + BM25), hybrid_alpha 0.7
# - MMR diversity, context reordering, Docling parsing
# - Agentic multi-hop RAG (agentic_pipeline "default", 3 iterations)
# - Rule-based query routing
```

### Balanced Profile

Standard plus hierarchical chunking with auto-merge: when a search returns enough of one parent chunk's children (half of them, at threshold 0.5), it returns the parent in their place. It also turns on the hallucination check and the adaptive routing strategy.

```python
from lasagna.sdk import config_for_balanced, KnowledgeClient

config = config_for_balanced()
client = KnowledgeClient(config=config)

# Features:
# - BGE-M3 embedding (1024d)
# - Hierarchical parent-child chunking, auto-merge threshold 0.5
# - Cross-encoder reranking, hybrid search, MMR
# - Agentic RAG with the structured graph and the hallucination check
# - Adaptive query routing
```

### Max Quality Profile

Every retrieval feature on, and every query sent to the comprehensive (agentic) tier. It makes the most LLM calls per query of the four.

```python
from lasagna.sdk import config_for_max_quality, KnowledgeClient

config = config_for_max_quality()
client = KnowledgeClient(config=config)

# Features:
# - BGE-M3 embedding (1024d)
# - HyDE (adaptive), query expansion, RAG fusion, CRAG, contextual retrieval
# - Reranking, hybrid search, MMR
# - Hierarchical chunking with auto-merge threshold 0.4
# - Hallucination and answer quality checks, 5 agent iterations
# - similarity_threshold 0.2, adaptive chunk count up to 30
```

### Adding Features to a Profile

You can start from any profile and turn on more features:

```python
from lasagna.sdk import config_for_balanced, KnowledgeClient

config = config_for_balanced()

# Contextual retrieval: an LLM call per chunk at index time
config.enable_contextual_retrieval = True

# RAG fusion: several LLM-written query variants, merged with RRF
config.enable_rag_fusion = True
config.rag_fusion_n_queries = 3

# Corrective RAG: grade the retrieved chunks, with web search as a fallback
# (the fallback needs pip install "lasagna-rag[web]")
config.enable_crag = True
config.crag_web_search_fallback = True

client = KnowledgeClient(config=config)
```

### Saving Custom Profiles

```python
config_manager = client.config_manager

# Save your custom configuration as a reusable profile
await config_manager.save_profile(
    "my_profile",
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
await config_manager.set_profile("my_profile")
await config_manager.set_profile("speed")  # Switch back

# List all available profiles
profiles = await config_manager.list_profiles()
```

---

## See Also

- [User Guide](user_guide.md)
- [Features](features.md)
- [Deployment](deployment.md)
