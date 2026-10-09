# Extending the Knowledge SDK

Guide for researchers and ML engineers extending RAG pipeline components.

## Table of Contents

1. [Component Registry](#component-registry)
2. [Swappable Components](#swappable-components)
3. [KnowledgePipeline](#knowledgepipeline)
4. [RAGAS Evaluation](#ragas-evaluation)
5. [Ablation Studies](#ablation-studies)
6. [Agentic Pipeline Extensibility](#agentic-pipeline-extensibility)
7. [Pipeline Serialization](#pipeline-serialization)
8. [Using Profiles as Research Baselines](#using-profiles-as-research-baselines)
9. [Contributing Components](#contributing-components)
10. [Quick Reference](#quick-reference)
11. [See Also](#see-also)

---

## Component Registry

The SDK uses a **Registry + Factory** pattern at two levels:

1. **Vector backend registry** — swap the underlying vector database (Milvus, Qdrant, etc.)
2. **Retrieval component registry** — swap pipeline components (rerankers, embedders, chunkers, query enhancers)

### Retrieval Component Registry

Register custom pipeline components without modifying package source code:

```python
from lasagna.retrieval.registry import component_registry

# --- Register a custom reranker ---
class MyColBERTReranker:
    def rerank(self, query, results, top_k=5, config=None):
        # Your reranking logic here
        ...

component_registry.register_reranker("my_colbert", MyColBERTReranker,
                                      description="ColBERT-based reranker")

# --- Register a custom query enhancer ---
class MyHyDE:
    def enhance(self, query):
        ...

component_registry.register_enhancer("my_hyde", MyHyDE)

# --- List all registered components (names per component type) ---
print(component_registry.list_all())
# {
#   "reranker": ["cross_encoder", "combined", ..., "my_colbert"],
#   "embedder": ["default"],
#   "chunker": ["semantic", "hierarchical", "structure_aware"],
#   "query_enhancer": ["hyde", "query_expansion", ..., "my_hyde"],
#   ...
# }

# --- Create instances via factory ---
reranker = component_registry.create_reranker("my_colbert")

# --- Use via config ---
# In settings.lasagna.toml:
#   custom_reranker = "my_colbert"
#   custom_query_enhancer = "my_hyde"
```

Four component categories are supported:

| Category | Registration | Required Interface |
|----------|-------------|--------------------|
| Rerankers | `register_reranker(name, cls)` | `rerank(query, results, top_k)` |
| Embedders | `register_embedder(name, cls)` | `encode_query(text)`, `encode_documents_batch(texts)` |
| Chunkers | `register_chunker(name, cls)` | `chunk(text) -> List[str]` |
| Enhancers | `register_enhancer(name, cls)` | `enhance(query)` |

Built-in components are auto-registered. The factory handles constructor signature mismatches gracefully — unknown kwargs are filtered automatically.

### Vector Backend Registry

#### Listing Available Backends

```python
from lasagna.vector_backends import get_available_backends

# List all registered backends
backends = get_available_backends()
print(backends)
# ['milvus', ...]: the built-in backends whose client library is installed
```

Built-in backends are registered automatically if their client library is installed:

| Backend | Required Package | Install |
|---------|-----------------|---------|
| `milvus` | `pymilvus` | included |
| `pinecone` | `pinecone` | `pip install "lasagna-rag[vector-backends]"` |
| `qdrant` | `qdrant-client` | `pip install "lasagna-rag[vector-backends]"` |
| `weaviate` | `weaviate-client` | `pip install "lasagna-rag[vector-backends]"` |
| `pgvector` | `psycopg` + `pgvector` | `pip install "lasagna-rag[pgvector]"` |
| `opensearch` | `opensearch-py` | `pip install "lasagna-rag[opensearch]"` |

### Registering Custom Backends

Implement the `VectorStoreAdapter` interface and register your backend:

```python
from lasagna.vector_backends.base import BackendCapabilities, VectorStoreAdapter
from lasagna.vector_backends import register_backend, get_backend

class MyCustomBackend(VectorStoreAdapter):
    """Custom vector backend implementation.

    Every member below is abstract in the base class, so the class cannot be
    created until all of them exist. The methods are synchronous. Replace
    each ``...`` with calls to your database.
    """

    @property
    def backend_name(self) -> str:
        return "my_custom"

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities()

    # Connection
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def is_connected(self) -> bool: ...

    # Collections and indexes
    def has_collection(self, collection_name) -> bool: ...
    def create_collection(self, collection_name, schema, index_config=None) -> None: ...
    def drop_collection(self, collection_name) -> None: ...
    def describe_collection(self, collection_name): ...  # returns CollectionInfo
    def get_collection_stats(self, collection_name) -> dict: ...
    def create_index(self, collection_name, field_name, index_config) -> None: ...
    def list_indexes(self, collection_name) -> list: ...

    # Data
    def insert(self, collection_name, data, partition_key=None) -> list: ...
    def delete(self, collection_name, filter_expr, partition_key=None) -> int: ...
    def query(self, collection_name, filter_expr=None, output_fields=None,
              limit=100, offset=0, partition_key=None) -> list: ...
    def search(self, collection_name, query_embedding, vector_field="embedding",
               filter_expr=None, limit=10, output_fields=None,
               search_params=None, partition_key=None) -> list: ...  # VectorSearchResult items

# Register with the global registry
register_backend("my_custom", MyCustomBackend)

# Use via the factory
backend = get_backend("my_custom", singleton=True)

# Or via KnowledgeConfig
from lasagna.sdk import KnowledgeConfig
config = KnowledgeConfig(vector_backend="my_custom")
```

### Backend Factory Options

```python
from lasagna.vector_backends import get_backend

# Get default backend (from config or "milvus")
backend = get_backend()

# Get specific backend with custom config
backend = get_backend("milvus", db_path="/path/to/db")

# Create a fresh (non-singleton) instance
backend = get_backend("milvus", singleton=False)

# Connection pooling (wraps backend in PooledVectorBackend)
backend = get_backend("milvus", pool_size=10)
```

### Plugin Auto-Discovery

The registry supports automatic discovery of third-party backend plugins. Any installed package with the `lasagna_vector_` prefix that exposes a `register_backend(registry)` function will be auto-discovered:

```python
from lasagna.vector_backends.registry import discover_backends

# Discover and register all installed plugins
count = discover_backends()
print(f"Discovered {count} backend plugins")

# Example plugin package: lasagna_vector_chromadb
# The package must define:
#   def register_backend(registry):
#       registry.register("chromadb", ChromaDBBackend)
```

---

The sections below cover the research workflow: registering, discovering, and swapping pipeline components without modifying framework internals. Every component type has a unique namespace, and your custom implementations are first-class citizens alongside built-in defaults.

### Registering a Custom Component

```python
from lasagna.sdk import registry
from lasagna.sdk import BaseReranker, ScoredDocument

@registry.register("reranker", "my_novel_reranker")
class MyNovelReranker(BaseReranker):
    def rerank(self, query, documents, top_k=5):
        # Your custom reranking logic here
        return sorted(documents, key=lambda d: d.score, reverse=True)[:top_k]
```

### Listing Registered Components

```python
# See all components in a namespace
registry.list("reranker")
# {'reranker': ['cross_encoder', 'combined', 'ensemble', 'mmr', 'colbert', 'llm_listwise', 'my_novel_reranker']}

# Get an instantiated component
instance = registry.get("reranker", "my_novel_reranker")

# Or get just the class (without instantiating)
cls = registry.get_class("reranker", "my_novel_reranker")
```

### Registry Namespaces

Each of the six swappable component types lives in its own namespace. Register into the correct one to make your component available throughout the pipeline.

| Namespace | Base Class | Description |
|-----------|------------|-------------|
| `embedder` | `BaseEmbedder` | Text-to-vector encoding |
| `chunker` | `BaseChunker` | Document segmentation |
| `reranker` | `BaseReranker` | Result reranking |
| `retriever` | `BaseRetriever` | Vector/hybrid retrieval |
| `document_processor` | `BaseDocumentProcessor` | Text extraction from files |
| `query_enhancer` | `BaseQueryEnhancer` | Query rewriting and expansion |
| `agentic_pipeline` | `BaseAgenticPipeline` | Pluggable agentic RAG pipelines |

**Note:** `graph_node` callables are stored separately from `COMPONENT_TYPES` to prevent accidental instantiation (since `registry.get()` would execute a callable node). Use the dedicated methods `registry.register_graph_node()`, `registry.get_graph_node()`, and `registry.list_graph_nodes()` to manage them.

---

## Swappable Components

The pipeline is composed of six **swappable component types**. Each has a base class with a well-defined interface. Implement the required methods and register your class to plug it into any pipeline configuration.

### 1. BaseEmbedder

Converts text into dense vector representations.

```python
import numpy as np
from lasagna.sdk import BaseEmbedder

class MyEmbedder(BaseEmbedder):
    def encode(self, texts: list[str], **kwargs) -> np.ndarray:
        """Encode a batch of texts into vectors.

        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        ...

    def encode_query(self, query: str, **kwargs) -> np.ndarray:
        """Encode a single query (may differ from document encoding).

        Returns:
            1D numpy array of shape (embedding_dim,)
        """
        ...

    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        return 768
```

**Key design note:** `encode_query` is separated from `encode` to support asymmetric embedding models (e.g., E5, GTE) where query and document encoders differ.

### 2. BaseChunker

Segments raw text into retrieval-ready chunks.

```python
from lasagna.sdk import BaseChunker, Chunk

class MyChunker(BaseChunker):
    def chunk(self, text: str, **kwargs) -> list[Chunk]:
        """Split text into Chunk objects.

        Returns a list of Chunk dataclass instances, each with:
        - content: str — the chunk text
        - chunk_index: int — position in document (0-based)
        - metadata: dict — chunk-level metadata
        - page_number: Optional[int]
        - start_char: int — start character offset
        - end_char: int — end character offset
        - section_header: Optional[str] — section heading
        - parent_chunk_id: Optional[str] — parent chunk reference
        - chunk_type: str — 'text', 'table', 'code', 'header'
        """
        ...
```

### 3. BaseReranker

Rescores and reorders candidate documents using a cross-encoder or learned scoring function.

```python
from lasagna.sdk import BaseReranker, ScoredDocument

class MyReranker(BaseReranker):
    def rerank(self, query: str, documents: list[ScoredDocument], top_k: int = 5) -> list[ScoredDocument]:
        """Rerank documents by relevance to the query.

        Args:
            query: The user query.
            documents: Candidate documents with initial scores.
            top_k: Maximum number of documents to return.

        Returns:
            Reranked list of ScoredDocument, truncated to top_k.
        """
        ...
```

### 4. BaseRetriever

Performs the core retrieval step — dense, sparse, or hybrid search over the vector store.

```python
from lasagna.sdk import BaseRetriever, ScoredDocument

class MyRetriever(BaseRetriever):
    def retrieve(self, query: str, collection: str, top_k: int = 5, **kwargs) -> list[ScoredDocument]:
        """Retrieve candidate documents for a query.

        Returns a list of ScoredDocument dataclass instances with:
        - content: str
        - score: float
        - document_id: str
        - chunk_id: str
        - metadata: dict
        """
        ...
```

### 5. BaseDocumentProcessor

Extracts structured text from raw files (PDF, DOCX, HTML, etc.).

```python
from lasagna.sdk import BaseDocumentProcessor, ProcessedDocument

class MyDocumentProcessor(BaseDocumentProcessor):
    def extract(self, file_path: str, **kwargs) -> ProcessedDocument:
        """Extract text and metadata from a file.

        Returns:
            ProcessedDocument dataclass with:
            - text: str — extracted text
            - metadata: dict — file metadata
            - pages: Optional[List[str]] — per-page text
            - tables: Optional[List[str]] — extracted tables
        """
        ...
```

### 6. BaseQueryEnhancer

Rewrites, expands, or decomposes queries before retrieval (e.g., HyDE, multi-query expansion).

```python
from lasagna.sdk import BaseQueryEnhancer

class MyQueryEnhancer(BaseQueryEnhancer):
    def enhance(self, query: str, **kwargs) -> list[str]:
        """Enhance a query into one or more retrieval queries.

        Returns:
            A list of query strings. The retriever will search for all of them
            and merge results.
        """
        ...
```

---

## KnowledgePipeline

`KnowledgePipeline` is the composition layer that wires together registered components into an end-to-end RAG pipeline. You can override any component slot by name, keeping all others at their defaults.

### Composing a Pipeline

```python
from lasagna.sdk import KnowledgePipeline

# Override only the reranker; everything else uses defaults
pipeline = KnowledgePipeline(reranker="my_novel_reranker")

# Override multiple components
pipeline = KnowledgePipeline(
    embedder="my_custom_embedder",
    chunker="semantic_v2",
    reranker="my_novel_reranker",
)

# Override the agentic pipeline (controls multi-hop retrieval strategy)
pipeline = KnowledgePipeline(
    agentic_pipeline="my_custom_rag",  # Any registered BaseAgenticPipeline
)

# Override individual graph nodes within the structured pipeline
# First register the custom node, then reference it by name
registry.register_graph_node("my_retrieve", my_custom_retrieve_fn)

pipeline = KnowledgePipeline(
    agentic_pipeline="default",
)
pipeline.set_node_override("retrieve", "my_retrieve")  # pass registered name
```

### Loading from a Profile

Use built-in profiles as standardized baselines:

```python
# Create a pipeline pre-configured from a built-in profile
pipeline = KnowledgePipeline.from_profile("balanced")
# Stores BGE-M3, mxbai-rerank-large, hierarchical chunking, etc.
```

### Ingesting Documents

```python
# Ingest a single document (runs Docling extraction, chunking, embedding, indexing)
pipeline.ingest("./corpus/report.pdf")

# Ingest an entire directory
pipeline.ingest("./corpus/")
```

### Querying

```python
# Run the full SOTA search pipeline (HyDE, hybrid, reranking, adaptive-K)
results = pipeline.query("What mechanisms drive attention head specialization?", top_k=10)
for r in results:
    print(f"[{r.score:.3f}] {r.chunk.content[:120]}...")
```

### Evaluating a Pipeline

```python
# Evaluate against a labeled dataset
results = pipeline.evaluate(dataset=my_dataset, evaluator=my_evaluator, metrics=["faithfulness"])
print(results.metrics)
# {"faithfulness": 0.87}
```

### Comparing Pipelines

```python
baseline = KnowledgePipeline()  # all defaults
experimental = KnowledgePipeline(reranker="my_novel_reranker")

for name, pipe in [("baseline", baseline), ("experimental", experimental)]:
    result = pipe.evaluate(dataset=my_dataset, metrics=["faithfulness", "context_precision"])
    print(f"{name}: faithfulness={result.metrics['faithfulness']:.3f}, precision={result.metrics['context_precision']:.3f}")
```

---

## RAGAS Evaluation

Lasagna ships with a RAGAS-compatible evaluator for standardized RAG quality assessment. It supports the core RAGAS metrics — faithfulness, answer relevancy, context precision, and context recall — with a single-call API.

### Basic Usage

```python
from lasagna.sdk import RAGASEvaluator

evaluator = RAGASEvaluator()
results = evaluator.evaluate_from_samples(
    questions=["What is RAG?"],
    answers=["RAG is Retrieval-Augmented Generation..."],
    contexts=[["RAG combines retrieval with generation..."]],
    ground_truths=["RAG is a technique that..."],
)
```

### Available Metrics

| Metric | What It Measures |
|--------|-----------------|
| `faithfulness` | Whether the answer is grounded in the retrieved context (no hallucination) |
| `answer_relevancy` | Whether the answer addresses the question directly |
| `context_precision` | Whether retrieved contexts are relevant to the question |
| `context_recall` | Whether all necessary information is present in the retrieved contexts |

### Batch Evaluation

```python
results = evaluator.evaluate_from_samples(
    questions=questions_list,
    answers=answers_list,
    contexts=contexts_list,
    ground_truths=ground_truths_list,
    # The default is faithfulness, context_recall and context_precision
    metrics=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
)

print(f"Faithfulness:       {results['faithfulness']:.3f}")
print(f"Answer Relevancy:   {results['answer_relevancy']:.3f}")
print(f"Context Precision:  {results['context_precision']:.3f}")
print(f"Context Recall:     {results['context_recall']:.3f}")
```

### Integration with KnowledgePipeline

```python
# Run end-to-end: pipeline retrieval + RAGAS scoring
pipeline = KnowledgePipeline(reranker="my_novel_reranker")
eval_results = pipeline.evaluate(
    dataset=my_dataset,
    metrics=["faithfulness", "context_precision", "context_recall"],
)
```

---

## Ablation Studies

The ablation module automates **component sweeps**: hold everything constant except one component slot, iterate over variant implementations, and collect comparative metrics.

### Running a Sweep

```python
from lasagna.sdk import sweep

results = sweep(
    pipeline=base_pipeline,
    component="reranker",
    variants=["mxbai-rerank-large", "ms-marco-MiniLM", "bge-reranker-v2"],
    dataset=my_dataset,
    metrics=["faithfulness", "context_precision"],
)
results.to_dataframe()
results.plot()
```

### Sweep Results

The `sweep` function returns an `AblationResults` object with built-in analysis methods:

```python
# `results` is the AblationResults that sweep() returned above
# Tabular summary
df = results.to_dataframe()
print(df)
#                    variant  faithfulness  context_precision  duration_s
# 0     mxbai-rerank-large         0.912              0.874       12.3
# 1       ms-marco-MiniLM         0.856              0.831        8.1
# 2        bge-reranker-v2         0.889              0.862        9.7

# Visual comparison
results.plot()  # Generates a grouped bar chart

# Find the best variant for a metric
results.best("faithfulness")
# 'mxbai-rerank-large'

# Paired sign-flip significance test, with a t interval
test = results.significance_test(
    "mxbai-rerank-large", "bge-reranker-v2",
    metric="faithfulness",
    scores_a=per_case_scores_a,  # list of per-sample scores
    scores_b=per_case_scores_b,
)
print(f"p={test['p_value']:.4f}, significant={test['significant']}")
print(f"Effect size (Cohen's d): {test['effect_size']:.3f}")
print(f"95% CI: {test['confidence_interval_95']}")
```

### Multi-Component Sweeps

```python
# Sweep over embedders while holding the reranker fixed
results_embedder = sweep(
    pipeline=base_pipeline,
    component="embedder",
    variants=["bge-large-en-v1.5", "e5-large-v2", "gte-large"],
    dataset=my_dataset,
    metrics=["faithfulness", "context_recall"],
)

# Combine results for a joint analysis
combined_df = pd.concat([
    results.to_dataframe().assign(swept="reranker"),
    results_embedder.to_dataframe().assign(swept="embedder"),
])
```

### Grid Search

For full factorial experiments across multiple component slots:

```python
from lasagna.sdk import grid_search

grid_results = grid_search(
    pipeline=base_pipeline,
    components={
        "embedder": ["bge-large-en-v1.5", "e5-large-v2"],
        "reranker": ["mxbai-rerank-large", "bge-reranker-v2"],
    },
    dataset=my_dataset,
    metrics=["faithfulness", "context_precision"],
    evaluator=my_evaluator,
)
grid_results.to_dataframe()
# Rows: one per (embedder, reranker) combination
# Variant names: "embedder=bge-large-en-v1.5__reranker=mxbai-rerank-large"
```

---

## Agentic Pipeline Extensibility

The `agentic_pipeline` component slot and `graph_node` callable registry enable systematic experimentation with retrieval strategies.

### Sweeping Pipelines

Compare different agentic pipelines using `sweep_pipelines()`:

```python
from lasagna.sdk import sweep_pipelines

results = sweep_pipelines(
    pipeline=base_pipeline,
    variants=["legacy_react", "default", "my_custom_rag"],
    dataset=my_dataset,
    metrics=["faithfulness", "context_precision"],
)
results.to_dataframe()
#                variant  faithfulness  context_precision  duration_s
# 0        legacy_react         0.812              0.795       4.2
# 1             default         0.912              0.874      12.3
# 2       my_custom_rag         0.935              0.891      10.1
```

### Sweeping Nodes

Hold the pipeline fixed and sweep a single graph node across implementations:

```python
from lasagna.sdk import registry, sweep_nodes

# Each variant is the name of a node registered with register_graph_node()
registry.register_graph_node("retrieve_v1", my_retrieve_v1)
registry.register_graph_node("retrieve_v2", my_retrieve_v2)

results = sweep_nodes(
    pipeline=base_pipeline,
    node_name="retrieve",
    variants=["retrieve_v1", "retrieve_v2"],
    dataset=my_dataset,
    metrics=["context_recall", "faithfulness"],
)
results.to_dataframe()
# For the built-in node, evaluate base_pipeline without an override
```

### Pipeline and Node Identity in Metrics

Search metrics (`AgenticRAGMetrics`) now include `pipeline_name` and `node_overrides` fields, so evaluation results are always traceable back to the exact pipeline configuration that produced them.

> For end-to-end custom pipeline setup (writing, registering, deploying), see the [BYOA Guide](custom_pipelines.md).

---

## Pipeline Serialization

Reproducibility is essential for research. Pipelines can be serialized to YAML for exact reproduction, sharing with collaborators, and inclusion in paper appendices.

### Save a Pipeline

```python
pipeline.to_yaml("experiment_v1.yaml")
```

The YAML captures the complete pipeline specification:

```yaml
# experiment_v1.yaml
components:
  chunker: semantic
  query_enhancer: hyde
  reranker: my_novel_reranker
config:
  chunk_overlap: 150
  chunk_size: 800
  enable_mmr: true
  hybrid_alpha: 0.7
  mmr_lambda: 0.5
  rerank_top_k: 10
node_overrides:        # only when set_node_override() was used
  retrieve: my_retrieve
```

Only components you set are listed. The file holds no timestamps or hardware details, so the same pipeline always writes the same file; run metadata goes in `PipelineResult.metadata` instead.

### Load a Pipeline

```python
reproduced = KnowledgePipeline.from_yaml("experiment_v1.yaml")

# Verify exact reproduction
original_results = pipeline.evaluate(dataset=my_dataset, metrics=["faithfulness"])
reproduced_results = reproduced.evaluate(dataset=my_dataset, metrics=["faithfulness"])
assert abs(original_results.metrics["faithfulness"] - reproduced_results.metrics["faithfulness"]) < 1e-6
```

### Sharing Experiments

```python
# Export for a collaborator
pipeline.to_yaml("shared_experiment.yaml")

# Import on a different machine
pipeline = KnowledgePipeline.from_yaml("shared_experiment.yaml")
# Missing components will raise a clear error with installation instructions
```

---

## Using Profiles as Research Baselines

Lasagna ships with **built-in configuration profiles** that serve as standardized baselines for comparison. Use them as control conditions in your experiments. For the full profile reference, see [Configuration Profiles](configuration.md#configuration-profiles).

### Built-in Profiles

| Profile | Embedding Model | Reranking | Hybrid | HyDE | Hierarchical | Typical Latency |
|---------|----------------|-----------|--------|------|-------------|-----------------|
| `speed` | MiniLM-L6 (384d) | No | No | No | No | ~50-100ms |
| `standard` | BGE-M3 (1024d) | mxbai-rerank-large | Yes | No | No | ~200-400ms |
| `balanced` | BGE-M3 (1024d) | mxbai-rerank-large | Yes | No | Yes | ~300-600ms |
| `max_quality` | BGE-M3 (1024d) | mxbai-rerank-large | Yes | Yes | Yes | ~1-3s |

### Using Profiles as Baselines

```python
from lasagna.sdk import KnowledgePipeline

# Establish baselines across all built-in profiles
baselines = {}
for profile in ["speed", "standard", "balanced", "max_quality"]:
    pipe = KnowledgePipeline.from_profile(profile)
    result = pipe.evaluate(dataset=my_dataset, evaluator=my_evaluator, metrics=["faithfulness", "context_precision"])
    baselines[profile] = result

# Compare your experimental pipeline against baselines
experimental = KnowledgePipeline(reranker="my_novel_reranker")
exp_result = experimental.evaluate(dataset=my_dataset, evaluator=my_evaluator, metrics=["faithfulness", "context_precision"])

for profile, baseline in baselines.items():
    delta = exp_result.metrics["faithfulness"] - baseline.metrics["faithfulness"]
    print(f"vs {profile:12s}: faithfulness delta = {delta:+.3f}")
```

### Custom Profiles

Save and reuse your own experiment configurations:

```python
from lasagna.sdk import KnowledgeClient

client = KnowledgeClient()
config_manager = client.config_manager

# Save a named profile
await config_manager.save_profile(
    "my_research_baseline",
    config={
        "enable_hyde": True,
        "max_agent_iterations": 5,
        "default_top_k": 80,
        "embedding_model": "BAAI/bge-large-en-v1.5",
        "embedding_dim": 1024,
    },
)

# List all available profiles
profiles = await config_manager.list_profiles()
for p in profiles:
    print(f"  {p['name']}: {p['description']}")

# Switch between profiles during experiments
await config_manager.set_profile("my_research_baseline")
```

---

## Contributing Components

We welcome contributions of new components, especially those backed by published research or reproducible benchmarks.

### Contribution Workflow

1. **Implement** your component by subclassing the appropriate base class.
2. **Register** it with the component registry using `@registry.register`.
3. **Test** it against at least one standard benchmark dataset.
4. **Document** the expected performance characteristics.
5. **Submit** a pull request with your implementation, tests, and a brief write-up.

### Example: Contributing a New Reranker

```python
# File: src/lasagna/retrieval/search/rerankers/my_reranker.py

from lasagna.sdk import registry
from lasagna.sdk import BaseReranker, ScoredDocument


@registry.register("reranker", "my-research-reranker")
class MyResearchReranker(BaseReranker):
    """Reranker based on [Your Paper Title] (arXiv:XXXX.XXXXX).

    This reranker uses <brief description of approach>.

    Reference:
        Author et al. "Paper Title." Conference/Journal, Year.
    """

    def __init__(self, model_name: str = "my-org/my-model", device: str = "auto"):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            # Lazy-load to avoid import-time overhead
            from transformers import AutoModelForSequenceClassification
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

    def rerank(self, query: str, documents: list[ScoredDocument], top_k: int = 5) -> list[ScoredDocument]:
        self._ensure_loaded()
        # Your reranking logic
        ...
        return reranked[:top_k]
```

### Adding Tests

```python
# File: tests/unit/knowledge/test_my_reranker.py

import pytest
from lasagna.sdk import registry


def test_reranker_registered():
    """Verify the component is discoverable."""
    cls = registry.get_class("reranker", "my-research-reranker")
    assert cls is not None


def test_reranker_basic():
    """Smoke test with synthetic data."""
    cls = registry.get_class("reranker", "my-research-reranker")
    reranker = cls()
    # Provide synthetic ScoredDocument instances
    results = reranker.rerank("test query", mock_documents, top_k=3)
    assert len(results) <= 3
    assert all(results[i].score >= results[i + 1].score for i in range(len(results) - 1))
```

### Pull Request Checklist

- [ ] Component subclasses the correct base class
- [ ] Registered with `@registry.register` in the appropriate namespace
- [ ] Unit tests pass (`python -m pytest tests/unit/knowledge/ -v`)
- [ ] Docstring includes paper reference (if applicable)
- [ ] No new hard dependencies added to the core package (use lazy imports)
- [ ] Benchmark results included in the PR description (dataset, metrics, comparison to baseline)

### Citing Lasagna

If you use Lasagna in your research, please cite:

```bibtex
@software{lasagna_knowledge,
  title  = {Lasagna: A Modular RAG Framework for Research and Production},
  author = {The Lasagna Authors},
  url    = {https://github.com/dolev31/lasagna},
  year   = {2025}
}
```

---

## Quick Reference

| Task | Code |
|------|------|
| Register a component | `@registry.register("reranker", "my_reranker")` |
| Build a pipeline | `KnowledgePipeline(reranker="my_reranker")` |
| Load a baseline profile | `KnowledgePipeline.from_profile("balanced")` |
| Ingest documents | `pipeline.ingest("./corpus/")` |
| Query | `pipeline.query("search text", top_k=10)` |
| Evaluate | `pipeline.evaluate(dataset=ds, metrics=["faithfulness"])` |
| Run ablation sweep | `sweep(pipeline, "reranker", variants=[...], dataset=ds)` |
| Run grid search | `grid_search(pipeline, {"reranker": [...], "chunker": [...]})` |
| Significance test | `results.significance_test("a", "b", "metric", scores_a, scores_b)` |
| Save for reproducibility | `pipeline.to_yaml("experiment.yaml")` |
| Load a saved pipeline | `KnowledgePipeline.from_yaml("experiment.yaml")` |
| RAGAS evaluation | `RAGASEvaluator().evaluate_from_samples(...)` |

---

## See Also

- [Custom Pipelines (BYOA)](custom_pipelines.md)
- [Features](features.md)
- [Evaluation](evaluation.md)
