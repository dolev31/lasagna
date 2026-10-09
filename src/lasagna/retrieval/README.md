# Retrieval Module

The retrieval layer provides hybrid search, reranking, query enhancement, and intelligent routing for the RAG Knowledge system. This module is **independent** from the larger Lasagna package and can be used standalone via the SDK or REST API.

## Architecture

```
                              Query
                                |
                    +-----------v-----------+
                    |     Query Router      |
                    |  (tier selection)      |
                    +---+-------+-------+---+
                        |       |       |
                      FAST  STANDARD  COMPREHENSIVE
                        |       |       |
                        v       v       v
                    +---+-------+-------+---+
                    |  Query Enhancement    |
                    |  HyDE | Expansion |   |
                    |  Decomposition |      |
                    |  Classification       |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |    Hybrid Search      |
                    |  Dense + BM25 (RRF)   |
                    |  Native | Fallback    |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |  Reranking Pipeline   |
                    |  Cross-encoder -->    |
                    |  Pre-MMR filter -->   |
                    |  MMR diversity        |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Coverage Scoring    |
                    |  Keyword | LLM-based  |
                    +-----------+-----------+
                                |
                            Results
```

## Subpackages

### `search/`

| Class | Description |
|-------|-------------|
| `HybridSearcher` | Unified hybrid search combining dense vectors with BM25 sparse retrieval via Reciprocal Rank Fusion (RRF). Supports Milvus native hybrid search (Standalone/Distributed) and rank_bm25 fallback (Milvus Lite). Backend-agnostic via BYOVDB. |
| `MilvusNativeHybridSearcher` | Milvus-specific implementation using built-in sparse vectors and `hybrid_search` API with RRF ranking. |
| `BM25FallbackIndex` | In-memory BM25 index per session using rank_bm25 library. Supports optional Porter stemming. Lazy rebuild after document removal. |
| `CrossEncoderReranker` | Cross-encoder based reranker (default: mxbai-rerank-large-v1). Joint query-document scoring for higher precision than bi-encoders. GPU/MPS/CPU auto-detection. |
| `MMRReranker` | Maximal Marginal Relevance for result diversity. Normalized relevance scores ensure comparability with cosine similarity diversity term. Pre-fetched embeddings avoid re-embedding. |
| `CombinedReranker` | Pipeline: cross-encoder reranking -> pre-MMR relevance filter (sigmoid >= 0.1) -> MMR diversity. Supports adaptive-k score-gap analysis (EMNLP 2025). |
| `EnsembleReranker` | Multi-model reranking ensemble with weighted score averaging. Reduces model-specific biases. |

### `enhancement/`

| Class | Description |
|-------|-------------|
| `HyDEGenerator` | Hypothetical Document Embeddings — generates hypothetical answers to bridge query/document embedding gap. Supports multi-document HyDE with embedding averaging or max-similarity selection. Thread-safe LLM initialization. |
| `QueryExpander` | LLM-based query paraphrasing for improved recall. Generates alternative phrasings with different terminology. |
| `QueryDecomposer` | Breaks complex multi-part queries into simpler sub-queries for targeted retrieval. |
| `QueryEnhancer` | Combined pipeline orchestrating HyDE, expansion, and decomposition. Provides both async and sync interfaces. |
| `QueryClassifier` | Rule-based query type classification (FACTUAL, CONCEPTUAL, NAVIGATIONAL, COMPARATIVE). Zero LLM overhead. Recommends retrieval parameters per type (alpha, HyDE strategy, expansion). |
| `CoverageScorer` | Keyword-based aspect coverage analysis. Extracts query aspects, checks chunk coverage via word-boundary matching with pre-compiled regex. Provides confidence scores and follow-up suggestions. |
| `LLMCoverageScorer` | LLM-powered alternative for semantic coverage analysis. Understands synonyms and implicit aspects. Produces aspect-level coverage ratios with consistency-derived confidence. |

### `routing/`

| Class | Description |
|-------|-------------|
| `QueryRouter` | Singleton router selecting FAST, STANDARD, or COMPREHENSIVE retrieval tier. Integrates signal extraction, strategy delegation, forced tier overrides, and agentic RAG downgrade logic. |
| `RuleBasedStrategy` | Deterministic decision tree (<1ms). Routes based on query type, word count, multi-hop indicators, and complexity. |
| `AdaptiveStrategy` | Extends rule-based with confidence-aware tier escalation. Low-confidence decisions are escalated one tier. Informed by Adaptive-RAG (NAACL 2024). |
| `RoutingTier` | Enum: FAST (~50-100ms, vector-only), STANDARD (~200-400ms, full pipeline), COMPREHENSIVE (~1-5s, agentic multi-hop). |
| `RoutingSignals` | Extracted features: word count, query type, confidence, multi-hop indicators, document filter presence, estimated complexity. |

## Key Algorithms

| Algorithm | Reference | Usage |
|-----------|-----------|-------|
| Reciprocal Rank Fusion (RRF) | Cormack et al. 2009 | Combining dense + sparse search results with configurable k parameter |
| HyDE | Gao et al. 2022 | Hypothetical Document Embeddings with multi-doc averaging for robust query representation |
| MMR | Carbonell & Goldstein 1998 | Balancing relevance and diversity via lambda-weighted scoring |
| Adaptive-k | EMNLP 2025 | Score-gap analysis to dynamically determine optimal result count |
| Adaptive-RAG | NAACL 2024 | Query complexity routing to select retrieval tier |
| BM25 (Okapi) | Robertson et al. 1995 | Sparse keyword retrieval with optional Porter stemming |

## Configuration

Every retrieval parameter is configurable across multiple surfaces. Configuration precedence (highest to lowest):

1. **Session overrides** (per-session patches via UI/API)
2. **Environment variables** (`RAG_*`)
3. **TOML file** (`settings.lasagna.toml`)
4. **Built-in defaults**

### Key Parameters

| Parameter | Default | Surfaces | Description |
|-----------|---------|----------|-------------|
| `enable_hybrid_search` | `true` | TOML, ENV, SDK, REST, Session, UI | Enable dense+BM25 hybrid search |
| `hybrid_alpha` | `0.7` | TOML, ENV, SDK, REST, Session, UI | Dense vs sparse weight (1.0=pure dense) |
| `enable_reranking` | `true` | TOML, ENV, SDK, REST, Session, UI | Enable cross-encoder reranking |
| `reranker_model` | `mixedbread-ai/mxbai-rerank-large-v1` | TOML, ENV, SDK, Session, UI | Cross-encoder model |
| `rerank_top_k` | `5` | TOML, ENV, SDK, REST, Session, UI | Results after reranking |
| `enable_mmr` | `true` | TOML, ENV, SDK, REST, Session, UI | Enable MMR diversity |
| `mmr_lambda` | `0.5` | TOML, ENV, SDK, REST, Session, UI | Relevance (1) vs diversity (0) |
| `enable_hyde` | `false` | TOML, ENV, SDK, REST, Session, UI | Enable HyDE |
| `hyde_model` | `"default"` | TOML, ENV, SDK, Session | LLM for HyDE generation (`"default"` = match Lasagna's active model) |
| `hyde_n_docs` | `3` | TOML, ENV, SDK, Session | Number of hypothetical documents |
| `enable_query_routing` | `true` | TOML, ENV, SDK, REST, Session, UI | Enable intelligent routing |
| `routing_strategy` | `rule_based` | TOML, ENV, SDK, REST, Session, UI | `rule_based` or `adaptive` |
| `default_top_k` | `20` | TOML, ENV, SDK, REST, Session, UI | Initial retrieval count |
| `similarity_threshold` | `0.3` | TOML, ENV, SDK, REST, Session, UI | Minimum score threshold |
| `enable_context_reordering` | `true` | TOML, ENV, SDK, REST, Session, UI | Lost-in-the-middle fix |
| `enable_rag_fusion` | `false` | TOML, ENV, SDK, REST, Session, UI | Multi-query RAG Fusion |
| `enable_crag` | `false` | TOML, ENV, SDK, REST, Session, UI | Corrective RAG |
| `bm25_max_sessions` | `1000` | TOML, ENV | Max in-memory BM25 session indices |
| `bm25_session_ttl_seconds` | `86400` | TOML, ENV | BM25 session idle TTL |

### Built-in Quality Profiles

| Setting | speed | standard | balanced | max_quality |
|---------|-------|----------|----------|-------------|
| `enable_hybrid_search` | false | true | true | true |
| `enable_reranking` | false | true | true | true |
| `enable_mmr` | false | true | true | true |
| `enable_hyde` | false | false | false | true |
| `use_agentic_rag` | false | true | true | true |
| `default_top_k` | 20 | 20 | 20 | 20 |
| `similarity_threshold` | 0.3 | 0.3 | 0.3 | 0.2 |

## Component Registry

Custom rerankers and query enhancers can be registered via the component registry:

```python
from lasagna.retrieval.registry import component_registry

# Register a custom reranker
component_registry.register_reranker("my_colbert", MyColBERTReranker)

# Register a custom query enhancer
component_registry.register_query_enhancer("my_enhancer", MyEnhancer)

# Use via config
config = KnowledgeConfig(reranker="my_colbert")

# List available components
component_registry.list_rerankers()
```

Built-in registrations: `cross_encoder`, `combined`, `ensemble` (rerankers); `hyde`, `query_expansion`, `query_decomposition`, `combined` (enhancers). ABC-adapted variants (`cross_encoder_adapted`, `combined_adapted`, `ensemble_adapted`, `mmr_adapted`) are also available.

## Thread Safety

All singletons use double-checked locking (DCL):

```python
_instance = None
_lock = threading.Lock()

def get_instance():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = MyClass()
    return _instance
```

Key thread-safety design decisions:
- `HybridSearcher` session indices protected by `_session_lock`
- `CrossEncoderReranker` model loading protected by `_config_lock`
- `HyDEGenerator` LLM initialization protected by `_llm_lock`
- `MMRReranker.rerank_mmr()` uses local variables for config parameters (no singleton mutation)
- `QueryRouter._strategy_cache` pre-populated at class definition (no write contention)

## Performance

| Component | Complexity | Notes |
|-----------|-----------|-------|
| `HybridSearcher` session cleanup | O(n) sessions | Periodic cleanup every hour, max 1000 sessions |
| `BM25FallbackIndex.search()` | O(n) corpus | Per-session index, lazy rebuild after removal |
| `BM25FallbackIndex.remove_document()` | O(n) chunks | Invalidates index; lazy rebuild on next search |
| `CrossEncoderReranker.rerank()` | O(n) results | Batch prediction with configurable batch_size |
| `MMRReranker.rerank_mmr()` | O(k*n) | k=selected, n=candidates. Pre-fetched embeddings avoid re-computation |
| `QueryClassifier.classify()` | O(p) patterns | Compiled regex, 32 patterns, <1ms |
| `CoverageScorer.score_coverage()` | O(a*c*k) | a=aspects, c=chunks, k=keywords. Pre-compiled patterns |
| `QueryRouter.route()` | O(p) patterns | Signal extraction + deterministic rules, <1ms |
