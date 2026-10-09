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

When a sweep arm or a pipeline names a retriever component, its ranked list replaces the hybrid search box: it is the first stage, and it is not fused with BM25 again afterwards. Reranking and the stages after it run as usual.

## Built-in Components

Every component below is registered in the core registry (`lasagna.core.registry.registry`) and implements its stage's interface, so any two of a stage are interchangeable in a sweep. Each reads the arm's settings. One that needs a model, package, key or LLM raises when it is missing, instead of returning the input order and scoring as the control.

### Retrievers (`retrievers.py`)

| Name | Method | Settings (default) |
|------|--------|--------------------|
| `dense` | Dense vector search with the configured embedder (DPR's bi-encoder setting, Karpukhin et al. 2020). The default first stage as a component: use it as the control in a retriever sweep | the embedder's settings |
| `bm25` | Okapi BM25 over the session's chunks (Robertson and Zaragoza 2009), lowercased and Porter-stemmed as the hybrid index tokenizes | `bm25_k1` (1.2), `bm25_b` (0.75), `enable_bm25_stemming` (true) |
| `splade` | SPLADE learned sparse retrieval (Formal et al. 2021): query and passages encoded by a masked language model, scored by dot product | `splade_model` (`naver/splade-cocondenser-ensembledistil`, non-commercial license) |
| `hybrid_rrf` | Dense and BM25 lists fused by Reciprocal Rank Fusion (Cormack et al. 2009), unweighted | `rrf_k` (60), `hybrid_fusion_depth` (1000) |
| `hybrid_convex` | Convex combination of normalized dense and BM25 scores (Bruch et al. 2023). A chunk missing from one list has that score computed, not set to zero | `hybrid_alpha` (0.7), `hybrid_convex_normalization` (`tmm`, or `minmax`, `zscore`), `hybrid_fusion_depth` (1000) |

BM25 and SPLADE score every passage of the session, so `bm25_k1`, `bm25_b` and `splade_model` take effect on every vector backend, including ones whose native BM25 fixes its parameters at index time.

### Rerankers

| Name | Method | Settings (default) |
|------|--------|--------------------|
| `cross_encoder` | Cross-encoder scoring. Loads current Hugging Face rerankers by name (for example `BAAI/bge-reranker-v2-m3`); `mixedbread-ai/mxbai-rerank-base-v2` needs sentence-transformers 5.4 or later and is refused on older versions | `reranker_model` (`mixedbread-ai/mxbai-rerank-large-v1`), `reranker_trust_remote_code` (false) |
| `colbert` | ColBERT late interaction, token-level MaxSim (Khattab and Zaharia 2020) | `colbert_model` (`colbert-ir/colbertv2.0`) |
| `monot5` | monoT5 (Nogueira et al. 2020): the log-probability of "true" from a seq2seq relevance model | `monot5_model` (`castorini/monot5-base-msmarco-10k`) |
| `llm_listwise` | RankGPT (Sun et al. 2023): the LLM orders windows of passages, sliding from the bottom of the list to the top | `llm_reranker_model` (empty: the configured LLM), `llm_reranker_window_size` (20), `llm_reranker_step_size` (10) |
| `llm_pointwise` | LLM relevance grade per passage on a 0 to 3 scale (Thomas et al. 2023, in UMBRELA's form); ties keep retrieval order | `llm_reranker_model` |
| `flashrank` | FlashRank's small ONNX cross-encoders on the CPU. Needs `pip install flashrank` | `flashrank_model` (`ms-marco-TinyBERT-L-2-v2`) |
| `cohere` | Cohere's hosted Rerank API. Needs `pip install cohere` and `COHERE_API_KEY` (or `CO_API_KEY`) | `cohere_rerank_model` (`rerank-v3.5`) |
| `mmr` | Maximal Marginal Relevance diversity reranking (Carbonell and Goldstein 1998) | `mmr_lambda` (0.5), `mmr_score_normalization` (`sigmoid`) |
| `ensemble` | Weighted ensemble of cross-encoders | `ensemble_reranker_*` |
| `combined` | Cross-encoder, then a relevance filter, then MMR | the cross-encoder and MMR settings |

Built-ins that load a model record `component:model` as the reranker that ran, so two arms with different checkpoints stay distinguishable.

### Query Enhancers

Each returns the original query first, then what its method adds. The pipeline searches every returned query and fuses the lists with weighted reciprocal rank fusion: the original at weight 2, each addition at weight 1. That fusion is held fixed across enhancers, so comparing two measures what each adds to the original query. A method that searches only its rewrite in its paper (Query2doc, Rewrite-Retrieve-Read) is therefore measured fused with the original here. The LLM enhancers use the stage's model (`hyde_model`) at temperature 0, so a sweep across enhancers holds the model fixed. The pipeline caps the additions at its variation count (`max_query_variations`, or `rag_fusion_n_queries` when `enable_rag_fusion` is on).

| Name | Method | Settings (default) |
|------|--------|--------------------|
| `hyde` | HyDE (Gao et al. 2022): the query plus one hypothetical answer passage | `hyde_model` |
| `query2doc` | Query2doc (Wang et al. 2023): the query followed by an LLM-written pseudo-document, with the paper's prompt | `query2doc_format` (`dense`, or `sparse`), `query2doc_query_repeats` (5), `query2doc_max_tokens` (128) |
| `step_back` | Step-back prompting (Zheng et al. 2023): the question and a more generic step-back question | `hyde_model` |
| `rewrite` | Rewrite-Retrieve-Read (Ma et al. 2023): search queries rewritten by a frozen LLM | `hyde_model` |
| `multi_query` | RAG-Fusion (Rackauckas 2024): LLM-generated search queries | `hyde_model` |
| `query_expansion` | LLM paraphrases of the query | `max_query_variations` (3) |
| `query_decomposition` | Least-to-most prompting's decomposition (Zhou et al. 2022): the paper's few-shot prompt, parsed into the ordered sub-questions | `hyde_model` |
| `combined` | Diverse LLM reformulations of the query (RAG-Fusion style) | the pipeline's variation count |

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
| `ColBERTReranker`, `LLMListwiseReranker` | ColBERT late interaction, and RankGPT's sliding-window LLM reranking. |
| `DenseRetriever`, `BM25Retriever`, `SPLADERetriever`, `HybridRRFRetriever`, `HybridConvexRetriever` | `retrievers.py`: the five retriever components. |
| `MonoT5Reranker`, `LLMPointwiseReranker`, `FlashRankReranker`, `CohereReranker` | `rerankers.py`: the monoT5, LLM pointwise, FlashRank and Cohere reranker components. |

### `enhancement/`

| Class | Description |
|-------|-------------|
| `HyDEGenerator` | Hypothetical Document Embeddings — generates hypothetical answers to bridge query/document embedding gap. Supports multi-document HyDE with embedding averaging or max-similarity selection. Thread-safe LLM initialization. |
| `QueryExpander` | LLM-based query paraphrasing for improved recall. Generates alternative phrasings with different terminology. |
| `QueryDecomposer` | Least-to-most decomposition (Zhou et al. 2022): the paper's few-shot prompt, parsed into the ordered sub-questions a question needs. |
| `StepBackEnhancer`, `Query2DocEnhancer`, `RewriteEnhancer`, `MultiQueryEnhancer` | `augmentation.py`: the `step_back`, `query2doc`, `rewrite` and `multi_query` components, each with its paper's prompt. |
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
| BM25 (Okapi) | Robertson et al. 1995; Robertson and Zaragoza 2009 | Sparse keyword retrieval with optional Porter stemming |
| SPLADE | Formal et al. 2021, 2022 | Learned sparse retrieval (`splade` retriever, `sparse_method = "splade"`) |
| Convex score fusion | Bruch et al. 2023 | `hybrid_convex` retriever |
| monoT5 | Nogueira et al. 2020 | `monot5` reranker |
| RankGPT | Sun et al. 2023 | `llm_listwise` reranker |
| Query2doc, Step-back, Rewrite-Retrieve-Read, RAG-Fusion, least-to-most | Wang et al. 2023; Zheng et al. 2023; Ma et al. 2023; Rackauckas 2024; Zhou et al. 2022 | Query enhancer components |

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

Custom retrievers, rerankers and query enhancers can be registered via the component registry. In a sweep, name the component instead (`rerankers: [cross_encoder, my_colbert]`, or `{name: mine, retriever: my_retriever}`):

```python
from lasagna.retrieval.registry import component_registry
from lasagna.sdk import KnowledgeConfig

# Register a custom reranker
component_registry.register_reranker("my_colbert", MyColBERTReranker)

# Register a custom query enhancer
component_registry.register_enhancer("my_enhancer", MyEnhancer)

# Use via config (the enhancer is chosen with custom_query_enhancer in settings.lasagna.toml)
config = KnowledgeConfig(custom_reranker="my_colbert")

# List available components
component_registry.list_rerankers()
```

`component_registry` is the core registry (`lasagna.core.registry.registry`) under another name, so a component registered either way is the one a sweep names. The built-in retrievers, rerankers and query enhancers are listed under [Built-in Components](#built-in-components). The older `*_adapted` names still resolve to the same classes.

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
