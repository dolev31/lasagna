# retrieval: first-stage search, reranking, query enhancement, routing

The query-time stages other than generation and the agentic loop: the `retriever`, `reranker`
and `query_enhancer` components, the hybrid dense plus BM25 searcher the default pipeline uses,
the query router, and the optional GraphRAG and RAPTOR indexes. `api/_search_orchestrator.py`
composes these into a search; this package holds the pieces.

The registered components and the paper each implements: `lasagna component list retriever`
(or `reranker`, `query_enhancer`), and [docs/guide.md](../../../docs/guide.md#3-the-stages-you-can-swap).
Settings and defaults: [docs/reference/settings.md](../../../docs/reference/settings.md).
What each profile turns on: [docs/configuration.md](../../../docs/configuration.md#profiles).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `search/hybrid.py` | `HybridSearcher` (dense plus sparse, fused with RRF; per-session BM25 indexes with LRU and idle TTL, `bm25_max_sessions`, `bm25_session_ttl_seconds`), `MilvusNativeHybridSearcher` (Milvus server-side sparse and `hybrid_search`), `BM25FallbackIndex` (rank_bm25 in memory, for backends without native BM25 such as Milvus Lite), `SPLADESparseEncoder` |
| `search/retrievers.py` | Retriever components: `DenseRetriever`, `BM25Retriever`, `SPLADERetriever`, `HybridRRFRetriever`, `HybridConvexRetriever` |
| `search/ranker.py` | `CrossEncoderReranker`, `MMRReranker`, `CombinedReranker` (cross-encoder, relevance filter, MMR), `EnsembleReranker`, `ColBERTReranker`, `LLMListwiseReranker`, and the `*Adapter` classes registered as components |
| `search/rerankers.py` | `MonoT5Reranker`, `LLMPointwiseReranker`, `FlashRankReranker`, `CohereReranker` |
| `enhancement/enhancer.py` | `HyDEGenerator`, `QueryExpander`, `QueryDecomposer`, `QueryEnhancer` (the combined pipeline), and their component adapters |
| `enhancement/augmentation.py` | `StepBackEnhancer`, `Query2DocEnhancer`, `RewriteEnhancer`, `MultiQueryEnhancer` on a shared `_PromptedEnhancer` |
| `enhancement/classifier.py` | `QueryClassifier`: rule-based query type, with recommended retrieval parameters; no LLM |
| `enhancement/coverage.py`, `enhancement/llm_coverage.py` | `CoverageScorer` (keyword aspects) and `LLMCoverageScorer`, used by the agentic loop to decide whether to search again |
| `routing/` | `QueryRouter` (`router.py`, process singleton), `RuleBasedStrategy` and `AdaptiveStrategy` (`strategies.py`), signal extraction (`signals.py`), `RoutingTier`, `RoutingSignals`, `RoutingDecision` (`models.py`) |
| `graph/` | GraphRAG: `EntityExtractor` at ingestion, `InMemoryGraphStore` per session (NetworkX, `graphrag` extra), `GraphRetriever` expansion at query time; on with `enable_graph_retrieval` |
| `raptor/` | RAPTOR (Sarthi et al. 2024): `RaptorTreeBuilder` adds recursive summary chunks at ingestion when `enable_raptor` is on; `RaptorRetriever` |
| `registry.py` | `component_registry`: the core registry (`lasagna.core.registry.registry`) under its older name |

## Built-in components

Registered in `core/registry.py` as lazy `module:Class` factories:

| Stage | Names | Classes |
| ----- | ----- | ------- |
| `retriever` | `dense`, `bm25`, `splade`, `hybrid_rrf`, `hybrid_convex` | `search/retrievers.py` |
| `reranker` | `cross_encoder`, `combined`, `ensemble`, `mmr`, `colbert`, `llm_listwise` | `*RerankerAdapter` in `search/ranker.py` |
| `reranker` | `monot5`, `llm_pointwise`, `flashrank`, `cohere` | `search/rerankers.py` |
| `query_enhancer` | `hyde`, `query_expansion`, `query_decomposition`, `combined` | `*Adapter` in `enhancement/enhancer.py` |
| `query_enhancer` | `step_back`, `query2doc`, `rewrite`, `multi_query` | `enhancement/augmentation.py` |

The older `*_adapted` names resolve to the same classes. Each component reads its settings from
the arm's config; the description of every setting in
[docs/reference/settings.md](../../../docs/reference/settings.md) names the component that
reads it.

## Design notes

**A named retriever is the first stage.** When an arm names a retriever component, the
orchestrator calls it instead of the vector store, does not impose its own filters and search
parameters on it, and turns hybrid fusion off for that search, so `bm25` is never BM25 fused
with BM25. Built-in retrievers set `_forwards_session_config` and receive the arm's config and
a `RetrievalContext` (backend, collection, session filter, query embedding, the shared
`HybridSearcher`). Reranking and later stages run as usual.

**Components raise rather than return the input order.** A reranker or enhancer whose model,
package, key or LLM is missing raises. Returning candidates unchanged would score the control
under the component's name. Built-ins that load a model record `component:model` as what ran.

**Enhancers are compared under one fusion.** Every enhancer returns the original query first,
then its additions. The orchestrator searches each and fuses with weighted RRF, the original at
weight 2.0 and each addition at 1.0 (`search_multi_query`), so comparing enhancers measures
what each adds. The LLM enhancers use `hyde_model` at temperature 0, which holds the model
fixed across a sweep.

**Sparse scoring covers every backend.** `bm25` and `splade` score every passage of the
session in process, so `bm25_k1`, `bm25_b` and `splade_model` apply even on backends whose
native BM25 fixes parameters at index time. `BM25FallbackIndex` rebuilds lazily after a
removal and uses a version counter, so a build that raced a mutation is not committed.

**Routing tiers.** `QueryRouter.route()` picks FAST, STANDARD or COMPREHENSIVE from cheap
signals (no LLM). FAST turns heavy features off for that search; COMPREHENSIVE hands the query
to the agentic pipeline (`SearchOrchestrator._search_comprehensive`). A caller can force a tier
with `routing_tier=`; `routing_default_tier` applies only when `enable_query_routing` is off.
`enable_crag` forces COMPREHENSIVE, since CRAG runs only in the agentic graph.

**Thread safety.** `HybridSearcher` guards its session map with `_session_lock`;
`BM25FallbackIndex` serializes builds with `_build_lock`; `CrossEncoderReranker` loads under
`_config_lock`; the LLM-backed enhancers create their client under `_llm_lock`; routing
strategies are stateless and created once in `QueryRouter._strategy_cache`. Model inference
runs inside `core.accelerator.device_guard` on Apple MPS.
