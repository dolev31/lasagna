# `utils/` — Cross-Cutting Utilities

Shared utilities for the RAG knowledge system: observability (metrics, tracing),
score normalization, token counting, model resolution, and structured logging.
Every other knowledge module depends on `utils/`. All utilities follow a
**zero-cost-by-default** design: disabled features have zero overhead (no-op
backends, no extra imports, no allocations).

## Module Map

| File | Lines | Purpose | Key Exports |
|---|---|---|---|
| `monitoring.py` | 215 | Prometheus metrics facade, structured JSON logging, text helpers | `METRICS`, `MetricsClient`, `configure_metrics`, `StageTimer`, `trunc_text`, `hash_text`, `log_json` |
| `tracing.py` | 237 | OpenTelemetry distributed tracing facade (OTLP, Jaeger, Zipkin, Console) | `TRACER`, `Tracer`, `configure_tracing` |
| `normalization.py` | 233 | Score normalization (sigmoid, minmax, MMR, adaptive-k) | `normalize_score_sigmoid`, `normalize_scores_for_mmr`, `adaptive_k_selection` |
| `tokenization.py` | 218 | Token counting with tiktoken fallback | `estimate_tokens`, `estimate_tokens_batch`, `fits_in_context`, `truncate_to_token_limit` |
| `model_utils.py` | 201 | LLM model reference resolution via provider abstraction | `resolve_model_settings` |
| `logging_utils.py` | 39 | Knowledge-scoped loguru DEBUG handler | `configure_knowledge_logging` |

**Total: ~1,145 lines across 7 files, 26 public exports.**

## Architecture

```
      +-----------+    +-----------+    +-------------+    +------------+
      | monitoring|    |  tracing  |    | normalization|    | tokenization|
      |  (facade) |    |  (facade) |    | (pure funcs) |    |  (cached)  |
      +-----+-----+    +-----+-----+    +------+------+    +-----+------+
            |                |                  |                 |
     +------+------+   +----+-------+           |                 |
     | No-op (def) |   | No-op      |     All stateless     +----+--------+
     | Prometheus  |   | OTel/OTLP  |     (thread-safe)     | tiktoken    |
     +-------------+   | Jaeger     |                       | (optional)  |
                        | Zipkin     |                       | heuristic   |
                        | Console    |                       | (fallback)  |
                        +------------+                       +-------------+

                     +-------------+
                     | model_utils |
                     |  (provider) |
                     +------+------+
                            |
                    ConfigProvider / LLMProvider
                      (from providers/)
```

**Design patterns:**

- **Facade + Singleton**: `METRICS` and `TRACER` are module-level singletons.
  Backends are swapped at runtime via `configure_metrics()` / `configure_tracing()`.
  All call-site references remain valid after a backend swap.
- **Double-checked locking**: `_PrometheusBackend` uses `threading.Lock` for
  lazy metric registration, preventing "Duplicated timeseries" errors.
- **Graceful degradation**: Optional dependencies (`prometheus-client`, `tiktoken`,
  `opentelemetry-*`) are wrapped in `try/except ImportError` with no-op or
  heuristic fallback.
- **Pure functions**: All normalization functions are stateless and thread-safe.
- **LRU caching**: Tokenizer encoders are cached via `@lru_cache(maxsize=4)`.

## Singletons

| Singleton | Accessor | Configure | Default Mode |
|---|---|---|---|
| `METRICS` | `from lasagna.utils import METRICS` | `configure_metrics(enable_prometheus=True)` | No-op (zero overhead) |
| `TRACER` | `from lasagna.utils import TRACER` | `configure_tracing(enable=True, ...)` | No-op (zero overhead) |

## Configuration Reference

| Parameter | Default | TOML | ENV | SDK | REST | UI |
|---|---|---|---|---|---|---|
| `enable_prometheus` | `false` | Y | `RAG_ENABLE_PROMETHEUS` | `KnowledgeConfig(enable_prometheus=True)` | Y | Y |
| `enable_tracing` | `false` | Y | `RAG_ENABLE_TRACING` | `KnowledgeConfig(enable_tracing=True)` | Y | Y |
| `tracing_exporter` | `"otlp"` | Y | `RAG_TRACING_EXPORTER` | -- | -- | -- |
| `tracing_endpoint` | `"http://localhost:4317"` | Y | `RAG_TRACING_ENDPOINT` | -- | -- | -- |
| `tracing_service_name` | `"rag-knowledge"` | Y | `RAG_TRACING_SERVICE_NAME` | -- | -- | -- |
| `mmr_score_normalization` | `"sigmoid"` | Y | `RAG_MMR_SCORE_NORMALIZATION` | `KnowledgeConfig(mmr_score_normalization=...)` | -- | Y |
| `log_level` | `"INFO"` | Y | `RAG_LOG_LEVEL` | -- | -- | Y |

`tracing_*` detail parameters are TOML+ENV only by design (infrastructure concern, not per-session).

Additional env vars: `LASAGNA_SKIP_LOGGING=1` bypasses the knowledge log handler.
`RAG_LANGFUSE_TRACING=true` enables Langfuse tracing integration.

## Code Examples

### Metrics

```python
from lasagna.utils import METRICS, configure_metrics, StageTimer

# Enable Prometheus (typically once at startup)
configure_metrics(enable_prometheus=True)

# Emit metrics anywhere in the pipeline
METRICS.counter("rag.search.queries", tags={"session": session_id})

timer = StageTimer()
results = search(query)
METRICS.histogram("rag.search.latency_ms", timer.elapsed_ms(), tags={"stage": "rerank"})
```

### Tracing

```python
from lasagna.utils import TRACER, configure_tracing

configure_tracing(enable=True, exporter="otlp", endpoint="http://localhost:4317")

with TRACER.span("dense_search", attributes={"top_k": 75}) as span:
    results = vector_store.search(query_embedding, top_k=75)
    span.set_attribute("result_count", len(results))
```

### Score Normalization

```python
from lasagna.utils import (
    normalize_score_sigmoid,
    normalize_scores_for_mmr,
    adaptive_k_selection,
)

# Normalize cross-encoder logits for display
display_score = normalize_score_sigmoid(-2.5)  # -> ~0.076

# Normalize for MMR (makes scores comparable to cosine similarity)
normalized = normalize_scores_for_mmr(raw_scores, strategy="sigmoid_minmax")

# Determine optimal number of results via score-gap analysis
k = adaptive_k_selection(scores, min_k=2, max_k=20, buffer=2)
```

### Token Counting

```python
from lasagna.utils import estimate_tokens, fits_in_context, truncate_to_token_limit

tokens = estimate_tokens("Your text here")  # accurate with tiktoken, ~4 chars/token fallback

if fits_in_context(chunk_text, max_tokens=4096, safety_margin=0.95):
    context.append(chunk_text)
else:
    chunk_text = truncate_to_token_limit(chunk_text, max_tokens=4096)
```

### Model Resolution

```python
from lasagna.utils import resolve_model_settings

# Supports: None, "default", "openai:gpt-6-luna", "agent.action.model", dict
settings = resolve_model_settings(
    model_ref="openai:gpt-6-luna",
    default_settings=default,
    max_tokens=1000,
    context="Query enhancement",
)
```

## Design Decisions

1. **Facade pattern for observability** — `METRICS` and `TRACER` use facade + singleton
   so call-sites never check whether observability is enabled. No-op backends have zero
   overhead (no allocations, single virtual dispatch).

2. **Module-level singletons** — `METRICS` and `TRACER` are instantiated at import time.
   `configure_*()` swaps the internal backend while keeping all existing references valid.
   This avoids dependency injection boilerplate across ~20 call-sites.

3. **Adaptive-k over fixed top-k** — `adaptive_k_selection()` implements the EMNLP 2025
   Adaptive-k approach: finds natural score gaps in the relevance distribution to prevent
   both over-retrieval (hallucination) and under-retrieval (information loss).

4. **tiktoken with heuristic fallback** — Accurate token counting matters for context
   window management (exceeding limits causes hard failures). tiktoken provides exact
   counts; the 4-chars/token heuristic handles environments without the dependency.

5. **Prometheus metric deduplication** — `_PrometheusBackend` uses double-checked locking
   to prevent "Duplicated timeseries" errors from concurrent metric registration. Metrics
   are lazily created on first use with thread-safe `threading.Lock`.

## Integration Points

| Consumer | Utils Used |
|---|---|
| `api/rag_backend.py` | `configure_metrics`, `configure_tracing` (startup) |
| `api/_search_orchestrator.py` | `METRICS`, `TRACER`, `log_json`, `trunc_text`, `hash_text` |
| `api/_document_pipeline.py` | `METRICS`, `TRACER` |
| `api/rest/app.py` | `METRICS` (request middleware) |
| `agentic/agentic_rag.py` | `resolve_model_settings`, `estimate_tokens`, `normalize_score_relative`, `adaptive_k_selection` |
| `agentic/nodes/select_chunks.py` | `adaptive_k_selection` |
| `agentic/nodes/refine_chunks.py` | `resolve_model_settings`, `estimate_tokens` |
| `agentic/nodes/retrieve.py` | `estimate_tokens` |
| `agentic/document_awareness.py` | `trunc_text` |
| `retrieval/search/ranker.py` | `normalize_score_sigmoid`, `normalize_scores_minmax`, `normalize_scores_for_mmr` |
| `retrieval/enhancement/enhancer.py` | `resolve_model_settings` |
| `core/config.py` | `configure_knowledge_logging` |
| `mcp/server.py` | `hash_text` |
