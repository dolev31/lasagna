# utils: observability, accounting, small shared helpers

Helpers every other package imports: metrics and tracing facades, the per-request record of
stages that fell back, token accounting and the experiment's LLM response cache, score
normalization, token estimation, model-name resolution and logging setup. Optional
dependencies (`prometheus-client`, `opentelemetry-*`, `tiktoken`) are imported inside
`try/except ImportError`, and a disabled feature costs a no-op call.

Observability and cache settings with their defaults:
[docs/reference/settings.md](../../../docs/reference/settings.md). Repeatable runs and the
response cache from a user's side:
[docs/configuration.md](../../../docs/configuration.md#temperature-and-repeatable-runs).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `monitoring.py` | `METRICS` (`MetricsClient` facade: no-op, or Prometheus after `configure_metrics(enable_prometheus=True)`), `StageTimer`, `log_json`, `log_json_debug`, `hash_text`, `trunc_text`, `summarize_document_ids` |
| `tracing.py` | `TRACER` (`Tracer` facade over OpenTelemetry), `configure_tracing`; exporters `otlp`, `console`, `zipkin`, and `jaeger` (deprecated, warns) |
| `degraded.py` | `record(stage, reason)` and `collecting()`: counts a stage that fell back (`lasagna_degraded_fallbacks_total{stage=...}`) and collects it for the response's `degraded_stages` |
| `token_usage.py` | `UsageMeter`, `attach_usage_meter`, `instrument_openai_client` (the RAGAS judge client): tokens and LLM calls; tokens only, never prices |
| `llm_cache.py` | The experiment's LLM response cache: `LLMCacheSettings`, `llm_cache_scope`, `attach_llm_cache`, one SQLite file (`default_cache_path()`: `~/.lasagna/cache/llm.sqlite`) |
| `normalization.py` | `normalize_score_sigmoid`, `normalize_scores_minmax`, `normalize_scores_for_mmr`, `normalize_score_for_display`, `normalize_score_relative`, `adaptive_k_selection` |
| `tokenization.py` | `estimate_tokens` (tiktoken when installed, else `len(text) // 4`), `estimate_tokens_batch`, `count_tokens_for_messages`, `fits_in_context`, `truncate_to_token_limit` |
| `model_utils.py` | `resolve_model_settings` (`None`, `"default"`, `"platform:model"`, a dotted settings path, or a dict), `get_default_model_name` (`MODEL_NAME`, then the config provider, then a fallback), `message_text` |
| `logging_utils.py` | `configure_knowledge_logging(level)`: a DEBUG handler for `lasagna.*` loggers; `LASAGNA_SKIP_LOGGING=1` skips it |

## Design notes

**Facades keep call sites unconditional.** `METRICS` and `TRACER` are created at import.
`configure_metrics()` and `configure_tracing()`, called from `RAGBackend` init with
`enable_prometheus` and `enable_tracing`, swap the backend behind them, so references held
anywhere stay valid and no call site checks whether observability is on. The Prometheus backend
registers each metric once under a lock to avoid duplicated time series.

**Fallbacks are reported, not hidden.** HyDE, GraphRAG, hybrid search and the agentic tier
catch their own failures and continue with less. Each calls `degraded.record`, which counts it
and, inside `collecting()`, adds the stage to what a REST response reports. The collection is a
`ContextVar`, so it follows a search into its worker thread.

**Tokens are counted where models are built.** `SimpleLLMProvider.get_chat_model()` attaches
the usage meter, so every LLM call in the pipeline and the harness is counted, including ones
added later. Dollars are computed only on request (`prices:` in a sweep, in
`knowledge_evaluation_sdk/cost.py`) and labeled as an estimate.

**The response cache replays only what is deterministic.** Inside `llm_cache_scope` (an
experiment), a call is cached only when it asked for temperature 0 and the model kept it
(`deterministic_only`, the default). The run seed is part of the key, so run *k* replays run
*k*. A replayed response carries its original usage, so token counts read the same cold or
warm, and hits and misses are counted separately. Outside a scope the cache defers to
LangChain's global cache, so `lasagna serve` never writes prompts to disk.
