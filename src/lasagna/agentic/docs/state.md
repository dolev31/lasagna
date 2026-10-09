# RAGGraphState Reference

Defined in `graph_state.py`. A `TypedDict` with `total=False` so every field is optional at construction time and nodes can return partial state updates.

---

## Field Groups

### Input (immutable after init)

| Field | Type | Description |
| ----- | ---- | ----------- |
| `original_question` | `str` | User's question |
| `session_id` | `str` | Session identifier |
| `document_filter` | `Optional[List[str]]` | Restrict search to specific document IDs |
| `top_k` | `int` | Number of chunks to retrieve per hop |
| `trace_id` | `Optional[str]` | Distributed tracing identifier |

### Routing

| Field | Type | Description |
| ----- | ---- | ----------- |
| `routing_decision` | `Any` | Pre-computed routing tier from upstream orchestrator (FAST/STANDARD/COMPREHENSIVE). Never recomputed inside the graph. |

### Session Config

| Field | Type | Description |
| ----- | ---- | ----------- |
| `session_config` | `Any` | `RAGConfig` instance (session-scoped) |

### LLM

| Field | Type | Description |
| ----- | ---- | ----------- |
| `llm` | `Any` | `BaseChatModel` shared across all nodes |

### Query Planning

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `query_complexity` | `str` | `"moderate"` | Complexity level: `"simple"`, `"moderate"`, `"complex"` |
| `sub_queries` | `List[str]` | `[question]` | Decomposed sub-queries (or original question as single-element list) |
| `current_sub_query_index` | `int` | `0` | Index into `sub_queries` |
| `current_query` | `str` | `question` | Active search query (may be rewritten) |

### Retrieval

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `retrieved_chunks` | `List[RetrievedChunkData]` | `[]` | Raw results from current hop (reset each iteration) |
| `hop_data` | `List[HopData]` | `[]` | Accumulated across all hops |
| `hop_count` | `int` | `0` | Number of search iterations performed |

### Refinement

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `refined_chunks` | `List[RetrievedChunkData]` | `[]` | Chunks after refinement |
| `refinement_method` | `str` | from config | Active method: `"vanilla"`, `"refined"`, `"auto"` |
| `refinement_metrics` | `Dict[str, Any]` | `{}` | Per-hop: tokens_before/after, chunks_filtered, etc. |

### Citations

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `global_citation_index` | `int` | `1` | Monotonically increasing across hops |
| `all_hop_citations` | `List[Dict[str, Any]]` | `[]` | Accumulated citation dicts from all hops |
| `seen_content_hashes` | `Set[str]` | `set()` | Content hashes for cross-hop deduplication |

### CRAG Grading

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `document_grades` | `List[str]` | `[]` | Per-chunk: `"correct"`, `"ambiguous"`, `"incorrect"` |
| `overall_grade` | `str` | `""` | Aggregate grade for routing |
| `graded_chunks` | `List[RetrievedChunkData]` | `[]` | Chunks after grading/filtering |
| `crag_web_fallback_used` | `bool` | `False` | Whether web fallback was triggered |
| `crag_grading_latency_ms` | `float` | `0.0` | CRAG grading latency |

### Generation

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `generation` | `str` | `""` | LLM-generated answer text |
| `agent_summary` | `str` | `""` | Copy of generation (matches `AgenticRAGResult` field name) |
| `generation_citations` | `List[str]` | `[]` | Extracted `[N]` references from generation |

### Quality Gates

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `hallucination_check` | `str` | `""` | `"supported"`, `"partially_supported"`, `"not_supported"` |
| `answer_quality` | `str` | `""` | `"addresses_question"`, `"partially_addresses"`, `"does_not_address"` |
| `quality_check_attempts` | `int` | `0` | Shared counter for both quality gates |

### Coverage

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `coverage_ratio` | `float` | `0.0` | How well chunks cover query aspects (0.0-1.0) |
| `coverage_confidence` | `float` | `0.0` | Confidence in coverage assessment (0.0-1.0) |
| `uncovered_aspects` | `List[str]` | `[]` | Missing topics (fed to rewrite_query) |
| `suggested_followup` | `Optional[str]` | `None` | Query suggestion for gap-filling |

### Chunk Selection

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `marked_chunks` | `List[int]` | `[]` | Result numbers selected by LLM (agent_decides strategy) |
| `filtered_citations` | `List[Dict[str, Any]]` | `[]` | Citations after strategy-based filtering |

### Flow Control

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `retrieval_attempts` | `int` | `0` | Current hop counter |
| `max_retrieval_attempts` | `int` | `3` | Budget limit (may be adjusted by plan_retrieval) |
| `rewrite_count` | `int` | `0` | Query rewrite counter |
| `max_rewrites` | `int` | `3` | Rewrite budget limit |
| `stop_reason` | `str` | `""` | Terminal condition: `"max_iterations"`, `"query_stabilized"`, `"sufficient_coverage"`, `"graph_complete"` |

### Metrics

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `timing_spans` | `Dict[str, float]` | `{}` | Accumulated timing: `search_ms`, `refinement_ms`, etc. |
| `rag_metrics_list` | `List[Dict[str, Any]]` | `[]` | Per-hop metrics (mirrors configurable `rag_metrics`) |
| `total_start_time` | `float` | `0.0` | `time.time()` at graph entry |

### Query Deduplication

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `previous_queries` | `List[str]` | `[]` | All queries searched so far |
| `previous_query_embeddings` | `List[Any]` | `[]` | Embedding vectors for semantic dedup |

### Final Output

| Field | Type | Default | Description |
| ----- | ---- | ------- | ----------- |
| `formatted_context` | `str` | `""` | `<knowledge_base>` XML context block |
| `final_chunks` | `List[Dict[str, Any]]` | `[]` | Final chunks with full content |
| `final_citations` | `List[Dict[str, Any]]` | `[]` | Reindexed citation metadata |
| `final_metrics` | `AgenticRAGMetrics` | `None` | Comprehensive metrics object |

---

## build_initial_state()

```python
def build_initial_state(
    question: str,
    session_id: str,
    top_k: int,
    session_config: Any,
    llm: Any,
    *,
    document_filter: Optional[List[str]] = None,
    trace_id: Optional[str] = None,
    routing_decision: Any = None,
    max_retrieval_attempts: int = 3,
    max_rewrites: int = 3,
) -> RAGGraphState:
```

Centralizes default-value logic so `AgenticRAG.retrieve()` and tests can construct state consistently. Sets all fields to their default values as shown in the tables above.
