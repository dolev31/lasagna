# Configuration, Data Classes, and Metrics

## Configuration Reference

### Backend/Session Config Keys

Set via `settings.lasagna.toml`, environment variables, or the session config API. These keys live in the `[rag]` section of the TOML config and are accessed through `RAGConfig` properties.

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `use_agentic_rag` | bool | `true` | Enable agentic multi-hop retrieval |
| `max_agent_iterations` | int | `3` | Maximum retrieval hops per query |
| `agentic_pipeline` | str | `"legacy_react"` | Pipeline selection: `"legacy_react"`, `"default"` (structured graph), a published loop (`"self_ask"`, `"ircot"`, `"flare"`, `"adaptive"`, `"self_rag"`, `"search_o1"`), or any registered `BaseAgenticPipeline` name |
| `agentic_models` | dict | `{}` | Model pins keyed by graph node name or by published loop name; empty uses the pipeline's model |
| `agentic_prompts` | dict | `{}` | Prompt overrides keyed by graph node name (the published loops keep their papers' prompts) |
| `use_structured_graph` | bool | `false` | *Legacy alias*: `true` maps to `agentic_pipeline="default"`, `false` maps to `"legacy_react"` |
| `agentic_hyde_strategy` | str | `"adaptive"` | HyDE strategy: `off`, `always`, `first_hop`, `adaptive` |
| `chunk_refinement_method` | str | `"auto"` | Refinement mode: `vanilla`, `refined`, `auto` |
| `refinement_model` | str | `MODEL_NAME`, else `"gpt-6-luna"` | LLM model for chunk refinement |
| `agent_temperature` | float | `0.1` | Temperature for agent LLM calls |
| `max_query_length` | int | `12` | Token threshold for HyDE adaptive mode |
| `enable_crag` | bool | `false` | Enable Corrective RAG document grading |
| `crag_web_search_fallback` | bool | `false` | Allow web search fallback on poor CRAG grade |
| `enable_query_decomposition` | bool | `false` | LLM-based sub-query decomposition for complex questions |
| `use_final_search` | bool | `false` | Run one more search on the original question after the hops |
| `final_search_mode` | str | `"merge"` | How final search results merge: `merge` or `replace` |
| `enable_hallucination_check` | bool | `false` | Self-RAG ISSUP grounding check (structured graph only) |
| `enable_answer_quality_check` | bool | `false` | Self-RAG ISUSE relevance check (structured graph only) |
| `max_quality_check_attempts` | int | `2` | Max retry attempts when quality gates fail |
| `coverage_scorer_mode` | str | `"keyword"` | Coverage scoring: `keyword` (fast) or `llm` (+200-500ms) |

#### `agentic_pipeline` Precedence

The effective pipeline is resolved in the following order:

1. **Explicit `agentic_pipeline` value**: set via TOML (`agentic_pipeline = "default"`), SDK config (`KnowledgeConfig(agentic_pipeline="default")`), REST API (`PATCH /api/v1/config`), config bridge, or session profile override.
2. **`use_structured_graph` boolean**: if `agentic_pipeline` is not explicitly set, `use_structured_graph = true` maps to `"default"` and `false` maps to `"legacy_react"`.
3. **Default**: if neither key is set, the system defaults to `"legacy_react"`.

When both `agentic_pipeline` and `use_structured_graph` are explicitly set, `agentic_pipeline` wins and `use_structured_graph` is ignored.

The resolved pipeline name is available in search metrics via `AgenticRAGMetrics.pipeline_name`.

#### Published Loop Settings

Each published loop reads its own settings, with its paper's defaults, instead of `max_agent_iterations`. All of them belong to the `agentic_pipeline` stage, so a sweep arm that changes one is recorded as an agentic change.

| Option | Type | Default | Loop | Description |
| ------ | ---- | ------- | ---- | ----------- |
| `self_ask_max_follow_ups` | int | `5` | `self_ask` | Follow-up questions before the loop asks for the final answer (the paper sets no bound) |
| `ircot_paragraphs_per_step` | int | `6` | `ircot`, `adaptive` | Paragraphs retrieved per step (K) |
| `ircot_max_steps` | int | `8` | `ircot`, `adaptive` | Reasoning steps before stopping |
| `ircot_max_paragraphs` | int | `15` | `ircot`, `adaptive` | Cap on paragraphs collected |
| `flare_retrieval_threshold` | float | `0.8` | `flare` | Retrieve when any token of the draft sentence is below this probability (theta) |
| `flare_mask_threshold` | float | `0.4` | `flare` | Mask tokens below this probability out of the query (beta) |
| `flare_max_sentences` | int | `8` | `flare` | Sentences written before stopping |
| `flare_docs_per_retrieval` | int | `2` | `flare` | Documents per retrieval |
| `flare_confidence_source` | str | `"auto"` | `flare` | `auto`, `logprobs` or `self_reported`; which one ran is recorded in `loop_trace` |
| `adaptive_rag_classifier_model` | str | `""` | `adaptive` | Model for the complexity classifier; empty uses the loop's model |
| `adaptive_rag_single_step_docs` | int | `5` | `adaptive` | Documents for the single-step route |
| `self_rag_passages` | int | `5` | `self_rag` | Passages retrieved and critiqued |
| `self_rag_weight_rel` | float | `1.0` | `self_rag` | Weight of the relevance critique (ISREL) |
| `self_rag_weight_sup` | float | `1.0` | `self_rag` | Weight of the support critique (ISSUP) |
| `self_rag_weight_use` | float | `0.5` | `self_rag` | Weight of the usefulness critique (ISUSE) |
| `self_rag_retrieval` | str | `"adaptive"` | `self_rag` | `adaptive` asks the model whether to retrieve; `always` always retrieves |
| `search_o1_max_searches` | int | `10` | `search_o1` | Searches before further queries get the authors' refusal message (their code's multi-hop default; 5 for single-hop QA) |
| `search_o1_max_turns` | int | `15` | `search_o1` | Generations before the loop stops (the authors' code; the paper sets none) |
| `search_o1_docs_per_search` | int | `10` | `search_o1` | Passages retrieved per search, each read by Reason-in-Documents (the code's `top_k`) |

`adaptive` and `self_rag` are prompted variants: their papers train a classifier and a critic model, and these prompt a general LLM instead. See [the agentic README](../README.md#design-notes) for what that leaves out.

#### Auto-Refinement Thresholds

Used when `chunk_refinement_method = "auto"`:

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `auto_refine_avg_length_threshold` | int | `1000` | Refine if avg chunk length exceeds this |
| `auto_refine_avg_score_threshold` | float | `0.5` | Refine if avg score is below this |
| `auto_refine_chunk_count_threshold` | int | `7` | Refine if chunk count exceeds this |

#### Chunk Retrieval Strategy `[rag.chunk_cite_retrieval]`

| Option | Type | Default | Description |
| ------ | ---- | ------- | ----------- |
| `strategy` | str | `"adaptive"` | Strategy: `fixed`, `above_threshold`, `adaptive`, `agent_decides` |
| `fixed_size` | int | `5` | Number of chunks for `fixed` strategy |
| `threshold_score` | float | `0.5` | Score cutoff for `above_threshold` |
| `max_chunks` | int | `20` | Ceiling for `adaptive` / `above_threshold` |
| `min_chunks` | int | `2` | Floor for `adaptive` |
| `score_gap_buffer` | int | `2` | Extra chunks past score gap (`adaptive`) |
| `min_gap_ratio` | float | `0.15` | Minimum relative gap to trigger cutoff |

### SDK-Available Keys

These keys are fields on the SDK's `KnowledgeConfig` dataclass and can be set when creating a `KnowledgeClient` or via `await client.set_config(...)`:

| SDK Field | Backend Equivalent | SDK Default |
| --------- | ------------------ | ----------- |
| `use_agentic_rag` | `use_agentic_rag` | `True` |
| `agentic_pipeline` | `agentic_pipeline` | `None` (resolved from `use_structured_graph`, so `"legacy_react"`) |
| `use_structured_graph` | `use_structured_graph` | `False` |
| `max_agent_iterations` | `max_agent_iterations` | `3` |
| `chunk_refinement_method` | `chunk_refinement_method` | `"auto"` |
| `refinement_model` | `refinement_model` | `MODEL_NAME`, else `"gpt-6-luna"` |
| `agent_temperature` | `agent_temperature` | `0.1` |
| `enable_crag` | `enable_crag` | `False` |
| `crag_web_search_fallback` | `crag_web_search_fallback` | `False` |
| `enable_query_decomposition` | `enable_query_decomposition` | `False` |
| `hyde_strategy` | `agentic_hyde_strategy` | `HyDEStrategy.ADAPTIVE` |
| `enable_hallucination_check` | `enable_hallucination_check` | `False` |
| `enable_answer_quality_check` | `enable_answer_quality_check` | `False` |
| `coverage_scorer_mode` | `coverage_scorer_mode` | `"keyword"` |

**Backend-only keys:** `max_quality_check_attempts`, `final_search_mode`, and `use_final_search` have no `KnowledgeConfig` field. Set them in `settings.lasagna.toml`, as environment variables (`RAG_<KEY>` or `LASAGNA_<KEY>`), or through the session config REST API. `agentic_pipeline` supersedes the legacy `use_structured_graph` boolean.

---

## Citation System

### Citation Class

Defined in `citation_manager.py`. Key fields:

| Field | Type | Description |
| ----- | ---- | ----------- |
| `citation_id` | `str` | Collision-resistant ID: `{session}_{query}_{content_hash}_{sequence}` |
| `index` | `int` | Display index (1, 2, 3...) |
| `chunk_id` | `str` | Source chunk identifier |
| `document_id` | `str` | Source document identifier |
| `document_name` | `str` | Source document filename |
| `content` | `str` | Full chunk content |
| `content_preview` | `str` | Truncated preview |
| `score` | `float` | Raw retrieval score |
| `normalized_score` | `float` | Normalized to [0, 1] for display |
| `chunk_index` | `int` | Position within document |
| `page_number` | `Optional[int]` | Page number (if available) |
| `section_header` | `Optional[str]` | Section heading |
| `stable_id` | `Optional[str]` | Content-hash for deduplication |
| `hop_index` | `Optional[int]` | Which retrieval hop produced this |
| `highlight_ranges` | `Optional[List[Tuple[int, int]]]` | Character spans for keyword highlights |

### Deduplication

Cross-hop deduplication uses content hashing (`_generate_content_hash`): normalizes whitespace and produces a 16-char hex SHA256 hash. The `seen_content_hashes` set in graph state prevents the same chunk from appearing in citations across different hops.

### Selection Strategies

After all hops complete, `select_chunks_node` filters accumulated citations:

- **`adaptive`** (default): Score-gap analysis finds the natural "knee" in the relevance distribution
- **`fixed`**: Always returns exactly N citations
- **`above_threshold`**: Returns all citations above a score threshold, capped at max
- **`agent_decides`**: LLM evaluates which chunks genuinely support the generated answer

### Citation Reindexing

`format_output_node` reindexes citations to contiguous `[1, 2, 3...]` regardless of original indices, ensuring clean display in the final `<knowledge_base>` context block.

---

## Metrics: AgenticRAGMetrics

Metrics object computed by `format_output_node`. All timing values are in milliseconds.

### Iteration Tracking

| Field | Description |
| ----- | ----------- |
| `agentic_enabled` | Whether agentic mode was active |
| `final_search_used` | Whether the final search on the original question ran |
| `agent_iterations_used` | Number of agent iterations (ReAct mode) |
| `hop_count` | Number of retrieval hops (graph mode) |
| `max_iterations_configured` | Configured iteration budget |

### Latency Breakdown

| Field | Description |
| ----- | ----------- |
| `total_latency_ms` | End-to-end latency |
| `hop_search_latencies_ms` | Per-hop search timing list |
| `hop_refinement_latencies_ms` | Per-hop refinement timing list |
| `agent_orchestration_ms` | LangGraph routing and tool dispatch |
| `agent_llm_calls_ms` | LLM call time (planning, synthesis) |
| `final_search_ms` | Final search time |
| `context_formatting_ms` | Context formatting time |
| `accounted_latency_ms` | Sum of all measured components |
| `unaccounted_latency_ms` | `total - accounted` (should be small) |

### Token Metrics

| Field | Description |
| ----- | ----------- |
| `total_tokens_before` | Estimated tokens before refinement |
| `total_tokens_after` | Estimated tokens after refinement |
| `total_tokens_saved` | Tokens saved by refinement |
| `reduction_percentage` | Token reduction as percentage |
| `total_chars_before` | Characters before refinement |
| `total_chars_after` | Characters after refinement |

### Refinement Tracking

| Field | Description |
| ----- | ----------- |
| `total_chunks_filtered` | Chunks removed as NOT_RELEVANT |
| `refinement_failures_count` | LLM refinement call failures |
| `refinement_skipped` | Whether auto-refinement decided not to refine |
| `refinement_skip_reason` | Why: `"low_variance"`, `"few_chunks"`, `"short_content"` |

### Coverage and Citations

| Field | Description |
| ----- | ----------- |
| `unique_docs_across_hops` | Distinct documents retrieved across all hops |
| `unique_pages_across_hops` | Distinct pages across all hops |
| `citation_count` | Number of citations in final output |
| `citation_missing_page_rate` | Fraction of citations without page numbers |
| `stop_reason` | Terminal condition: `"max_iterations"`, `"query_stabilized"`, `"graph_complete"`, etc. |

### Retrieval Features

| Field | Description |
| ----- | ----------- |
| `hybrid_search_enabled` | BM25 + dense hybrid active |
| `reranking_enabled` | Cross-encoder reranking active |
| `mmr_enabled` | Max Marginal Relevance active |
| `hyde_enabled` | HyDE query expansion active |

### CRAG Metrics

| Field | Description |
| ----- | ----------- |
| `crag_enabled` | Whether CRAG grading was active |
| `crag_grades` | Per-hop overall grades list |
| `crag_web_fallback_triggered` | Whether web fallback fired |
| `crag_grading_latency_ms` | CRAG grading latency |

---

## Data Classes

### AgenticRAGResult

Returned by `AgenticRAG.retrieve()` and `RAGBackend.async_agentic_search()`.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `formatted_context` | `str` | `<knowledge_base>` XML block for LLM consumption |
| `chunks` | `List[Dict]` | Final chunks with full content and metadata |
| `citations` | `List[Dict]` | Citation metadata (document, page, score, etc.) |
| `agent_summary` | `str` | Agent's synthesized answer |
| `search_count` | `int` | Number of searches performed |
| `tokens_before` | `int` | Estimated tokens before refinement |
| `tokens_after` | `int` | Estimated tokens after refinement |
| `tokens_saved` | `int` | Tokens saved by refinement |
| `reduction_percentage` | `float` | Token reduction percentage |
| `chunks_filtered` | `int` | Chunks removed as irrelevant |
| `hops` | `List[HopData]` | Full multi-hop tracking data |
| `metrics` | `AgenticRAGMetrics` | All metrics (see above) |

### HopData

Per-hop tracking data accumulated across retrieval iterations.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `hop_index` | `int` | Search iteration number |
| `query_used` | `str` | The search query for this hop |
| `retrieved_chunks` | `List[RetrievedChunkData]` | Chunks retrieved in this hop |
| `search_latency_ms` | `float` | Search timing |
| `refinement_latency_ms` | `float` | Refinement timing |
| `tokens_before_refinement` | `int` | Tokens before refinement |
| `tokens_after_refinement` | `int` | Tokens after refinement |
| `chunks_filtered` | `int` | Chunks filtered in this hop |
| `refinement_skipped` | `bool` | Whether auto-refinement skipped |
| `refinement_skip_reason` | `str` | Why refinement was skipped |
| `hyde_used` | `bool` | Whether HyDE was applied |
| `hyde_reason` | `str` | Why HyDE was/wasn't used |
| `crag_grade` | `str` | CRAG grade for this hop |
| `crag_grades_detail` | `List[str]` | Per-chunk CRAG grades |
| `crag_web_fallback_used` | `bool` | Whether web fallback fired |

Properties: `unique_doc_ids`, `unique_page_numbers`

### RetrievedChunkData

Individual chunk with citation metadata.

| Field | Type | Description |
| ----- | ---- | ----------- |
| `chunk_id` | `str` | Chunk identifier |
| `document_id` | `str` | Source document |
| `document_name` | `str` | Source filename |
| `content` | `str` | Chunk text (may be refined) |
| `score` | `float` | Retrieval score |
| `chunk_index` | `int` | Position within document |
| `page_number` | `Optional[int]` | Page number |
| `section_header` | `Optional[str]` | Section heading |
| `metadata` | `Dict[str, Any]` | Additional metadata |
| `original_content` | `Optional[str]` | Pre-refinement content (for tracking) |
| `is_refined` | `bool` | Whether this chunk was refined |
| `refinement_kept` | `bool` | `False` if filtered as NOT_RELEVANT |
