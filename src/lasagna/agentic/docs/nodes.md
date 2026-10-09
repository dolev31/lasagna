# Node Reference

13 graph nodes implemented across 11 files in `nodes/`. Each node is an async function that receives `RAGGraphState` and returns a partial state update dict.

---

## plan_retrieval_node

**File:** `nodes/plan_retrieval.py`

**Purpose:** Map upstream routing tier to query complexity, set iteration budget, and optionally decompose the question into sub-queries.

**Reads:** `routing_decision`, `original_question`, `max_retrieval_attempts`, `session_config`, `llm`

**Writes:** `query_complexity`, `sub_queries`, `current_sub_query_index`, `current_query`, `max_retrieval_attempts`

**Logic:** Translates routing tier (FAST/STANDARD/COMPREHENSIVE) to complexity ("simple"/"moderate"/"complex") and applies adaptive iteration budgets per tier. When `enable_query_decomposition` is enabled and complexity is "complex", calls the LLM to decompose the question into 2-4 focused sub-queries. Sets `current_query` to the first sub-query (or original question if no decomposition).

---

## retrieve_node

**File:** `nodes/retrieve.py`

**Purpose:** Execute a single retrieval hop against the knowledge base with deduplication, HyDE, and coverage scoring.

**Reads:** `current_query`, `session_id`, `document_filter`, `top_k`, `trace_id`, `previous_queries`, `previous_query_embeddings`, `hop_count`, `hop_data`, `session_config`, `max_retrieval_attempts`, `retrieval_attempts`, `timing_spans`, `llm`

**Writes:** `retrieved_chunks`, `hop_count`, `previous_queries`, `previous_query_embeddings`, `coverage_ratio`, `coverage_confidence`, `uncovered_aspects`, `suggested_followup`, `timing_spans`, `retrieval_attempts`, `stop_reason`

**Logic:** Performs exact and semantic query deduplication (cosine similarity >= 0.92 threshold). Makes adaptive HyDE decision based on strategy config. Executes backend search via `RAGBackend.search()`. Converts results to `RetrievedChunkData` objects. Computes coverage metrics to inform downstream rewrite targeting. Sets `stop_reason = "query_stabilized"` and returns early on deduplicated queries.

---

## refine_chunks_node

**File:** `nodes/refine_chunks.py`

**Purpose:** Apply chunk refinement (vanilla, auto, or refined) with citation building, deduplication, and token tracking.

**Reads:** `retrieved_chunks`, `refinement_method`, `current_query`, `session_config`, `hop_count`, `global_citation_index`, `all_hop_citations`, `seen_content_hashes`, `hop_data`, `rag_metrics_list`, `timing_spans`

**Writes:** `refined_chunks`, `global_citation_index`, `all_hop_citations`, `seen_content_hashes`, `hop_data`, `refinement_metrics`, `rag_metrics_list`, `timing_spans`

**Logic:** Three paths: **vanilla** passes chunks through with dedup + citation building; **auto** applies a heuristic gate (chunk count > 7, avg length > 1000, high score variance) and falls back to refined when triggered; **refined** runs parallel LLM refinement per chunk with NOT_RELEVANT filtering. Content-hash deduplication prevents duplicates across hops. Builds `HopData` entry and records token reduction metrics. Safety net: if all chunks filtered, keeps the top-scoring original.

---

## grade_documents_node

**File:** `nodes/grade_documents.py`

**Purpose:** Apply CRAG quality grading to refined chunks.

**Reads:** `refined_chunks`, `current_query`, `session_config`

**Writes:** `overall_grade`, `graded_chunks`, `document_grades`, `crag_grading_latency_ms`, `crag_web_fallback_used`, `hop_data` (backfill)

**Logic:** When CRAG is disabled or no chunks exist, fast-paths with `overall_grade = "correct"` and passes all chunks through. When enabled, grades each chunk in parallel via LLM and aggregates by majority vote. Filters non-relevant chunks on "correct"/"ambiguous" grades. On "incorrect" grade, preserves all chunks for potential web fallback. Backfills CRAG metadata onto the latest `HopData` entry.

---

## generate_node

**File:** `nodes/generate.py`

**Purpose:** Generate a grounded LLM answer with `[N]` citation references.

**Reads:** `llm`, `original_question`, `graded_chunks` (fallback: `refined_chunks`, then `retrieved_chunks`), `timing_spans`

**Writes:** `generation`, `agent_summary`, `generation_citations`, `timing_spans`

**Logic:** Formats chunks as numbered context blocks (`[1]\ncontent\n[2]\ncontent...`). Invokes LLM with a grounded-generation prompt instructing it to cite sources with `[N]` references. Extracts all `[N]` citation references from the response via regex. Cascades through available chunk sources in priority order.

---

## rewrite_query_node

**File:** `nodes/rewrite_query.py`

**Purpose:** LLM-based query refinement targeting uncovered aspects.

**Reads:** `rewrite_count`, `llm`, `original_question`, `current_query`, `hop_data`, `uncovered_aspects`

**Writes:** `current_query`, `rewrite_count`

**Logic:** Builds a results summary from `hop_data` and a missing-info description from `uncovered_aspects`. Calls LLM with the query refinement prompt. If the LLM returns "SUFFICIENT", keeps the current query unchanged (enabling stabilization detection in the next retrieve). Increments `rewrite_count` for budget tracking.

---

## select_chunks_node

**File:** `nodes/select_chunks.py`

**Purpose:** Filter accumulated citations using the configured selection strategy.

**Reads:** `session_config`, `all_hop_citations`, `generation`, `llm`, `marked_chunks`

**Writes:** `filtered_citations`, `marked_chunks`

**Logic:** Applies one of four strategies from `chunk_cite_retrieval` config:

- **`agent_decides`** -- LLM evaluates which chunks actually support the generation
- **`fixed`** -- Takes first `fixed_size` citations
- **`above_threshold`** -- Filters by `threshold_score`, capped at `max_chunks`
- **`adaptive`** -- Score-gap analysis via `adaptive_k_selection()` with configurable buffer

Falls back to first 5 citations if strategy is unrecognized.

---

## format_output_node

**File:** `nodes/format_output.py`

**Purpose:** Terminal node -- assemble final result with reindexed citations and comprehensive metrics.

**Reads:** `filtered_citations`, `hop_data`, `session_config`, `total_start_time`, `rag_metrics_list`, `stop_reason`, `final_search_results`, `seen_content_hashes`

**Writes:** `formatted_context`, `final_chunks`, `final_citations`, `final_metrics`, `stop_reason`

**Logic:** Looks up full chunk content from `hop_data` for each citation. Reindexes citations to contiguous `[1, 2, 3...]` indices. Merges optional final-search results using configured mode (`merge` or `replace`). Builds `<knowledge_base>` XML context block. Computes `AgenticRAGMetrics` with latency breakdown, token reduction stats, CRAG flags, and quality booster status. Sets `stop_reason = "graph_complete"` if not already set.

---

## final_search_node

**File:** `nodes/final_search.py`

**Purpose:** Optional comprehensive search on the original question before output assembly.

**Reads:** `session_config`, `original_question`, `session_id`, `document_filter`, `top_k`, `trace_id`, `hop_data`

**Writes:** `final_search_results`

**Logic:** Executes when `use_final_search` or `enable_final_search` is enabled. Makes adaptive HyDE decision with `is_final=True` flag. Performs backend search using original question. Converts results to citation-format dicts with `full_content` field and `hop_index = -1` sentinel for downstream merge identification.

---

## web_fallback_node

**File:** `nodes/web_fallback.py`

**Purpose:** CRAG web search fallback triggered on "incorrect" document grade.

**Reads:** `current_query`, `graded_chunks`, `all_hop_citations`, `global_citation_index`, `seen_content_hashes`, `hop_count`

**Writes:** `graded_chunks`, `crag_web_fallback_used`, `all_hop_citations`, `global_citation_index`, `seen_content_hashes`

**Logic:** Performs DuckDuckGo web search for the current query. Strips citations from the failed hop (identified by `hop_count`) while preserving prior-hop citations. Builds web citations with content-hash deduplication. Replaces `graded_chunks` with web results. Returns original chunks unchanged if no web results are found.

---

## advance_sub_query_node

**File:** `nodes/advance_sub_query.py`

**Purpose:** Move to the next decomposed sub-query.

**Reads:** `sub_queries`, `current_sub_query_index`

**Writes:** `current_sub_query_index`, `current_query`

**Logic:** Increments `current_sub_query_index` and sets `current_query` to the next sub-query. Routes back to `retrieve` for the next retrieval loop.

---

## check_hallucination_node

**File:** `nodes/quality_gates.py`

**Purpose:** Self-RAG ISSUP grounding verification -- checks whether the generation is supported by source documents.

**Reads:** `llm`, `generation`, `graded_chunks` (fallback: `refined_chunks`, then `retrieved_chunks`), `quality_check_attempts`, `timing_spans`

**Writes:** `hallucination_check`, `quality_check_attempts`, `timing_spans`

**Logic:** Formats source chunks and generation into the ISSUP prompt. LLM classifies as `"supported"`, `"partially_supported"`, or `"not_supported"`. Uses longest-first label matching to avoid substring false positives. Increments `quality_check_attempts` for budget tracking.

---

## check_answer_quality_node

**File:** `nodes/quality_gates.py`

**Purpose:** Self-RAG ISUSE relevance verification -- checks whether the generation addresses the original question.

**Reads:** `llm`, `generation`, `original_question`, `quality_check_attempts`, `timing_spans`

**Writes:** `answer_quality`, `quality_check_attempts`, `timing_spans`

**Logic:** Formats original question and generation into the ISUSE prompt. LLM classifies as `"addresses_question"`, `"partially_addresses"`, or `"does_not_address"`. Uses safe defaults and robust label extraction. Increments `quality_check_attempts` for budget tracking.
