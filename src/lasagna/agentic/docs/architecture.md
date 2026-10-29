# Architecture: Structured Graph Pipeline

## Pipeline Overview

The structured graph (enabled via `use_structured_graph = true`, or `agentic_pipeline = "default"`) is a LangGraph `StateGraph` with 13 nodes and 6 conditional edges, built by `build_rag_graph()` in `agentic/graph_builder.py`. All nodes read from and write to `RAGGraphState`.

This document covers that graph and the legacy ReAct mode (`agentic_pipeline = "legacy_react"`, the default). The six published loops (`self_ask`, `ircot`, `flare`, `adaptive`, `self_rag`, `search_o1`) do not use this graph. Each is a short loop in `agentic/loops/`, listed in [the agentic README](../README.md#which-pipeline-runs).

```
START
  |
  v
plan_retrieval           Map routing tier to complexity, optionally decompose query
  |
  v
retrieve                 Search knowledge base (HyDE, dedup, coverage scoring)
  |
  v
refine_chunks            LLM-based chunk refinement (vanilla / auto / refined)
  |
  v
grade_documents          CRAG quality grading
  |
  |--- [route_after_grading] ---+--- grade="correct" ---------> generate
  |                             |--- grade="incorrect"           |
  |                             |    + web_fallback enabled ---> web_fallback --> generate
  |                             |--- else + budget remains ----> rewrite_query
  |                             +--- else (budget exhausted) --> generate
  |
  v
generate                 LLM answer generation with [N] citations
  |
  |--- [route_after_generation] ---+--- hallucination_check ON --> check_hallucination
  |                                |--- answer_quality_check ON -> check_answer_quality
  |                                +--- neither ----------------> select_chunks
  |
  v (if quality gates enabled)
check_hallucination      ISSUP grounding verification
  |
  |--- [route_after_hallucination_check] ---+--- "not_supported" + budget --> rewrite_query
  |                                         |--- answer_quality ON --------> check_answer_quality
  |                                         +--- else ---------------------> select_chunks
  |
  v (if enabled)
check_answer_quality     ISUSE relevance verification
  |
  |--- [route_after_quality_check] ---+--- both pass ---------> select_chunks
  |                                   |--- budget exhausted ---> select_chunks
  |                                   |--- hallucination bad --> rewrite_query
  |                                   +--- quality bad -------> generate (retry)
  |
  v
select_chunks            Filter citations by strategy (adaptive / fixed / above_threshold / agent_decides)
  |
  |--- [route_after_select_chunks] ---+--- more sub-queries + budget --> advance_sub_query
  |                                   +--- else -----------------------> final_search
  |
  v (if sub-queries remain)
advance_sub_query        Move to next decomposed sub-query
  |
  v
  +---> retrieve (loops back)

  v (when sub-queries exhausted or no decomposition)
final_search             Optional wide search on the original question
  |
  v
format_output            Assemble final result, reindex citations, compute metrics
  |
  v
END


Rewrite loop:
  rewrite_query
    |--- [route_after_rewrite] ---+--- budget remains --> retrieve (loops back)
                                  +--- budget exhausted -> generate
```

---

## Edge Routing Table

| Edge Function | Source Node | Possible Targets | Decision Logic |
| ------------- | ---------- | ---------------- | -------------- |
| `route_after_grading` | `grade_documents` | `generate`, `web_fallback`, `rewrite_query` | CRAG grade + web fallback flag + retrieval budget |
| `route_after_generation` | `generate` | `check_hallucination`, `check_answer_quality`, `select_chunks` | Quality gate config flags (checks in order) |
| `route_after_hallucination_check` | `check_hallucination` | `check_answer_quality`, `rewrite_query`, `select_chunks` | ISSUP label + quality check budget |
| `route_after_quality_check` | `check_answer_quality` | `select_chunks`, `rewrite_query`, `generate` | Both labels + budget (hallucination -> rewrite, quality -> regenerate) |
| `route_after_select_chunks` | `select_chunks` | `advance_sub_query`, `final_search` | Remaining sub-queries + global hop budget |
| `route_after_rewrite` | `rewrite_query` | `retrieve`, `generate` | Rewrite budget (`max_rewrites`) |

**Budget enforcement:** Most edges enforce iteration or budget caps. `route_after_generation` is the exception: it routes purely on config flags (which quality gates are enabled). `route_after_select_chunks` enforces the global retrieval budget (`max_retrieval_attempts`) across all sub-queries, preventing large decompositions from bypassing the hop limit.

---

## Techniques

### Corrective RAG (CRAG)

Per-chunk LLM grading evaluates whether each retrieved chunk is relevant to the query. Individual grades are aggregated by majority vote into an overall grade:

- **"correct"** (>50% of chunks graded correct): proceed to generation
- **"ambiguous"**: rewrite query and retry if budget allows
- **"incorrect"**: trigger web fallback (DuckDuckGo) if enabled, otherwise rewrite

When web fallback fires, citations from the failed hop are stripped and replaced with web results. Prior-hop citations are preserved.

Config: `enable_crag`, `crag_web_search_fallback`

### Self-RAG Quality Gates

Two opt-in post-generation checks inspired by Self-RAG (Asai et al., 2023). They are not the Self-RAG method: the graph grades a retrieved set, writes one answer from all of it and then checks that answer. The `self_rag` loop is the method's short-form inference, with one answer per passage chosen by critique.

**ISSUP (hallucination check):** Evaluates whether every claim in the generated answer is directly supported by the source documents. Labels: `"supported"`, `"partially_supported"`, `"not_supported"`. On `"not_supported"`, routes to rewrite_query if budget allows.

**ISUSE (answer quality check):** Evaluates whether the answer adequately addresses the original question. Labels: `"addresses_question"`, `"partially_addresses"`, `"does_not_address"`. On `"does_not_address"`, routes to generate for regeneration.

Both gates increment `quality_check_attempts` and respect `max_quality_check_attempts` (default: 2) to prevent infinite loops.

Config: `enable_hallucination_check`, `enable_answer_quality_check`, `max_quality_check_attempts`

### HyDE (Hypothetical Document Embeddings)

Generates a hypothetical answer document to use for embedding-based search instead of the raw query. Four strategies:

- **`off`**: disabled
- **`always`**: HyDE on every hop
- **`first_hop`**: HyDE on first hop only
- **`adaptive`** (default): HyDE when (a) the query is short (<=`max_query_length` tokens) on the first or final hop, or (b) the prior hop returned too few chunks or documents

Config: `agentic_hyde_strategy`, `max_query_length`

### Query Decomposition

For complex queries, the LLM decomposes the original question into 2-4 focused sub-queries. Each sub-query is processed as a separate retrieval loop (retrieve -> refine -> grade -> generate -> select_chunks), with results accumulated across loops. The global retrieval budget is shared across all sub-queries.

Config: `enable_query_decomposition`

### Chunk Refinement

Three modes for processing retrieved chunks:

- **`vanilla`**: pass-through with deduplication and citation building. No LLM cost.
- **`refined`**: parallel LLM-based refinement. Each chunk is independently refined with context from the query. Chunks receiving `NOT_RELEVANT` classification are filtered. Tracks token reduction metrics.
- **`auto`** (default): heuristic gate. Refines only when chunks are numerous (>7), long (avg >1000 chars), or have high score variance, and otherwise behaves as vanilla.

Safety net: if all chunks are filtered as irrelevant, the top-scoring original chunk is kept.

Config: `chunk_refinement_method`, `refinement_model`

### Coverage Scoring

Keyword-based or LLM-based assessment of how well retrieved chunks cover the aspects of the query. Computes `coverage_ratio` (0.0-1.0), `coverage_confidence`, and `uncovered_aspects`.

**Coverage informs rewrite targeting:** the `uncovered_aspects` list is fed to `rewrite_query_node` to generate queries that target missing information. Coverage does NOT directly control stopping; stopping is determined by retrieval budgets and sub-query progression.

Config: `coverage_scorer_mode` (`"keyword"` or `"llm"`)

### Document Awareness

Provides agents with context about available documents before they search. Extracts document snippets using configurable strategies (`first_tokens`, `llm_summary`, `hybrid`) and generates adaptive search guidance:

- **0 searches performed:** Lists documents with previews and usage instructions
- **1 search:** Encourages answering but allows follow-up searches
- **2+ searches:** Strong encouragement to answer with current results

Config: `[rag.document_awareness]` section in settings

---

## Legacy ReAct Mode

When `use_structured_graph = false` (default) and no other `agentic_pipeline` is named, the system uses a LangChain agent with two tools:

**`search_knowledge_base(query, config)`** performs a single retrieval hop with:
- Query deduplication (exact + semantic)
- HyDE decision
- Backend search via RAGBackend
- CRAG grading (if enabled)
- Chunk refinement
- Citation tracking
- Coverage scoring
- Query stabilization detection

**`mark_chunk_for_citation(result_number, config)`**: for the `agent_decides` citation strategy, the agent marks which results to include in final context.

The agent LLM receives a system prompt (`AGENTIC_RAG_SYSTEM_PROMPT`) that instructs it on search strategy, iteration budget, and when to stop. The agent decides autonomously when to perform additional searches or finalize results.
