# agentic: multi-hop retrieval loops

This package holds the `agentic_pipeline` stage: retrieval that searches more than once per
question, refines or grades what it found, and returns an answer with the passages it read.
It ships two built-in pipelines (a LangChain ReAct agent and a LangGraph structured graph) and
five published loops (Self-Ask, IRCoT, FLARE, Adaptive-RAG, Self-RAG), all registered in
`lasagna.core.registry` under the `agentic_pipeline` namespace.

Settings and their defaults are in [docs/reference/settings.md](../../../docs/reference/settings.md);
how to write a component is in [docs/guide.md](../../../docs/guide.md#4-write-a-component).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `agentic_rag.py` | `AgenticRAG` (entry point and pipeline dispatch), the ReAct tools `search_knowledge_base` and `mark_chunk_for_citation`, `AgenticRAGResult`, `AgenticRAGMetrics`, `HopData`, `RetrievedChunkData`, `LegacyReActPipeline` |
| `graph_builder.py` | `build_rag_graph()` compiles the structured `StateGraph` (13 nodes, 6 conditional edges); `VALID_NODE_NAMES`; `BuiltinStructuredPipeline` (registered as `default`) |
| `graph_state.py` | `RAGGraphState` TypedDict and `build_initial_state()` |
| `edges.py` | The six `route_after_*` functions that pick the next node |
| `nodes/` | One file per graph node (12 files, 13 nodes: `quality_gates.py` holds `check_hallucination` and `check_answer_quality`) |
| `loops/` | The five published loops, one file each, and `_base.py` (`LoopPipeline`): LLM and settings resolution, search through the pipeline, result building |
| `prompts.py` | Every graph prompt, overridable through the `agentic_prompts` setting; `NODE_PLACEHOLDERS` lists the fields each node supplies |
| `citation_manager.py` | `Citation`, `CitationManager`, `CitationVerifier`, stable citation ids, highlight ranges, deduplication |
| `document_awareness.py` | Per-session document snippets and the awareness context given to the agent |
| `speculative_rag.py` | `SpeculativeRAG` (parallel drafts plus a verifier); used only by the experiment runner when `enable_speculative_rag` is set |

Deeper references: [docs/architecture.md](docs/architecture.md) (graph and edge routing),
[docs/nodes.md](docs/nodes.md), [docs/state.md](docs/state.md), [docs/config.md](docs/config.md).

## Which pipeline runs

`AgenticRAG.__init__` resolves the pipeline from the session config. `RAGConfig.agentic_pipeline`
returns, in order: a session override, the `agentic_pipeline` setting, `"default"` when
`use_structured_graph` is true, else `"legacy_react"`. Any name other than `legacy_react` is
looked up with `registry.get_class("agentic_pipeline", name)`; an unknown name raises `KeyError`.

- `legacy_react`: a LangChain agent with two tools; the LLM decides when to stop.
- `default`: the structured graph. Routing between nodes is rule-based, with quality gates
  (`enable_crag`, `enable_hallucination_check`, `enable_answer_quality_check`) switching branches on.
- `self_ask`, `ircot`, `flare`, `adaptive`, `self_rag`: the published loops below.
- Any other registered name: a user `BaseAgenticPipeline` subclass
  (see [../sdk/docs/custom_pipelines.md](../sdk/docs/custom_pipelines.md)).

## Design notes

**Loops search through the pipeline.** Each hop calls the pipeline's own search (the standard
tier: hybrid search, reranking, fusion) with the arm's settings and components. An agentic arm
therefore measures its loop inside the full pipeline, never against a private retriever. The LLM
is `AgenticPipelineInput.llm`; `agentic_models` pins another model per loop name. A loop with no
LLM raises rather than scoring plain retrieval under its name.

**Budgets are per loop.** `max_agent_iterations` applies to the built-in pipelines only. Each
published loop has its own settings (`self_ask_*`, `ircot_*`, `flare_*`, `adaptive_rag_*`,
`self_rag_*`); the defaults follow each paper.

**Adaptive-RAG and Self-RAG are prompted variants.** The papers train a T5 classifier and a
reflection-token model. Here `adaptive` prompts an LLM with the paper's class definitions, and
`self_rag` prompts an LLM with the authors' critic instructions: no reflection-token
probabilities, a yes/no retrieve decision, and no segment-level beam search. A number from
either loop describes the prompted variant.

**FLARE confidence.** With `flare_confidence_source = "auto"`, FLARE uses token log probabilities
when the model returns them and falls back to self-reported unsure spans. The signal used is
recorded in `AgenticRAGMetrics.loop_trace["confidence_source"]`.

**No silent fallback under evaluation.** In `_search_comprehensive`
(`api/_search_orchestrator.py`), `KeyError`, `ValueError` and `TypeError` are re-raised as
configuration errors. Other failures fall back to standard search, except when `strict_stages()`
is on (evaluation), where they fail the task, so standard results are never scored as the
agentic arm's.

**Prompt overrides fail early.** `validate_prompt()` in `prompts.py` checks an override against
`NODE_PLACEHOLDERS`: a field the node does not supply raises `PromptError` naming the node,
instead of a `KeyError` deep in a run. An override that uses none of the node's fields is
allowed and logged as a warning.

**Node overrides.** `build_rag_graph(node_overrides={...})` and
`KnowledgePipeline.set_node_override(node, registered_name)` (with
`registry.register_graph_node`) replace single graph nodes; names outside `VALID_NODE_NAMES`
raise `ValueError`.

## Adding a loop

Subclass `LoopPipeline` in `loops/`, register it in `core/registry.py` next to the other five
(with its `paper` id), and add its settings to `core/config.py`. `loops/self_ask.py` is the
shortest example. `lasagna component validate` checks only that a pipeline's `run` is `async def`
and takes the input, so test a loop by running it in a sweep.

## Callers

`RAGBackend.async_agentic_search` (`api/rag_backend.py`), `KnowledgeClient.agentic_search`
(`sdk/client.py`), the MCP `search` tool when `use_agentic_rag` is true (`mcp/server.py`), and
`_search_comprehensive` for the comprehensive routing tier.
