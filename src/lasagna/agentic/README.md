# Agentic RAG -- Multi-Hop Retrieval with Quality Gates

A multi-hop, self-correcting retrieval pipeline that iteratively searches, refines, and grades knowledge base results to produce grounded answers with citations. Implements Corrective RAG, Self-RAG quality gates, HyDE, query decomposition, and adaptive chunk refinement. Beside it ship five published loops, Self-Ask, IRCoT, FLARE, Adaptive-RAG and Self-RAG, so a new loop can be compared with the methods a paper would cite (see [Published Loops](#published-loops)).

---

## Execution Modes

The `agentic_pipeline` config key (which supersedes the legacy `use_structured_graph` boolean) selects one of these execution modes:

```
  agentic_pipeline = "legacy_react"            (default when use_structured_graph = false)
 +------------------------------------+
 |        Legacy ReAct Mode           |
 |                                    |
 |  LangChain agent loop with tools:  |
 |   - search_knowledge_base          |
 |   - mark_chunk_for_citation        |
 |                                    |
 |  Agent decides when to stop.       |
 +------------------------------------+

  agentic_pipeline = "default"                 (default when use_structured_graph = true)
 +------------------------------------+
 |    Built-in Structured Graph       |
 |                                    |
 |  LangGraph StateGraph with 13      |
 |  nodes and 6 conditional edges.    |
 |  Explicit pipeline with CRAG,      |
 |  quality gates, query decomp.      |
 |                                    |
 |  Deterministic, debuggable flow.   |
 +------------------------------------+

  agentic_pipeline = "self_ask" | "ircot" | "flare" | "adaptive" | "self_rag"
 +------------------------------------+
 |        Published Loops             |
 |                                    |
 |  Each paper's own loop, with its   |
 |  prompts and defaults. Every hop   |
 |  searches through the pipeline.    |
 +------------------------------------+

  agentic_pipeline = "<custom_name>"           (user-registered pipeline)
 +------------------------------------+
 |        Custom Pipeline             |
 |                                    |
 |  Any BaseAgenticPipeline subclass  |
 |  registered in the component       |
 |  registry under the                |
 |  "agentic_pipeline" namespace.     |
 +------------------------------------+
```

- **Legacy ReAct** (`"legacy_react"`): A LangChain agent with `search_knowledge_base` and `mark_chunk_for_citation` tools. The LLM decides search strategy and stopping.
- **Built-in Structured Graph** (`"default"`): An explicit LangGraph `StateGraph` with 13 nodes and 6 conditional edges. All routing is rule-based with configurable quality gates. See [docs/architecture.md](docs/architecture.md) for the full pipeline.
- **Published Loops** (`"self_ask"`, `"ircot"`, `"flare"`, `"adaptive"`, `"self_rag"`): Self-Ask, IRCoT, FLARE, Adaptive-RAG and Self-RAG, each as its paper describes it. See [Published Loops](#published-loops) below.
- **Custom Pipeline** (any other registered name): A user-supplied `BaseAgenticPipeline` subclass, registered via the component registry. See [Custom Pipelines](#custom-pipelines) below.

**Precedence:** The `agentic_pipeline` value is resolved as follows:
1. Explicit `agentic_pipeline` setting (TOML, SDK, REST API, session override)
2. `use_structured_graph = true` maps to `"default"`, `false` maps to `"legacy_react"`
3. If neither is set, defaults to `"legacy_react"`

---

## Module Structure

| File | Purpose |
| ---- | ------- |
| `agentic_rag.py` | Core: `AgenticRAG` class, ReAct tools, metrics, helper functions |
| `graph_builder.py` | Compiles LangGraph StateGraph (13 nodes, 6 conditional edges) |
| `graph_state.py` | `RAGGraphState` TypedDict (50+ fields) + `build_initial_state()` |
| `edges.py` | 6 conditional routing functions for state transitions |
| `citation_manager.py` | `Citation` class, stable IDs, highlight extraction, deduplication |
| `document_awareness.py` | Document snippets, LLM summaries, agent awareness context |
| `nodes/` (11 files, 13 nodes) | Individual graph node implementations (`quality_gates.py` contains 2) |
| `loops/` | The five published loops, one file each, and `_base.py`: the LLM, search and result handling they share |

---

## Quick Config Reference

| Option | Default | What it does |
| ------ | ------- | ------------ |
| `use_agentic_rag` | `true` | Enable agentic multi-hop retrieval |
| `max_agent_iterations` | `3` | Max retrieval hops per query |
| `agentic_pipeline` | `"legacy_react"` | Pipeline selection: `"legacy_react"`, `"default"` (structured graph), or any registered name |
| `use_structured_graph` | `false` | *Legacy alias* -- `true` maps to `agentic_pipeline="default"` |
| `enable_crag` | `false` | CRAG document quality grading |
| `enable_hallucination_check` | `false` | Self-RAG ISSUP grounding check (graph only) |
| `enable_answer_quality_check` | `false` | Self-RAG ISUSE relevance check (graph only) |
| `enable_query_decomposition` | `false` | LLM-based sub-query decomposition |
| `chunk_refinement_method` | `"auto"` | Refinement: `vanilla` / `refined` / `auto` |
| `agentic_hyde_strategy` | `"adaptive"` | HyDE: `off` / `always` / `first_hop` / `adaptive` |
| `use_final_search` | `false` | Final comprehensive search on original question |

Full configuration reference: [docs/config.md](docs/config.md). The published loops have settings of their own, listed below.

---

## Published Loops

Five loops implement published methods, each with its paper's prompts and defaults. Select one with `agentic_pipeline` (TOML, SDK, REST), or name it in a sweep arm (`{name: ircot, agentic_pipeline: ircot}`), which also turns on `use_agentic_rag` for that arm.

| Name | Method | What it does | Settings (default) |
| ---- | ------ | ------------ | ------------------ |
| `self_ask` | Self-Ask (Press et al. 2022, arXiv:2210.03350) | The model asks follow-up questions. Each is searched, and one LLM call reads the passages into a short intermediate answer, until the model writes its final answer. | `self_ask_max_follow_ups` (5) |
| `ircot` | IRCoT (Trivedi et al. 2023, arXiv:2212.10509) | Retrieves for the question, then alternates one chain-of-thought sentence with a search for that sentence, until a sentence states the answer or the step limit is reached. A separate reader then answers from every collected paragraph. | `ircot_paragraphs_per_step` (6), `ircot_max_steps` (8), `ircot_max_paragraphs` (15) |
| `flare` | FLARE (Jiang et al. 2023, arXiv:2305.06983) | Writes the answer a sentence at a time. When a draft sentence has a low-confidence token, it searches with that sentence, unsure tokens masked out, and writes the sentence again. | `flare_retrieval_threshold` (0.8), `flare_mask_threshold` (0.4), `flare_max_sentences` (8), `flare_docs_per_retrieval` (2), `flare_confidence_source` (`auto`) |
| `adaptive` | Adaptive-RAG (Jeong et al. 2024, arXiv:2403.14403) | A classifier labels the question A, B or C, which picks no retrieval, one search, or IRCoT. | `adaptive_rag_classifier_model` (empty: the loop's model), `adaptive_rag_single_step_docs` (5), and the `ircot_*` settings |
| `self_rag` | Self-RAG (Asai et al. 2023, arXiv:2310.11511) | Decides whether to retrieve, writes one answer per retrieved passage, critiques each for relevance, support and usefulness, and keeps the best-scored answer. | `self_rag_passages` (5), `self_rag_weight_rel` (1.0), `self_rag_weight_sup` (1.0), `self_rag_weight_use` (0.5), `self_rag_retrieval` (`adaptive`, or `always`) |

These budgets are each loop's own; `max_agent_iterations` does not apply to them.

**Adaptive-RAG and Self-RAG are prompted variants of trained-model methods.** Adaptive-RAG's paper trains a T5 classifier, and Self-RAG's paper trains a model to emit reflection tokens. Neither paper has a prompted version. Here `adaptive` prompts an LLM with the paper's own definitions of the three classes, and `self_rag` prompts an LLM with the authors' critic instructions. So `self_rag` has no reflection-token probabilities: each critique counts at its label's value, the retrieve decision is yes or no instead of a probability against a threshold, and the long-form, segment-level beam search is not implemented. A number from either loop describes the prompted variant, not the trained model.

**FLARE needs token probabilities to run as published.** With `flare_confidence_source: auto`, the loop uses the model's token log probabilities when the model returns them. A model that returns none falls back to naming the spans of its draft it is unsure of. Which signal ran is recorded in `loop_trace` as `confidence_source` (`logprobs`, `self_reported` or `mixed`). Set `logprobs` or `self_reported` to force one.

**What every loop gets from the pipeline:**

- **Search.** Each hop calls the pipeline's own search: the retrieval the standard tier runs (hybrid search, reranking, fusion), with the arm's settings and the arm's components (reranker, query enhancer, retriever and embedder). An agentic arm therefore measures its loop inside the complete pipeline, never against a private retriever.
- **LLM.** The model the orchestrator built for the query (`AgenticPipelineInput.llm`). `agentic_models` pins another model for one loop, keyed by the loop's name (for example `{"self_rag": "<model>"}`). With no LLM a loop raises, rather than scoring plain retrieval under its name.
- **Settings.** `AgenticPipelineInput.session_config`: the session's config with the arm's per-query overrides applied.

Each loop returns its answer (`agent_summary`) and the passages it read (`chunks`), so an evaluation judges the answer against what the loop retrieved. What the loop did, including every query it searched, is in `AgenticRAGMetrics.loop_trace`. Under evaluation, an agentic search that fails fails the task instead of falling back to standard results.

---

## Custom Pipelines

Register a completely custom agentic retrieval pipeline by subclassing `BaseAgenticPipeline` and registering it in the `agentic_pipeline` namespace:

```python
from lasagna.core.base_components import (
    BaseAgenticPipeline, AgenticPipelineInput,
)
from lasagna.core.registry import registry

@registry.register("agentic_pipeline", "my_custom_rag")
class MyCustomRAG(BaseAgenticPipeline):
    def __init__(self, session_config=None, **kwargs):
        self.config = session_config

    async def run(self, input: AgenticPipelineInput) -> "AgenticRAGResult":
        from lasagna.agentic.agentic_rag import AgenticRAGResult
        # Your retrieval logic — must return AgenticRAGResult
        return AgenticRAGResult(
            formatted_context="...",
            chunks=[...],
            citations=[...],
            agent_summary="...",
            search_count=1,
        )
```

Then select it via config:

```toml
# settings.lasagna.toml
agentic_pipeline = "my_custom_rag"
```

Or at runtime via SDK / REST API:

```python
config = KnowledgeConfig(agentic_pipeline="my_custom_rag")
```

> For a comprehensive guide including registration methods, deploy patterns, and troubleshooting, see the [BYOA Guide](../sdk/docs/custom_pipelines.md).

The published loops are written this way too. They share `LoopPipeline` in `lasagna/agentic/loops/_base.py`, which resolves the loop's LLM and settings, runs each search through the pipeline, and builds the `AgenticRAGResult`; reading `loops/self_ask.py` is a short way to see the whole pattern. `lasagna component validate` does not check agentic pipelines yet, so test a new loop by running it in a sweep.

---

## Custom Nodes

Individual graph nodes can be overridden without replacing the entire pipeline. The built-in structured graph exposes 13 node slots via `VALID_NODE_NAMES`. Override any node by registering a callable or passing `node_overrides` to `build_rag_graph()`:

```python
from lasagna.core.registry import registry

# Define a custom async node function
async def my_retrieve_node(state):
    # Custom retrieval logic — must accept and return RAGGraphState
    ...

# Register it with a unique name
registry.register_graph_node("my_retrieve", my_retrieve_node)
```

Then use it via `KnowledgePipeline`:

```python
pipeline = KnowledgePipeline(agentic_pipeline="default")
pipeline.set_node_override("retrieve", "my_retrieve")  # pass registered name
```

Or override at graph build time:

```python
from lasagna.agentic.graph_builder import build_rag_graph

graph = build_rag_graph(
    session_config=config,
    node_overrides={"retrieve": my_custom_retrieve_fn},
)
```

Node overrides are validated against `VALID_NODE_NAMES`; unknown names raise `ValueError`.

---

## Integration Points

The agentic RAG system can be invoked through 4 paths:

**1. RAGBackend (backend API)**

```python
from lasagna.api.rag_backend import RAGBackend

backend = RAGBackend()
result = await backend.async_agentic_search(
    query="What factors affected Q3 performance?",
    session_id="session-123",
    top_k=10,
)
```

**2. SDK Client**

```python
from lasagna.sdk import KnowledgeClient

async with KnowledgeClient() as client:
    results = await client.agentic_search(
        "Compare revenue across all quarters",
        max_iterations=5,
    )
```

**3. MCP Server** -- when `use_agentic_rag = true`, the `search_knowledge` MCP tool uses agentic retrieval automatically.

**4. Direct instantiation**

```python
from lasagna.agentic import AgenticRAG

agent = AgenticRAG.create(
    max_iterations=3,
    enable_final_search=False,
    session_config=config,
)
result = await agent.retrieve(
    question="What is the quarterly revenue?",
    session_id="session-123",
)
```

---

## Deep Dive Documentation

| Document | Content |
| -------- | ------- |
| [docs/architecture.md](docs/architecture.md) | Structured graph pipeline diagram, edge routing, SOTA techniques |
| [docs/nodes.md](docs/nodes.md) | 13-node reference with reads, writes, and key logic |
| [docs/state.md](docs/state.md) | `RAGGraphState` field groups and `build_initial_state()` defaults |
| [docs/config.md](docs/config.md) | Full configuration reference, SDK availability, citations, metrics, data classes |
