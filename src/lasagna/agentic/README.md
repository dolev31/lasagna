# Agentic RAG -- Multi-Hop Retrieval with Quality Gates

A multi-hop, self-correcting retrieval pipeline that iteratively searches, refines, and grades knowledge base results to produce grounded answers with citations. Implements Corrective RAG, Self-RAG quality gates, HyDE, query decomposition, and adaptive chunk refinement.

---

## Execution Modes

The system supports three execution modes, selected by the `agentic_pipeline` config key (which supersedes the legacy `use_structured_graph` boolean):

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

Full configuration reference: [docs/config.md](docs/config.md)

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
