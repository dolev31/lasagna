# Bring Your Own Agent (BYOA)

Replace the entire agentic retrieval pipeline with your own implementation while keeping document ingestion, vector search infrastructure, and configuration management intact.

> Only one pipeline is active per request/session. Custom pipeline code must be registered on the backend process before use.

---

## 1. Writing a Custom Pipeline

Subclass `BaseAgenticPipeline` and implement `run()`.

**Constructor contract:** your `__init__` must accept `session_config` (a `RAGConfig` instance) and `**kwargs`. The runtime passes `session_config=` and `node_overrides=` when it creates the pipeline, and may pass more keywords in later releases, so accept and ignore the ones you do not use.

```python
from lasagna.core.base_components import (
    BaseAgenticPipeline,
    AgenticPipelineInput,
)
from lasagna.agentic.agentic_rag import AgenticRAGResult


class MyCustomRAG(BaseAgenticPipeline):
    """Single-hop retrieval with custom reranking logic."""

    def __init__(self, session_config=None, **kwargs):
        self.config = session_config

    async def run(self, input: AgenticPipelineInput) -> AgenticRAGResult:
        # Access the RAG backend for vector search
        from lasagna.api.rag_backend import get_rag_backend

        backend = get_rag_backend()
        results = backend.search(
            query=input.question,
            session_id=input.session_id,
            top_k=input.top_k,
            document_ids=input.document_filter,
        )

        # Build chunks and citations from search results
        # SearchResult fields: chunk (RecordedDocumentChunk), score, document
        chunks = [
            {"content": r.chunk.content, "score": r.score, "document_id": r.chunk.document_id}
            for r in results
        ]
        citations = [
            {"text": r.chunk.content[:200], "document_id": r.chunk.document_id, "score": r.score}
            for r in results
        ]
        context = "\n\n".join(r.chunk.content for r in results)

        return AgenticRAGResult(
            formatted_context=context,
            chunks=chunks,
            citations=citations,
            agent_summary=f"Retrieved {len(results)} chunks for: {input.question}",
            search_count=1,
        )
```

### Input (`AgenticPipelineInput`)

| Field | Type | Description |
|-------|------|-------------|
| `question` | `str` | User's query |
| `session_id` | `str` | Session namespace for document isolation |
| `top_k` | `int` | Number of results to return (default: 5) |
| `document_filter` | `Optional[List[str]]` | Restrict to specific document IDs |
| `trace_id` | `Optional[str]` | Distributed tracing ID |
| `routing_decision` | `Any` | Upstream query classification result |
| `session_config` | `Any` | `RAGConfig` instance with all settings |
| `llm` | `Any` | `BaseChatModel` instance for LLM calls |

### Output (`AgenticRAGResult`)

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `formatted_context` | `str` | Yes | Combined text context for downstream use |
| `chunks` | `List[Dict]` | Yes | Retrieved chunk dicts |
| `citations` | `List[Dict]` | Yes | Citation metadata |
| `agent_summary` | `str` | Yes | Human-readable summary of what the agent did |
| `search_count` | `int` | Yes | Number of search operations performed |
| `tokens_before` | `int` | No | Token count before refinement (default: 0) |
| `tokens_after` | `int` | No | Token count after refinement (default: 0) |
| `tokens_saved` | `int` | No | Tokens removed during refinement (default: 0) |
| `reduction_percentage` | `float` | No | Context reduction as a percentage (default: 0.0) |
| `chunks_filtered` | `int` | No | Number of chunks removed during refinement (default: 0) |
| `hops` | `List[HopData]` | No | Per-hop tracking data |
| `metrics` | `AgenticRAGMetrics` | No | Detailed timing/quality metrics |

> **`HopData`** and **`AgenticRAGMetrics`** are optional tracking types defined in `lasagna.agentic.agentic_rag`. Simple pipelines can omit both: they default to an empty list and `None`. Multi-hop pipelines that want per-hop observability should populate `hops` with one `HopData` per search iteration (fields: `hop_index`, `query_used`, `retrieved_chunks`, latency timings). `AgenticRAGMetrics` aggregates timing breakdown and token accounting across the full pipeline run.

---

## 2. Registering Your Pipeline

### Method A: Decorator (simplest)

```python
from lasagna.core.registry import registry

@registry.register("agentic_pipeline", "my_custom_rag")
class MyCustomRAG(BaseAgenticPipeline):
    ...
```

The decorated module must be imported before the pipeline is resolved. Place the import in your application's startup code (e.g., a FastAPI `lifespan`, a `conftest.py`, or your main entry point).

### Method B: Imperative (for conditional registration)

```python
from lasagna.core.registry import registry

# Only register if a dependency is available
if has_colbert:
    registry.register_pipeline(
        "colbert_rag",
        ColBERTRAGPipeline,
        description="Multi-hop retrieval with ColBERT reranking",
    )
```

### Method C: Entry-points (for pip-installable packages)

In your package's `pyproject.toml`:

```toml
[project.entry-points."lasagna_components"]
"agentic_pipeline.my_custom_rag" = "my_package.pipeline:MyCustomRAG"
```

The format is `{category}.{name} = "module:ClassName"`. The registry auto-discovers entry-points in the `lasagna_components` group at initialization. Third-party packages just need `pip install my-package` and the pipeline becomes available automatically.

---

## 3. Selecting Your Pipeline

Once registered, select the pipeline via any config surface:

| Surface | Syntax | Lands in |
|---------|--------|----------|
| SDK | `KnowledgeConfig(agentic_pipeline="my_custom_rag")` or `await client.set_config(agentic_pipeline="my_custom_rag")` | Session override |
| REST | `PATCH /api/v1/config {"agentic_pipeline": "my_custom_rag"}` | Session override |
| CLI, MCP | `lasagna import` of a configuration JSON that sets it, or the MCP `patch_rag_config` tool | Session override |
| Env var | `LASAGNA_AGENTIC_PIPELINE=my_custom_rag` (or `RAG_AGENTIC_PIPELINE`) | Environment |
| TOML | `agentic_pipeline = "my_custom_rag"` in `settings.lasagna.toml` | Settings file |

When two of them set it, the level in the last column decides, in the order Lasagna resolves every setting: per-query overrides, then session overrides, then environment variables, then the settings file (see [Where a setting comes from](../../../../docs/configuration.md#where-a-setting-comes-from)). When none sets `agentic_pipeline`, it is `"default"` if `use_structured_graph` is true, else `"legacy_react"` (`RAGConfig.agentic_pipeline` in `core/config.py`).

> **Registration does not activate.** Registering a pipeline makes it *available* but does not select it. You must explicitly choose it via one of the surfaces above for it to handle searches.

---

## 4. Verifying Registration

### CLI

```bash
python -m lasagna pipelines list
```

Output (shortened; the five published loops `self_ask`, `ircot`, `flare`, `adaptive` and `self_rag` are also listed):
```
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Name          ┃ Description                            ┃ Class                     ┃ Active ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ default       │ Built-in structured RAG graph (13      │ BuiltinStructuredPipeline │        │
│               │ nodes, CRAG, quality gates)            │                           │        │
│ legacy_react  │ Legacy ReAct agent with monolithic     │ LegacyReActPipeline       │ ✓      │
│               │ search tool                            │                           │        │
│ my_custom_rag │ Single-hop with ColBERT                │ MyCustomRAG               │        │
└───────────────┴────────────────────────────────────────┴───────────────────────────┴────────┘

Active pipeline: legacy_react
```

### REST API

```bash
curl http://localhost:8000/api/v1/pipelines
```

```json
{
  "pipelines": [
    {"name": "default", "description": "Built-in structured RAG graph (13 nodes, CRAG, quality gates)", "class_name": "BuiltinStructuredPipeline"},
    {"name": "legacy_react", "description": "Legacy ReAct agent with monolithic search tool", "class_name": "LegacyReActPipeline"},
    ...
    {"name": "my_custom_rag", "description": "Single-hop with ColBERT", "class_name": "MyCustomRAG"}
  ],
  "active": "legacy_react",
  "total": 8
}
```

### Python

```python
from lasagna.core.registry import registry
print(registry.list_pipelines())
```

---

## 5. Deploy Patterns

### Source integration (development / monorepo)

Place your pipeline module in the project and import it at startup:

```python
# In your app's startup code:
import my_pipelines.custom_rag  # triggers @registry.register decorator
```

### pip package with entry-points (production)

Create a separate package with entry-points (see Method C above). Install alongside Lasagna:

```dockerfile
# Dockerfile
FROM python:3.11-slim
RUN pip install lasagna-rag my-custom-pipeline
ENV LASAGNA_AGENTIC_PIPELINE=my_custom_rag
```

### Docker with source mount (prototyping)

```yaml
# docker-compose.yml
services:
  rag-api:
    image: lasagna:latest
    volumes:
      - ./my_pipeline.py:/app/custom/my_pipeline.py
    environment:
      - PYTHONPATH=/app/custom
      - LASAGNA_AGENTIC_PIPELINE=my_custom_rag
```

> **Important:** mounting a file and setting `LASAGNA_AGENTIC_PIPELINE` is not enough on its own: the module must be **imported** so the `@registry.register` decorator runs. Either add the import to the REST app's startup (e.g., FastAPI lifespan), or use entry-points (Method C) which are auto-discovered without explicit imports.

---

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `KeyError: "Agentic pipeline 'X' is not registered"` | Config references a pipeline name that hasn't been registered | Ensure the module with `@registry.register(...)` is imported before first search. Use `pipelines list` to check. |
| Pipeline registered locally but not in REST API | The module isn't importable in the uvicorn process | Use entry-points (auto-discovered) or add the import to the REST app's startup. |
| A search fails after your `run()` returns | `run()` returns a dict or another type | Import and return `AgenticRAGResult` from `lasagna.agentic.agentic_rag`. |
| Pipeline shows in `list` but queries don't use it | Another source sets `agentic_pipeline` ahead of yours | Check session overrides first (`lasagna doctor -s SESSION` shows what is in force), then `LASAGNA_`/`RAG_` environment variables, then the settings file. |
| Entry-point not discovered | Wrong group name or naming format | Group must be `lasagna_components`, key format is `agentic_pipeline.{name}`. Run `pip show my-package` to verify installation. |

---

## Further Reading

- [Which pipeline runs](../../agentic/README.md#which-pipeline-runs): how a pipeline is resolved, and node-level overrides
- [Agentic Pipeline Extensibility](extending.md#agentic-pipeline-extensibility): ablation with `sweep_pipelines()` and `sweep_nodes()`
- [REST API Endpoints](../../api/docs/endpoints.md): every endpoint, including `GET /api/v1/pipelines`
- [Agentic Config Reference](../../agentic/docs/config.md): `agentic_pipeline` precedence rules

## See Also

- [Extending the SDK](extending.md)
- [Features](features.md)
