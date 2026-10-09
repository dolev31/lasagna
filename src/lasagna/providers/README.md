# Providers -- Dependency Injection Layer

## Overview

The `providers/` package is the abstraction layer. It defines three provider interfaces (ABCs) that abstract configuration, LLM access, and session management. Any host application can implement these interfaces to integrate the knowledge package -- or simply use the built-in `Simple*` / `InMemory*` implementations for standalone operation.

Zero configuration is required for basic use: providers auto-configure from environment variables on first access.

## Architecture

### Provider Interfaces

| Interface | File | Purpose |
|-----------|------|---------|
| `ConfigProvider` | `config_provider.py` | Paths (db, uploads, package root), RAG settings, LLM model settings |
| `LLMProvider` | `llm_provider.py` | Chat model creation, model validation |
| `SessionConfigProvider` | `session_provider.py` | Per-session RAG state, document focus/exclusion, config overrides |

### Registry (`__init__.py`)

Global singleton registry with getter/setter functions:

```
get_config_provider()  / set_config_provider()
get_llm_provider()     / set_llm_provider()
get_session_provider() / set_session_provider()
configure_providers()  -- set all at once
reset_providers()      -- clear all (for testing)
```

**Auto-configuration** happens lazily on the first getter call:
1. Detects LLM platform from environment API keys (first match wins)
2. Creates `SimpleConfigProvider` with paths from `RAG_DB_DIR` / `RAG_UPLOAD_DIR` (default: `~/.rag_knowledge/`)
3. Creates `InMemorySessionProvider` as fallback

Thread-safe via double-checked locking.

### Design

```
External Code / Lasagna
    |
    v  configure_providers(config=..., llm=..., session=...)
    |  (or auto-config from env vars)
    v
Provider Registry (singletons)
    |
    v  get_*_provider()
    |
    v
RAGConfig / AgenticRAG / RAGBackend / SearchOrchestrator
```

## Implementations

### Standalone (`Simple*` / `InMemory*`)

**SimpleConfigProvider** -- In-memory paths and settings. Immutable after construction; getters return deep copies.

**SimpleLLMProvider** -- Supports 18+ LLM platforms out of the box:

| Platform | Env Var | Default Model | Tier |
|----------|---------|---------------|------|
| openai | `OPENAI_API_KEY` | gpt-4o-mini | 1 (native) |
| anthropic | `ANTHROPIC_API_KEY` | claude-sonnet-4-20250514 | 1 (native) |
| azure | `AZURE_OPENAI_API_KEY` | gpt-4o-mini | 1 (native) |
| google | `GOOGLE_API_KEY` | gemini-2.0-flash | 1 (native) |
| groq | `GROQ_API_KEY` | llama-3.3-70b-versatile | 1 (native) |
| watsonx | `WATSONX_API_KEY` | llama-4-maverick-17b-128e | 1 (native) |
| mistral | `MISTRAL_API_KEY` | mistral-large-latest | 1 (native) |
| cohere | `COHERE_API_KEY` | command-r-plus | 1 (native) |
| nvidia | `NVIDIA_API_KEY` | llama-3.1-70b-instruct | 1 (native) |
| openrouter | `OPENROUTER_API_KEY` | claude-3.5-sonnet | 2 (OpenAI-compat) |
| together | `TOGETHER_API_KEY` | Llama-3.3-70B-Instruct-Turbo | 2 (OpenAI-compat) |
| deepseek | `DEEPSEEK_API_KEY` | deepseek-chat | 2 (OpenAI-compat) |
| fireworks | `FIREWORKS_API_KEY` | llama-v3p1-70b-instruct | 2 (OpenAI-compat) |
| perplexity | `PERPLEXITY_API_KEY` | sonar | 2 (OpenAI-compat) |
| ollama | `OLLAMA_BASE_URL` | llama3.2 | 3 (local) |
| huggingface | `HF_MODEL_ID` | Phi-3-mini-4k-instruct | 3 (local) |
| llamacpp | `LLAMACPP_MODEL_PATH` | *(requires path)* | 3 (local) |
| vllm | `VLLM_BASE_URL` | default | 3 (local) |

- **Tier 1** uses dedicated `langchain_*` packages (installed on demand)
- **Tier 2** uses `ChatOpenAI` with a custom `base_url` -- zero extra dependencies
- **Tier 3** runs models locally -- no API key needed. Install extras: `pip install lasagna-rag[local]`

**InMemorySessionProvider** -- Thread-safe in-memory session store with optional LRU eviction (`max_sessions` parameter). No persistence; data lost on restart.

### Persistent Sessions

Set `RAG_SESSION_BACKEND=persistent` to use file-backed sessions via `PersistentSessionProvider` (wraps `KnowledgeConfigManager`). Default is in-memory.

### Integration with Other Agents

Any agent framework can use the knowledge package. Just set env vars and start the MCP server:

```bash
RAG_DB_DIR=/my/data python -m lasagna.mcp --port 8005
```

Or pass paths as CLI args:

```bash
python -m lasagna.mcp --db-dir /my/data --upload-dir /my/uploads --session-backend persistent
```

## Quick Start

### Auto-config (just set an API key)

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."

from lasagna.sdk import KnowledgeClient

async with KnowledgeClient() as client:
    await client.upload("doc.pdf")
    results = await client.search_documents("query")
```

### Explicit config

```python
from pathlib import Path
from lasagna.providers import (
    configure_providers,
    SimpleConfigProvider,
    SimpleLLMProvider,
    InMemorySessionProvider,
)

configure_providers(
    config=SimpleConfigProvider(
        db_dir=Path("/data/rag/dbs"),
        upload_dir=Path("/data/rag/uploads"),
        rag_settings={"chunk_size": 800, "enable_reranking": True},
    ),
    llm=SimpleLLMProvider(default_provider="anthropic"),
    session=InMemorySessionProvider(max_sessions=1000),
)
```

### Local models

```python
# Ollama (just run `ollama serve` and pull a model)
export OLLAMA_BASE_URL=http://localhost:11434

from lasagna.providers import SimpleLLMProvider
llm = SimpleLLMProvider(default_provider="ollama")
model = llm.get_chat_model({"platform": "ollama", "model_name": "llama3.2"})

# HuggingFace Transformers (in-process, pip install lasagna-rag[local-huggingface])
llm = SimpleLLMProvider(default_provider="huggingface")
model = llm.get_chat_model({
    "platform": "huggingface",
    "model_id": "microsoft/Phi-3-mini-4k-instruct",
    "device_map": "auto",
})

# llama.cpp GGUF (in-process, pip install lasagna-rag[local-llamacpp])
llm = SimpleLLMProvider(default_provider="llamacpp")
model = llm.get_chat_model({
    "platform": "llamacpp",
    "model_path": "/path/to/model.gguf",
    "n_gpu_layers": -1,  # offload all layers to GPU
})

# vLLM or any OpenAI-compatible local server
export VLLM_BASE_URL=http://localhost:8000/v1
llm = SimpleLLMProvider(default_provider="vllm")
model = llm.get_chat_model({"platform": "vllm", "model_name": "my-model"})
```

### Custom provider

```python
from pathlib import Path
from lasagna.providers import ConfigProvider

class MyConfigProvider(ConfigProvider):
    @property
    def db_dir(self) -> Path: ...
    @property
    def upload_dir(self) -> Path: ...
    @property
    def package_root(self) -> Path: ...
    def get_rag_settings(self) -> dict: ...
    def get_llm_settings(self, key: str) -> dict | None: ...
```

## Configuration Surfaces

Providers feed into the broader configuration system:

| Surface | How it uses providers |
|---------|----------------------|
| **TOML** (`settings.lasagna.toml`) | `ConfigProvider.get_rag_settings()` merges with TOML defaults |
| **SDK** (`KnowledgeConfig`) | `_config_bridge.py` maps SDK fields to session overrides via `SessionConfigProvider` |
| **REST API** (`PATCH /api/v1/config`) | Writes to `SessionConfigProvider.set_config_overrides()` |
| **MCP** (tools) | Reads config via `RAGConfig`, which reads from providers |
| **CLI** (`python -m lasagna`) | `init` wizard configures providers interactively |
| **Configurator UI** | Full profile management via `SessionConfigProvider` |

**Config priority** (highest to lowest): Session overrides > Environment variables (`RAG_*`) > TOML settings > Code defaults.

## Design Decisions

**Models created per-call (no caching).** LLM chat model constructors (`ChatOpenAI`, `ChatAnthropic`, etc.) are lightweight wrappers that store config. The underlying HTTP clients (`httpx`) manage connection pooling internally. Caching would add complexity for key management (different temperatures, max_tokens, model names) with no measurable benefit.

**Deep copy pattern.** `SimpleConfigProvider` and `InMemorySessionProvider` return deep/shallow copies from all getters. This prevents callers from accidentally mutating provider-internal state.

**Thread safety.** The provider registry uses a global lock for writes and relies on CPython's GIL for atomic reference reads. `InMemorySessionProvider` uses a fine-grained lock per instance. `SimpleConfigProvider` is immutable after construction.

**Init-time vs runtime settings.** Some settings (`vector_backend`, `database_backend`, `cache_backend`) are process-wide and immutable after the backend singleton is created. All search/retrieval features are per-session and changeable at runtime via any configuration surface.
