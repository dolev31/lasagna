# sdk: the Python client

`KnowledgeClient` and `KnowledgeConfig`, the async API an application or an agent framework
uses: upload, search, context for an LLM prompt, agentic search, citations and per-session
configuration. In process it drives `api.RAGBackend`; `KnowledgeClient.from_url()` drives a
remote server over REST with the same interface. The subpackages re-export the internal
packages under stable SDK names, and `integrations/` connects LangChain, LlamaIndex and MCP in
both directions.

Using the client: [README.md](../../../README.md) and [docs/guide.md](../../../docs/guide.md).
Older user-facing pages live in [docs/](docs/user_guide.md) (`getting_started.md`,
`user_guide.md`, `configuration.md`, `integrations.md`, `custom_pipelines.md`, and others).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `__init__.py` | Public SDK surface (lazy where an import is heavy) |
| `client.py` | `KnowledgeClient`: backend creation, config bridging, `upload`, `search_documents`, `get_context`, `agentic_search`, the `documents`, `search` and `citations` managers; `from_url()` |
| `config.py` | `KnowledgeConfig` dataclass and the presets `config_for_speed`, `config_for_standard`, `config_for_balanced`, `config_for_max_quality`, `config_for_standalone`, `config_for_production` |
| `_config_bridge.py` | Maps `KnowledgeConfig` fields onto backend settings: `INIT_TIME_FIELDS`, `RUNTIME_FIELD_MAP`, `SPECIAL_MAPPINGS`, `NESTED_MAPPINGS`, `SDK_ONLY_FIELDS` |
| `config_validation.py` | Static rules checked when a config is built, and runtime checks that log when an adaptive stage overrides a user setting |
| `remote_backend.py` | `RemoteRAGBackend`: the synchronous backend interface the managers expect, implemented over `httpx` (no torch) |
| `managers/` | `DocumentManager`, `SearchManager`, `ConfigManager`, `CitationManager` |
| `types.py`, `exceptions.py` | Enums and result dataclasses; `KnowledgeError` and its subclasses |
| `integrations/` | `langchain.py` (Lasagna as a LangChain retriever and loader), `inbound.py` (`from_langchain`, `from_llamaindex`: a LangChain or LlamaIndex component as a sweep stage), `mcp.py`, `mcp_client.py` (`MCPKnowledgeClient`) |
| `agentic/`, `backends/`, `evaluation/`, `processing/`, `retrieval/`, `utils/` | Re-exports of `lasagna.agentic`, `vector_backends`, `knowledge_evaluation_sdk`, `processing`, `retrieval` and helpers |
| `component_registry.py` | Re-export of the core registry |
| `_version.py` | `__version__` from the installed distribution's metadata |

## Design notes

**Two tiers of configuration.** `KnowledgeClient._get_backend()` applies the config in two
steps. Storage paths and `INIT_TIME_FIELDS` go into `rag_config._env_overrides` before the
`RAGBackend` singleton exists, because they are read once and are process-wide (vector backend,
embedding model and the like). Runtime fields are written per session through
`SessionConfigProvider.set_config_overrides()` after it exists, and keys bridged earlier but no
longer set are removed. Changing `session_id` re-applies the runtime tier to the new session.

**Only fields the caller set are bridged.** A field is bridged when it was passed or assigned,
or differs from the dataclass default; untouched fields are not, so SDK defaults never mask a
profile or the TOML file.

**Adding a `KnowledgeConfig` field.** Add it to `config.py`, then to exactly one of
`INIT_TIME_FIELDS`, `RUNTIME_FIELD_MAP` or `SDK_ONLY_FIELDS` in `_config_bridge.py` (plus
`SPECIAL_MAPPINGS` or `NESTED_MAPPINGS` when the backend key differs or is nested).
`tests/test_every_sdk_setting_reaches_the_backend.py` fails for a field in none of them.

**Remote and local share the managers.** `RemoteRAGBackend` implements the same synchronous
methods as `RAGBackend` (`upload_document`, `search`, `get_documents`, ...), so the managers do
not know which they hold. A client method that needs a local-only feature checks
`KnowledgeClient._is_remote`.

**Embedding-size guard.** Applying runtime config can raise `EmbeddingSizeConflict`
(`api/_profile_guard.py`) when the change would leave the session's stored vectors at another
size; the client surfaces it rather than writing the override.
