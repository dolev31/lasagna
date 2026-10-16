# providers: configuration, LLM and session interfaces

Three small interfaces that decouple the package from where its paths, chat models and
per-session state come from, a process-wide registry holding one instance of each, and the
standalone implementations used when nothing else is configured. A host application can supply
its own implementation of any of the three with `configure_providers()`.

Platforms, default models and the API key each one reads:
[docs/configuration.md](../../../docs/configuration.md#llm-providers-and-models). What
`temperature=0` sends to each model, and when a run cannot be pinned:
[docs/configuration.md](../../../docs/configuration.md#temperature-and-repeatable-runs).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `__init__.py` | Registry: `get_/set_config_provider`, `get_/set_llm_provider`, `get_/set_session_provider`, `configure_providers`, `reset_providers`; auto-configuration; `LLMProviderNotConfigured` |
| `config_provider.py` | `ConfigProvider` ABC (`db_dir`, `upload_dir`, `data_dir`, `package_root`, `get_rag_settings`, `get_llm_settings`); `SimpleConfigProvider` |
| `llm_provider.py` | `LLMProvider` ABC (`get_chat_model`, `get_llm`, `validate_model_settings`); `SimpleLLMProvider` with `SUPPORTED_PLATFORMS` (platform to default model) and `_OPENAI_COMPATIBLE` (base URL and key variable per OpenAI-compatible platform) |
| `model_capabilities.py` | Which sampling parameters a model accepts (`ACCEPTED`, `NEEDS_EFFORT_NONE`, `REJECTED`); `sampling_kwargs` builds them and says whether the run is pinned |
| `session_provider.py` | `SessionConfigProvider` ABC (resolved config, RAG toggle, focused and excluded documents, config overrides); `InMemorySessionProvider`; `PersistentSessionProvider` |

## Auto-configuration

The first `get_*_provider()` call runs `_auto_configure_from_env()` under `_provider_lock`, and
fills in only the providers not already set:

1. Config: `SimpleConfigProvider` with `core.paths.db_dir()` and `core.paths.upload_dir()`
   (under `~/.lasagna` unless overridden; see
   [docs/configuration.md](../../../docs/configuration.md#where-data-lives)).
2. LLM: `SimpleLLMProvider(default_provider=...)` for the first variable of
   `_API_KEY_TO_PLATFORM` that is set (`OPENAI_API_KEY` first, then `ANTHROPIC_API_KEY`,
   `GROQ_API_KEY`, ...; watsonx accepts `WATSONX_APIKEY` and `WATSONX_API_KEY`; the local
   platforms come last). With none set, `get_llm_provider()` raises `LLMProviderNotConfigured`.
3. Session: `InMemorySessionProvider`, unless `RAG_SESSION_BACKEND=persistent`, which selects
   `PersistentSessionProvider`. The `lasagna` CLI (`cli.py`) and `lasagna-mcp`
   (`--session-backend`, default `persistent`) set it to `persistent`, so a profile chosen by
   one command applies in the next. A library user who imports `lasagna` directly gets the
   in-memory provider. The variable is read from the environment directly; it has no
   `LASAGNA_` alias.

`reset_providers()` clears all three and the done flag; tests use it.

## Configuring providers in code

Call `configure_providers()` before anything reads a provider; auto-configuration then fills in
only what you left out.

```python
from pathlib import Path
from lasagna.providers import (
    InMemorySessionProvider, SimpleConfigProvider, SimpleLLMProvider, configure_providers,
)

configure_providers(
    config=SimpleConfigProvider(db_dir=Path("/data/rag/dbs"), upload_dir=Path("/data/rag/uploads")),
    llm=SimpleLLMProvider(default_provider="anthropic"),
    session=InMemorySessionProvider(max_sessions=1000),
)
```

A provider of your own subclasses the ABC. A `ConfigProvider` implements the `db_dir`,
`upload_dir` and `package_root` properties and `get_rag_settings()` and
`get_llm_settings(key)`; an `LLMProvider` implements `get_chat_model()` and
`validate_model_settings()`; a `SessionConfigProvider` implements every method of the ABC in
`session_provider.py`.

`get_rag_settings()` ranks below the settings file: a `RAGConfig` property reads the session
override, then the environment, then `settings.lasagna.toml`, and only then the merged defaults
that include the provider's settings. A key the TOML file sets is therefore never taken from
the provider; set it in the environment or as a session override instead.

## Design notes

**Session overrides flow through the session provider.** `RAGConfig.with_session(id)` reads a
session's overrides from `get_session_provider().get_resolved_config(id)`. With the in-memory
provider, a write to `KnowledgeConfigManager` is not visible to `RAGConfig` until it is also
written to the provider: `PATCH /api/v1/config` (`api/rest/routes/config.py`) writes both, and
skips the second write when the provider sets `_delegates_to_manager` (as
`PersistentSessionProvider` does). Any other write path must do the same.

**`PersistentSessionProvider` resolves its manager lazily.** It is constructed inside
`_auto_configure_from_env()` while `_provider_lock` is held; building the manager there would
call `get_config_provider()` and re-enter the lock.

**One place builds chat models.** Every chat model the pipeline and the harness use comes from
an `LLMProvider.get_chat_model()`, which is what makes token accounting
(`utils.token_usage.attach_usage_meter`) and the experiment response cache
(`utils.llm_cache.attach_llm_cache`) complete. `SimpleLLMProvider` attaches both itself, and
the cache only replays one of its models that decodes greedily: temperature 0 was asked for,
`model_capabilities` says the model accepts it, and the LangChain client kept it. Any other
subclass's `get_chat_model` is wrapped (`LLMProvider.__init_subclass__`), so a provider set
with `set_llm_provider` gets both as well; its model is replayed when temperature 0 was asked
for and the model's `temperature` is 0. Models are built per call and not cached; the HTTP clients pool
connections themselves.

**Unknown models keep temperature.** `model_capabilities` sends `temperature` for an id it does
not know: dropping it on a guess would unpin a model that honors it. When a provider changes
what a model accepts, update the table there and the default in `SUPPORTED_PLATFORMS`, with the
provider's deprecation page in a comment. `tests/test_no_retired_model_defaults.py` fails when
a retired id appears anywhere under `src/lasagna`, or a deprecated one outside the two files
allowed to name it.

**Copies, not references.** `SimpleConfigProvider.get_rag_settings()` returns a deep copy and
`InMemorySessionProvider` returns copies of its lists and dicts, so callers cannot mutate
provider state. `InMemorySessionProvider(max_sessions=N)` evicts the least recently used
session; 0 (the default) keeps all.

**Process-wide settings.** `database_backend` (fixed by the first `get_rag_database()`; the
session-config persistence mixin is picked from `RAG_DATABASE_BACKEND` when
`core.session_config` is imported) and `cache_backend` hold for the life of the process.
Retrieval and generation settings are per session and can change between queries.
