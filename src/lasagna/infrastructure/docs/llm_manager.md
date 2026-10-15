# LLM Manager -- Deep Dive

## Overview

`LLMManager` is a singleton that creates and caches LangChain `BaseChatModel` instances across 8 platforms. It is a **self-contained copy** for standalone deployment of the knowledge package — no dependencies on any host application.

`LLMProvider` (`providers/llm_provider.py`) abstracts over it, so a host application can supply its own LLM plumbing instead.

---

## Import Constraints

`llm_manager.py` imports at module level:

```python
from langchain_openai import ChatOpenAI, AzureChatOpenAI   # Required
from langchain_ibm import ChatWatsonx                       # Required
from langchain_groq import ChatGroq                          # Optional (try/except)
from langchain_google_genai import ChatGoogleGenerativeAI    # Optional (try/except)
```

`langchain_openai` and `langchain_ibm` are **required** — importing the module fails without them. This is why `LLMManager` is NOT exported from `infrastructure/__init__.py`.

---

## Platform Support Matrix

| Platform | Class | Required Env Vars | Optional Env Vars | Default Model |
| -------- | ----- | ----------------- | ----------------- | ------------- |
| `openai` | `ChatOpenAI` | `OPENAI_API_KEY` (implicit) | `MODEL_NAME`, `OPENAI_BASE_URL`, `OPENAI_API_VERSION` | `gpt-6-luna` |
| `azure` | `AzureChatOpenAI` | `AZURE_OPENAI_API_KEY` (implicit) | `MODEL_NAME` | `gpt-5.6-luna` |
| `groq` | `ChatGroq` | `GROQ_API_KEY` (implicit) | `MODEL_NAME` | `openai/gpt-oss-20b` |
| `watsonx` | `ChatWatsonx` | `WATSONX_PROJECT_ID`, `WATSONX_APIKEY` (implicit) | `MODEL_NAME` | `meta-llama/llama-4-maverick-17b-128e-instruct-fp8` |
| `google-genai` | `ChatGoogleGenerativeAI` | `GOOGLE_API_KEY` | `MODEL_NAME` | `gemini-2.5-pro` |
| `openrouter` | `ChatOpenAI` (wrapper) | `OPENROUTER_API_KEY` | `MODEL_NAME`, `OPENROUTER_BASE_URL`, `OPENROUTER_SITE_URL`, `OPENROUTER_APP_NAME` | `anthropic/claude-sonnet-5.5` |

### Model Name Resolution

Priority: `MODEL_NAME` env var > TOML `model_name` > platform default. The openai, azure and openrouter defaults are `SimpleLLMProvider.SUPPORTED_PLATFORMS`, where the source of each is cited.

---

## Singleton Pattern

Uses class-level `__new__` with double-checked locking:

```python
class LLMManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

`__init__` runs only once (guarded by `_initialized` flag).

---

## Model Cache

### Structure

`OrderedDict[str, BaseChatModel]` with LRU eviction.

### Cache Key

MD5 hash of model settings dict (with prompt keys removed) plus resolved model name, API version, and base URL:

```text
key = MD5(JSON({
    ...model_settings (minus prompt keys)...,
    "resolved_model_name": "gpt-6-luna",
    "resolved_api_version": "2024-08-06",
    "resolved_base_url": null,
}))
```

### Thread Safety

All model access is under `_cache_lock`. `set_llm()` and `clear_pre_instantiated_model()` also acquire `_cache_lock`, so model source switching is deterministic under concurrent access. Pre-instantiated path uses clone-per-call to prevent parameter bleed.

```
get_model(settings)
├── max_tokens is None? → ValueError
├── Acquire _cache_lock
│   ├── Pre-instantiated model set? → clone model → update params → return clone
│   ├── Cache hit? → move_to_end (LRU), update params, return
│   └── Cache miss?
│       ├── Create new LLM instance
│       ├── At max capacity? → evict LRU entries
│       ├── Store in cache
│       └── Update params, return
└── Release _cache_lock
```

### Capacity

Default: 50 entries (configurable via `RAG_LLM_CACHE_MAX_SIZE` env var). Invalid or non-positive values fall back to the default with a warning logged.

Each cached model is a LangChain `BaseChatModel` instance (~1-10 KB depending on platform). At 50 entries, memory is negligible.

### Parameter Mutation

`_update_model_parameters()` mutates the model in-place (`model.temperature = ...`, `model.max_tokens = ...`).

- **Cached models**: mutation happens under `_cache_lock`, preventing cross-thread contamination.
- **Pre-instantiated model** (`set_llm()`): a per-call clone is created via `model_copy(deep=True)` (Pydantic v2) or `copy.deepcopy()`, so parameter mutation does not leak across threads.

---

## Reasoning Model Detection

`_is_reasoning_model()` reads `lasagna.providers.model_capabilities`, the package's one table of what each model accepts, and also keeps the prefixes this manager always skipped (`o1`, `o3`, `gpt-5`) for ids the table does not know.

- A model that takes temperature only with reasoning off (`gpt-6-luna`, `gpt-6-sol`, `gpt-5.6-luna`, `gpt-5.5`, ...) is built with `reasoning_effort="none"` and the temperature, and later parameter updates set its temperature as usual.
- Any other reasoning model gets no temperature, at creation or on update.
- `max_completion_tokens` is used instead of `max_tokens` for Azure reasoning models.

---

## Error Handling

| Scenario | Behavior |
| -------- | -------- |
| `max_tokens` not in settings | `ValueError` (explicit, not assert) |
| `WATSONX_PROJECT_ID` missing | `ValueError` with descriptive message |
| `OPENROUTER_API_KEY` missing | `ValueError` with descriptive message |
| Unknown platform | `ValueError(f"Unsupported platform: {platform}")` |
| Model creation fails | Exception propagates to caller (no retry) |
| Groq not installed | `ChatGroq = None`; fails at creation time if platform is "groq" |
| Google GenAI not installed | `ChatGoogleGenerativeAI = None`; fails at creation time |
