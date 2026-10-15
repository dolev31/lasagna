# configurator: the profile editor web page

A React page, served by the REST app at `/configurator`, for editing a session's RAG settings
and saving them as profiles. The parameter list lives in Python (`schema.py`); the frontend
fetches it at runtime and renders a control for each entry, so adding a parameter to the page
is a Python change.

Users start it with `lasagna serve` (the configurator is on by default; `--no-configurator`
turns it off). Profiles themselves: [docs/configuration.md](../../../docs/configuration.md#profiles).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `__init__.py` | `mount_configurator(app, prefix="/configurator")`: includes `routes.router` and, when `frontend/dist/` exists, serves it as static files |
| `schema.py` | `ParameterDef` and `EnumOption`, `TABS`, `GROUPS`, `PARAMETERS`; `get_schema_dict()`, `get_reindex_keys()`, `get_defaults()` |
| `validation.py` | `validate_config(config, baseline)`: type and range checks per parameter, type checks for every other config key against its default, cross-field rules, reindex detection; returns `ValidationResult` |
| `routes.py` | `GET {prefix}/api/schema`, `POST {prefix}/api/validate` |
| `serve.py` | `main()`: runs `lasagna.api.rest.app:app` under uvicorn with `RAG_CONFIGURATOR=true` (default host 127.0.0.1, port 8000, next free port if taken). There is no `__main__.py`, so run it as `python -m lasagna.configurator.serve` |
| `frontend/` | React, TypeScript and Vite source. `npm run build` writes `frontend/dist/`, which the wheel ships (`configurator/frontend/dist/**/*` in `pyproject.toml`) |

Frontend entry points: `src/api/client.ts` (every call to the backend), `src/api/types.ts`
(TypeScript shapes of the Python responses), `src/hooks/useConfig.ts` (state, edits, saves),
`src/components/tabs/` (one container per tab).

Reference pages: [docs/schema.md](docs/schema.md), [docs/architecture.md](docs/architecture.md),
[docs/validation.md](docs/validation.md).

## Routes the page calls

| Mount | Routes | Source |
| ----- | ------ | ------ |
| `/configurator/api` | `GET /schema`, `POST /validate` | `routes.py` |
| `/api/config/knowledge` | `GET ""`, `PUT /active`, `POST /profile`, `POST /profile/delete`, `POST /profile/rename`, `POST /reset`, `GET /export`, `POST /import` | `api/rest/routes/knowledge_config.py` |
| `/api/knowledge` | `POST /reindex` | `api/rest/routes/reindex.py` |

The last two groups sit outside `/api/v1` because `client.ts` uses those paths. All three
resolve identity through `api/rest/dependencies.py`, so with `RAG_REQUIRE_AUTH` on they need a
session token like every other route ([../api/rest/DESIGN.md](../api/rest/DESIGN.md)).

## Design notes

**Python is the source of truth, with two copies in the frontend.**
`frontend/src/schema/reindexKeys.ts` repeats the reindex keys (the `requires_reindex`
parameters in `schema.py`) and the protected profile names, so the page can mark them before
the schema loads. `EssentialTab.tsx` holds an embedding-model-to-dimension map to fill in
`embedding_dim` when the model changes. Change either alongside `schema.py` and the embedding
defaults.

**Validation does not save.** `POST /validate` reports errors, warnings and which changed keys
need a reindex. Saving goes through `KnowledgeConfigManager` (`core/session_config/`), which
enforces its own checks, including the embedding-size guard in `api/_profile_guard.py`.

**Built-in profiles cannot be deleted or renamed.** The manager refuses it for any name in its
default profiles (`_is_protected_profile_name`); the frontend's `PROTECTED_PROFILES` set
(`default`, `speed`, `standard`, `balanced`, `max_quality`) only hides the buttons.

**Reindex deletes and re-uploads.** `POST /api/knowledge/reindex` re-processes every document
in the caller's session from its stored file, with `keep_file=True` on the delete, in a worker
thread.
