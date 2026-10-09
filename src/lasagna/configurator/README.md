# RAG Configurator

Schema-driven UI for managing RAG configuration profiles. The backend (`schema.py`) defines all 91 tunable parameters — the React frontend fetches this schema at runtime and renders controls dynamically. Two client-side exceptions: `reindexKeys.ts` hardcodes the 18 reindex key names (for instant UI feedback before schema loads), and `EssentialTab.tsx` hardcodes an embedding model-to-dimension mapping (for auto-fill on model change).

## Module Structure

| File | Purpose |
|------|---------|
| `__init__.py` | `mount_configurator(app, prefix)` — registers API routes and serves the built frontend |
| `schema.py` | Single source of truth: 91 `ParameterDef` entries, tabs, groups, helpers (`get_schema_dict`, `get_reindex_keys`, `get_defaults`) |
| `validation.py` | 8 cross-field validation rules + type/range checks + reindex detection |
| `routes.py` | 2 endpoints: `GET /configurator/api/schema`, `POST /configurator/api/validate` |
| `serve.py` | CLI helper invoked by `RAG_CONFIGURATOR=true python3 -m lasagna serve` (runs uvicorn with configurator mounted) |
| `frontend/` | React + TypeScript + Vite SPA (see below) |

### Frontend Tree

```
frontend/
  src/
    App.tsx                      Root component — layout + tab routing
    index.tsx                    Entry point
    api/
      client.ts                  10 API functions (fetch → backend)
      types.ts                   TypeScript interfaces matching Python shapes
    components/
      fields/                    Parameter renderers (BooleanField, SliderField, EnumField, ...)
      layout/                    Header, Sidebar, Footer, TabBar
      profiles/                  Profile selector + CRUD UI
      shared/                    Reusable atoms (Tooltip, StatusBadge, ...)
      tabs/                      Tab-level containers (EssentialTab, SearchTab, ...)
    hooks/
      useConfig.ts               Central state manager — fetches config, applies edits, saves profiles
    schema/
      reindexKeys.ts             18 REINDEX_KEYS + 5 PROTECTED_PROFILES
    styles/
      configurator.css           Full stylesheet
      variables.css              CSS custom properties (colors, spacing)
```

## Quick Start

### Standalone (development)

```bash
RAG_CONFIGURATOR=true python3 -m lasagna serve
# → http://localhost:8000/configurator
```

This uses `serve.py` which runs the full REST API with the configurator mounted.

### Embedded in an existing FastAPI app

```python
from lasagna.configurator import mount_configurator

app = FastAPI()
mount_configurator(app)  # mounts at /configurator by default
```

## API Surface

The frontend communicates with **three** route groups:

| Route Group | Mount Point | Endpoints | Source |
|-------------|-------------|-----------|--------|
| Configurator | `/configurator/api/` | `GET /schema`, `POST /validate` | `configurator/routes.py` |
| Knowledge Config | `/api/config/knowledge/` | `GET ""`, `PUT /active`, `POST /profile`, `POST /profile/delete`, `POST /profile/rename`, `POST /reset`, `GET /export`, `POST /import` | `api/rest/routes/knowledge_config.py` |
| Reindex | `/api/knowledge/` | `POST /reindex` | `api/rest/routes/reindex.py` |

## Integration Points

- **`KnowledgeConfigManager`** (`core/session_config/`): Profile CRUD, session-scoped config state, validation enforcement
- **`RAGConfig`** (`core/config.py`): Singleton that resolves effective config via the priority chain (see [architecture docs](docs/architecture.md))
- **`RAGBackend`** (`api/rag_backend.py`): Document operations used by the reindex endpoint

## Profile System

5 built-in profiles are protected from deletion and renaming: `default`, `speed`, `standard`, `balanced`, `max_quality`. Custom profiles can be created, renamed, and deleted freely.

## Security Notes

- No authentication — designed as a local development tool
- No CSRF protection — same-origin API only
- Session IDs from `localStorage` (frontend) or headers (backend)
- Reindex endpoint has path traversal protection (`os.path.realpath` + upload dir check)

## Documentation

- [Parameter Reference (91 params)](docs/schema.md)
- [Architecture & Data Flow](docs/architecture.md)
- [Validation Rules (8 cross-field rules)](docs/validation.md)
