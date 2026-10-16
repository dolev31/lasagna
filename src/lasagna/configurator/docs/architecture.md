# Architecture & Data Flow

## System Overview

```
 Browser (React SPA)                    Python Backend (FastAPI)
 ──────────────────                     ───────────────────────
 App.tsx                                api/rest/app.py
   ├─ useConfig.ts (state)                ├─ configurator/routes.py      (schema + validate)
   ├─ client.ts (HTTP)  ─── fetch ───►    ├─ routes/knowledge_config.py  (profile CRUD)
   └─ reindexKeys.ts                      └─ routes/reindex.py           (document re-processing)
                                               │
                                               ▼
                                          KnowledgeConfigManager
                                          (core/session_config/)
                                               │
                                               ▼
                                          RAGConfig (core/config.py)
```

## Full API Surface

### 1. Configurator Routes (`configurator/routes.py`)

Mounted at `/configurator/api/` by `mount_configurator()` when the server runs with `RAG_CONFIGURATOR=true` (or `lasagna serve --configurator`). The built frontend is served at `/configurator`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/configurator/api/schema` | Returns all 100 parameters with types, constraints, defaults, and the 4 tabs and 18 groups |
| POST | `/configurator/api/validate` | Validates config dict, returns errors/warnings/reindex info |

### 2. Knowledge Config Routes (`api/rest/routes/knowledge_config.py`)

Mounted at `/api/config/knowledge/`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/api/config/knowledge` | Full config state: profiles, active profile, session overrides |
| PUT | `/api/config/knowledge/active` | Set active profile + optional config overrides |
| POST | `/api/config/knowledge/profile` | Create or update a named profile |
| POST | `/api/config/knowledge/profile/delete` | Delete a custom profile |
| POST | `/api/config/knowledge/profile/rename` | Rename a custom profile |
| POST | `/api/config/knowledge/reset` | Reset config (scope: "active" or "all") |
| GET | `/api/config/knowledge/export` | Export full config as JSON |
| POST | `/api/config/knowledge/import` | Import config from JSON |

### 3. Reindex Route (`api/rest/routes/reindex.py`)

Mounted at `/api/knowledge/`.

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/knowledge/reindex` | Re-process all documents in a session with current config |

Security: the session ID comes from the identity headers or query parameter, never the request body, so a caller cannot reindex another session by naming it. Each source file is read back through the storage backend that holds it and copied to a temporary file under its base name (`os.path.basename`) before it is processed again.

## End-to-End Data Flow

### User Edits a Parameter

```
1. User adjusts slider in UI
2. useConfig.ts updates local state, and compares the edit with the
   saved config against REINDEX_KEYS; the UI shows the reindex banner
3. User clicks Save
4. client.ts POST /configurator/api/validate  →  validation.py
   (errors stop the save and are shown)
5. client.ts PUT /api/config/knowledge/active  →  knowledge_config.py
6. KnowledgeConfigManager persists session overrides
7. If reindex needed, the user starts it from the banner:
   client.ts POST /api/knowledge/reindex  →  reindex.py
8. RAGBackend re-processes documents with new config
```

### User Switches Profile

```
1. User selects profile in Sidebar
2. client.ts PUT /api/config/knowledge/active (body: {active_profile: "speed"})
3. KnowledgeConfigManager sets active profile + loads its config
4. useConfig.ts receives full state, updates all controls
```

## Configuration Resolution Order

`RAGConfig` (singleton in `core/config.py`) resolves each setting through a 4-level priority chain, defined in `_get_knowledge_setting()`:

```
Priority (highest first):
  1. Session overrides: per-session UI changes via KnowledgeConfigManager
  2. Environment vars:  LASAGNA_<NAME> or RAG_<NAME> (LASAGNA_ wins when both are set)
  3. TOML settings:     settings.lasagna.toml
  4. Code defaults:     the default_config dict in _load_config()
```

The merge in `_load_config()` builds the base dict as `{**default_config, **knowledge_rag, **rag_settings, **env_overrides}`.

Session/profile overrides are managed by `KnowledgeConfigManager` (`core/session_config/`), which writes session-scoped state. `RAGConfig` reads it via `_get_session_override`.

## Frontend Component Tree

```
App.tsx                          (under components/)
  ├─ layout/Header.tsx           Logo + session info
  ├─ layout/Sidebar.tsx          Profile list + CRUD buttons
  ├─ layout/TabBar.tsx           4 tabs (Essential, Search & Retrieval, Processing, Advanced)
  ├─ tabs/*Tab.tsx               One component per tab; each lists its groups
  │    └─ fields/*.tsx           FieldGroup, ToggleSwitch, SelectField, SliderField,
  │                              NumberField, TextField, HelpTooltip
  ├─ shared/ReindexBanner.tsx    Reindex prompt
  ├─ shared/Toast.tsx            Save and error notices
  ├─ shared/CreateProfileDialog.tsx, shared/ConfirmDialog.tsx
  └─ layout/Footer.tsx           Save/reset actions
```

State is centralized in `hooks/useConfig.ts`, which:
- Fetches full config on mount via `GET /api/config/knowledge`
- Tracks local edits against the last saved config
- Validates before saving via `POST /configurator/api/validate`
- Saves via `PUT /api/config/knowledge/active`

## Session & Profile Lifecycle

```
1. Frontend generates session ID (configurator-<random>) → localStorage
2. The config API calls pass it as the ?session_id= query parameter
3. Backend resolves KnowledgeConfigManager per session
4. Profiles are global (shared across sessions)
5. Session overrides are per-session (cleared on reset)
6. 5 built-in profiles are protected: default, speed, standard, balanced, max_quality
```

## Schema-Driven Rendering

Parameter metadata (labels, types, ranges, help text) is fetched from the backend at runtime; the frontend does not hardcode it. Three things are hardcoded in the frontend:

1. **`schema/reindexKeys.ts`** hardcodes the 20 reindex key names (and the 5 protected profile names) so the UI can show reindex warnings instantly, before the schema response arrives.
2. **`components/tabs/EssentialTab.tsx`** hardcodes an embedding model → dimension mapping (the `dims` object) so changing the model auto-fills the dimension field without a round-trip.
3. **Each `components/tabs/*Tab.tsx`** lists the group keys it renders, with their titles.

All three must be kept in sync with `schema.py`. A drift guard test (`TestReindexKeySync` in `tests/test_configurator.py`) parses `reindexKeys.ts` directly to enforce parity for the first.

```
schema.py PARAMETERS
  → get_schema_dict()
    → GET /configurator/api/schema (JSON)
      → useConfig.ts parses response
        → TabBar renders tabs from schema.tabs
          → each *Tab.tsx filters the parameters by its group keys
            → renderParam picks ToggleSwitch/SelectField/SliderField/NumberField/TextField by param.type
```

A new parameter needs a `ParameterDef` in `schema.py`. The frontend picks it up when its group is one a tab already lists; a new group, a reindex key or a model-dimension mapping needs a frontend change too.
