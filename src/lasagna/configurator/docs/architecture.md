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

Mounted at `/configurator/api/` by `mount_configurator()`.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/configurator/api/schema` | Returns all 91 parameters with types, constraints, defaults, tabs, groups |
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

Security: Session ID from auth headers (not request body) to prevent cross-tenant access. Path traversal protection via `os.path.realpath` + upload directory containment check.

## End-to-End Data Flow

### User Edits a Parameter

```
1. User adjusts slider in UI
2. useConfig.ts updates local state (optimistic)
3. client.ts POST /configurator/api/validate  →  validation.py
4. UI shows errors/warnings/reindex banner
5. User clicks Save
6. client.ts PUT /api/config/knowledge/active  →  knowledge_config.py
7. KnowledgeConfigManager persists session overrides
8. If reindex needed: client.ts POST /api/knowledge/reindex  →  reindex.py
9. RAGBackend re-processes documents with new config
```

### User Switches Profile

```
1. User selects profile in Sidebar
2. client.ts PUT /api/config/knowledge/active (body: {active_profile: "speed"})
3. KnowledgeConfigManager sets active profile + loads its config
4. useConfig.ts receives full state, updates all controls
```

## Configuration Resolution Order

`RAGConfig` (singleton in `core/config.py`) resolves each setting through a 4-level priority chain. Defined in `_get_knowledge_setting()` (config.py:544-575):

```
Priority (highest first):
  1. Session overrides  — per-session UI changes via KnowledgeConfigManager
  2. Environment vars   — RAG_* prefix (12-factor)
  3. TOML settings      — settings.lasagna.toml
  4. Code defaults      — _load_config() hardcoded dict
```

The merge in `_load_config()` (config.py:452) builds the base dict as `{**default_config, **knowledge_rag, **rag_settings, **env_overrides}`.

Session/profile overrides are managed by `KnowledgeConfigManager` (`core/session_config/`), which writes session-scoped state. `RAGConfig` reads it via `_get_session_override`.

## Frontend Component Tree

```
App.tsx
  ├─ layout/Header.tsx           Logo + session info
  ├─ layout/Sidebar.tsx          Profile list + CRUD buttons
  ├─ layout/TabBar.tsx           4 tabs (Essential, Search, Processing, Advanced)
  ├─ tabs/*Tab.tsx               Tab containers → render groups
  │    └─ fields/*Field.tsx      Parameter renderers (BooleanField, SliderField, EnumField, etc.)
  ├─ shared/Tooltip.tsx          Help text popover (technical vs plain toggle)
  ├─ shared/StatusBadge.tsx      Reindex/validation status indicators
  └─ layout/Footer.tsx           Save/reset actions + reindex prompt
```

State is centralized in `hooks/useConfig.ts`, which:
- Fetches full config on mount via `GET /api/config/knowledge`
- Tracks local edits as a delta object
- Validates on every change via `POST /configurator/api/validate`
- Saves via `PUT /api/config/knowledge/active`

## Session & Profile Lifecycle

```
1. Frontend generates session ID → localStorage
2. All API calls include X-Session-Id header
3. Backend resolves KnowledgeConfigManager per session
4. Profiles are global (shared across sessions)
5. Session overrides are per-session (cleared on reset)
6. 5 built-in profiles are protected: default, speed, standard, balanced, max_quality
```

## Schema-Driven Rendering

Parameter metadata (labels, types, ranges, help text) is fetched from the backend at runtime — the frontend does not hardcode it. Two exceptions exist for UX reasons:

1. **`reindexKeys.ts`** — hardcodes the 18 reindex key names so the UI can show reindex warnings instantly, before the schema response arrives.
2. **`EssentialTab.tsx`** — hardcodes an embedding model → dimension mapping (`dims` object at line 70) so changing the model auto-fills the dimension field without a round-trip.

Both must be kept in sync with `schema.py`. A drift guard test (`TestReindexKeySync`) parses `reindexKeys.ts` directly to enforce parity.

```
schema.py PARAMETERS
  → get_schema_dict()
    → GET /configurator/api/schema (JSON)
      → useConfig.ts parses response
        → TabBar renders tabs from schema.tabs
          → GroupContainer renders groups from schema.groups
            → FieldRenderer dispatches to BooleanField/SliderField/EnumField based on param.type
```

Adding a new parameter requires only adding a `ParameterDef` to `schema.py`. The frontend picks it up automatically (unless it involves reindex keys or model-dimension mappings, which need manual client-side updates).
