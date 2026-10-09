# REST API — Design Overview

## Architecture

```
Client
  |
  |  POST /api/v1/sessions   (handshake — get a token)
  |  X-Session-Token: eyJ...  (all subsequent requests)
  |
  v
FastAPI (app.py)
  |
  |-- Middleware chain (outermost → innermost):
  |     request-id → rate-limit → timing → handler
  |
  |-- Dependencies (dependencies.py):
  |     get_session_id()  — token → header → query → "default"
  |     get_user_id()     — Bearer → token → header → anon_<uuid>
  |
  |-- Routes (/api/v1/):
  |     health.py      — liveness + readiness probes
  |     sessions.py    — POST /sessions (identity handshake)
  |     documents.py   — CRUD for documents
  |     search.py      — semantic search endpoints
  |     citations.py   — citation retrieval
  |     config.py      — profile & config management
  |
  |-- Configurator Routes (outside /api/v1/):
  |     knowledge_config.py — /api/config/knowledge/* (profile CRUD, import/export)
  |     reindex.py          — /api/knowledge/reindex (re-process documents)
  |
  |-- session_token.py  — stateless JWT encode/decode (shared)
  |
  v
Core layer (session_config, rag_backend, rate_limiter, ...)
```

## Identity Model

### Problem

Without a handshake, unauthenticated clients share the identity
`"__anonymous__"`, causing profile collisions, config bleed-through,
and broken ownership checks in multi-tenant deployments.

### Solution: Stateless Session Tokens

A lightweight `POST /api/v1/sessions` endpoint issues a signed JWT
encoding `(session_id, user_id)`.  The token is self-contained — the
server decodes it on every request without any database lookup.

```
POST /api/v1/sessions
Body (all optional): { "session_id": "...", "user_id": "..." }
Response 201:        { "token": "eyJ...", "session_id": "...",
                       "user_id": "...", "expires_at": "..." }
```

Omitted fields auto-generate UUIDs (`sess_<hex12>`, `user_<hex12>`).

### Priority Chains

Identity is resolved from the highest-trust source available:

| `get_user_id()`                        | `get_session_id()`              |
|----------------------------------------|---------------------------------|
| 1. JWT Bearer `sub` (upstream auth)    | 1. `X-Session-Token` → `sid`   |
| 2. `X-Session-Token` → `sub`          | 2. `X-Session-ID` header       |
| 3. `X-User-ID` header                 | 3. `session_id` query param    |
| 4. Auto-assign `anon_<uuid>`          | 4. `"default"`                  |

**Zero breaking changes** — existing `X-Session-ID` + `X-User-ID`
clients keep working.  Bare requests get a unique `anon_<uuid>` user_id
per request (no longer `"__anonymous__"`), preventing identity collisions.

### Token Claims

```json
{
  "sub": "<user_id>",
  "sid": "<session_id>",
  "typ": "session",
  "iat": 1700000000,
  "exp": 1700604800,
  "iss": "rag-knowledge"
}
```

- `typ: "session"` distinguishes from upstream auth JWTs.
- Signed with `RAG_JWT_SECRET` env var (ephemeral fallback in dev).
- Default TTL: 7 days (`RAG_SESSION_TOKEN_TTL_SECONDS`).

## Key Files

| File | Responsibility |
|------|---------------|
| `session_token.py` | Token create/decode + shared `extract_session_id()` used by both middleware and dependencies. No FastAPI dependency — importable from MCP. |
| `dependencies.py` | FastAPI `Depends()` functions: `get_session_id()`, `get_user_id()`, `get_backend()`. Delegates session extraction to `session_token.extract_session_id()`. |
| `app.py` | App factory, middleware chain, lifespan. `_extract_session_id()` calls the shared helper for rate-limiter session resolution. |
| `routes/sessions.py` | `POST /api/v1/sessions` — handshake endpoint. |

## Middleware Chain

Registered in `create_app()`.  FastAPI middleware executes
outermost-first:

```
Request-ID  →  Rate-Limit  →  Timing  →  Route Handler
```

1. **Request-ID** — attaches `X-Request-ID` (auto-generated or echoed).
2. **Rate-Limit** — per-session token bucket.  Extracts session_id via
   the shared helper.  Returns 429 with `Retry-After` when exceeded.
   Exempt paths: `/health`, `/docs`, `/mcp`, `/configurator`,
   `/api/v1/sessions`.
3. **Timing** — records `X-Response-Time` header and feeds latency to
   the Prometheus histogram + rate-limiter load estimator.

## MCP Integration

The MCP server (`mcp/server.py`) reuses the same identity model:

- `_resolve_session(session_id, session_token)` decodes the token or
  falls back to `(session_id, anon_<uuid>)`.
- All tools that use `get_knowledge_config_manager()` pass the resolved
  `user_id` and `session_id` so PostgreSQL-mode managers are
  correctly scoped.
- `save_rag_profile()` and `delete_rag_profile()` forward `user_id`
  to the profile ownership system.

## Flow Diagrams

### With handshake

```
Client                              Server
  |  POST /api/v1/sessions           |
  |  {"user_id": "alice"}            |
  |--------------------------------->|  Generate sess_<hex>, sign JWT
  |  201 {token, session_id, ...}    |
  |<---------------------------------|
  |                                  |
  |  GET /api/v1/config              |
  |  X-Session-Token: eyJ...         |
  |--------------------------------->|  Decode → (alice, sess_<hex>)
  |  200 {config for alice}          |
  |<---------------------------------|
```

### Without handshake (bare client)

```
Client                              Server
  |  GET /api/v1/config              |
  |  (no headers)                    |
  |--------------------------------->|  session_id = "default"
  |                                  |  user_id   = "anon_7f3a..."
  |  200 {config}                    |  (unique per request)
  |<---------------------------------|
```

### Backward-compatible (legacy headers)

```
Client                              Server
  |  GET /api/v1/config              |
  |  X-Session-ID: my-session        |
  |  X-User-ID: bob                  |
  |--------------------------------->|  session_id = "my-session"
  |                                  |  user_id   = "bob"
  |  200 {config for bob}            |
  |<---------------------------------|
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_JWT_SECRET` | (ephemeral) | HMAC secret for signing session tokens. **Set in production** so tokens survive restarts and work across workers. |
| `RAG_SESSION_TOKEN_TTL_SECONDS` | `604800` (7d) | Token lifetime. |

## Security Notes

- Session tokens are stateless and **cannot be revoked** short of
  rotating `RAG_JWT_SECRET`.  Keep TTL reasonable.
- The handshake endpoint is exempt from session-based rate limiting
  (it creates the session identity).  Consider IP-based rate limiting
  at the load balancer level.
- Bearer JWT decode enforces `require: ["sub"]` to reject malformed
  upstream tokens.
- All silent `except` blocks log at `DEBUG` level for security auditing.

## Configurator REST Routes

The configurator UI (React SPA at `/configurator`) calls a set of
REST endpoints that bridge to `KnowledgeConfigManager`.  These are
mounted **directly on the app** (not under `/api/v1/`) because the
frontend hardcodes the paths in `client.ts`.

### Knowledge Config (`/api/config/knowledge`)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/config/knowledge` | Full config state (profiles, active, overrides) |
| PUT | `/api/config/knowledge/active` | Switch active profile + optional overrides |
| POST | `/api/config/knowledge/profile` | Create or update a named profile |
| POST | `/api/config/knowledge/profile/delete` | Delete a custom profile |
| POST | `/api/config/knowledge/profile/rename` | Rename a custom profile |
| POST | `/api/config/knowledge/reset` | Reset config (`scope: "active"` or `"all"`) |
| GET | `/api/config/knowledge/export` | Export full config as JSON |
| POST | `/api/config/knowledge/import` | Import config from JSON |

All endpoints resolve `session_id` and `user_id` via the same
`Depends(get_session_id)` / `Depends(get_user_id)` chains described
above.  Profile mutation endpoints (`save`, `delete`) pass `user_id`
for ownership tracking.

### Reindex (`/api/knowledge/reindex`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/knowledge/reindex` | Re-process all documents in a session |

Body: `{ "session_id": "...", "force_rebuild": true }`

Blocking I/O (file reads, vector DB operations) is offloaded via
`asyncio.to_thread()` to avoid blocking the event loop.
