# API Layer Configuration Reference

All configuration that affects the `api/` package: environment variables, middleware settings, session tokens, and validation constraints. Each entry names the file that reads it; line numbers are left out because they drift.

---

## Environment Variables

These are server settings read from the environment, under their `RAG_` names only. (The `LASAGNA_` alias that pipeline settings accept does not apply to them.)

| Variable | Default | Source | Description |
|----------|---------|--------|-------------|
| `RAG_JWT_SECRET` | ephemeral random | `rest/session_token.py` | HMAC-SHA256 signing key for session tokens. If unset, a random secret is generated per process, so tokens stop working on restart. With `RAG_REQUIRE_AUTH` on, an unset, placeholder or shorter-than-32-character secret makes `create_app()` refuse to start. |
| `RAG_SESSION_TOKEN_TTL_SECONDS` | `604800` (7 days) | `rest/session_token.py` | Session token expiry in seconds. |
| `RAG_REQUIRE_AUTH` | `false` | `rest/auth_mode.py` | When `true`, only a verified `X-Session-Token` names a session, and `POST /api/v1/sessions` needs an `X-API-Key`. When `false`, binding to a non-loopback address is a startup error. |
| `RAG_API_KEYS` | unset | `rest/auth_mode.py` | Comma-separated keys (each at least 32 characters) that may mint session tokens when `RAG_REQUIRE_AUTH` is on. |
| `RAG_ADMIN_TOKEN` | unset | `rest/routes/gc_admin.py`, `rest/routes/sessions.py` | Value of `X-Admin-Token` for the `/api/v1/admin/*` routes and for naming a session in `POST /api/v1/sessions`. Unset, placeholder or shorter than 32 characters: the admin routes answer 503. |
| `RAG_ALLOW_ANONYMOUS_SESSION` | `false` | `rest/auth_mode.py` | With auth on, keep the anonymous `default` session usable. |
| `RAG_BIND_IS_PROTECTED` | `false` | `rest/auth_mode.py` | Assert that something outside the process (a loopback port mapping, a NetworkPolicy, a firewall) restricts access, so the server may bind beyond loopback with auth off. |
| `RAG_CORS_ORIGINS` | `""` (no cross-origin access) | `rest/app.py` | Comma-separated allowed CORS origins. Empty denies cross-origin requests and leaves same-origin ones alone. `*` logs a warning, and is refused at startup when `RAG_REQUIRE_AUTH` is on. |
| `RAG_EMBED_MCP` | `false` | `rest/app.py` | When `true`, mount the MCP SSE server at `/mcp` inside the FastAPI process, for single-process deployments such as Milvus Lite, where only one process may open the store. |
| `RAG_CONFIGURATOR` | `false` | `rest/app.py` | When `true`, mount the configurator UI at `/configurator` and its API routes. |
| `RAG_ALLOW_HEADER_IDENTITY` | `false` | `rest/dependencies.py` | When `true`, accept the `X-User-ID` header for user identity. When `false`, the header is rejected with 401. |
| `RAG_MAX_BATCH_FILES` | `20` | `rest/limits.py` | Files in one `POST /api/v1/documents/batch`. |
| `RAG_MAX_BATCH_BYTES` | `209715200` (200 MiB) | `rest/limits.py` | Total body of one batch upload. |
| `RAG_SEARCH_TIMEOUT` | `120` | `rest/limits.py` | Seconds a REST search may run before it answers 504. |
| `LASAGNA_LOG_FORMAT` | `text` | `rest/limits.py` | `json` writes one JSON object per log line. |
| `PROMETHEUS_MULTIPROC_DIR` | unset | `rest/app.py` | Directory for Prometheus multi-process metric aggregation (gunicorn deployments). |

An invalid value for one of the `rest/limits.py` variables is an error at startup that names the variable. A single upload is limited by the `max_file_size` setting (`RAG_MAX_FILE_SIZE`, 50 MB by default).

---

## Session Token Configuration

| Setting | Value | Source |
|---------|-------|--------|
| Algorithm | HS256 | `rest/session_token.py` |
| Default TTL | 604800s (7 days) | `rest/session_token.py` |
| Max session_id length | 128 chars | `rest/session_token.py` |
| Required JWT claims | `sub`, `sid`, `exp`, `typ` | `rest/session_token.py` |
| Token type claim | `typ: "session"` | `rest/session_token.py` |
| Issuer claim | `iss: "lasagna"` (`"rag-knowledge"` still accepted) | `rest/session_token.py` |

JWT payload structure:

```json
{
  "sub": "<user_id>",
  "sid": "<session_id>",
  "typ": "session",
  "iat": 1700000000,
  "exp": 1700604800,
  "iss": "lasagna"
}
```

With `RAG_REQUIRE_AUTH` on, the payload also carries `prn`, the principal that minted it (an API key's digest, or `admin`); rate limits are counted per principal.

---

## Middleware Configuration

### Exempt Prefixes

Paths exempt from rate limiting and timing metrics (`_EXEMPT_PREFIXES` in `rest/app.py`), besides `/` and `/metrics`:

```
/health, /api/v1/health, /docs, /openapi.json, /redoc
```

Note: `/api/v1/sessions` is intentionally NOT exempt, so session creation is rate-limited.

### Middleware Execution Order

Outermost-first (each wraps the next):

1. **Request-ID**: attaches `X-Request-ID` (the client's value when well-formed, else a generated one)
2. **Body-Size-Limit**: 413 before reading a body over the route's limit
3. **Rate-Limit**: per-tenant adaptive token bucket, returns 429 + `Retry-After`
4. **Timing**: records `X-Response-Time` header, feeds Prometheus + rate limiter
5. **CORS**: applies `RAG_CORS_ORIGINS`
6. **Handler**: FastAPI route execution

---

## Validation Constraints

### Session & Identity (`_validation.py`, `rest/dependencies.py`)

| Constraint | Value | Source |
|------------|-------|--------|
| Session ID max length | 128 chars | `rest/session_token.py` |
| Session ID pattern | `[a-zA-Z0-9_\-\.]+` | `_validation.py` |
| User ID max length | 128 chars | `rest/dependencies.py` |
| Document ID pattern | `doc_[0-9a-f]{16}` | `_validation.py` |

### Upload Limits (`rest/models.py`, `rest/routes/documents.py`, `rest/limits.py`)

| Constraint | Value | Source |
|------------|-------|--------|
| Max file size | `rag_config.max_file_size` (50 MB) | `rest/routes/documents.py` (streamed check) |
| Max request body | upload routes: the file limit (batch: `RAG_MAX_BATCH_BYTES`) plus 1 MiB for multipart overhead; every other body: 1 MiB | `rest/limits.py` |
| Chunk pagination limit | 1 to 500 | `rest/routes/documents.py` |
| Chunk pagination offset | >= 0 | `rest/routes/documents.py` |

### Configurator Models (`rest/models.py`)

| Model | Constraint |
|-------|-----------|
| `ResetConfigRequest.scope` | regex `^(active\|all)$` |
| `ReindexRequest.session_id` | max 256 chars (deprecated) |
| `ImportConfigRequest` | at most 50 profiles; depth 5 and 10,000 bytes per profile and for `session_overrides`, checked by `_check_depth` |
| `ConfigPatchRequest` | `extra="forbid"` (rejects unknown keys) |

---

## Rate Limiting

Rate limiting is off by default (`rate_limiting_enabled = false`). Its settings flow from `core/config.py` to the middleware:

- The token bucket is set by the `rate_limiting_*` settings (`rate_limiting_strategy` `adaptive` or `fixed`, `rate_limiting_rpm` 60 per tenant, `rate_limiting_burst` 10, `rate_limiting_burst_multiplier`, `rate_limiting_overload_threshold`, `rate_limiting_idle_threshold`). Set them in the `[rag]` table of a settings file or as environment variables (`RAG_RATE_LIMITING_ENABLED=true`, or the `LASAGNA_` name).
- The adaptive strategy scales the bucket with a load estimate built from observed request latency.
- The timing middleware feeds latency measurements to the rate limiter via `record_request_latency()`.
- When rate-limited, the response includes a `Retry-After` header (integer seconds).

---

## Cross-References

- Session identity flow: [rest/DESIGN.md](../rest/DESIGN.md)
- Core configuration keys: `settings.lasagna.toml`
- Agentic RAG config: [agentic/docs/config.md](../../agentic/docs/config.md)
- SDK configuration: [sdk/docs/configuration.md](../../sdk/docs/configuration.md)
