# API Layer Configuration Reference

All configuration that affects the `api/` package — environment variables, middleware settings, session tokens, and validation constraints. Every default is verified against source code.

---

## Environment Variables

| Variable | Default | Source | Description |
|----------|---------|--------|-------------|
| `RAG_JWT_SECRET` | ephemeral random | `session_token.py:38` | HMAC-SHA256 signing key for session tokens. If unset, a random 32-byte hex secret is generated per-process (tokens lost on restart). |
| `RAG_SESSION_TOKEN_TTL_SECONDS` | `604800` (7 days) | `session_token.py:29` | Session token expiry in seconds. |
| `RAG_CORS_ORIGINS` | `""` → `["*"]` | `app.py:232` | Comma-separated allowed CORS origins. Empty string falls back to wildcard (logs a warning). |
| `RAG_EMBED_MCP` | `false` | `app.py:298` | When `true`, mount the MCP SSE server at `/mcp` inside the FastAPI process. Required for ARM64/Milvus Lite single-process deployments. |
| `RAG_CONFIGURATOR` | `false` | `app.py:313` | When `true`, mount the configurator UI at `/configurator` and its API routes. |
| `RAG_ALLOW_HEADER_IDENTITY` | `false` | `dependencies.py:21` | When `true`, accept `X-User-ID` header for user identity. When `false`, the header is rejected with 401. |
| `PROMETHEUS_MULTIPROC_DIR` | unset | `app.py:276` | Directory for Prometheus multi-process metric aggregation (gunicorn deployments). |

---

## Session Token Configuration

| Setting | Value | Source |
|---------|-------|--------|
| Algorithm | HS256 | `session_token.py:72` |
| Default TTL | 604800s (7 days) | `session_token.py:29` |
| Max session_id length | 128 chars | `session_token.py:103` |
| Required JWT claims | `sub`, `sid`, `exp`, `typ` | `session_token.py:89` |
| Token type claim | `typ: "session"` | `session_token.py:70` |
| Issuer claim | `iss: "rag-knowledge"` | `session_token.py:69` |

JWT payload structure:

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

---

## Middleware Configuration

### Exempt Prefixes

Paths exempt from rate limiting and timing metrics (`app.py:38`):

```
/health, /api/v1/health, /docs, /openapi.json, /redoc, /mcp, /configurator
```

Note: `/api/v1/sessions` is intentionally NOT exempt — session creation must be rate-limited.

### Middleware Execution Order

Outermost-first (each wraps the next):

1. **Request-ID** — attaches `X-Request-ID` (forwarded or UUID)
2. **Rate-Limit** — per-tenant adaptive token bucket, returns 429 + `Retry-After`
3. **Timing** — records `X-Response-Time` header, feeds Prometheus + rate limiter
4. **Handler** — FastAPI route execution

---

## Validation Constraints

### Session & Identity (`_validation.py`, `dependencies.py`)

| Constraint | Value | Source |
|------------|-------|--------|
| Session ID max length | 128 chars | `session_token.py:103` |
| Session ID pattern | `[a-zA-Z0-9_\-\.]+` | `_validation.py` |
| User ID max length | 128 chars | `dependencies.py:93,118` |
| Document ID pattern | `doc_[0-9a-f]{16}` | `_validation.py` |

### Upload Limits (`models.py`, `documents.py`)

| Constraint | Value | Source |
|------------|-------|--------|
| Max file size | `rag_config.max_file_size` | `documents.py:71` (streamed check) |
| Chunk pagination limit | 1–500 | `documents.py:157` |
| Chunk pagination offset | >= 0 | `documents.py:156` |

### Configurator Models (`models.py`)

| Model | Constraint | Source |
|-------|-----------|--------|
| `ResetConfigRequest.scope` | regex `^(active\|all)$` | `models.py:278` |
| `ReindexRequest.session_id` | max 256 chars (deprecated) | `models.py:282` |
| `ImportConfigRequest` | depth + size validation via `_check_depth` | `models.py:305` |
| `ConfigPatchRequest` | `extra="forbid"` (rejects unknown keys) | `models.py` |

---

## Rate Limiting

Rate limiting configuration flows through from `core/config.py` to the middleware:

- Token bucket parameters are configured in `settings.lasagna.toml` under `[rate_limiting]`
- The rate limiter adapts based on observed request latency (load-based scaling)
- The timing middleware feeds latency measurements to the rate limiter via `record_request_latency()`
- When rate-limited, the response includes `Retry-After` header (integer seconds)

---

## Cross-References

- Session identity flow: [rest/DESIGN.md](../rest/DESIGN.md)
- Core configuration keys: `settings.lasagna.toml`
- Agentic RAG config: [agentic/docs/config.md](../../agentic/docs/config.md)
- SDK configuration: [sdk/docs/configuration.md](../../sdk/docs/configuration.md)
