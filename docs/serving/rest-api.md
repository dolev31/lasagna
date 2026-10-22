# REST API

`lasagna serve` runs an HTTP server with 41 endpoints: 32 under `/api/v1/` and 9
that are not versioned. Every endpoint takes and returns JSON, except the two
upload routes, which take multipart form data.

## Start the server

```bash
lasagna serve
```

The server listens on `127.0.0.1:8000`. That address is reachable only from your
machine, and it needs no login. Useful options:

| Option | Meaning |
|---|---|
| `--port` | Port to listen on. |
| `--host` | Address to bind. A non-loopback address needs auth on. |
| `--configurator` / `--no-configurator` | Serve the settings UI at `/configurator`. |
| `--open` | Open the browser. |
| `--mcp` | Also serve the [MCP server](mcp.md) at `/mcp/sse`. |
| `--session`, `-s` | The session for requests that name none. |

A running server serves interactive docs at `/docs` and the schema at
`/openapi.json`. When this page and that schema disagree, the schema is right.
The `/docs` page loads its scripts from cdn.jsdelivr.net, so it needs a network
connection.

## Upload and search

```bash
lasagna upload report.pdf -s my-corpus
```

```bash
curl -X POST http://127.0.0.1:8000/api/v1/search \
  -H "X-Session-ID: my-corpus" \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the report conclude?", "top_k": 5}'
```

Uploads are indexed in the background by default. The upload response comes back
with `status` set to `processing`. Poll `GET /api/v1/documents/{document_id}/status`
until it reads `ready`.

A document id works as an idempotency key. Uploading again with an id the session
already owns replaces that document. Without an id, a file whose bytes the session
already holds is not uploaded again, and the response names the original in
`duplicate_of`.

The search body has these fields:

| Field | Default | Meaning |
|---|---|---|
| `query` | (required) | The question, 1 to 10,000 characters. |
| `top_k` | `5` | Results to return, 1 to 100. |
| `min_score` | none | Drop results scoring below this. |
| `document_ids` | none | Search only these documents. |
| `use_hyde` | `true` | Use HyDE query expansion. |
| `use_reranking` | `true` | Rerank with a cross-encoder. |
| `routing_tier` | none | Force `fast`, `standard` or `comprehensive`. |

A stage that failed and fell back is named in `degraded_stages`, so an empty list
means every configured stage ran.

## Endpoints at a glance

| Group | Routes |
|---|---|
| Health | `GET /api/v1/health/live`, `/health/ready`, `/health`, and `GET /api/v1/stats` |
| Sessions | `POST /api/v1/sessions` |
| Documents | `POST /api/v1/documents`, `POST /documents/batch`, `GET /documents`, and `GET`, `DELETE`, `/chunks`, `/status`, `/reindex`, `/archive`, `/restore` under `/documents/{document_id}` |
| Admin | Garbage collection, orphan repair, the lifecycle policy and session deletion under `/api/v1/admin/` |
| Search | `POST /api/v1/search`, `/search/citations`, `/search/multi-query`, `/search/agentic` |
| Citations | `GET /api/v1/citations` and `/citations/{citation_id}` |
| Config | `GET` and `PATCH /api/v1/config`, `GET /config/profiles`, `PUT /config/profiles/{profile_name}` |
| Pipelines | `GET /api/v1/pipelines` |
| Configurator | Eight routes under `/api/config/knowledge` |
| Reindex | `POST /api/knowledge/reindex` |

The [endpoint reference](https://github.com/dolev31/lasagna/blob/main/src/lasagna/api/docs/endpoints.md)
documents each route with its request, response and status codes.

## Sessions and identity

A session scopes documents and configuration. The server reads the session from
the first of these that is present:

1. `X-Session-Token` header (a signed token).
2. `X-Session-ID` header.
3. `session_id` query parameter.
4. The name `default`.

A session id is at most 128 characters. A token that is present but invalid or
expired is refused with 401. It does not fall through to the next source.

## Authentication

Out of the box, auth is off, and the server binds only to loopback. Turn auth on
before you expose the server to anyone else:

```bash
export RAG_REQUIRE_AUTH=true
export RAG_JWT_SECRET="$(openssl rand -hex 32)"
export RAG_API_KEYS="$(openssl rand -hex 32)"
lasagna serve --host 0.0.0.0
```

With `RAG_REQUIRE_AUTH=true`:

- Only a verified `X-Session-Token` names a session. A bare `X-Session-ID` is
  refused with 401, and so is the `default` fallback, unless
  `RAG_ALLOW_ANONYMOUS_SESSION=true`.
- `POST /api/v1/sessions` mints tokens and needs an `X-API-Key` header with a key
  from `RAG_API_KEYS`. Keys are at least 32 characters.
- Tokens are signed with `RAG_JWT_SECRET` and last 7 days by default
  (`RAG_SESSION_TOKEN_TTL_SECONDS`). An unset or placeholder secret is a startup
  error. Without auth, an unset secret gets a random one per process, so tokens
  stop working on restart.
- The health probes stay open.

Instance-wide admin routes need an `X-Admin-Token` header that matches
`RAG_ADMIN_TOKEN`. If that variable is unset, they answer 503.

With auth off, the server refuses to start on a non-loopback address unless
`RAG_BIND_IS_PROTECTED=true` says that something outside the process restricts
access. A request with a non-loopback `Host` header gets 421 unless the host is in
`RAG_ALLOWED_HOSTS`. Set `RAG_CORS_ORIGINS` to allow browser clients.

These security variables are read under their `RAG_` names only.

## Limits

| Limit | Default | Setting |
|---|---|---|
| One uploaded file | 50 MiB | `RAG_MAX_FILE_SIZE` |
| One batch upload | 200 MiB and 20 files | `RAG_MAX_BATCH_BYTES`, `RAG_MAX_BATCH_FILES` |
| Other request bodies | 1 MiB | fixed |
| One search | 120 seconds | `RAG_SEARCH_TIMEOUT` |
| Concurrent uploads | 5 | `RAG_MAX_CONCURRENT_UPLOADS` |
| Sessions per API key | 100 | `RAG_MAX_SESSIONS_PER_KEY` |
| Request rate | 60 per minute, burst 10, adaptive | `RAG_RATE_LIMITING_ENABLED` |

Rate limiting is off in the code and on in the Docker and Kubernetes
configurations. Each session also has quotas: 1,000 documents, 5 GiB, 100,000
chunks, 600 searches per minute and 200 uploads per hour.

## Status codes

| Code | Meaning |
|---|---|
| 400 | Invalid input: a bad file type, a session id over 128 characters, a malformed query. |
| 401 | Missing, invalid or expired credentials. |
| 403 | You named an identity without a valid admin token. |
| 404 | The document, citation or profile does not exist in this session. |
| 409 | A conflict: a restore after the grace window, or an embedding size that does not fit the collection. |
| 410 | The source file is gone, so the document cannot be reindexed. |
| 413 | The file, batch or body is over its limit. |
| 422 | The body failed validation, or a config key is unknown. |
| 429 | Rate limited or over quota. Rate limits include `Retry-After`. |
| 500 | An unhandled backend error. |
| 503 | The backend is starting, a dependency is down, the processing pool is full, or the embedding model is not downloaded. |
| 504 | The search ran past `RAG_SEARCH_TIMEOUT`. |

A server never downloads a model inside a request. If the session's embedding
model is not on disk, search answers 503 with the command that fixes it.
