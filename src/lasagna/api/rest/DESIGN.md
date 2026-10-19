# REST API: design note

How the FastAPI app in this directory resolves identity, which middleware runs on a request,
and what `RAG_REQUIRE_AUTH` changes. The routes themselves are listed in
[../README.md](../README.md); the security policy is in [SECURITY.md](../../../../SECURITY.md).

## Request path

```
Client
  |  POST /api/v1/sessions          (handshake: get a token)
  |  X-Session-Token: eyJ...        (every later request)
  v
create_app() in app.py
  |-- CORS (RAG_CORS_ORIGINS, else the cors_origins argument, else no cross-origin access)
  |-- middleware, outermost first:
  |     request-id -> body size limit -> rate limit -> timing -> route
  |-- dependencies.py: get_backend, get_session_id, get_user_id, get_profile_owner
  |-- routes/ under /api/v1: health, sessions, documents, search, citations,
  |     config, pipelines, admin (gc_admin.py)
  |-- configurator routes outside /api/v1: /api/config/knowledge/*, /api/knowledge/reindex
  v
RAGBackend (lasagna.api.rag_backend), KnowledgeConfigManager, rate limiter
```

## Identity

`POST /api/v1/sessions` returns a JWT signed with HS256 that carries `(session_id, user_id)`.
The server verifies it on each request with no database lookup.

- Omitted ids are generated as `sess_<12 hex>` and `user_<12 hex>`; with auth on, `user_id` is the API key's digest (`key_<12 hex>`), so one key's sessions share one set of saved profiles. Naming a `session_id` or
  `user_id` in the body requires `X-Admin-Token` matching `RAG_ADMIN_TOKEN`, else 403: sessions
  are the isolation boundary, so choosing one is impersonation.
- Claims: `sub` (user), `sid` (session), `typ: "session"`, `iat`, `exp`, `iss: "lasagna"`, and
  `prn` (the principal that minted it) when auth is on. Decoding requires `sub`, `sid`, `exp`
  and `typ`, and accepts the issuers in `ACCEPTED_ISSUERS` (`lasagna`, and `rag-knowledge` for
  tokens minted before 0.1.0).
- Lifetime: `RAG_SESSION_TOKEN_TTL_SECONDS` (default 604800, seven days). Tokens cannot be
  revoked short of rotating `RAG_JWT_SECRET`.
- With `RAG_JWT_SECRET` unset, `session_token._get_secret()` generates a per-process secret and
  logs a warning. That is refused (`MissingSigningSecret`) when `RAG_REQUIRE_AUTH` is on, and a
  placeholder or a secret shorter than 32 characters is refused too: with several replicas or
  workers, a per-process secret rejects tokens minted by the others.

Resolution order (`dependencies.py`, `session_token.extract_session_id`):

| `get_session_id()` | `get_user_id()` |
| ------------------ | --------------- |
| 1. `X-Session-Token` -> `sid` | 1. `Authorization: Bearer` JWT `sub` (verified with `RAG_JWT_SECRET`) |
| 2. `X-Session-ID` header | 2. `X-Session-Token` -> `sub` |
| 3. `session_id` query parameter | 3. `X-User-ID`, only with `RAG_ALLOW_HEADER_IDENTITY=true` (else 401) |
| 4. `"default"` | 4. a fresh `anon_<16 hex>` per request |

A presented `X-Session-Token` that does not verify is a 401, never a fall-through to the
header path: an expired token silently landing in `default` would write documents to the
wrong namespace. Session ids longer than 128 characters are a 400.

`get_profile_owner()` decides who owns saved profiles: the verified user with auth on, and the
shared `__anonymous__` owner with auth off (a per-request `anon_` id could never edit its own
profile on the next request).

## RAG_REQUIRE_AUTH

`auth_mode.py` holds the switch. It is read at call time, not import time.

- Off (default): `session_id` is a namespace, not a credential. `lasagna serve` (`cli.py`) and
  `lasagna-mcp` call `require_auth_or_refuse_to_start()`, which raises `UnsafeBindError` for a
  non-loopback bind unless `RAG_BIND_IS_PROTECTED=true` asserts that something outside the
  process (a loopback port mapping, a NetworkPolicy, a firewall) limits access. In a container
  the bind address is always `0.0.0.0`, so the process cannot check this itself.
- `create_app()` itself only checks the signing secret (`_require_a_shared_signing_secret`) and
  the limits (`limits.validate()`), because the shipped Dockerfile runs gunicorn on the app
  directly and never passes through `lasagna serve`.
- On: minting a session needs `X-API-Key` from `RAG_API_KEYS` (or `X-Admin-Token`), and the key's
  principal goes into the token. `reject_unverified_identity()` refuses any session id that did
  not come from a verified token, including `default`, unless `RAG_ALLOW_ANONYMOUS_SESSION` is
  set.
- CORS does not replace this. It decides whether a page may read a response, and a "simple"
  cross-origin request (a multipart upload, a bodyless POST) still runs. With auth on, identity
  travels in a custom header, which forces a preflight that the CORS policy denies.

## Middleware

Registered in `create_app()`:

1. **Request id**: keeps a client `X-Request-ID` that looks like an id, else generates one; it is
   bound to every log line of the request and echoed in the response.
2. **Body size limit** (`limits.BodySizeLimitMiddleware`): answers 413 from `Content-Length`
   before the body is read, and counts chunked bodies as they arrive. Starlette would otherwise
   spool a whole multipart upload to disk before the route runs.
3. **Rate limit**: `core.rate_limiter` token buckets, off unless the limiter is enabled. With
   auth off the bucket is the session id; with auth on it is the token's principal (`prn:`),
   the verified session (`sid:`), or the client address (`ip:`) for unverified requests, so
   minting sessions does not buy requests. 429 carries `Retry-After`, rounded up.
4. **Timing**: `X-Response-Time`, a latency histogram labeled by route template and status
   (not raw path, which would create one series per scanned URL), and the limiter's load input.

Exempt from rate limit and timing: `/`, `/metrics`, and the prefixes in `_EXEMPT_PREFIXES`
(`/health`, `/api/v1/health`, `/docs`, `/openapi.json`, `/redoc`). `/api/v1/sessions`, `/mcp`
and `/configurator` are deliberately not exempt: they mint identities or carry destructive
operations.

## Optional mounts

- `RAG_EMBED_MCP=true` mounts the MCP SSE app at `/mcp` in the same process (needed where only
  one process may open Milvus Lite). It sets the MCP transport to `http`, and with auth off adds
  the loopback Host/Origin guard against DNS rebinding.
- `RAG_CONFIGURATOR=true` mounts the configurator UI at `/configurator`
  (`lasagna.configurator.mount_configurator`).

## Configurator routes

`routes/knowledge_config.py` (`/api/config/knowledge`, eight routes: get, set active, save,
delete, rename, reset, export, import) and `routes/reindex.py` (`POST /api/knowledge/reindex`)
are mounted on the app, outside `/api/v1`, because the configurator frontend uses those paths.
They resolve identity with the same dependencies; profile writes use `get_profile_owner`.
Reindex takes the session from the credential, not the body (the body's `session_id` field is
deprecated and ignored), and runs the delete and re-upload of each document in
`asyncio.to_thread()`.

## MCP

`mcp/server.py` reuses `session_token.decode_session_token`. Its `_resolve_session()` takes a
valid `session_token`, raises `MCPAuthError` for an invalid one, and otherwise uses the given
`session_id` (or `RAG_SESSION_ID`) with a deterministic `anon_<hash>` user id, so profile
ownership stays stable across calls. `session_token.py` has no FastAPI import for this reason.
