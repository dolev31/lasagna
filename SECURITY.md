# Security policy

## Reporting a vulnerability

Please report security issues privately through
[GitHub Security Advisories](https://github.com/dolev31/lasagna/security/advisories/new)
rather than opening a public issue.

We aim to acknowledge a report within 3 working days and to agree a disclosure
timeline with you once the issue is confirmed.

Useful details: affected version, reproduction steps, and what an attacker could
achieve.

## Scope

Lasagna ingests untrusted documents, calls external model providers, and can
expose a REST API and MCP server. Reports we are particularly interested in:

- Document parsing that escapes its sandbox or achieves code execution
- Cross-tenant or cross-session leakage of documents, chunks or configuration
- Authentication or authorization bypass on the REST API or MCP server
- Injection via document content, filenames or search queries
- Secrets disclosed through logs, traces or error responses

Out of scope: vulnerabilities in third-party model weights or dependencies
(report those upstream), and denial of service from deliberately oversized
inputs on an instance you control.

## Operational notes

- **No default credentials.** Every credential is required configuration; the
  deployment manifests fail to start rather than falling back to a default. If
  you find a code path that accepts a built-in default, treat it as a bug and
  report it.
- **A session cannot reconfigure the server.** The settings a session may
  set for itself (REST config routes, the MCP config tools, the configurator,
  a config import) are retrieval and generation settings. The vector backend,
  its connection settings, the collection, paths, URLs, credentials,
  `*_trust_remote_code` and server limits are refused there, with the key
  named; they come from the environment and the settings file only.
- **The API is unauthenticated by default** and intended to run on a trusted
  network. `X-Session-ID` is accepted as identity with no credential, so
  anything that can reach the port can read or delete any session's documents.

  Set **`RAG_REQUIRE_AUTH=true`** to change that. Identity then has to come from
  a token issued by `POST /api/v1/sessions`, and a `session_id` merely asserted
  in a header or query parameter is refused with 401. `RAG_JWT_SECRET` must be
  set to the same value in every replica: it is generated per process
  otherwise, so a multi-replica deployment rejects roughly half of its own
  tokens. The app refuses to start rather than doing that.

  With auth on, **minting a token needs an API key**: send `X-API-Key` with one
  of the comma-separated keys in `RAG_API_KEYS` (each at least 32 characters;
  placeholders are ignored). With no usable key configured the endpoint returns
  503, so an exposed instance cannot be left open by omission. The key's
  principal (a digest of the key, never the key) is written into the token, and
  rate limits are counted per principal, so minting more sessions buys no more
  requests. Storage and document quotas are still counted per session.
  Removing a key from `RAG_API_KEYS` ends every session it minted: the
  principal is checked against the configured keys whenever a token is read,
  on REST and MCP. Tokens minted with `RAG_ADMIN_TOKEN` name no key (their
  principal is `admin`); changing `RAG_JWT_SECRET` ends those, and every other
  token. A token minted while auth was off names no principal at all, and is
  refused once auth is on with keys configured, so switching auth on ends
  those sessions without a new secret.

  `RAG_ADMIN_TOKEN` is a **narrower** thing and is not a password for the API.
  It guards the operations whose blast radius is the whole instance rather than
  one session: `POST /api/v1/admin/gc/run`, `GET /api/v1/admin/gc/orphans`,
  `POST /api/v1/admin/gc/repair`, `GET` and `PATCH /api/v1/admin/gc/policy`,
  `DELETE /api/v1/admin/sessions/{id}`, minting a token that names someone
  else, and the five MCP tools that mirror those. Leaving it unset does
  not open those routes; it makes them return 503.

- **Where the bind check does and does not run.** `lasagna serve` refuses to
  start on a non-loopback interface with `RAG_REQUIRE_AUTH` unset, because the
  bind address is an argument it can read. Serving `lasagna.api.rest.app:app`
  with your own gunicorn or uvicorn (which is what the bundled `Dockerfile`
  does) bypasses that check: inside a container the app always sees itself
  bound to `0.0.0.0` regardless of how the port is published, so it cannot tell
  an exposed deployment from a loopback one. The shipped k8s manifests therefore
  set `RAG_REQUIRE_AUTH=true` in the ConfigMap, and `docker-compose.yml`
  publishes the API on `127.0.0.1` by default.

- **Cross-origin requests are denied by default.** Name the origins you serve in
  `RAG_CORS_ORIGINS`. A wildcard combined with `RAG_REQUIRE_AUTH` is a startup
  error: this API authenticates with a header rather than a cookie, so a
  wildcard would let any page a user visits drive authenticated requests from
  their browser.

  **CORS is not a CSRF defense, and it is worth being precise about why.** It
  governs whether the browser hands the *response* back to the calling page; it
  does not stop the request executing. Requests that require a preflight are
  blocked (`DELETE /api/v1/documents/{id}` among them), but a
  "simple" request needs no preflight, and `POST /api/v1/documents`
  (multipart/form-data) and `POST /api/v1/documents/{id}/archive` (no body) both
  qualify. On an instance running **without** `RAG_REQUIRE_AUTH`, a page the
  operator visits can therefore still write to it blind.

  `RAG_REQUIRE_AUTH` is what closes this: identity then has to arrive in
  `X-Session-Token`, a custom header that forces a preflight, and `?session_id=`
  stops being accepted, including the anonymous `default` namespace. This is
  one more reason the unauthenticated default is only defensible on loopback.

- **Forwarding headers are trusted only from a proxy you name.** By default
  `lasagna serve` ignores `X-Forwarded-For`, so the client address used for rate
  limits is the connecting socket's. Behind a reverse proxy, set
  `RAG_FORWARDED_ALLOW_IPS` to the proxy's address (comma-separated for
  several). Trusting every local address instead let any client on the host
  choose its own address, and with it a fresh rate-limit bucket per request.
  When you run the app under your own gunicorn or uvicorn, their
  `--forwarded-allow-ips` setting governs this instead.
- **With auth off, the HTTP surfaces accept only loopback `Host` names.** The
  REST API, the embedded `/mcp` and `lasagna-mcp --transport sse` (on any bind
  address) answer a request naming any other host with 421. That blocks DNS
  rebinding, where a web page points its own host name at 127.0.0.1. Inside a
  container network, where clients use a service name, or behind a reverse
  proxy that forwards its own name, add it with `RAG_ALLOWED_HOSTS`
  (comma-separated `host` or `host:*` entries); compose sets `api:*,mcp:*`.
  The health and `/metrics` paths answer any `Host`, for probes and scrapes by
  pod address. Origins in `RAG_CORS_ORIGINS` stay allowed. With auth on, the
  `Host` is not checked: every request needs a token.
- **MCP over stdio is deliberately not authenticated.** The client launched the
  process and owns its pipes, so the process boundary is the credential. Network
  transports (`--transport sse`, or the `/mcp` mount inside the REST app) are
  held to the same rules as REST. On any network transport, with or without
  auth, `upload_document` reads server-side paths only inside
  `RAG_MCP_UPLOAD_ROOT`; unset, it reads none.
- **The MCP server serves no deletion tools by default.** The `safe` tool set
  leaves out the six tools that delete data or reset configuration; `--tools
  all` or `LASAGNA_MCP_TOOLS=all` serves them.
- **Request bodies are limited before they are read.** An upload over
  `RAG_MAX_FILE_SIZE`, a batch over `RAG_MAX_BATCH_BYTES` and any other body
  over 1 MiB is refused with 413 from its `Content-Length`, or as soon as a
  chunked body passes the limit. A batch of more than `RAG_MAX_BATCH_FILES`
  files is refused with 400. The defaults are in
  [the deployment guide](src/lasagna/deploy/README.md).
- **Model weights are third-party code** downloaded at runtime and executed in
  your process. Pin versions and use a private mirror for sensitive deployments.
- **Installed plugins execute on import.** Lasagna discovers components and
  benchmarks through the `lasagna_components` and `lasagna_benchmarks` entry
  point groups, and loading an entry point runs that package's module-level
  code. This is how Python packaging works rather than anything specific to
  Lasagna, but it is worth stating plainly because third-party plugins are
  invited: **anything installed in your environment can run when the registry
  loads.** Treat a Lasagna plugin with the same care as any other dependency,
  and prefer a locked environment for anything sensitive.

  `lasagna plugins list` shows exactly which third-party entry points are
  present and which failed to load. Discovery failures are deliberately
  non-fatal so one broken plugin cannot stop the registry; set
  `LASAGNA_STRICT_PLUGINS=1` to raise instead of continuing.
