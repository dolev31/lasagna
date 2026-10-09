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
- Authentication or authorisation bypass on the REST API or MCP server
- Injection via document content, filenames or search queries
- Secrets disclosed through logs, traces or error responses

Out of scope: vulnerabilities in third-party model weights or dependencies —
report those upstream — and denial of service from deliberately oversized
inputs on an instance you control.

## Operational notes

- **No default credentials.** Every credential is required configuration; the
  deployment manifests fail to start rather than falling back to a default. If
  you find a code path that accepts a built-in default, treat it as a bug and
  report it.
- **The API is unauthenticated by default** and intended to run on a trusted
  network. `X-Session-ID` is accepted as identity with no credential, so
  anything that can reach the port can read or delete any session's documents.

  Set **`RAG_REQUIRE_AUTH=true`** to change that. Identity then has to come from
  a token issued by `POST /api/v1/sessions`, and a `session_id` merely asserted
  in a header or query parameter is refused with 401. `RAG_JWT_SECRET` must be
  set to the same value in every replica — it is generated per process
  otherwise, so a multi-replica deployment rejects roughly half of its own
  tokens. The app refuses to start rather than doing that.

  `RAG_ADMIN_TOKEN` is a **narrower** thing and is not a password for the API.
  It guards the operations whose blast radius is the whole instance rather than
  one session: `POST /gc/run`, `GET /gc/orphans`, `POST /gc/repair`, `GET` and
  `PATCH /gc/policy`, `DELETE /api/v1/sessions/{id}`, minting a token that names
  someone else, and the five MCP tools that mirror those. Leaving it unset does
  not open those routes; it makes them return 503.

- **Where the bind check does and does not run.** `lasagna serve` refuses to
  start on a non-loopback interface with `RAG_REQUIRE_AUTH` unset, because the
  bind address is an argument it can read. Serving `lasagna.api.rest.app:app`
  with your own gunicorn or uvicorn — which is what the bundled `Dockerfile`
  does — bypasses that check: inside a container the app always sees itself
  bound to `0.0.0.0` regardless of how the port is published, so it cannot tell
  an exposed deployment from a loopback one. The shipped k8s manifests therefore
  set `RAG_REQUIRE_AUTH=true` in the ConfigMap, and `docker-compose.yml`
  publishes the API on `127.0.0.1` by default.

- **Cross-origin requests are denied by default.** Name the origins you serve in
  `RAG_CORS_ORIGINS`. A wildcard combined with `RAG_REQUIRE_AUTH` is a startup
  error: this API authenticates with a header rather than a cookie, so a
  wildcard would let any page a user visits drive authenticated requests from
  their browser.

  **CORS is not a CSRF defence, and it is worth being precise about why.** It
  governs whether the browser hands the *response* back to the calling page; it
  does not stop the request executing. Requests that require a preflight are
  genuinely blocked — `DELETE /api/v1/documents/{id}` among them — but a
  "simple" request needs no preflight, and `POST /api/v1/documents`
  (multipart/form-data) and `POST /api/v1/documents/{id}/archive` (no body) both
  qualify. On an instance running **without** `RAG_REQUIRE_AUTH`, a page the
  operator visits can therefore still write to it blind.

  `RAG_REQUIRE_AUTH` is what closes this: identity then has to arrive in
  `X-Session-Token`, a custom header that forces a preflight, and `?session_id=`
  stops being accepted — including the anonymous `default` namespace. This is
  one more reason the unauthenticated default is only defensible on loopback.

- **MCP over stdio is deliberately not authenticated.** The client launched the
  process and owns its pipes, so the process boundary is the credential. Network
  transports (`--transport sse`, or the `/mcp` mount inside the REST app) are
  held to the same rules as REST.
- **Model weights are third-party code** downloaded at runtime and executed in
  your process. Pin versions and use a private mirror for sensitive deployments.
- **Installed plugins execute on import.** Lasagna discovers components and
  benchmarks through the `lasagna_components` and `lasagna_benchmarks` entry
  point groups, and loading an entry point runs that package's module-level
  code. This is how Python packaging works rather than anything specific to
  Lasagna, but it is worth stating plainly now that third-party plugins are
  invited: **anything installed in your environment can run when the registry
  loads.** Treat a Lasagna plugin with the same care as any other dependency,
  and prefer a locked environment for anything sensitive.

  `lasagna plugins list` shows exactly which third-party entry points are
  present and which failed to load. Discovery failures are deliberately
  non-fatal so one broken plugin cannot stop the registry — set
  `LASAGNA_STRICT_PLUGINS=1` to raise instead of continuing.
