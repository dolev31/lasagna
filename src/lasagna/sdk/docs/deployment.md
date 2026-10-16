# Deployment

Running Lasagna as a server (the Docker Compose stack, the Kubernetes manifests,
authentication, request limits, health checks, metrics and upgrade notes) is documented
in one place: the [deployment guide](../../deploy/README.md). The security model is in
[SECURITY.md](../../../../SECURITY.md). This page covers only using the SDK against a
server that is already running.

## Connecting from Python

`KnowledgeClient.from_url(url, api_key=None, session_id=None, session_token=None)` returns a client that sends
its calls to the server's REST API under `/api/v1` with `httpx`, so it loads no models
locally. For the compose stack on its default port:
`KnowledgeClient.from_url("http://127.0.0.1:8000", session_id="my-project")`.

- The session goes to the server in the `X-Session-ID` header. Without `session_id`,
  each `from_url` call makes up a different one of the form `remote_<8 hex digits>`, so
  pass one to find your documents again.
- With `RAG_REQUIRE_AUTH=true` (the Kubernetes ConfigMap's setting) the server takes no
  `X-Session-ID` as identity. Pass `api_key`, one of the server's `RAG_API_KEYS`: on the
  first 401 the client mints a session token with it (`POST /api/v1/sessions` with
  `X-API-Key`), sends it as `X-Session-Token` from then on, and `client.session_id`
  becomes the token's session. Or pass `session_token`, a token minted already. Only an
  admin may name the session, so with auth on leave `session_id` out; naming one is
  refused with 403.
- With `RAG_REQUIRE_AUTH` off (the compose default, published on 127.0.0.1) the client
  mints nothing and `session_id` is used as given. `api_key` is also sent as
  `Authorization: Bearer <api_key>`, which the server does not check.
- Search settings are the server's: see [Configuration](configuration.md) for what
  `set_config()` does on a remote client.

## See also

- [Configuration](configuration.md)
- [Backends](backends.md)
