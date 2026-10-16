# Deployment

Running Lasagna as a server (the Docker Compose stack, the Kubernetes manifests,
authentication, request limits, health checks, metrics and upgrade notes) is documented
in one place: the [deployment guide](../../deploy/README.md). The security model is in
[SECURITY.md](../../../../SECURITY.md). This page covers only using the SDK against a
server that is already running.

## Connecting from Python

`KnowledgeClient.from_url(url, api_key=None, session_id=None)` returns a client that sends
its calls to the server's REST API under `/api/v1` with `httpx`, so it loads no models
locally. For the compose stack on its default port:
`KnowledgeClient.from_url("http://127.0.0.1:8000", session_id="my-project")`.

- The session goes to the server in the `X-Session-ID` header. Without `session_id`,
  each `from_url` call makes up a different one of the form `remote_<8 hex digits>`, so
  pass one to find your documents again.
- `api_key` is sent as `Authorization: Bearer <api_key>`. The server does not check it
  against `RAG_API_KEYS`; that list only governs `POST /api/v1/sessions`.
- The client sends no `X-Session-Token`, so it works with a server that has
  `RAG_REQUIRE_AUTH` off (the compose default, published on 127.0.0.1). A server with
  `RAG_REQUIRE_AUTH=true`, as the Kubernetes ConfigMap sets, answers 401.
- Search settings are the server's: see [Configuration](configuration.md) for what
  `set_config()` does on a remote client.

## See also

- [Configuration](configuration.md)
- [Backends](backends.md)
