# Security

Report vulnerabilities privately through
[GitHub Security Advisories](https://github.com/dolev31/lasagna/security/advisories/new).
Do not open a public issue. The project aims to acknowledge a report within three
working days.

## What to know before you serve

- **The REST API is unauthenticated by default.** It is meant for a trusted
  network. In that mode, `X-Session-ID` is accepted as identity with no credential,
  so anything that can reach the port can read or delete any session's documents.
  Turn on `RAG_REQUIRE_AUTH` before you publish it. See
  [REST API](../serving/rest-api.md#authentication).
- **No default credentials.** Every credential is required configuration.
- **A session cannot reconfigure the server.** The settings a session may set for
  itself are retrieval and generation settings. The vector backend, its connection
  settings, paths, URLs, credentials and server limits come from the environment
  and the settings file only.
- **The MCP server has two tool sets.** The safe set leaves out the tools that
  delete or reconfigure. See [MCP server](../serving/mcp.md#choose-a-tool-set).

Read the full
[security policy](https://github.com/dolev31/lasagna/blob/main/SECURITY.md).
