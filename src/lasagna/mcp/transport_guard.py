"""What the MCP surface is allowed to do, decided by how it is reached.

The 33 tools in ``server.py`` had no authentication of any kind, and five of
them are operations the REST API refuses without ``X-Admin-Token``:

===========================  ==========================================
MCP tool                     REST equivalent, and its gate
===========================  ==========================================
``bulk_delete_session``      ``DELETE /api/v1/sessions/{id}`` — admin
``get_lifecycle_policy``     ``GET /gc/policy`` — admin
``set_lifecycle_policy``     ``PATCH /gc/policy`` — admin
``get_gc_status``            the server-wide GC audit log — admin
``rebuild_collection``       no REST equivalent; see below
===========================  ==========================================

`api/rest/app.py` mounts this server at ``/mcp`` in the same process that
serves those routes, so the admin gate on the REST side was decorative: the
identical operation was one unauthenticated MCP call away.

``rebuild_collection`` is the worst of the five and has no REST equivalent at
all. It reads ``config.collection_name``, which resolves to the single global
``rag_documents_v2`` — sessions are a metadata filter *within* that collection,
not separate collections — and calls ``recreate_collection_for_config``. One
call from any session drops every session's documents. It has no ``confirm``
flag, where the strictly less destructive ``bulk_delete_session`` does.

**The rule.** Enforcement turns on when the surface is both networked and
running with ``RAG_REQUIRE_AUTH``:

* **stdio** — the client launched this process and talks to it over its own
  pipes. The process boundary is the credential; there is no one else to
  authenticate, and requiring a token would only break every local MCP client
  config in existence.
* **networked, auth off** — unchanged. `auth_mode.require_auth_or_refuse_to_start`
  makes this combination a startup error on any non-loopback bind, so "auth
  off" means "loopback", and loopback is the case stdio's reasoning covers.
* **networked, auth on** — the operator has declared this instance exposed.
  ``session_id`` must come from a verified token, exactly as on REST, and the
  five tools above additionally require the admin token REST demands.

That leaves one gap worth stating rather than hiding: an embedder who calls
`create_app()` and serves it on a public interface with their own uvicorn never
passes through the CLI's startup refusal. The library cannot see the bind
address from inside the app, so this is documented in SECURITY.md rather than
enforced.
"""

from __future__ import annotations

import os
import secrets
from typing import Optional

from lasagna.api.rest.auth_mode import REQUIRE_AUTH_ENV, reject_unverified_identity, require_auth

#: Tools whose blast radius is the whole instance rather than one session, each
#: matching an operation the REST API gates behind ``X-Admin-Token``.
ADMIN_SCOPE_TOOLS = frozenset(
    {
        "bulk_delete_session",
        "get_lifecycle_policy",
        "set_lifecycle_policy",
        "get_gc_status",
        "rebuild_collection",
    }
)

#: Default is the strict direction. ``main()`` sets this before serving, but the
#: module is also imported by `api/rest/app.py`'s ``/mcp`` mount, where nothing
#: calls ``main()`` — so an unset transport must mean "networked", never stdio.
_transport = "http"


def set_transport(name: str) -> None:
    """Record how this server is being served. Called once, before serving."""
    global _transport
    _transport = (name or "").strip().lower() or "http"


def get_transport() -> str:
    return _transport


def is_networked() -> bool:
    """Whether something other than the launching process can reach these tools."""
    return _transport != "stdio"


def enforcement_active() -> bool:
    """Whether to demand credentials.

    Both conditions matter. Without ``is_networked`` every local MCP client
    config breaks for no gain; without ``require_auth`` a loopback development
    server starts refusing its own operator.
    """
    return is_networked() and require_auth()


class MCPAuthError(ValueError):
    """Raised instead of returning an error string.

    The tools in this module return JSON strings for ordinary failures, which a
    model is free to summarise, retry past, or ignore. A refusal must not be
    something the caller can talk its way around, so it propagates as an
    exception and FastMCP reports it as a tool error.

    Subclasses ``ValueError`` because ``_resolve_session`` already raises that
    for a missing session, so callers that catch it keep working.
    """


#: Directory a networked, authenticated MCP surface may read server-side files
#: from. Unset means it may read none.
UPLOAD_ROOT_ENV = "RAG_MCP_UPLOAD_ROOT"


def guard_server_path(file_path: str) -> str:
    """Resolve a server-side path a caller asked the server to ingest.

    ``upload_document`` takes a path on the server's own disk. Over stdio that
    is the client's own machine. On a networked surface with auth on, any
    caller can mint a session token, so a bare path let an anonymous caller
    ingest and read back any file the process could open (credential JSON,
    config files). There, the path must resolve inside ``RAG_MCP_UPLOAD_ROOT``.
    """
    resolved = os.path.realpath(file_path)
    if not enforcement_active():
        return resolved
    root = os.environ.get(UPLOAD_ROOT_ENV, "").strip()
    if not root:
        raise MCPAuthError(
            "upload_document cannot read server-side paths on a networked, "
            f"authenticated MCP surface unless {UPLOAD_ROOT_ENV} names a directory "
            "they must be inside. Upload through the REST API instead, or set it."
        )
    root_real = os.path.realpath(root)
    if os.path.commonpath([resolved, root_real]) != root_real:
        raise MCPAuthError(f"upload_document: '{file_path}' is outside {UPLOAD_ROOT_ENV}.")
    return resolved


def guard_identity(
    resolved: str,
    *,
    from_token: bool,
    tool: str = "this tool",
    admin_verified: bool = False,
) -> None:
    """Refuse a session id the caller merely asserted.

    Delegates the decision to `auth_mode.reject_unverified_identity` so the MCP
    and REST surfaces cannot drift apart on what counts as identity — which is
    how the two ended up disagreeing in the first place.
    """
    if not enforcement_active():
        return
    # An operator who presented the admin token has proven instance-wide
    # authority, which subsumes "who am I". Requiring a session token on top
    # would make MCP stricter than REST, where the admin routes take no session
    # at all -- and would leave an operator holding the right credential unable
    # to run the tool it authorises.
    if admin_verified:
        return
    reason = reject_unverified_identity(resolved, from_token=from_token)
    if reason:
        raise MCPAuthError(
            f"{tool}: {reason} Pass it as the `session_token` argument. "
            f"(This server is reachable over {_transport!r} with "
            f"{REQUIRE_AUTH_ENV} set.)"
        )


def guard_admin(admin_token: Optional[str], *, tool: str) -> None:
    """Require the same credential REST requires for the same operation.

    An unset ``RAG_ADMIN_TOKEN`` refuses rather than admits, matching the 503
    the REST admin routes return in that state. Opening up when the operator
    forgot to configure a secret is the wrong direction for an operation that
    can delete every session.
    """
    if not enforcement_active():
        return

    from lasagna.api.rest.auth_mode import configured_secret

    expected = configured_secret("RAG_ADMIN_TOKEN")
    if not expected:
        raise MCPAuthError(
            f"{tool} affects the whole instance, not one session, and "
            f"RAG_ADMIN_TOKEN is not configured on this server (or is a "
            f"placeholder or shorter than 32 characters). The REST "
            f"equivalent returns 503 in this state. Set RAG_ADMIN_TOKEN and "
            f"pass it as the `admin_token` argument."
        )
    if not admin_token or not secrets.compare_digest(admin_token, expected):
        raise MCPAuthError(
            f"{tool} requires the admin token. It affects the whole instance "
            f"rather than a single session, so it takes the same credential as "
            f"the REST route that performs it."
        )
