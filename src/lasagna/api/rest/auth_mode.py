"""Whether `session_id` is a namespace or an authorization claim.

Until this existed it was a namespace. `extract_session_id` accepted an
`X-Session-ID` header or a `?session_id=` query parameter as the caller's
identity, with no credential of any kind, so:

    DELETE /api/v1/documents/{id}  -H "X-Session-ID: victim-tenant"

succeeded against any deployed instance. The signed-token machinery in
`session_token.py` was well built and simply bypassed — which is why the 403 in
`routes/sessions.py`, which stops an anonymous caller *minting* a token naming
someone else, guarded a door nobody needed to use.

`SECURITY.md` lists "authentication or authorisation bypass" and "cross-tenant
leakage" as in-scope vulnerabilities, so the stated threat model and the code
disagreed. `tests/stress/test_tenant_isolation.py` encodes the intended model
and tests the wrong half of it: it proves session B's *search* cannot surface
session A's documents, and its own client authenticates by setting the header,
so it never tests session B *claiming to be* session A.

**Default is off.** Local development, the README quickstart and every existing
script keep working unchanged. What closes the hole in practice is not the
default but `require_auth_or_refuse_to_start`: binding to a non-loopback
interface with auth off is a startup error, not a warning. That is the pattern
`mcp/server.py` already uses for `--enable-docs-api`, and its comment there
explains the reasoning — a warning in a startup log is not read by the person
who is affected by it.
"""

from __future__ import annotations

import ipaddress
import os
from typing import Optional

from loguru import logger

from lasagna.core.config import TRUTHY as _TRUTHY

#: Environment variable turning identity enforcement on.
REQUIRE_AUTH_ENV = "RAG_REQUIRE_AUTH"


def require_auth() -> bool:
    """Whether a verified credential is required to name a session.

    Read at call time rather than import time so tests and embedding
    applications can change it without reimporting the module.
    """
    return os.environ.get(REQUIRE_AUTH_ENV, "").strip().lower() in _TRUTHY


def is_loopback(host: str) -> bool:
    """Whether binding to ``host`` keeps the service off the network.

    Unrecognised values are treated as NOT loopback. A hostname this cannot
    resolve is more likely to be a public interface than the local one, and the
    safe direction for an unknown value is to demand authentication.
    """
    if not host:
        return False
    candidate = host.strip().strip("[]")
    if candidate in ("localhost", "::1"):
        return True
    try:
        return ipaddress.ip_address(candidate).is_loopback
    except ValueError:
        return False


class UnsafeBindError(RuntimeError):
    """Raised when a server would be exposed without authentication."""


#: Asserts that something outside this process restricts who can reach the
#: bind address — a Docker port published on loopback, a k8s NetworkPolicy, a
#: firewall. See `require_auth_or_refuse_to_start`.
BIND_IS_PROTECTED_ENV = "RAG_BIND_IS_PROTECTED"


def bind_is_externally_protected() -> bool:
    return os.environ.get(BIND_IS_PROTECTED_ENV, "").strip().lower() in _TRUTHY


def require_auth_or_refuse_to_start(host: str, *, surface: str = "API") -> None:
    """Refuse to start an unauthenticated server on a public interface.

    Deliberately an exception rather than a warning. Every route on this
    service is reachable without a credential when auth is off, including
    `DELETE /api/v1/documents/{id}`, and the person who would be harmed by that
    is not the person reading the startup log.

    **Containers are the exception, and getting this wrong broke the shipped
    stack.** A process in a container always sees itself bound to ``0.0.0.0`` on
    the container network no matter how the port is published, so the bind
    address carries no information about exposure — ``docker-compose.yml``
    publishes the MCP service on ``127.0.0.1:8005`` and still passes
    ``--host=0.0.0.0``, which this check refused outright. The container cannot
    determine the answer, so the operator asserts it with
    ``RAG_BIND_IS_PROTECTED``. That it must be set explicitly is the point: it
    is a claim someone makes, not a default that quietly holds.

    Raises:
        UnsafeBindError: binding beyond loopback with ``RAG_REQUIRE_AUTH`` unset
            and no external protection asserted.
    """
    if is_loopback(host) or require_auth() or bind_is_externally_protected():
        return

    raise UnsafeBindError(
        f"Refusing to start: the {surface} would bind {host!r}, which is "
        f"reachable from the network, and {REQUIRE_AUTH_ENV} is not set.\n"
        f"\n"
        f"Without it, `X-Session-ID` is accepted as identity with no "
        f"credential, so any caller can read and delete any session's "
        f"documents.\n"
        f"\n"
        f"Either:\n"
        f"  * bind loopback and put a reverse proxy in front (the documented "
        f"deployment),\n"
        f"  * set {REQUIRE_AUTH_ENV}=true and RAG_JWT_SECRET, then issue "
        f"session tokens via POST /api/v1/sessions, or\n"
        f"  * set {BIND_IS_PROTECTED_ENV}=true if something outside this "
        f"process already restricts who can reach {host!r} -- a Docker port "
        f"published on loopback, a NetworkPolicy, a firewall. In a container "
        f"the bind address is always 0.0.0.0 regardless of how the port is "
        f"published, so this process cannot check that for you.\n"
        f"\n"
        f"See SECURITY.md."
    )


#: Allow the anonymous ``default`` namespace to keep working with auth on.
#: Off by default: see `reject_unverified_identity`.
ALLOW_ANONYMOUS_ENV = "RAG_ALLOW_ANONYMOUS_SESSION"


def allow_anonymous_session() -> bool:
    return os.environ.get(ALLOW_ANONYMOUS_ENV, "").strip().lower() in _TRUTHY


def reject_unverified_identity(
    resolved: str,
    *,
    from_token: bool,
    what: str = "session_id",
) -> Optional[str]:
    """The reason this identity may not be used, or ``None`` if it may.

    Returns a reason rather than raising so the REST and MCP surfaces can shape
    their own errors — they have different error contracts and neither should
    import the other's.

    **The ``default`` session is not exempt.** It was, on the stated grounds
    that "refusing it would break health checks" — which was simply false:
    `routes/health.py` depends only on ``get_backend`` and never resolves a
    session at all, and every health path is in ``_EXEMPT_PREFIXES`` besides.

    The consequence of that mistake was that ``RAG_REQUIRE_AUTH`` did far less
    than SECURITY.md and the k8s ConfigMap claim for it. A caller sending no
    credential at all still resolved to ``default`` and could ingest, read and
    delete documents there. It stopped a caller *impersonating a named tenant*,
    which is worth having, but not unauthenticated write access to an exposed
    instance — and that is what the switch is advertised as preventing.

    Set ``RAG_ALLOW_ANONYMOUS_SESSION`` to restore the old behaviour where a
    genuinely public read-only corpus is the intent. It has to be asked for.
    """
    if not require_auth() or from_token:
        return None
    if resolved == "default" and allow_anonymous_session():
        return None
    if resolved == "default":
        return (
            f"{REQUIRE_AUTH_ENV} is on, so every request needs a session token "
            f"issued by POST /api/v1/sessions, including requests that send no "
            f"{what} at all, which would otherwise land in the shared "
            f"'default' namespace. Set {ALLOW_ANONYMOUS_ENV} if that namespace "
            f"is meant to be publicly writable."
        )
    return (
        f"{what} {resolved!r} was supplied without a verified session token. "
        f"{REQUIRE_AUTH_ENV} is on, so identity must come from a token issued "
        f"by POST /api/v1/sessions."
    )


#: Extra ``Host`` values an auth-off HTTP surface accepts, comma-separated:
#: ``name`` for that name without a port, ``name:*`` for any port.
ALLOWED_HOSTS_ENV = "RAG_ALLOWED_HOSTS"
_LOOPBACK_HOST_NAMES = ("127.0.0.1", "localhost", "[::1]")


def allowed_hosts() -> list:
    """The ``Host`` patterns an auth-off server answers to.

    The loopback names, with or without a port, then ``RAG_ALLOWED_HOSTS``.
    The same list feeds FastMCP's check on the MCP surfaces
    (`transport_guard.loopback_browser_guard`) and `host_is_allowed` on REST,
    so one setting governs both.
    """
    extra = [v.strip() for v in os.environ.get(ALLOWED_HOSTS_ENV, "").split(",") if v.strip()]
    return [h for name in _LOOPBACK_HOST_NAMES for h in (name, f"{name}:*")] + extra


def host_is_allowed(host: Optional[str], patterns: Optional[list] = None) -> bool:
    """Whether a request's ``Host`` header matches ``patterns`` (`allowed_hosts`).

    Matched as FastMCP matches it: exactly, or ``name:*`` for ``name:<port>``.
    A missing header is refused, as there.
    """
    if not host:
        return False
    host = host.strip().lower()
    for pattern in allowed_hosts() if patterns is None else patterns:
        pattern = pattern.lower()
        if host == pattern:
            return True
        if pattern.endswith(":*") and host.startswith(pattern[:-1]):
            if host[len(pattern) - 1 :].isdigit():
                return True
    return False


def foreign_host_message(host: Optional[str]) -> str:
    """Why a request naming ``host`` was refused, and the setting that admits it."""
    shown = repr((host or "")[:100]) if host else "no Host header"
    return (
        f"This server does not answer to {shown}. With {REQUIRE_AUTH_ENV} off it "
        f"answers only to localhost, 127.0.0.1, [::1] and the names in "
        f"{ALLOWED_HOSTS_ENV}, so a web page cannot reach it by pointing its own "
        f"domain at this machine (DNS rebinding). To reach it under this name, add "
        f"the name to {ALLOWED_HOSTS_ENV} (comma-separated; name:* allows any "
        f"port), or set {REQUIRE_AUTH_ENV}=true."
    )


def warn_once_if_unauthenticated(host: str) -> None:
    """Say plainly what an auth-off local server exposes."""
    if require_auth():
        return
    logger.warning(
        f"Serving on {host} without authentication. `X-Session-ID` is accepted "
        f"as identity with no credential, so anything that can reach this port "
        f"can read and delete any session. Set {REQUIRE_AUTH_ENV}=true before "
        f"exposing it."
    )


#: Values that ship in templates and docs; a secret equal to one of them is
#: public knowledge, so it is treated exactly like an unset secret.
_PLACEHOLDER_SECRETS = frozenset(
    {"change_me", "changeme", "change-me", "secret", "password", "admin"}
)
#: Shorter than this, a shared secret is guessable or a placeholder.
MIN_SECRET_LENGTH = 32


def secret_matches(presented: str, expected: str) -> bool:
    """Constant-time comparison of a presented secret with the configured one.

    `secrets.compare_digest` on two ``str`` raises TypeError when either holds
    a non-ASCII character, which reached the client as a 500 for an
    ``X-Admin-Token`` of Latin-1 bytes. Bytes have no such restriction.
    """
    import secrets

    return secrets.compare_digest(presented.encode("utf-8"), expected.encode("utf-8"))


#: Comma-separated API keys; with auth on, minting a session needs one of them.
API_KEYS_ENV = "RAG_API_KEYS"


def configured_api_keys() -> list:
    """The usable keys in ``RAG_API_KEYS``, by the same rule as `configured_secret`.

    A placeholder or a key shorter than ``MIN_SECRET_LENGTH`` is dropped, so a
    template value cannot open the instance.
    """
    keys = []
    for raw in os.environ.get(API_KEYS_ENV, "").split(","):
        key = raw.strip()
        if key and key.lower() not in _PLACEHOLDER_SECRETS and len(key) >= MIN_SECRET_LENGTH:
            keys.append(key)
    return keys


#: How a principal minted from an API key starts; the other principal is ``admin``.
KEY_PRINCIPAL_PREFIX = "key_"


def _key_principal(key: str) -> str:
    import hashlib

    return KEY_PRINCIPAL_PREFIX + hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]


def principal_for_api_key(presented: Optional[str]) -> Optional[str]:
    """The principal a presented API key belongs to, or ``None``.

    The principal is a short digest of the key, never the key: it goes into
    session tokens, rate-limit buckets and logs. Every configured key is
    compared, in constant time, so the time taken does not reveal which
    position in the list a near miss matched.
    """
    if not presented:
        return None
    matched = None
    for key in configured_api_keys():
        if secret_matches(presented, key):
            matched = key
    if matched is None:
        return None
    return _key_principal(matched)


def principal_is_revoked(principal: Optional[str]) -> bool:
    """Whether a token's principal names an API key no longer in ``RAG_API_KEYS``.

    A token outlives the key that minted it (seven days by default), so
    removing a leaked key revoked nothing until this was checked on every
    decode. Only key principals are checked: ``admin`` and tokens minted with
    auth off name no key. With auth off, keys mint nothing and are not read.
    """
    if not require_auth() or not isinstance(principal, str):
        return False
    if not principal.startswith(KEY_PRINCIPAL_PREFIX):
        return False
    return principal not in {_key_principal(key) for key in configured_api_keys()}


def configured_secret(name: str) -> str:
    """The value of a secret env var, or "" when it is unset or unusable.

    The shipped k8s manifest sets ``RAG_JWT_SECRET`` and ``RAG_ADMIN_TOKEN`` to
    ``CHANGE_ME``. Accepting that value let anyone who read the manifest forge
    session tokens and open the admin routes, while SECURITY.md promises no
    default credentials. A placeholder or a short value now counts as unset, so
    the existing refusals (no signing secret under auth, admin routes disabled)
    apply to it.
    """
    value = os.environ.get(name, "").strip()
    if not value:
        return ""
    if value.lower() in _PLACEHOLDER_SECRETS or len(value) < MIN_SECRET_LENGTH:
        return ""
    return value
