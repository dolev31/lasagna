"""Configuration management for Lasagna.

Settings for every stage: embedding models, chunking, hybrid search, reranking
and the rest, with their defaults, environment variables and validation.

The configuration supports session-scoped overrides that allow users to modify
RAG settings via the UI without permanently altering the settings.lasagna.toml file.

Provider Abstraction:
    This module uses the provider abstraction layer for configuration sources.
    Auto-configures from environment variables (RAG_DB_DIR, RAG_UPLOAD_DIR, etc.).
    For custom configuration, call configure_providers() before instantiating RAGConfig.
"""

import copy
import os
from pathlib import Path
from threading import Lock
from typing import Dict, Any, List, Optional, Tuple, TYPE_CHECKING
from dynaconf import Dynaconf
from loguru import logger
from lasagna.utils.logging_utils import configure_knowledge_logging

if TYPE_CHECKING:
    from lasagna.providers import ConfigProvider


# =============================================================================
# Collection identity: one vector collection per embedding model
# =============================================================================

#: The collection every embedding model shared before 0.1.0. The shipped
#: settings file and every built-in profile still carry it, so it means "not
#: chosen": a configuration that names it gets its own model's collection
#: (`collection_name_for`), unless an earlier version's documents in it were
#: adopted for that model (`adopt_collection`).
LEGACY_COLLECTION_NAME = "rag_documents_v2"

#: (store, embedding model, dimension) -> the collection that model adopted.
#: Filled by the collection manager, which reads and writes the stamps; the
#: config itself never opens a store.
_adopted_collections: Dict[Tuple[str, str, int], str] = {}
_adopted_lock = Lock()


def collection_name_for(model: Optional[str], dim: Any) -> str:
    """The collection documents embedded by ``model`` at ``dim`` live in.

    Vectors from two models are not comparable even at the same size
    (bge-large and bge-m3 are both 1024), so the model is part of the name, not
    only the dimension. The name keeps to lowercase letters, digits and
    underscores, which every supported backend accepts, and stays short enough
    for a Postgres identifier once pgvector prefixes it.
    """
    import hashlib
    import re

    raw = str(model or "default")
    slug = re.sub(r"[^a-z0-9]+", "_", raw.lower()).strip("_") or "default"
    if len(slug) > 32:
        slug = f"{slug[:23]}_{hashlib.sha256(raw.encode()).hexdigest()[:8]}"
    return f"{LEGACY_COLLECTION_NAME}__{slug}__{int(dim)}"


def is_derived_collection_name(name: Optional[str]) -> bool:
    """Whether ``name`` was derived from a model rather than chosen."""
    return bool(name) and str(name).startswith(f"{LEGACY_COLLECTION_NAME}__")


def resolve_collection_name(
    name: Optional[str], model: Optional[str], dim: Any, store: Optional[str] = None
) -> str:
    """The collection a configuration uses, given the name it carries.

    A name other than the legacy default is the user's choice and is returned
    unchanged; the evaluation harness names one per arm this way. Otherwise the
    model's own collection, or the legacy one if this model adopted it.
    """
    if name and name != LEGACY_COLLECTION_NAME:
        return str(name)
    with _adopted_lock:
        adopted = _adopted_collections.get((str(store or ""), str(model), int(dim)))
    return adopted or collection_name_for(model, dim)


def adopt_collection(name: str, model: Optional[str], dim: Any, store: Optional[str] = None):
    """Record that ``model`` at ``dim`` uses the existing collection ``name``."""
    with _adopted_lock:
        _adopted_collections[(str(store or ""), str(model), int(dim))] = name


def collection_name_for_settings(settings: Dict[str, Any]) -> str:
    """The collection a session whose resolved settings are ``settings`` uses.

    For checks made before a change is applied (a profile switch), when there
    is no ``RAGConfig`` for the new settings yet. Keys the settings leave out
    fall back to the process configuration, as a session's do.
    """
    base = get_rag_config()
    name = settings.get("collection_name")
    if name is None:
        name = base._get_knowledge_setting(
            "rag", "collection_name", default=base._config["collection_name"]
        )
    model = settings.get("embedding_model") or base.embedding_model
    dim = int(settings.get("embedding_dim") or base.embedding_dim)
    truncate = settings.get("embedding_truncate_dim", base.embedding_truncate_dim)
    if truncate and int(truncate) < dim:
        dim = int(truncate)
    return resolve_collection_name(name, model, dim, store=base._config.get("db_path"))


def forget_adopted_collections() -> None:
    """Drop every adoption recorded in this process (the stamps stay)."""
    with _adopted_lock:
        _adopted_collections.clear()


# =============================================================================
# Environment settings: parsing and validation
# =============================================================================

#: What an environment variable may say for true, and for false. The one set
#: every on/off variable in the package is read with.
TRUTHY = frozenset({"1", "true", "yes", "on"})
FALSY = frozenset({"0", "false", "no", "off"})


def env_flag(name: str, default: bool = False) -> bool:
    """An on/off environment variable, read the way every Lasagna switch is.

    ``TRUTHY`` and ``FALSY``, any case and surrounding space; unset or anything
    else is ``default``. Each module had its own set: ``yes`` and ``1`` turned
    ``OPENSEARCH_VERIFY_CERTS`` off, and ``on`` was ignored by most switches.
    """
    value = os.environ.get(name, "").strip().lower()
    if value in TRUTHY:
        return True
    if value in FALSY:
        return False
    return default


class ConfigError(ValueError):
    """A setting has a value it cannot take. The message names it and the fix."""


#: Numeric settings with a range; a value outside it is refused.
_VALID_RANGES: Dict[str, Tuple[float, float]] = {
    "chunk_size": (50, 100_000),
    "chunk_overlap": (0, 50_000),
    "default_top_k": (1, 1000),
    "rerank_top_k": (1, 1000),
    "embedding_dim": (32, 8192),
    "hybrid_alpha": (0.0, 1.0),
    "mmr_lambda": (0.0, 1.0),
    "similarity_threshold": (-1.0, 1.0),
    "connection_pool_size": (1, 500),
    "routing_confidence_threshold": (0.0, 1.0),
    "max_concurrent_uploads": (1, 20),
}

#: Settings whose value is one of a fixed set (compared without case).
_CHOICES: Dict[str, Tuple[str, ...]] = {
    "database_backend": ("sqlite", "postgresql"),
    "cache_backend": ("memory", "redis"),
    "tenant_isolation": ("none", "partition"),
    "log_level": ("TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"),
    "routing_default_tier": ("fast", "standard", "comprehensive"),
    "routing_strategy": ("rule_based", "adaptive"),
}

#: Settings whose default is None, with the type a value must have.
#: "none" or "null" (or an empty value) sets them back to None.
_NULLABLE_TYPES: Dict[str, type] = {
    "embedding_truncate_dim": int,
    "semantic_breakpoint_amount": float,
    "chunk_tokenizer": str,
    "proposition_model": str,
    "contextual_model": str,
    "contextual_max_document_chars": int,
}

#: Settings read only through a property default, with a value of their type.
#: Listed so their environment variables work; they appear in no defaults dict.
_PROPERTY_ONLY_SETTINGS: Dict[str, Any] = {
    "agentic_pipeline": "",
    "refinement_model": "",
}

#: Environment variables that are not settings: what the servers, the MCP
#: server and the vector stores read. Name -> (group, default, description,
#: older names). ``scripts/gen_settings_doc.py`` writes them into
#: docs/reference/settings.md, and their ``RAG_`` names make up
#: ``NON_SETTING_ENV``, so a variable the package starts reading is listed here
#: or ``tests/test_config_fail_fast.py`` fails.
SERVER_ENV: Dict[str, Tuple[str, str, str, Tuple[str, ...]]] = {
    "RAG_REQUIRE_AUTH": (
        "REST API and authentication",
        "false",
        "When true, only a verified X-Session-Token names a session, and POST "
        "/api/v1/sessions needs an X-API-Key. When false, binding beyond loopback is "
        "refused at startup unless RAG_BIND_IS_PROTECTED is set.",
        (),
    ),
    "RAG_API_KEYS": (
        "REST API and authentication",
        "unset",
        "Comma-separated keys, each at least 32 characters, that may mint session tokens "
        "when RAG_REQUIRE_AUTH is on. Rate limits are counted per key; storage quotas per "
        "session.",
        (),
    ),
    "RAG_JWT_SECRET": (
        "REST API and authentication",
        "a random secret per process",
        "Signs session tokens (HMAC-SHA256). Unset, tokens stop working when the process "
        "restarts; with RAG_REQUIRE_AUTH on, an unset, placeholder or shorter-than-32-"
        "character secret stops the server at startup.",
        (),
    ),
    "RAG_SESSION_TOKEN_TTL_SECONDS": (
        "REST API and authentication",
        "604800",
        "How long a session token is valid, in seconds (7 days).",
        (),
    ),
    "RAG_ADMIN_TOKEN": (
        "REST API and authentication",
        "unset",
        "The X-Admin-Token for /api/v1/admin/* and for naming a session in POST "
        "/api/v1/sessions. Unset, a placeholder or shorter than 32 characters: the admin "
        "routes answer 503.",
        (),
    ),
    "RAG_ALLOW_ANONYMOUS_SESSION": (
        "REST API and authentication",
        "false",
        "With RAG_REQUIRE_AUTH on, keep the anonymous default session usable.",
        (),
    ),
    "RAG_BIND_IS_PROTECTED": (
        "REST API and authentication",
        "false",
        "Says that something outside the process (a loopback port mapping, a "
        "NetworkPolicy, a firewall) restricts access, so the server may bind beyond "
        "loopback with auth off.",
        (),
    ),
    "RAG_ALLOW_HEADER_IDENTITY": (
        "REST API and authentication",
        "false",
        "Accept the X-User-ID header as the user identity; otherwise it is refused with 401.",
        (),
    ),
    "RAG_CORS_ORIGINS": (
        "REST API and authentication",
        "empty",
        "Comma-separated origins allowed cross-origin requests (REST and MCP over HTTP). "
        "Empty allows none; * is refused at startup when RAG_REQUIRE_AUTH is on.",
        (),
    ),
    "RAG_FORWARDED_ALLOW_IPS": (
        "REST API and authentication",
        "127.0.0.1",
        "Proxy addresses whose X-Forwarded-For is trusted. The image reads gunicorn's "
        "FORWARDED_ALLOW_IPS first.",
        (),
    ),
    "RAG_MAX_BATCH_FILES": (
        "REST API and authentication",
        "20",
        "Files in one POST /api/v1/documents/batch; more is refused with 400.",
        (),
    ),
    "RAG_MAX_BATCH_BYTES": (
        "REST API and authentication",
        "209715200",
        "Total body of one batch upload (200 MiB); more is refused with 413 before the "
        "body is read.",
        (),
    ),
    "RAG_SEARCH_TIMEOUT": (
        "REST API and authentication",
        "120",
        "Seconds a REST search may run before it answers 504.",
        (),
    ),
    "RAG_EMBED_MCP": (
        "REST API and authentication",
        "false",
        "Mount the MCP server at /mcp inside the REST process, for single-process "
        "deployments such as Milvus Lite.",
        (),
    ),
    "RAG_CONFIGURATOR": (
        "REST API and authentication",
        "false",
        "Mount the configurator UI at /configurator, with its API routes.",
        (),
    ),
    "LASAGNA_LOG_FORMAT": (
        "REST API and authentication",
        "text",
        "json writes one JSON object per log line; text shows a request's id before the "
        "message. Any other value stops the server at startup.",
        (),
    ),
    "PROMETHEUS_MULTIPROC_DIR": (
        "REST API and authentication",
        "unset",
        "The directory gunicorn workers share for Prometheus metrics (set in the image).",
        (),
    ),
    "LASAGNA_MCP_TOOLS": (
        "MCP server",
        "safe",
        "safe leaves out the six tools that delete data or reset configuration; all "
        "serves every tool. The same as lasagna-mcp --tools.",
        (),
    ),
    "LASAGNA_ENABLE_DOCS_API": (
        "MCP server",
        "false",
        "Turn on the /api/call tool tester, which calls any tool without "
        "authentication. For development only.",
        (),
    ),
    "RAG_SESSION_ID": (
        "MCP server",
        "unset",
        "The session a tool call uses when it names none.",
        (),
    ),
    "RAG_MCP_UPLOAD_ROOT": (
        "MCP server",
        "unset",
        "The directory a networked MCP server may ingest server-side files from; unset, "
        "it reads none.",
        (),
    ),
    "RAG_ALLOWED_HOSTS": (
        "MCP server",
        "unset",
        "Extra Host values (host or host:*) a loopback HTTP surface accepts, "
        "comma-separated, such as a container's service name.",
        (),
    ),
    "MILVUS_URI": (
        "Vector store connections",
        "unset",
        "A Milvus server's address, used when vector_backend_config has no uri. Without "
        "one the milvus backend opens Milvus Lite in the data directory.",
        (),
    ),
    "MILVUS_TOKEN": ("Vector store connections", "unset", "The token for MILVUS_URI.", ()),
    "QDRANT_URL": (
        "Vector store connections",
        "unset",
        "A Qdrant server's address; without one the qdrant backend stores its data in "
        "the data directory.",
        (),
    ),
    "QDRANT_API_KEY": ("Vector store connections", "unset", "The API key for QDRANT_URL.", ()),
    "WEAVIATE_URL": ("Vector store connections", "unset", "A Weaviate server's address.", ()),
    "WEAVIATE_API_KEY": (
        "Vector store connections",
        "unset",
        "The API key for WEAVIATE_URL.",
        (),
    ),
    "PINECONE_API_KEY": (
        "Vector store connections",
        "unset",
        "The Pinecone API key the pinecone backend needs.",
        (),
    ),
    "PGVECTOR_DSN": (
        "Vector store connections",
        "postgresql://rag:rag@localhost:5432/rag",
        "The PostgreSQL DSN of the pgvector backend; RAG_DATABASE_DSN is used when it is unset.",
        (),
    ),
    "OPENSEARCH_HOSTS": (
        "Vector store connections",
        "http://localhost:9200",
        "Comma-separated OpenSearch hosts, with OPENSEARCH_USER and OPENSEARCH_PASSWORD.",
        (),
    ),
    "OPENSEARCH_USE_SSL": (
        "Vector store connections",
        "false",
        "Connect to OpenSearch over SSL.",
        (),
    ),
    "OPENSEARCH_VERIFY_CERTS": (
        "Vector store connections",
        "false",
        "Verify the OpenSearch server's certificate.",
        (),
    ),
    "OPENSEARCH_TIMEOUT": (
        "Vector store connections",
        "30",
        "Seconds an OpenSearch request may take.",
        (),
    ),
    "OPENSEARCH_ENABLE_NEURAL_SPARSE": (
        "Vector store connections",
        "false",
        "Add neural sparse vectors to OpenSearch hybrid search (dense, sparse and BM25).",
        (),
    ),
    "LASAGNA_DATA_DIR": (
        "Paths and the process",
        "~/.lasagna",
        "Where Lasagna keeps its data.",
        ("RAG_DATA_DIR",),
    ),
    "LASAGNA_DB_DIR": (
        "Paths and the process",
        "<data dir>/dbs",
        "The databases and the local vector stores.",
        ("RAG_DB_DIR",),
    ),
    "LASAGNA_UPLOAD_DIR": (
        "Paths and the process",
        "<data dir>/uploads",
        "Stored uploads.",
        ("RAG_UPLOAD_DIR",),
    ),
    "LASAGNA_SETTINGS_FILE": (
        "Paths and the process",
        "the packaged settings.lasagna.toml",
        "The settings file to read instead of the packaged one.",
        ("RAG_SETTINGS_FILE",),
    ),
    "LASAGNA_LLM_CACHE": (
        "Paths and the process",
        "on",
        "0, false, no or off turns the LLM response cache off for every run.",
        (),
    ),
    "LASAGNA_STRICT_PLUGINS": (
        "Paths and the process",
        "unset",
        "Set: a plugin that fails to load raises instead of being reported and skipped.",
        (),
    ),
    "RAG_LANGFUSE_TRACING": (
        "Paths and the process",
        "false",
        "Send LLM traces to Langfuse (needs the langfuse package and its keys).",
        (),
    ),
    "RAG_LLM_CACHE_MAX_SIZE": (
        "Paths and the process",
        "50",
        "How many LLM clients the process keeps for reuse.",
        (),
    ),
    "RAG_CRAG_GRADING_CONCURRENCY": (
        "Paths and the process",
        "4",
        "Parallel relevance-grading calls in corrective RAG (CRAG).",
        (),
    ),
}

#: ``RAG_*`` variables read for the package's own use, not for an operator:
#: set by the CLI (``RAG_SESSION_BACKEND``), by the host application that
#: embeds Lasagna (``RAG_MCP_PORT``, ``RAG_REST_URL``) or for debugging.
_INTERNAL_RAG_ENV = frozenset(
    {"RAG_MCP_PORT", "RAG_REST_URL", "RAG_SESSION_BACKEND", "RAG_VALIDATE_PAGE_MAP"}
)

#: ``RAG_*`` variables the package reads that are not settings of this file:
#: storage paths, server security, the evaluation judge, the MCP server. The
#: ``RAG_GC_``, ``RAG_STORAGE_`` and ``RAG_EVAL_`` families are read by prefix.
#: ``tests/test_config_fail_fast.py`` checks this against every variable read.
NON_SETTING_ENV = frozenset(
    {name for name in SERVER_ENV if name.startswith("RAG_")}
    | {old for *_, olds in SERVER_ENV.values() for old in olds if old.startswith("RAG_")}
    | _INTERNAL_RAG_ENV
)
NON_SETTING_ENV_PREFIXES = ("RAG_GC_", "RAG_STORAGE_", "RAG_EVAL_")

#: ``LASAGNA_*`` names that already meant something else before settings took
#: the prefix, and keep that meaning. ``LASAGNA_LOG_LEVEL`` is the CLI's console
#: threshold (tolerant of a typo, so a bad value never stops ``doctor``); the
#: ``log_level`` setting stays ``RAG_LOG_LEVEL``.
LASAGNA_RESERVED_ENV = frozenset({"LASAGNA_LOG_LEVEL"})

_UNSET = object()
_warned_unknown_env: set = set()


def _type_phrase(template: Any, key: str) -> str:
    if key in _NULLABLE_TYPES:
        kind = _NULLABLE_TYPES[key]
        base = {int: "a whole number", float: "a number", str: "a string"}[kind]
        return f"{base}, or none"
    if isinstance(template, bool):
        return "true or false (also 1/0, yes/no, on/off)"
    if isinstance(template, int):
        return "a whole number"
    if isinstance(template, float):
        return "a number"
    if isinstance(template, list):
        return 'a JSON list, for example ["en"]'
    if isinstance(template, dict):
        return 'a JSON object, for example {"M": 16}'
    return "a string"


def _parse_env_value(key: str, raw: str, template: Any) -> Any:
    """``raw`` as the setting ``key`` takes it, or ``_UNSET`` for an empty value.

    An empty value of a typed setting is the shape ``${VAR:-}`` leaves in a
    compose file, so it means "not set" rather than an error.
    """
    import json as _json

    text = raw.strip()
    expected = _type_phrase(template, key)
    if key in _NULLABLE_TYPES:
        if text.lower() in ("", "none", "null"):
            return None
        try:
            value: Any = _NULLABLE_TYPES[key](text)
        except ValueError:
            raise ConfigError(f"{key} takes {expected}") from None
    elif isinstance(template, bool):
        if not text:
            return _UNSET
        if text.lower() in TRUTHY:
            value = True
        elif text.lower() in FALSY:
            value = False
        else:
            raise ConfigError(f"{key} takes {expected}")
    elif isinstance(template, (int, float)):
        if not text:
            return _UNSET
        try:
            value = int(text) if isinstance(template, int) else float(text)
        except ValueError:
            raise ConfigError(f"{key} takes {expected} (default {template})") from None
    elif isinstance(template, (list, dict)):
        if not text:
            return _UNSET
        try:
            value = _json.loads(text)
        except ValueError:
            raise ConfigError(f"{key} takes {expected}; this is not valid JSON") from None
        if not isinstance(value, type(template)):
            raise ConfigError(f"{key} takes {expected}")
    else:
        value = raw

    _check_bounds(key, value)
    return value


def _check_bounds(key: str, value: Any) -> None:
    """Raise `ConfigError` when ``value`` is outside ``key``'s range or choices."""
    if key in _VALID_RANGES and isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = _VALID_RANGES[key]
        if not (lo <= value <= hi):
            raise ConfigError(f"{key} takes {lo} to {hi}")
    if key in _CHOICES and isinstance(value, str):
        allowed = _CHOICES[key]
        if value.lower() not in {a.lower() for a in allowed}:
            import difflib

            hint = difflib.get_close_matches(value.lower(), [a.lower() for a in allowed], n=1)
            raise ConfigError(
                f"{key} takes one of {', '.join(allowed)}"
                + (f" (did you mean {hint[0]}?)" if hint else "")
            )


def _check_file_value(key: str, value: Any, template: Any) -> None:
    """Raise `ConfigError` when a settings-file value breaks the rules an env value meets.

    The file's values are typed already (TOML), so this checks the type the
    default has (a whole number also counts as a number), then the range and
    choices `_parse_env_value` applies.
    """

    def _is(kind: type) -> bool:
        if isinstance(value, bool):
            return kind is bool
        if kind is float:
            return isinstance(value, (int, float))
        return isinstance(value, kind)

    if key in _NULLABLE_TYPES:
        if value is None or (isinstance(value, str) and value.strip().lower() in ("none", "null")):
            return
        ok = _is(_NULLABLE_TYPES[key])
    elif template is None:
        ok = True
    else:
        ok = _is(type(template) if isinstance(template, (bool, int, float, str)) else object)
        if isinstance(template, list):
            ok = isinstance(value, list)
        elif isinstance(template, dict):
            ok = isinstance(value, dict)
    if not ok:
        raise ConfigError(f"{key} takes {_type_phrase(template, key)}")
    _check_bounds(key, value)


def _warn_unknown_env(env_key: str, known: Dict[str, Any]) -> None:
    """Warn once that ``env_key`` is not a setting, naming the closest one."""
    if (
        env_key in NON_SETTING_ENV
        or env_key.startswith(NON_SETTING_ENV_PREFIXES)
        or env_key in _warned_unknown_env
    ):
        return
    _warned_unknown_env.add(env_key)
    import difflib

    match = difflib.get_close_matches(env_key[4:].lower(), list(known), n=1, cutoff=0.75)
    hint = f" Did you mean RAG_{match[0].upper()}?" if match else ""
    logger.warning(
        f"{env_key} is not a Lasagna setting and was ignored.{hint} "
        f"The settings are listed in docs/reference/settings.md."
    )


class RAGConfig:
    """RAG system configuration manager.

    Supports session-scoped overrides that can be applied via the UI.
    When a session ID is provided, the configuration will include any
    session-specific overrides.

    Provider Support:
        - When config_provider is None (default): Uses auto-detected provider
        - Auto-configures from environment variables on first access
        - For custom configuration: Pass a ConfigProvider implementation
    """

    def __init__(self, config_provider: Optional["ConfigProvider"] = None):
        """Initialize RAG configuration.

        Args:
            config_provider: Optional custom configuration provider.
                           If None, uses auto-detected provider from environment
                           variables (RAG_DB_DIR, RAG_UPLOAD_DIR, API keys).
        """
        self._config_provider = config_provider
        self._knowledge_settings = self._load_knowledge_settings()
        self._env_overrides: Dict[str, Any] = {}  # Populated in _load_config
        self._config = self._load_config()
        self._session_id: Optional[str] = None
        self._query_overrides: Dict[str, Any] = {}  # Per-query ephemeral overrides
        configure_knowledge_logging(self.log_level)

    def _get_provider(self) -> "ConfigProvider":
        """Get the configuration provider (lazy initialization).

        Returns:
            ConfigProvider instance
        """
        if self._config_provider is None:
            from lasagna.providers import get_config_provider

            self._config_provider = get_config_provider()
        return self._config_provider

    def with_session(self, session_id: Optional[str]) -> "RAGConfig":
        """Create a config instance bound to a specific session.
        This allows retrieval of session-specific overrides.

        Args:
            session_id: The session identifier

        Returns:
            A new RAGConfig instance bound to the session
        """
        config = RAGConfig.__new__(RAGConfig)
        config._config_provider = self._config_provider
        config._config = copy.deepcopy(self._config)
        config._env_overrides = copy.deepcopy(self._env_overrides)
        config._knowledge_settings = self._knowledge_settings
        config._session_id = session_id
        config._query_overrides = {}
        return config

    def settings_fingerprint(self) -> str:
        """A short hash of everything that can differ between two searches.

        Base settings are process-wide; what varies between calls is the
        session's resolved config and the per-query overrides. The search cache
        keys on this, so changing any setting (a reranker model, a threshold,
        the candidate depth) is a cache miss rather than a stale result.
        """
        import hashlib
        import json as _json

        parts: Dict[str, Any] = {"query": getattr(self, "_query_overrides", None) or {}}
        if self._session_id:
            try:
                from lasagna.providers import get_session_provider

                parts["session"] = get_session_provider().get_resolved_config(self._session_id)
            except Exception:
                parts["session"] = {"_unresolved": self._session_id}
        blob = _json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def with_query_overrides(self, overrides: Dict[str, Any]) -> "RAGConfig":
        """Create a config clone with per-query ephemeral overrides (highest priority).

        These overrides are scoped to a single query call and are never written
        to the session provider.  They take effect above session overrides,
        environment variables, and TOML settings.

        Args:
            overrides: Flat dict of config keys to override (e.g.
                ``{"enable_reranking": False, "hybrid_alpha": 0.5}``).

        Returns:
            A new RAGConfig instance that inherits all current settings plus
            the given overrides.  The original instance is not modified.
        """
        config = RAGConfig.__new__(RAGConfig)
        config._config_provider = self._config_provider
        # Shallow copy: _config values are primitives (str/int/bool/dict-of-primitives).
        # A shallow copy is sufficient to prevent mutation of the original dict while
        # avoiding the overhead of deep-copying large nested structures.
        config._config = dict(self._config)
        config._env_overrides = self._env_overrides  # read-only in practice
        config._knowledge_settings = self._knowledge_settings
        config._session_id = self._session_id
        config._query_overrides = dict(overrides)
        return config

    def _get_session_override(self, *keys: str, default: Any = None) -> Any:
        """Get a value from session overrides if available.

        Args:
            *keys: The nested keys to look up (e.g., 'chunk_size' or 'document_awareness', 'enabled')
            default: Default value if not found

        Returns:
            The session override value, or None if not found
        """
        # Per-query ephemeral overrides take highest priority (set via with_query_overrides()).
        # Use flat leaf-key lookup — matches the flat dicts used by the eval SDK.
        if getattr(self, "_query_overrides", None):
            leaf_key = keys[-1] if len(keys) > 1 else keys[0]
            if leaf_key in self._query_overrides:
                return self._query_overrides[leaf_key]

        if not self._session_id:
            return None

        def _lookup(resolved: Dict[str, Any], path: Tuple[str, ...]) -> Tuple[Optional[Any], bool]:
            value: Any = resolved
            for key in path:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return None, False
            return value, True

        try:
            from lasagna.providers import get_session_provider

            resolved = getattr(self, "_resolved_once", None)
            if resolved is None:
                resolved = get_session_provider().get_resolved_config(self._session_id)

            # Navigate nested keys
            value, ok = _lookup(resolved, keys)
            if ok:
                return value

            # Session overrides are stored flat (no leading 'rag') for UI settings.
            if keys and keys[0] == "rag":
                value, ok = _lookup(resolved, keys[1:])
                if ok:
                    return value

                if len(keys) > 1:
                    value, ok = _lookup(resolved, (keys[-1],))
                    if ok:
                        return value
            return None
        except Exception as e:
            logger.debug(f"Could not get session override for {keys}: {e}")
            return None

    def _load_knowledge_settings(self) -> Dynaconf:
        """Load settings from settings.lasagna.toml, or the file the environment names.

        The config file is located in the knowledge module folder, making the
        knowledge service self-contained and usable as an external service.

        When ``LASAGNA_SETTINGS_FILE`` (or ``RAG_SETTINGS_FILE``) is set, that
        file is loaded instead of the default ``settings.lasagna.toml``.  The
        path can be absolute or relative to the knowledge module directory.

        Returns:
            Dynaconf settings object
        """
        knowledge_module_dir = Path(__file__).parent.parent  # knowledge/ folder

        # Allow overriding the settings file via environment variable
        from lasagna.core.paths import env_get

        settings_file_env = env_get("SETTINGS_FILE")
        if settings_file_env:
            override_path = Path(settings_file_env)
            if not override_path.is_absolute():
                override_path = knowledge_module_dir / override_path
            if override_path.exists():
                logger.info(f"Loading settings from {override_path}")
                self._settings_path = str(override_path)
                return Dynaconf(
                    root_path=str(override_path.parent),
                    settings_files=[str(override_path)],
                )
            logger.warning(
                f"Settings file {settings_file_env} (LASAGNA_SETTINGS_FILE or "
                f"RAG_SETTINGS_FILE) not found; using the packaged settings.lasagna.toml"
            )

        knowledge_toml_path = knowledge_module_dir / "settings.lasagna.toml"

        if knowledge_toml_path.exists():
            logger.debug(f"Loading settings from knowledge module: {knowledge_toml_path}")
            self._settings_path = str(knowledge_toml_path)
            return Dynaconf(
                root_path=str(knowledge_module_dir),
                settings_files=[str(knowledge_toml_path)],
            )

        logger.debug("settings.lasagna.toml not found in knowledge module, using defaults only")
        return Dynaconf()

    def _load_config(self) -> Dict[str, Any]:
        """Load RAG configuration from settings.

        Uses the config provider to get RAG settings and paths.

        Returns:
            Configuration dictionary
        """
        # Get RAG-specific settings from provider (settings.memory.rag in the host application)
        try:
            provider = self._get_provider()
            rag_settings = provider.get_rag_settings()
        except RuntimeError:
            # No provider configured - use empty settings
            logger.warning("No config provider, using defaults only")
            rag_settings = {}

        # Ensure rag_settings is a dict
        if hasattr(rag_settings, "to_dict"):
            rag_settings = rag_settings.to_dict()
        elif not isinstance(rag_settings, dict):
            rag_settings = {}

        # Default configuration
        default_config = {
            # =================================================================
            # Vector Database
            # =================================================================
            "vector_backend": "milvus",  # Supported: milvus, pinecone, qdrant, weaviate, pgvector, opensearch
            "vector_backend_config": {},  # Backend-specific configuration
            "collection_name": "rag_documents_v2",
            "embedding_dim": 1024,  # must match embedding_model (bge-m3 and bge-large: 1024)
            "metric_type": "COSINE",  # vector similarity: COSINE, L2 or IP
            "index_type": "HNSW",  # dense index on servers; Milvus Lite always uses FLAT
            "index_params": {"M": 16, "efConstruction": 256},  # HNSW build parameters
            "search_params": {"ef": 128},  # HNSW search breadth
            # =================================================================
            # Embedding Model
            # =================================================================
            "embedding_provider": "huggingface",  # where embeddings are computed
            "embedding_model": "BAAI/bge-m3",  # dense+sparse+ColBERT, 8K context, 100+ languages
            "embedding_model_fallback": "BAAI/bge-large-en-v1.5",  # loaded if the model fails; never during evaluation
            "embedding_backend": "auto",  # 'auto', 'sentence-transformers', 'flag-embedding'
            "normalize_embeddings": True,
            "batch_size": 32,
            "use_gpu": True,
            "embedding_truncate_dim": None,  # None = use full dim; int = truncate+renorm (Matryoshka)
            # =================================================================
            # Chunking
            # =================================================================
            "chunking_strategy": "semantic",  # fixed, semantic, recursive, sentence_window
            "chunk_size": 800,
            "chunk_overlap": 0,  # Research: no measurable benefit from overlap (arXiv:2601.14123)
            "semantic_threshold": 0.75,
            # 'absolute' (semantic_threshold) or Kamradt's 'percentile',
            # 'standard_deviation', 'interquartile', 'gradient'
            "semantic_breakpoint_method": "absolute",
            "semantic_breakpoint_amount": None,  # None = the method's default (95, 3, 1.5, 95)
            "semantic_buffer_size": 1,  # neighbours embedded with each sentence (Kamradt)
            "min_chunk_size": 200,
            "max_chunk_size": 1500,
            "window_size": 3,
            "sentence_overlap": 1,
            "enable_hierarchical": True,
            "parent_chunk_size": 2000,
            "child_chunk_size": 500,
            "auto_merge_threshold": 0.5,
            "enable_late_chunking": False,  # Late chunking: embed full doc, pool per chunk (Jina, arxiv:2409.04701)
            # Tokens shared by consecutive model windows when a document is longer
            # than one (long late chunking); capped at half a window.
            "late_chunking_overlap": 512,
            # What chunk_size and chunk_overlap count, for the fixed and recursive
            # chunkers: 'chars', 'words' (runs of non-whitespace) or 'tokens'
            # (of chunk_tokenizer: a Hugging Face tokenizer, or 'tiktoken:<encoding>').
            "chunk_unit": "chars",
            "chunk_tokenizer": None,
            # Recursive chunker: paragraph, line, sentence, word, character. With
            # ["\n\n", "\n", " ", ""] and keep 'start' it is LangChain's
            # RecursiveCharacterTextSplitter default exactly.
            "recursive_separators": ["\n\n", "\n", ". ", "? ", "! ", " ", ""],
            "recursive_keep_separator": "end",  # 'end' or 'start': which piece a separator stays with
            # Markdown header chunker (LangChain MarkdownHeaderTextSplitter semantics)
            "markdown_header_levels": 6,  # split on # .. ###### up to this depth
            "markdown_strip_headers": True,  # header text goes to metadata, not the chunk
            "markdown_split_long_sections": True,  # split sections over chunk_size recursively
            # Proposition chunker (Dense X Retrieval, arXiv:2312.06648)
            "proposition_model": None,  # None = the configured default LLM
            "proposition_passage_size": 100,  # words per passage handed to the LLM (paper: 100)
            # Contextual chunker (Anthropic Contextual Retrieval)
            "contextual_base_chunker": "recursive",  # registered chunker that makes the chunks
            "contextual_model": None,  # None = the configured default LLM
            "contextual_max_document_chars": 50000,  # document shown per call; None = all of it
            "preserve_tables": True,
            "preserve_headers": True,
            "extract_metadata": True,  # keep page, section and file metadata on each chunk
            "max_file_size": 52428800,  # 50MB
            "max_content_length": 4000,  # Max chars per chunk in storage
            # =================================================================
            # Hybrid Search
            # =================================================================
            "enable_hybrid_search": True,
            "hybrid_alpha": 0.7,
            "sparse_method": "bm25",
            "default_top_k": 20,  # Research: 75+ introduces hard negatives (ICLR 2025)
            "similarity_threshold": 0.3,
            "rrf_k": 60,  # RRF constant (higher = more weight to lower ranks)
            "enable_bm25_stemming": True,  # Enable Porter stemming for BM25
            "bm25_max_sessions": 1000,  # Max in-memory BM25 session indices
            "bm25_session_ttl_seconds": 86400,  # Idle TTL for session indices (24h)
            # Built-in retriever components (`retriever: bm25`, `splade`, ...).
            "bm25_k1": 1.2,  # Okapi BM25 tf saturation (Robertson & Zaragoza 2009)
            "bm25_b": 0.75,  # Okapi BM25 length normalisation
            # SPLADE++ CoCondenser-EnsembleDistil (Formal et al. 2022)
            "splade_model": "naver/splade-cocondenser-ensembledistil",
            # Candidates each list contributes before fusion; both fusion papers
            # (Cormack et al. 2009, Bruch et al. 2023) fuse the top 1000.
            "hybrid_fusion_depth": 1000,
            # Score normalisation for hybrid_convex: 'tmm' (theoretical min-max,
            # Bruch et al.'s recommendation), 'minmax' or 'zscore'.
            "hybrid_convex_normalization": "tmm",
            # =================================================================
            # Reranking
            # =================================================================
            "enable_reranking": True,
            "reranker_model": "mixedbread-ai/mxbai-rerank-large-v1",  # BEIR NDCG@10 = 57.49
            "rerank_top_k": 5,  # Research: context cliff at ~2,500 tokens
            "reranker_batch_size": 16,
            "custom_reranker": "",  # Registry name for custom reranker (empty = use default)
            "custom_query_enhancer": "",  # Registry name for custom query enhancer (empty = use default)
            # Empty = resolve from MODEL_NAME / the configured provider. Never
            # name a specific vendor route here: a gateway-prefixed default like
            # "<vendor>/<model>" is rejected outright by every other provider, so
            # it fails for anyone who did not happen to share our deployment.
            "llm_reranker_model": "",  # Model for LLM listwise reranker
            # RankGPT's sliding window (Sun et al. 2023): 20 passages, moved by 10.
            "llm_reranker_window_size": 20,
            "llm_reranker_step_size": 10,
            "colbert_model": "colbert-ir/colbertv2.0",  # ColBERT late-interaction model
            "monot5_model": "castorini/monot5-base-msmarco-10k",  # Nogueira et al. 2020
            "flashrank_model": "ms-marco-TinyBERT-L-2-v2",  # FlashRank's default ONNX model
            "cohere_rerank_model": "rerank-v3.5",
            # Some cross-encoder rerankers ship their own modelling code (e.g.
            # jinaai/jina-reranker-v2-base-multilingual) and load only with it.
            "reranker_trust_remote_code": False,
            "enable_context_reordering": True,  # Lost-in-the-middle fix: place best results at start/end
            "enable_context_compression": False,  # Compress chunks to reduce noise/tokens
            "compression_method": "extractive",  # 'extractive' | 'abstractive'
            "compression_ratio": 0.3,  # Keep ~30% of original length
            "enable_graph_retrieval": False,  # GraphRAG entity-linked retrieval
            "graph_max_hops": 2,  # Max traversal depth in knowledge graph
            "graph_entity_types": ["PERSON", "ORG", "CONCEPT", "LOCATION", "TECHNOLOGY", "EVENT"],
            "graph_extraction_model": "",  # Model for entity extraction (empty = configured default)
            # Per-node agentic overrides, keyed by node name. Empty means "use
            # the shipped prompt / the pipeline default model". These are what
            # make an agentic technique testable: without them the prompt that
            # decides whether an answer is grounded cannot be varied, and two
            # runs using different prompts recorded nothing to tell them apart.
            "agentic_prompts": {},
            "agentic_models": {},
            "enable_raptor": False,  # RAPTOR: recursive summary tree indexing (arXiv:2401.18059)
            "raptor_max_cluster_size": 10,  # Max chunks per cluster in RAPTOR tree
            "raptor_max_levels": 5,  # Max tree depth
            # raptor_retrieval_mode: reserved for future tree-traversal query path.
            # Currently all RAPTOR nodes are searched via standard vector search (collapsed mode).
            "enable_speculative_rag": False,  # Speculative RAG: parallel drafts + verification
            "speculative_n_drafts": 3,  # Number of parallel draft answers
            "speculative_subset_size": 3,  # Context chunks per draft
            # Published agentic loops (lasagna.agentic.loops), with their papers'
            # defaults; each loop's own budget, not max_agent_iterations.
            "self_ask_max_follow_ups": 5,  # Self-Ask: follow-up questions before concluding
            "ircot_paragraphs_per_step": 6,  # IRCoT: K paragraphs retrieved per step
            "ircot_max_steps": 8,  # IRCoT: reasoning steps before stopping
            "ircot_max_paragraphs": 15,  # IRCoT: cap on paragraphs collected
            "flare_retrieval_threshold": 0.8,  # FLARE theta: retrieve if any token is below
            "flare_mask_threshold": 0.4,  # FLARE beta: mask tokens below this in the query
            "flare_max_sentences": 8,  # FLARE: sentences generated before stopping
            "flare_docs_per_retrieval": 2,  # FLARE: documents per retrieval
            "flare_confidence_source": "auto",  # 'auto' | 'logprobs' | 'self_reported'
            "adaptive_rag_classifier_model": "",  # Adaptive-RAG classifier (empty = loop's LLM)
            "adaptive_rag_single_step_docs": 5,  # Adaptive-RAG: documents for the single-step route
            "self_rag_passages": 5,  # Self-RAG: passages retrieved and critiqued
            "self_rag_weight_rel": 1.0,  # Self-RAG critique weights (ISREL, ISSUP, ISUSE)
            "self_rag_weight_sup": 1.0,
            "self_rag_weight_use": 0.5,
            "self_rag_retrieval": "adaptive",  # 'adaptive' (Retrieve judgement) | 'always'
            "enable_mmr": True,
            "mmr_lambda": 0.5,  # LangChain and LlamaIndex default; a pre-MMR filter keeps only relevant items
            "mmr_score_normalization": "sigmoid",  # Options: 'sigmoid', 'minmax', 'sigmoid_minmax', 'none'. Plain sigmoid preserves cross-encoder calibration.
            # =================================================================
            # Query Enhancement
            # =================================================================
            "enable_hyde": False,
            "hyde_model": "default",
            "hyde_n_docs": 3,  # Number of hypothetical docs for multi-HyDE (1 = legacy single-doc)
            "hyde_embedding_strategy": "average",  # Options: 'average', 'max_similarity'
            "enable_query_expansion": False,
            "max_query_variations": 3,
            "enable_query_decomposition": False,
            "enable_query_classification": True,  # Enable adaptive query-type classification
            "agentic_hyde_strategy": "adaptive",
            "enable_contextual_retrieval": False,  # Anthropic-style chunk enrichment at index time
            "enable_rag_fusion": False,  # Multi-query generation + RRF fusion
            "rag_fusion_n_queries": 3,  # Number of query variations for RAG Fusion
            # Query2doc (arXiv:2303.07678): the expansion form ('dense': q [SEP] d';
            # 'sparse': q repeated n times, then d'), n, and the pseudo-document's
            # token budget, all at the paper's values.
            "query2doc_format": "dense",
            "query2doc_query_repeats": 5,
            "query2doc_max_tokens": 128,
            "enable_crag": False,  # Corrective RAG: LLM grades retrieval quality
            "crag_web_search_fallback": False,  # Fall back to web search if retrieval quality is low
            # =================================================================
            # Query Routing (Adaptive-RAG)
            # =================================================================
            "enable_query_routing": True,  # Route queries to FAST/STANDARD/COMPREHENSIVE tiers
            "routing_strategy": "rule_based",  # 'rule_based' (deterministic) or 'adaptive' (confidence-aware)
            "routing_default_tier": "standard",  # Default tier when routing is disabled: fast, standard, comprehensive
            "routing_confidence_threshold": 0.5,  # Adaptive strategy escalates tier below this confidence
            "routing_allow_tier_override": True,  # Allow API callers to force a routing tier
            "routing_log_decisions": True,  # Log routing decisions for observability
            # =================================================================
            # Citation Tracking
            # =================================================================
            "citation_format": "stable_id",
            "citation_id_length": 8,
            "verify_citations": True,
            "track_citations_db": True,
            "show_page_numbers": True,
            "include_source_preview": True,  # a short excerpt of the source with each citation
            "citation_db_max_age_hours": 168,  # 1 week
            "citation_db_max_per_session": 1000,  # oldest citations beyond this are removed
            "citation_normalize_scores": True,  # scale citation scores to 0-1
            "citation_score_method": "minmax",  # 'minmax', 'sigmoid', 'clamp'
            # =================================================================
            # File Types
            # =================================================================
            # Only formats DocumentProcessor can identify and extract. .doc,
            # .rtf and .epub have no extractor, and .pptx and .xlsx are refused
            # by the upload type check before Docling could convert them, so
            # listing them advertised uploads that always failed.
            "supported_extensions": [
                ".pdf",
                ".docx",
                ".txt",
                ".md",
                ".csv",
                ".json",
                ".html",
                ".xml",
            ],
            # =================================================================
            # Storage
            # =================================================================
            "upload_dir_name": "rag_uploads",  # folder name for stored uploads
            "db_name": "rag.db",  # Unified database (consolidated from rag_metadata.db + rag_citations.db)
            # =================================================================
            # Parallel Upload
            # =================================================================
            "max_concurrent_uploads": 5,  # Maximum parallel document uploads (1-20)
            # =================================================================
            # OCR
            # =================================================================
            "enable_ocr": False,
            "ocr_engine": "easyocr",
            "ocr_languages": ["en"],
            "ocr_confidence_threshold": 0.5,
            # =================================================================
            # Processing
            # =================================================================
            "extract_images": False,  # extract images from documents
            "preserve_formatting": True,  # keep layout such as lists and line breaks in extracted text
            "async_processing": True,  # process uploads in the background
            "processing_timeout": 300,  # seconds a document may take to process
            # A server (serve, the MCP server) refuses a request whose model is not
            # downloaded rather than fetch gigabytes inside it.
            "allow_model_download_in_requests": False,
            # =================================================================
            # Docling Document Processing
            # =================================================================
            "docling_enabled": True,  # Docling is default; falls back to legacy if not installed
            "docling_mode": "auto",  # 'auto', 'local', 'production'
            "docling_pdf_backend": "docling_parse",  # 'docling_parse' or 'pypdfium2'
            "docling_ocr_enabled": True,  # Enable OCR within Docling
            "docling_table_structure": True,  # Enable TableFormer table extraction
            "docling_max_tokens": 512,  # Max tokens per chunk (HybridChunker)
            "docling_timeout": 300,  # Per-document timeout (seconds)
            # =================================================================
            # Caching
            # =================================================================
            "enable_embedding_cache": True,
            "embedding_cache_size": 10000,
            "enable_search_cache": True,
            "search_cache_ttl": 300,
            "search_cache_max_size": 1000,  # Max cached search results
            # =================================================================
            # Security Limits
            # =================================================================
            "max_document_ids_filter": 100,  # Max document IDs in filter to prevent expression overflow
            # =================================================================
            # Observability
            # =================================================================
            "enable_prometheus": False,  # No-op metrics by default; True activates Prometheus
            "enable_tracing": False,  # No-op tracing by default; True activates OpenTelemetry
            "tracing_exporter": "otlp",  # 'otlp', 'jaeger', 'zipkin', 'console'
            "tracing_endpoint": "http://localhost:4317",  # OTLP gRPC endpoint
            "tracing_service_name": "rag-knowledge",
            "log_level": "INFO",
            # =================================================================
            # Multi-Tenancy
            # =================================================================
            "tenant_isolation": "none",  # 'none' = metadata filter only; 'partition' = native backend partitions
            "quota_enabled": False,  # Per-tenant resource quotas
            # =================================================================
            # Rate Limiting
            # =================================================================
            "rate_limiting_enabled": False,  # Adaptive rate limiter
            "rate_limiting_strategy": "adaptive",  # 'adaptive' or 'fixed'
            "rate_limiting_rpm": 60.0,  # Base requests per minute per tenant
            "rate_limiting_burst": 10.0,  # Max burst token capacity
            "rate_limiting_burst_multiplier": 2.0,  # Capacity multiplier when idle
            "rate_limiting_overload_threshold": 0.8,  # System load that triggers throttling
            "rate_limiting_idle_threshold": 0.3,  # System load below which bursting is allowed
            # =================================================================
            # Database Backend
            # =================================================================
            "database_backend": "sqlite",  # 'sqlite' (default) or 'postgresql'
            "database_dsn": "",  # PostgreSQL DSN (when database_backend='postgresql')
            "database_pool_min": 2,  # Min connection pool size (PostgreSQL)
            "database_pool_max": 10,  # Max connection pool size (PostgreSQL)
            # =================================================================
            # Redis / Distributed Cache
            # =================================================================
            "cache_backend": "memory",  # 'memory' (default) or 'redis'
            "redis_url": "redis://localhost:6379",  # Redis connection URL
            "redis_pool_size": 10,  # Redis connection pool size
            # =================================================================
            # Celery / Background Processing
            # =================================================================
            "celery_enabled": False,  # False = synchronous processing (default)
            "celery_broker_url": "redis://localhost:6379/0",
            "celery_result_backend": "redis://localhost:6379/1",
            # =================================================================
            # Connection Pooling
            # =================================================================
            "connection_pool_size": 1,  # Vector backend pool size (1 = no pooling)
            # =================================================================
            # File Storage Backend
            # =================================================================
            "storage_backend": "local",  # 'local' (default), 's3', 'fsspec'
            "storage_backend_config": {},  # Backend-specific configuration
            # =================================================================
            # Fast Ingestion (eval-only pipeline tuning)
            # =================================================================
            "ingest_extract_workers": min(os.cpu_count() or 4, 8),  # Parallel chunking workers
            "ingest_embed_batch_size": 64,  # Cross-doc embedding batch size
            "ingest_insert_batch_size": 500,  # Chunks per vector store insert call
        }

        # Merge with settings from TOML (settings.memory.rag takes precedence over defaults)
        config = {**default_config, **rag_settings}

        # Also merge settings from settings.lasagna.toml if available
        # This ensures settings.lasagna.toml values are used when settings.memory.rag doesn't override them
        knowledge_rag = self._knowledge_settings.get("rag", {})
        if hasattr(knowledge_rag, "to_dict"):
            knowledge_rag = knowledge_rag.to_dict()
        elif not isinstance(knowledge_rag, dict):
            knowledge_rag = {}
        self._check_settings_file(knowledge_rag, default_config)

        # 12-factor: Load overrides from RAG_* environment variables. Every
        # setting is accepted: the defaults above, the ones only the settings
        # file defines (typed by its value there), and the property-only ones.
        # [rag.lifecycle] has its own RAG_GC_* variables.
        env_overrides = self._load_env_overrides(
            {
                **_PROPERTY_ONLY_SETTINGS,
                **{k: v for k, v in knowledge_rag.items() if k != "lifecycle"},
                **default_config,
            }
        )
        self._env_overrides = env_overrides  # Store for priority lookup in _get_knowledge_setting

        # Merge order: defaults < knowledge.toml < settings.memory.rag < env vars
        config = {**default_config, **knowledge_rag, **rag_settings, **env_overrides}

        # Resolve paths using provider
        try:
            provider = self._get_provider()
            upload_dir = provider.upload_dir
            db_dir = provider.db_dir
        except RuntimeError:
            # No provider - use current directory fallback
            logger.warning("No config provider, using current directory for paths")
            cwd = Path(os.getcwd())
            upload_dir = cwd / "rag_uploads"
            db_dir = cwd / "dbs"

        config["upload_dir"] = str(upload_dir)

        db_name = config.get("db_name", "rag.db")
        config["db_path"] = str(db_dir / db_name)

        # Unified database: citation_db_path now points to same file as db_path
        # Kept for backward compatibility - code can still reference citation_db_path
        config["citation_db_path"] = config["db_path"]

        # Ensure directories exist
        os.makedirs(config["upload_dir"], exist_ok=True)
        os.makedirs(str(db_dir), exist_ok=True)

        logger.info(
            f"RAG config initialized: upload_dir={config['upload_dir']}, db_path={config['db_path']}"
        )

        return config

    def _check_settings_file(self, values: Dict[str, Any], defaults: Dict[str, Any]) -> None:
        """Refuse settings-file values that the same setting in the environment would refuse.

        They were merged unchecked, so ``chunk_size = 5`` or
        ``database_backend = "postgres"`` in a settings file ran with no sign.
        The error names the file, each key and its value.
        """
        problems = []
        for key, value in values.items():
            if key not in defaults:
                continue
            try:
                _check_file_value(key, value, defaults[key])
            except ConfigError as exc:
                problems.append(f"[rag] {key} = {value!r}: {exc}")
        if problems:
            source = getattr(self, "_settings_path", None) or "the settings file"
            raise ConfigError(
                f"Invalid setting{'s' if len(problems) > 1 else ''} in {source}:\n  "
                + "\n  ".join(problems)
                + "\nFix the value in the file, or remove the line to use the default."
            )

    @staticmethod
    def _load_env_overrides(default_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load overrides from ``LASAGNA_*`` and ``RAG_*`` environment variables (12-factor).

        ``LASAGNA_<SETTING>`` and ``RAG_<SETTING>`` name the same setting, and
        ``LASAGNA_`` wins when both are set (and not empty). Keys of
        *default_config* are accepted, coerced to the type of their
        default (bool, int, float, str, JSON list or object). A value that does
        not parse, or is outside its range or its set of choices, raises
        :class:`ConfigError` naming every bad variable; it used to be logged
        and ignored, so a typo ran the default without a visible sign. A
        ``RAG_*`` name that is no setting warns once, with the closest one.

        Example:
            ``RAG_DATABASE_BACKEND=postgresql``  →  ``{'database_backend': 'postgresql'}``
            ``RAG_ENABLE_RERANKING=true``         →  ``{'enable_reranking': True}``
        """
        overrides: Dict[str, Any] = {}
        problems: List[str] = []
        for env_key, env_value in sorted(os.environ.items()):
            if env_key.startswith("LASAGNA_"):
                config_key = env_key[len("LASAGNA_") :].lower()
                if config_key not in default_config or env_key in LASAGNA_RESERVED_ENV:
                    continue  # LASAGNA_* also names things that are not settings
            elif env_key.startswith("RAG_"):
                config_key = env_key[len("RAG_") :].lower()
                if config_key not in default_config:
                    _warn_unknown_env(env_key, default_config)
                    continue
                alias = f"LASAGNA_{env_key[len('RAG_') :]}"
                if alias not in LASAGNA_RESERVED_ENV and os.environ.get(alias):
                    continue  # the LASAGNA_ name of the same setting wins
            else:
                continue
            try:
                parsed = _parse_env_value(config_key, env_value, default_config[config_key])
            except ConfigError as exc:
                problems.append(f"{env_key}={env_value!r}: {exc}")
                continue
            if parsed is not _UNSET:
                overrides[config_key] = parsed

        if problems:
            raise ConfigError(
                "Invalid environment setting"
                + ("s" if len(problems) > 1 else "")
                + ":\n  "
                + "\n  ".join(problems)
                + "\nFix the value, or unset the variable to use the default."
            )

        if overrides:
            logger.info(f"Applied {len(overrides)} environment overrides: {list(overrides.keys())}")
        return overrides

    def _get_knowledge_setting(self, *keys, default=None):
        """Safely get nested setting from knowledge settings.

        Priority order (highest to lowest):
        1. Session overrides (per-session UI changes)
        2. Environment variable overrides (RAG_* 12-factor)
        3. TOML settings (settings.lasagna.toml)
        4. default parameter (typically self._config[key], which merges
           defaults < knowledge.toml < settings.memory.rag < env vars)
        """
        # 1. Check session overrides first (highest priority)
        session_value = self._get_session_override(*keys)
        if session_value is not None:
            return session_value

        # 2. Check env var overrides — use the leaf key for flat lookup
        #    e.g., keys=('rag', 'enable_reranking') → leaf='enable_reranking'
        if self._env_overrides and len(keys) >= 1:
            leaf_key = keys[-1] if len(keys) > 1 else keys[0]
            if leaf_key in self._env_overrides:
                return self._env_overrides[leaf_key]

        # 3. Fall back to TOML settings
        obj = self._knowledge_settings
        for key in keys:
            if hasattr(obj, key):
                obj = getattr(obj, key)
            elif isinstance(obj, dict) and key in obj:
                obj = obj[key]
            else:
                return default
        return obj if obj is not None else default

    # =========================================================================
    # Vector Database Properties
    # =========================================================================
    @property
    def vector_backend(self) -> str:
        """Vector database backend to use.

        Supported backends:
            - 'milvus' (default): Milvus Lite, Standalone, or Distributed
            - 'pinecone': Pinecone managed service
            - 'qdrant': Qdrant vector database
            - 'weaviate': Weaviate vector database
            - 'pgvector': PostgreSQL pgvector extension
            - 'opensearch': OpenSearch search engine (Lucene BM25 + k-NN vectors)

        Set via:
            - Environment variable: RAG_VECTOR_BACKEND (legacy: LASAGNA_VECTOR_BACKEND)
            - Config file: settings.lasagna.toml [rag] vector_backend
        """
        return self._get_knowledge_setting(
            "rag", "vector_backend", default=self._config["vector_backend"]
        )

    @property
    def vector_backend_config(self) -> Dict[str, Any]:
        """Backend-specific configuration.

        For Milvus:
            - db_path: Path to Milvus Lite database
            - uri: Milvus server URI
            - token: Authentication token

        For Pinecone:
            - api_key: Pinecone API key
            - environment: Pinecone environment
            - index_name: Index name

        For Qdrant:
            - url: Qdrant server URL
            - api_key: API key (optional)
            - collection_name: Collection name

        For Weaviate:
            - url: Weaviate server URL
            - api_key: API key (optional)
        """
        return self._get_knowledge_setting(
            "rag", "vector_backend_config", default=self._config["vector_backend_config"]
        )

    @property
    def collection_name(self) -> str:
        """The vector collection this configuration reads and writes.

        One per embedding model, ``rag_documents_v2__<model>__<dim>``, so a
        session on another profile never meets vectors it cannot compare with.
        A name other than ``rag_documents_v2`` set in the settings file, the
        environment or the session is used as given.
        """
        # Four settings from one resolution of the session's config: this is
        # read several times per search.
        self._resolved_once = None
        if self._session_id:
            try:
                from lasagna.providers import get_session_provider

                self._resolved_once = get_session_provider().get_resolved_config(self._session_id)
            except Exception:  # noqa: BLE001 - each lookup then resolves (and logs) itself
                self._resolved_once = None
        try:
            return resolve_collection_name(
                self._get_knowledge_setting(
                    "rag", "collection_name", default=self._config["collection_name"]
                ),
                self.embedding_model,
                self.effective_embedding_dim,
                store=self._config.get("db_path"),
            )
        finally:
            self._resolved_once = None

    @property
    def effective_embedding_dim(self) -> int:
        """The size vectors are stored at: the Matryoshka truncation when set."""
        dim = int(self.embedding_dim)
        truncate = self.embedding_truncate_dim
        return int(truncate) if truncate and int(truncate) < dim else dim

    @property
    def embedding_dim(self) -> int:
        return self._get_knowledge_setting(
            "rag", "embedding_dim", default=self._config["embedding_dim"]
        )

    @property
    def embedding_truncate_dim(self) -> Optional[int]:
        """Matryoshka dimension truncation. None = use full dim."""
        return self._get_knowledge_setting(
            "rag", "embedding_truncate_dim", default=self._config["embedding_truncate_dim"]
        )

    @property
    def metric_type(self) -> str:
        return self._get_knowledge_setting(
            "rag", "metric_type", default=self._config["metric_type"]
        )

    @property
    def index_type(self) -> str:
        return self._get_knowledge_setting("rag", "index_type", default=self._config["index_type"])

    @property
    def index_params(self) -> Dict[str, Any]:
        return self._get_knowledge_setting(
            "rag", "index_params", default=self._config["index_params"]
        )

    @property
    def search_params(self) -> Dict[str, Any]:
        return self._get_knowledge_setting(
            "rag", "search_params", default=self._config["search_params"]
        )

    # =========================================================================
    # Embedding Model Properties
    # =========================================================================
    @property
    def embedding_provider(self) -> str:
        return self._get_knowledge_setting(
            "rag", "embedding_provider", default=self._config["embedding_provider"]
        )

    @property
    def embedding_model(self) -> str:
        return self._get_knowledge_setting(
            "rag", "embedding_model", default=self._config["embedding_model"]
        )

    @property
    def embedding_model_fallback(self) -> str:
        return self._get_knowledge_setting(
            "rag", "embedding_model_fallback", default=self._config["embedding_model_fallback"]
        )

    @property
    def embedding_backend(self) -> str:
        """Embedding backend: 'auto', 'sentence-transformers', or 'flag-embedding'."""
        return self._get_knowledge_setting(
            "rag", "embedding_backend", default=self._config["embedding_backend"]
        )

    @property
    def normalize_embeddings(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "normalize_embeddings", default=self._config["normalize_embeddings"]
        )

    @property
    def batch_size(self) -> int:
        return self._get_knowledge_setting("rag", "batch_size", default=self._config["batch_size"])

    @property
    def use_gpu(self) -> bool:
        # A hardware decision, not a research dimension: an explicit
        # RAG_USE_GPU wins over profile and per-query overrides. Profiles
        # carry use_gpu from the settings file, so as a per-query override it
        # outranked the operator's RAG_USE_GPU=false and moved models onto a
        # GPU backend the operator had turned off.
        from lasagna.core.paths import env_get

        env = (env_get("USE_GPU") or "").strip().lower()
        if env:
            return env in TRUTHY
        return self._get_knowledge_setting("rag", "use_gpu", default=self._config["use_gpu"])

    # =========================================================================
    # Chunking Properties
    # =========================================================================
    @property
    def chunking_strategy(self) -> str:
        return self._get_knowledge_setting(
            "rag", "chunking_strategy", default=self._config["chunking_strategy"]
        )

    @property
    def chunk_size(self) -> int:
        return self._get_knowledge_setting("rag", "chunk_size", default=self._config["chunk_size"])

    @property
    def chunk_overlap(self) -> int:
        return self._get_knowledge_setting(
            "rag", "chunk_overlap", default=self._config["chunk_overlap"]
        )

    @property
    def semantic_threshold(self) -> float:
        return self._get_knowledge_setting(
            "rag", "semantic_threshold", default=self._config["semantic_threshold"]
        )

    @property
    def semantic_breakpoint_method(self) -> str:
        """How the semantic chunker picks breakpoints: 'absolute' or a Kamradt method."""
        return self._get_knowledge_setting(
            "rag",
            "semantic_breakpoint_method",
            default=self._config["semantic_breakpoint_method"],
        )

    @property
    def semantic_breakpoint_amount(self) -> Optional[float]:
        """The breakpoint method's parameter; None uses the method's default."""
        return self._get_knowledge_setting(
            "rag",
            "semantic_breakpoint_amount",
            default=self._config["semantic_breakpoint_amount"],
        )

    @property
    def semantic_buffer_size(self) -> int:
        """Neighbouring sentences embedded with each sentence, on either side."""
        return self._get_knowledge_setting(
            "rag", "semantic_buffer_size", default=self._config["semantic_buffer_size"]
        )

    @property
    def min_chunk_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "min_chunk_size", default=self._config["min_chunk_size"]
        )

    @property
    def max_chunk_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "max_chunk_size", default=self._config["max_chunk_size"]
        )

    @property
    def enable_hierarchical(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_hierarchical", default=self._config["enable_hierarchical"]
        )

    @property
    def parent_chunk_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "parent_chunk_size", default=self._config["parent_chunk_size"]
        )

    @property
    def child_chunk_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "child_chunk_size", default=self._config["child_chunk_size"]
        )

    @property
    def auto_merge_threshold(self) -> float:
        return self._get_knowledge_setting(
            "rag", "auto_merge_threshold", default=self._config["auto_merge_threshold"]
        )

    @property
    def enable_late_chunking(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_late_chunking", default=self._config["enable_late_chunking"]
        )

    @property
    def chunk_unit(self) -> str:
        """What chunk_size and chunk_overlap count: 'chars', 'words' or 'tokens'."""
        return self._get_knowledge_setting("rag", "chunk_unit", default=self._config["chunk_unit"])

    @property
    def chunk_tokenizer(self) -> Optional[str]:
        """Tokenizer for chunk_unit='tokens': a Hugging Face name, or 'tiktoken:<encoding>'."""
        return self._get_knowledge_setting(
            "rag", "chunk_tokenizer", default=self._config["chunk_tokenizer"]
        )

    @property
    def recursive_separators(self) -> List[str]:
        """Separators the recursive chunker tries, coarsest first."""
        return self._get_knowledge_setting(
            "rag", "recursive_separators", default=self._config["recursive_separators"]
        )

    @property
    def recursive_keep_separator(self) -> str:
        """Whether a separator ends the piece before it ('end') or starts the next ('start')."""
        return self._get_knowledge_setting(
            "rag", "recursive_keep_separator", default=self._config["recursive_keep_separator"]
        )

    @property
    def markdown_header_levels(self) -> int:
        """Deepest Markdown header level the markdown_header chunker splits on (1-6)."""
        return self._get_knowledge_setting(
            "rag", "markdown_header_levels", default=self._config["markdown_header_levels"]
        )

    @property
    def markdown_strip_headers(self) -> bool:
        """Leave header lines out of markdown_header chunk text (they are in metadata)."""
        return self._get_knowledge_setting(
            "rag", "markdown_strip_headers", default=self._config["markdown_strip_headers"]
        )

    @property
    def markdown_split_long_sections(self) -> bool:
        """Split Markdown sections longer than chunk_size with the recursive chunker."""
        return self._get_knowledge_setting(
            "rag",
            "markdown_split_long_sections",
            default=self._config["markdown_split_long_sections"],
        )

    @property
    def proposition_model(self) -> Optional[str]:
        """LLM that writes propositions; None uses the configured default."""
        return self._get_knowledge_setting(
            "rag", "proposition_model", default=self._config["proposition_model"]
        )

    @property
    def proposition_passage_size(self) -> int:
        """Words per passage the proposition chunker hands to the LLM."""
        return self._get_knowledge_setting(
            "rag", "proposition_passage_size", default=self._config["proposition_passage_size"]
        )

    @property
    def contextual_base_chunker(self) -> str:
        """Registered chunker whose chunks the contextual chunker situates."""
        return self._get_knowledge_setting(
            "rag", "contextual_base_chunker", default=self._config["contextual_base_chunker"]
        )

    @property
    def contextual_model(self) -> Optional[str]:
        """LLM that writes each chunk's context; None uses the configured default."""
        return self._get_knowledge_setting(
            "rag", "contextual_model", default=self._config["contextual_model"]
        )

    @property
    def contextual_max_document_chars(self) -> Optional[int]:
        """Characters of the document in each context prompt; None for all of it."""
        return self._get_knowledge_setting(
            "rag",
            "contextual_max_document_chars",
            default=self._config["contextual_max_document_chars"],
        )

    @property
    def late_chunking_overlap(self) -> int:
        """Tokens shared by consecutive windows in long late chunking."""
        return self._get_knowledge_setting(
            "rag", "late_chunking_overlap", default=self._config["late_chunking_overlap"]
        )

    @property
    def preserve_tables(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "preserve_tables", default=self._config["preserve_tables"]
        )

    @property
    def preserve_headers(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "preserve_headers", default=self._config["preserve_headers"]
        )

    @property
    def max_content_length(self) -> int:
        """Maximum content length per chunk in storage (characters)."""
        return self._get_knowledge_setting(
            "rag", "max_content_length", default=self._config["max_content_length"]
        )

    @property
    def window_size(self) -> int:
        """Number of sentences per chunk in sentence_window strategy."""
        return self._get_knowledge_setting(
            "rag", "window_size", default=self._config["window_size"]
        )

    @property
    def sentence_overlap(self) -> int:
        """Number of sentences to overlap in sentence_window strategy."""
        return self._get_knowledge_setting(
            "rag", "sentence_overlap", default=self._config["sentence_overlap"]
        )

    # =========================================================================
    # Hybrid Search Properties
    # =========================================================================
    @property
    def enable_hybrid_search(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_hybrid_search", default=self._config["enable_hybrid_search"]
        )

    @property
    def hybrid_alpha(self) -> float:
        return self._get_knowledge_setting(
            "rag", "hybrid_alpha", default=self._config["hybrid_alpha"]
        )

    @property
    def sparse_method(self) -> str:
        return self._get_knowledge_setting(
            "rag", "sparse_method", default=self._config["sparse_method"]
        )

    @property
    def rrf_k(self) -> int:
        """RRF constant for Reciprocal Rank Fusion.

        Higher values give more weight to lower-ranked results.
        Typical range: 20-100, default: 60.
        """
        return self._get_knowledge_setting("rag", "rrf_k", default=self._config["rrf_k"])

    @property
    def enable_bm25_stemming(self) -> bool:
        """Enable Porter stemming for BM25.

        Improves recall by matching word stems (e.g., "running" matches "run").
        """
        return self._get_knowledge_setting(
            "rag", "enable_bm25_stemming", default=self._config["enable_bm25_stemming"]
        )

    @property
    def bm25_max_sessions(self) -> int:
        """Maximum number of in-memory BM25 session indices to keep."""
        return self._get_knowledge_setting(
            "rag", "bm25_max_sessions", default=self._config["bm25_max_sessions"]
        )

    @property
    def bm25_session_ttl_seconds(self) -> int:
        """Idle TTL for BM25 session indices in seconds."""
        return self._get_knowledge_setting(
            "rag", "bm25_session_ttl_seconds", default=self._config["bm25_session_ttl_seconds"]
        )

    @property
    def bm25_k1(self) -> float:
        """BM25 term-frequency saturation for the bm25 and hybrid retrievers."""
        return self._get_knowledge_setting("rag", "bm25_k1", default=self._config["bm25_k1"])

    @property
    def bm25_b(self) -> float:
        """BM25 document-length normalisation for the bm25 and hybrid retrievers."""
        return self._get_knowledge_setting("rag", "bm25_b", default=self._config["bm25_b"])

    @property
    def splade_model(self) -> str:
        """Masked-language-model checkpoint the splade retriever encodes with."""
        return self._get_knowledge_setting(
            "rag", "splade_model", default=self._config["splade_model"]
        )

    @property
    def hybrid_fusion_depth(self) -> int:
        """Candidates each list contributes to hybrid_rrf / hybrid_convex fusion."""
        return self._get_knowledge_setting(
            "rag", "hybrid_fusion_depth", default=self._config["hybrid_fusion_depth"]
        )

    @property
    def hybrid_convex_normalization(self) -> str:
        """Score normalisation for hybrid_convex: 'tmm', 'minmax' or 'zscore'."""
        return self._get_knowledge_setting(
            "rag",
            "hybrid_convex_normalization",
            default=self._config["hybrid_convex_normalization"],
        )

    # =========================================================================
    # Reranking Properties
    # =========================================================================
    @property
    def enable_reranking(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_reranking", default=self._config["enable_reranking"]
        )

    @property
    def reranker_model(self) -> str:
        return self._get_knowledge_setting(
            "rag", "reranker_model", default=self._config["reranker_model"]
        )

    @property
    def custom_reranker(self) -> str:
        """Registry name for a custom reranker component (empty string = use default)."""
        return self._get_knowledge_setting(
            "rag", "custom_reranker", default=self._config["custom_reranker"]
        )

    @property
    def custom_query_enhancer(self) -> str:
        """Registry name for a custom query enhancer component (empty string = use default)."""
        return self._get_knowledge_setting(
            "rag", "custom_query_enhancer", default=self._config["custom_query_enhancer"]
        )

    @property
    def llm_reranker_model(self) -> str:
        """Model for LLM listwise reranker (empty config = configured default)."""
        from lasagna.utils.model_utils import get_default_model_name

        return (
            self._get_knowledge_setting(
                "rag", "llm_reranker_model", default=self._config["llm_reranker_model"]
            )
            or get_default_model_name()
        )

    @property
    def llm_reranker_window_size(self) -> int:
        """Sliding window size for listwise reranking."""
        return self._get_knowledge_setting(
            "rag", "llm_reranker_window_size", default=self._config["llm_reranker_window_size"]
        )

    @property
    def llm_reranker_step_size(self) -> int:
        """How far the listwise reranker's window moves toward the top each step."""
        return self._get_knowledge_setting(
            "rag", "llm_reranker_step_size", default=self._config["llm_reranker_step_size"]
        )

    @property
    def colbert_model(self) -> str:
        """ColBERT late-interaction model name."""
        return self._get_knowledge_setting(
            "rag", "colbert_model", default=self._config["colbert_model"]
        )

    @property
    def monot5_model(self) -> str:
        """Sequence-to-sequence checkpoint the monot5 reranker scores with."""
        return self._get_knowledge_setting(
            "rag", "monot5_model", default=self._config["monot5_model"]
        )

    @property
    def flashrank_model(self) -> str:
        """FlashRank model name for the flashrank reranker."""
        return self._get_knowledge_setting(
            "rag", "flashrank_model", default=self._config["flashrank_model"]
        )

    @property
    def cohere_rerank_model(self) -> str:
        """Cohere Rerank model for the cohere reranker."""
        return self._get_knowledge_setting(
            "rag", "cohere_rerank_model", default=self._config["cohere_rerank_model"]
        )

    @property
    def reranker_trust_remote_code(self) -> bool:
        """Let the cross-encoder run modelling code shipped with its checkpoint."""
        return self._get_knowledge_setting(
            "rag",
            "reranker_trust_remote_code",
            default=self._config["reranker_trust_remote_code"],
        )

    @property
    def rerank_top_k(self) -> int:
        return self._get_knowledge_setting(
            "rag", "rerank_top_k", default=self._config["rerank_top_k"]
        )

    @property
    def enable_context_reordering(self) -> bool:
        """Lost-in-the-middle fix: reorder results so best are at start and end."""
        return self._get_knowledge_setting(
            "rag", "enable_context_reordering", default=self._config["enable_context_reordering"]
        )

    @property
    def enable_graph_retrieval(self) -> bool:
        """Enable GraphRAG entity-linked retrieval."""
        return self._get_knowledge_setting(
            "rag", "enable_graph_retrieval", default=self._config["enable_graph_retrieval"]
        )

    @property
    def graph_max_hops(self) -> int:
        """Max traversal depth in knowledge graph."""
        return self._get_knowledge_setting(
            "rag", "graph_max_hops", default=self._config["graph_max_hops"]
        )

    @property
    def graph_entity_types(self) -> list:
        """Entity types for graph extraction."""
        return self._get_knowledge_setting(
            "rag", "graph_entity_types", default=self._config["graph_entity_types"]
        )

    @property
    def agentic_prompts(self) -> dict:
        """Per-node prompt overrides for the agentic graph, keyed by node name.

        See ``lasagna.agentic.prompts.describe_nodes()`` for the nodes and the
        placeholders each one supplies.
        """
        return self._get_knowledge_setting("rag", "agentic_prompts", default={}) or {}

    @property
    def agentic_models(self) -> dict:
        """Per-node model overrides for the agentic graph, keyed by node name.

        Nodes do different jobs -- a grounding check is a classification, a
        generation is not -- so measuring a small model on the gates against a
        large one on generation is a normal experiment.
        """
        return self._get_knowledge_setting("rag", "agentic_models", default={}) or {}

    @property
    def graph_extraction_model(self) -> str:
        """Model for entity extraction (empty config = configured default)."""
        from lasagna.utils.model_utils import get_default_model_name

        return (
            self._get_knowledge_setting(
                "rag", "graph_extraction_model", default=self._config["graph_extraction_model"]
            )
            or get_default_model_name()
        )

    @property
    def enable_raptor(self) -> bool:
        """Enable RAPTOR recursive summary tree indexing."""
        return self._get_knowledge_setting(
            "rag", "enable_raptor", default=self._config["enable_raptor"]
        )

    @property
    def raptor_max_cluster_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "raptor_max_cluster_size", default=self._config["raptor_max_cluster_size"]
        )

    @property
    def raptor_max_levels(self) -> int:
        return self._get_knowledge_setting(
            "rag", "raptor_max_levels", default=self._config["raptor_max_levels"]
        )

    # raptor_retrieval_mode: removed — no query-time consumer exists.
    # All RAPTOR nodes are searched via standard vector search (collapsed mode).
    # Restore this property when tree-traversal retrieval is wired into the search orchestrator.

    @property
    def enable_context_compression(self) -> bool:
        """Enable context compression to reduce noise/tokens after reranking."""
        return self._get_knowledge_setting(
            "rag", "enable_context_compression", default=self._config["enable_context_compression"]
        )

    @property
    def compression_method(self) -> str:
        """Compression method: 'extractive' or 'abstractive'."""
        return self._get_knowledge_setting(
            "rag", "compression_method", default=self._config["compression_method"]
        )

    @property
    def compression_ratio(self) -> float:
        """Target compression ratio (0.3 = keep 30%)."""
        return self._get_knowledge_setting(
            "rag", "compression_ratio", default=self._config["compression_ratio"]
        )

    @property
    def enable_mmr(self) -> bool:
        return self._get_knowledge_setting("rag", "enable_mmr", default=self._config["enable_mmr"])

    @property
    def mmr_lambda(self) -> float:
        return self._get_knowledge_setting("rag", "mmr_lambda", default=self._config["mmr_lambda"])

    @property
    def mmr_score_normalization(self) -> str:
        """MMR score normalization strategy.

        Options:
            - 'sigmoid': Apply sigmoid to each score (recommended, preserves calibration)
            - 'minmax': Apply min-max scaling
            - 'sigmoid_minmax': Apply sigmoid then minmax (destroys calibration)
            - 'none': No normalization (legacy behavior)
        """
        return self._get_knowledge_setting(
            "rag", "mmr_score_normalization", default=self._config["mmr_score_normalization"]
        )

    @property
    def reranker_batch_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "reranker_batch_size", default=self._config["reranker_batch_size"]
        )

    # =========================================================================
    # Query Enhancement Properties
    # =========================================================================
    @property
    def enable_hyde(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_hyde", default=self._config["enable_hyde"]
        )

    @property
    def hyde_model(self) -> str:
        return self._get_knowledge_setting("rag", "hyde_model", default=self._config["hyde_model"])

    @property
    def hyde_n_docs(self) -> int:
        """Number of hypothetical documents for multi-HyDE.

        Set to 1 for legacy single-document behavior.
        Recommended: 3-5 for better retrieval quality.
        """
        return self._get_knowledge_setting(
            "rag", "hyde_n_docs", default=self._config["hyde_n_docs"]
        )

    @property
    def hyde_embedding_strategy(self) -> str:
        """Strategy for combining multi-HyDE embeddings.

        Options:
            - 'average': Average all embeddings (recommended)
            - 'max_similarity': Use embedding most similar to query
        """
        return self._get_knowledge_setting(
            "rag", "hyde_embedding_strategy", default=self._config["hyde_embedding_strategy"]
        )

    @property
    def enable_query_classification(self) -> bool:
        """Enable adaptive query-type classification.

        When enabled, queries are classified into types (factual, conceptual,
        navigational, comparative) and retrieval parameters are adjusted accordingly.
        """
        return self._get_knowledge_setting(
            "rag",
            "enable_query_classification",
            default=self._config["enable_query_classification"],
        )

    @property
    def enable_query_expansion(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_query_expansion", default=self._config["enable_query_expansion"]
        )

    @property
    def max_query_variations(self) -> int:
        return self._get_knowledge_setting(
            "rag", "max_query_variations", default=self._config["max_query_variations"]
        )

    @property
    def enable_query_decomposition(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_query_decomposition", default=self._config["enable_query_decomposition"]
        )

    @property
    def agentic_hyde_strategy(self) -> str:
        return self._get_knowledge_setting(
            "rag",
            "agentic_hyde_strategy",
            default=self._config.get("agentic_hyde_strategy", "adaptive"),
        )

    @property
    def enable_contextual_retrieval(self) -> bool:
        """Enable Anthropic-style contextual retrieval.

        When enabled, each chunk is enriched with LLM-generated context
        at index time, reducing retrieval failure rates by ~67%.
        Reference: https://www.anthropic.com/news/contextual-retrieval
        """
        return self._get_knowledge_setting(
            "rag",
            "enable_contextual_retrieval",
            default=self._config["enable_contextual_retrieval"],
        )

    @property
    def enable_rag_fusion(self) -> bool:
        """Enable RAG Fusion: multi-query generation + Reciprocal Rank Fusion.

        Generates multiple query variations and merges results using RRF,
        improving comprehensiveness by ~8-10% accuracy.
        """
        return self._get_knowledge_setting(
            "rag", "enable_rag_fusion", default=self._config["enable_rag_fusion"]
        )

    @property
    def rag_fusion_n_queries(self) -> int:
        """Number of query variations to generate for RAG Fusion."""
        return self._get_knowledge_setting(
            "rag", "rag_fusion_n_queries", default=self._config["rag_fusion_n_queries"]
        )

    @property
    def query2doc_format(self) -> str:
        """Query2doc: 'dense' (q [SEP] d') or 'sparse' (q repeated, then d')."""
        return self._get_knowledge_setting(
            "rag", "query2doc_format", default=self._config["query2doc_format"]
        )

    @property
    def query2doc_query_repeats(self) -> int:
        """Query2doc: how many times the query precedes its pseudo-document (sparse form)."""
        return self._get_knowledge_setting(
            "rag", "query2doc_query_repeats", default=self._config["query2doc_query_repeats"]
        )

    @property
    def query2doc_max_tokens(self) -> int:
        """Query2doc: token budget for the generated pseudo-document."""
        return self._get_knowledge_setting(
            "rag", "query2doc_max_tokens", default=self._config["query2doc_max_tokens"]
        )

    @property
    def enable_crag(self) -> bool:
        """Enable Corrective RAG: LLM grades retrieval quality post-search.

        If retrieval quality is poor, the query is refined or the system
        falls back to web search (when crag_web_search_fallback is enabled).
        """
        return self._get_knowledge_setting(
            "rag", "enable_crag", default=self._config["enable_crag"]
        )

    @property
    def crag_web_search_fallback(self) -> bool:
        """Enable web search fallback for Corrective RAG.

        When CRAG determines retrieval quality is low, fall back to
        web search to supplement results.
        """
        return self._get_knowledge_setting(
            "rag", "crag_web_search_fallback", default=self._config["crag_web_search_fallback"]
        )

    # =========================================================================
    # Query Routing Properties (Adaptive-RAG)
    # =========================================================================

    @property
    def enable_query_routing(self) -> bool:
        """Enable intelligent query routing to FAST/STANDARD/COMPREHENSIVE tiers."""
        return self._get_knowledge_setting(
            "rag", "enable_query_routing", default=self._config["enable_query_routing"]
        )

    @property
    def routing_strategy(self) -> str:
        """Routing strategy: 'rule_based' (deterministic) or 'adaptive' (confidence-aware)."""
        return self._get_knowledge_setting(
            "rag", "routing_strategy", default=self._config["routing_strategy"]
        )

    @property
    def routing_default_tier(self) -> str:
        """Default routing tier when routing is disabled."""
        return self._get_knowledge_setting(
            "rag", "routing_default_tier", default=self._config["routing_default_tier"]
        )

    @property
    def routing_confidence_threshold(self) -> float:
        """Confidence threshold below which the adaptive strategy escalates the tier."""
        return self._get_knowledge_setting(
            "rag",
            "routing_confidence_threshold",
            default=self._config["routing_confidence_threshold"],
        )

    @property
    def routing_allow_tier_override(self) -> bool:
        """Allow API callers to force a specific routing tier."""
        return self._get_knowledge_setting(
            "rag",
            "routing_allow_tier_override",
            default=self._config["routing_allow_tier_override"],
        )

    @property
    def routing_log_decisions(self) -> bool:
        """Log routing decisions for observability."""
        return self._get_knowledge_setting(
            "rag", "routing_log_decisions", default=self._config["routing_log_decisions"]
        )

    # =========================================================================
    # Observability Properties
    # =========================================================================
    @property
    def enable_prometheus(self) -> bool:
        """Enable Prometheus metrics collection.

        When True, all existing METRICS.counter() / METRICS.histogram() calls
        throughout the search pipeline emit real Prometheus metrics.
        Requires: pip install prometheus-client>=0.20.0
        """
        return self._get_knowledge_setting(
            "rag", "enable_prometheus", default=self._config["enable_prometheus"]
        )

    @property
    def enable_tracing(self) -> bool:
        """Enable OpenTelemetry distributed tracing.

        When True, search pipeline stages are wrapped in trace spans for
        end-to-end latency visibility (viewable in Jaeger, Zipkin, etc.).
        Requires: pip install opentelemetry-api opentelemetry-sdk
        """
        return self._get_knowledge_setting(
            "rag", "enable_tracing", default=self._config["enable_tracing"]
        )

    @property
    def tracing_exporter(self) -> str:
        """Tracing exporter: 'otlp', 'jaeger', 'zipkin', or 'console'."""
        return self._get_knowledge_setting(
            "rag", "tracing_exporter", default=self._config["tracing_exporter"]
        )

    @property
    def tracing_endpoint(self) -> str:
        """OTLP/Jaeger/Zipkin collector endpoint."""
        return self._get_knowledge_setting(
            "rag", "tracing_endpoint", default=self._config["tracing_endpoint"]
        )

    @property
    def tracing_service_name(self) -> str:
        """Service name reported in traces."""
        return self._get_knowledge_setting(
            "rag", "tracing_service_name", default=self._config["tracing_service_name"]
        )

    # =========================================================================
    # Multi-Tenancy Properties
    # =========================================================================
    @property
    def tenant_isolation(self) -> str:
        """Tenant isolation mode: 'none' or 'partition'.

        - ``none``: Current metadata-filter behaviour (default, single-user).
        - ``partition``: Native backend partitions (Milvus, Pinecone) with
          metadata filter as defense-in-depth.
        """
        return self._get_knowledge_setting(
            "rag", "tenant_isolation", default=self._config["tenant_isolation"]
        )

    @property
    def quota_enabled(self) -> bool:
        """Enable per-tenant resource quotas (documents, storage, rate limits)."""
        return self._get_knowledge_setting(
            "rag", "quota_enabled", default=self._config["quota_enabled"]
        )

    # =========================================================================
    # Rate Limiting Properties
    # =========================================================================
    @property
    def rate_limiting_enabled(self) -> bool:
        """Enable adaptive rate limiting per tenant."""
        return self._get_knowledge_setting(
            "rag", "rate_limiting_enabled", default=self._config["rate_limiting_enabled"]
        )

    @property
    def rate_limiting_strategy(self) -> str:
        """Rate limiting strategy: 'adaptive' or 'fixed'."""
        return self._get_knowledge_setting(
            "rag", "rate_limiting_strategy", default=self._config["rate_limiting_strategy"]
        )

    @property
    def rate_limiting_rpm(self) -> float:
        """Base requests per minute per tenant."""
        return float(
            self._get_knowledge_setting(
                "rag", "rate_limiting_rpm", default=self._config["rate_limiting_rpm"]
            )
        )

    @property
    def rate_limiting_burst(self) -> float:
        """Maximum burst token capacity per tenant."""
        return float(
            self._get_knowledge_setting(
                "rag", "rate_limiting_burst", default=self._config["rate_limiting_burst"]
            )
        )

    @property
    def rate_limiting_burst_multiplier(self) -> float:
        """Capacity multiplier when system is idle (allows bursting)."""
        return float(
            self._get_knowledge_setting(
                "rag",
                "rate_limiting_burst_multiplier",
                default=self._config["rate_limiting_burst_multiplier"],
            )
        )

    @property
    def rate_limiting_overload_threshold(self) -> float:
        """System load score that triggers throttling (0.0–1.0)."""
        return float(
            self._get_knowledge_setting(
                "rag",
                "rate_limiting_overload_threshold",
                default=self._config["rate_limiting_overload_threshold"],
            )
        )

    @property
    def rate_limiting_idle_threshold(self) -> float:
        """System load score below which bursting is allowed (0.0–1.0)."""
        return float(
            self._get_knowledge_setting(
                "rag",
                "rate_limiting_idle_threshold",
                default=self._config["rate_limiting_idle_threshold"],
            )
        )

    # =========================================================================
    # Database Backend Properties
    # =========================================================================
    @property
    def database_backend(self) -> str:
        """Database backend: 'sqlite' (default) or 'postgresql'."""
        return self._get_knowledge_setting(
            "rag", "database_backend", default=self._config["database_backend"]
        )

    @property
    def database_dsn(self) -> str:
        """PostgreSQL connection string (DSN).  Required when database_backend='postgresql'."""
        return self._get_knowledge_setting(
            "rag", "database_dsn", default=self._config["database_dsn"]
        )

    @property
    def database_pool_min(self) -> int:
        """Minimum database connection pool size (PostgreSQL)."""
        return int(
            self._get_knowledge_setting(
                "rag", "database_pool_min", default=self._config["database_pool_min"]
            )
        )

    @property
    def database_pool_max(self) -> int:
        """Maximum database connection pool size (PostgreSQL)."""
        return int(
            self._get_knowledge_setting(
                "rag", "database_pool_max", default=self._config["database_pool_max"]
            )
        )

    # =========================================================================
    # Redis / Distributed Cache Properties
    # =========================================================================
    @property
    def cache_backend(self) -> str:
        """Cache backend: 'memory' (default) or 'redis'."""
        return self._get_knowledge_setting(
            "rag", "cache_backend", default=self._config["cache_backend"]
        )

    @property
    def redis_url(self) -> str:
        """Redis connection URL."""
        return self._get_knowledge_setting("rag", "redis_url", default=self._config["redis_url"])

    @property
    def redis_pool_size(self) -> int:
        """Redis connection pool size."""
        return int(
            self._get_knowledge_setting(
                "rag", "redis_pool_size", default=self._config["redis_pool_size"]
            )
        )

    # =========================================================================
    # Celery / Background Processing Properties
    # =========================================================================
    @property
    def celery_enabled(self) -> bool:
        """Enable Celery for background document processing."""
        return self._get_knowledge_setting(
            "rag", "celery_enabled", default=self._config["celery_enabled"]
        )

    @property
    def celery_broker_url(self) -> str:
        """Celery broker URL (typically Redis)."""
        return self._get_knowledge_setting(
            "rag", "celery_broker_url", default=self._config["celery_broker_url"]
        )

    @property
    def celery_result_backend(self) -> str:
        """Celery result backend URL (typically Redis)."""
        return self._get_knowledge_setting(
            "rag", "celery_result_backend", default=self._config["celery_result_backend"]
        )

    # =========================================================================
    # Connection Pooling Properties
    # =========================================================================
    @property
    def connection_pool_size(self) -> int:
        """Vector backend connection pool size.  1 = no pooling (default)."""
        return int(
            self._get_knowledge_setting(
                "rag", "connection_pool_size", default=self._config["connection_pool_size"]
            )
        )

    # =========================================================================
    # File Storage Backend Properties
    # =========================================================================
    @property
    def storage_backend(self) -> str:
        """File storage backend: 'local' (default), 's3', or 'fsspec'.

        Supported backends:
            - 'local': Local filesystem (zero-config, built-in)
            - 's3': S3-compatible (AWS S3, MinIO, R2, Backblaze, Wasabi, 15+ services)
            - 'fsspec': Universal (GCS, Azure, FTP, SFTP, HDFS, 30+ backends)

        Set via:
            - Environment variable: RAG_STORAGE_BACKEND
            - Config file: settings.lasagna.toml [rag] storage_backend
        """
        return self._get_knowledge_setting(
            "rag", "storage_backend", default=self._config["storage_backend"]
        )

    @property
    def storage_backend_config(self) -> Dict[str, Any]:
        """Backend-specific storage configuration.

        For S3:
            - bucket: S3 bucket name (required)
            - prefix: Key prefix (default: "rag-documents")
            - endpoint_url: S3-compatible endpoint (for MinIO, R2, etc.)
            - region_name: AWS region (default: "us-east-1")

        For fsspec:
            - protocol: Filesystem protocol (e.g. "gcs", "az", "ftp")
            - base_path: Base path within the filesystem
            - storage_options: Dict of fsspec storage options

        Set via:
            - Environment variable: RAG_STORAGE_BACKEND_CONFIG (JSON string)
            - Config file: settings.lasagna.toml [rag] storage_backend_config
        """
        return self._get_knowledge_setting(
            "rag", "storage_backend_config", default=self._config["storage_backend_config"]
        )

    # =========================================================================
    # Logging Properties
    # =========================================================================
    @property
    def log_level(self) -> str:
        value = self._get_knowledge_setting("rag", "log_level", default=self._config["log_level"])
        return str(value).upper()

    # =========================================================================
    # Citation Properties
    # =========================================================================
    @property
    def citation_format(self) -> str:
        return self._get_knowledge_setting(
            "rag", "citation_format", default=self._config["citation_format"]
        )

    @property
    def citation_id_length(self) -> int:
        return self._get_knowledge_setting(
            "rag", "citation_id_length", default=self._config["citation_id_length"]
        )

    @property
    def verify_citations(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "verify_citations", default=self._config["verify_citations"]
        )

    @property
    def track_citations_db(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "track_citations_db", default=self._config["track_citations_db"]
        )

    @property
    def show_page_numbers(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "show_page_numbers", default=self._config["show_page_numbers"]
        )

    @property
    def citation_db_path(self) -> str:
        return self._config["citation_db_path"]

    @property
    def citation_db_max_age_hours(self) -> int:
        return self._get_knowledge_setting(
            "rag", "citation_db_max_age_hours", default=self._config["citation_db_max_age_hours"]
        )

    @property
    def citation_db_max_per_session(self) -> int:
        return self._get_knowledge_setting(
            "rag",
            "citation_db_max_per_session",
            default=self._config["citation_db_max_per_session"],
        )

    @property
    def citation_normalize_scores(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "citation_normalize_scores", default=self._config["citation_normalize_scores"]
        )

    @property
    def citation_score_method(self) -> str:
        return self._get_knowledge_setting(
            "rag", "citation_score_method", default=self._config["citation_score_method"]
        )

    # =========================================================================
    # Search Properties
    # =========================================================================
    @property
    def default_top_k(self) -> int:
        return self._get_knowledge_setting(
            "rag", "default_top_k", default=self._config["default_top_k"]
        )

    @property
    def similarity_threshold(self) -> float:
        return self._get_knowledge_setting(
            "rag", "similarity_threshold", default=self._config["similarity_threshold"]
        )

    @property
    def max_file_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "max_file_size", default=self._config["max_file_size"]
        )

    @property
    def supported_extensions(self) -> List[str]:
        return self._get_knowledge_setting(
            "rag", "supported_extensions", default=self._config["supported_extensions"]
        )

    # =========================================================================
    # Storage Properties
    # =========================================================================
    @property
    def upload_dir(self) -> str:
        return self._config["upload_dir"]

    @property
    def db_path(self) -> str:
        return self._config["db_path"]

    @property
    def max_concurrent_uploads(self) -> int:
        """Maximum number of parallel document uploads (1-20)."""
        return self._get_knowledge_setting(
            "rag", "max_concurrent_uploads", default=self._config["max_concurrent_uploads"]
        )

    # =========================================================================
    # OCR Properties
    # =========================================================================
    @property
    def enable_ocr(self) -> bool:
        return self._get_knowledge_setting("rag", "enable_ocr", default=self._config["enable_ocr"])

    @property
    def ocr_engine(self) -> str:
        return self._get_knowledge_setting("rag", "ocr_engine", default=self._config["ocr_engine"])

    @property
    def ocr_languages(self) -> List[str]:
        return self._get_knowledge_setting(
            "rag", "ocr_languages", default=self._config["ocr_languages"]
        )

    @property
    def ocr_confidence_threshold(self) -> float:
        return self._get_knowledge_setting(
            "rag", "ocr_confidence_threshold", default=self._config["ocr_confidence_threshold"]
        )

    # =========================================================================
    # Processing Properties
    # =========================================================================
    @property
    def extract_images(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "extract_images", default=self._config["extract_images"]
        )

    @property
    def preserve_formatting(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "preserve_formatting", default=self._config["preserve_formatting"]
        )

    @property
    def async_processing(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "async_processing", default=self._config["async_processing"]
        )

    # =========================================================================
    # Docling Properties
    # =========================================================================
    @property
    def docling_enabled(self) -> bool:
        """Enable Docling for structure-aware document processing."""
        return self._get_knowledge_setting(
            "rag", "docling_enabled", default=self._config["docling_enabled"]
        )

    @property
    def docling_mode(self) -> str:
        """Docling deployment mode: 'auto', 'local', or 'production'."""
        return self._get_knowledge_setting(
            "rag", "docling_mode", default=self._config["docling_mode"]
        )

    @property
    def docling_pdf_backend(self) -> str:
        """Docling PDF backend: 'docling_parse' or 'pypdfium2'."""
        return self._get_knowledge_setting(
            "rag", "docling_pdf_backend", default=self._config["docling_pdf_backend"]
        )

    @property
    def docling_ocr_enabled(self) -> bool:
        """Enable OCR within Docling pipeline."""
        return self._get_knowledge_setting(
            "rag", "docling_ocr_enabled", default=self._config["docling_ocr_enabled"]
        )

    @property
    def docling_table_structure(self) -> bool:
        """Enable TableFormer table structure extraction."""
        return self._get_knowledge_setting(
            "rag", "docling_table_structure", default=self._config["docling_table_structure"]
        )

    @property
    def docling_max_tokens(self) -> int:
        """Max tokens per chunk for Docling HybridChunker."""
        return int(
            self._get_knowledge_setting(
                "rag", "docling_max_tokens", default=self._config["docling_max_tokens"]
            )
        )

    @property
    def docling_timeout(self) -> int:
        """Per-document processing timeout in seconds."""
        return int(
            self._get_knowledge_setting(
                "rag", "docling_timeout", default=self._config["docling_timeout"]
            )
        )

    # =========================================================================
    # Caching Properties
    # =========================================================================
    @property
    def enable_embedding_cache(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_embedding_cache", default=self._config["enable_embedding_cache"]
        )

    @property
    def embedding_cache_size(self) -> int:
        return self._get_knowledge_setting(
            "rag", "embedding_cache_size", default=self._config["embedding_cache_size"]
        )

    @property
    def allow_model_download_in_requests(self) -> bool:
        """Whether a server may download a missing model while it handles a request."""
        return self._get_knowledge_setting(
            "rag",
            "allow_model_download_in_requests",
            default=self._config["allow_model_download_in_requests"],
        )

    @property
    def enable_search_cache(self) -> bool:
        return self._get_knowledge_setting(
            "rag", "enable_search_cache", default=self._config["enable_search_cache"]
        )

    @property
    def search_cache_ttl(self) -> int:
        return self._get_knowledge_setting(
            "rag", "search_cache_ttl", default=self._config["search_cache_ttl"]
        )

    @property
    def search_cache_max_size(self) -> int:
        """Maximum number of cached search results."""
        return self._get_knowledge_setting(
            "rag", "search_cache_max_size", default=self._config["search_cache_max_size"]
        )

    @property
    def max_document_ids_filter(self) -> int:
        """Maximum document IDs allowed in filter expressions.

        Prevents Milvus filter expression overflow when
        too many document IDs are passed, which can cause query failures.
        """
        return self._get_knowledge_setting(
            "rag", "max_document_ids_filter", default=self._config["max_document_ids_filter"]
        )

    # =========================================================================
    # Agentic RAG Properties (existing)
    # =========================================================================
    @property
    def use_agentic_rag(self) -> bool:
        return self._get_knowledge_setting("rag", "use_agentic_rag", default=True)

    @property
    def use_final_search(self) -> bool:
        return self._get_knowledge_setting("rag", "use_final_search", default=False)

    @property
    def chunk_cite_retrieval(self) -> Dict[str, Any]:
        """How many retrieved chunks are kept: a fixed count, or cut at a score gap."""
        ccr = self._get_knowledge_setting("rag", "chunk_cite_retrieval", default=None)
        if ccr:
            if hasattr(ccr, "to_dict"):
                return ccr.to_dict()
            return ccr
        return {
            "strategy": "adaptive",
            "fixed_size": 5,
            "threshold_score": 0.5,
            "max_chunks": 20,
            "min_chunks": 2,
            "score_gap_buffer": 2,
            "min_gap_ratio": 0.15,
        }

    @property
    def chunk_refinement_method(self) -> str:
        return self._get_knowledge_setting("rag", "chunk_refinement_method", default="auto")

    @property
    def refinement_model(self) -> str:
        from lasagna.utils.model_utils import get_default_model_name

        return self._get_knowledge_setting(
            "rag", "refinement_model", default=get_default_model_name()
        )

    @property
    def max_agent_iterations(self) -> int:
        """Maximum number of search iterations for agentic RAG."""
        return self._get_knowledge_setting("rag", "max_agent_iterations", default=3)

    @property
    def agent_temperature(self) -> float:
        """Temperature for agent/refinement model (low for determinism)."""
        return self._get_knowledge_setting("rag", "agent_temperature", default=0.1)

    @property
    def max_query_length(self) -> int:
        """Maximum query length in tokens for refinement queries."""
        return self._get_knowledge_setting("rag", "max_query_length", default=12)

    @property
    def auto_refine_avg_length_threshold(self) -> int:
        """Average chunk length threshold to trigger auto-refinement."""
        return self._get_knowledge_setting("rag", "auto_refine_avg_length_threshold", default=1000)

    @property
    def auto_refine_avg_score_threshold(self) -> float:
        """Average score threshold below which to trigger auto-refinement."""
        return self._get_knowledge_setting("rag", "auto_refine_avg_score_threshold", default=0.5)

    @property
    def auto_refine_chunk_count_threshold(self) -> int:
        """Chunk count threshold above which to trigger auto-refinement."""
        return self._get_knowledge_setting("rag", "auto_refine_chunk_count_threshold", default=7)

    # =========================================================================
    # Structured Graph Properties (Phase 1d)
    # =========================================================================
    @property
    def use_structured_graph(self) -> bool:
        """Use structured StateGraph instead of flat ReAct agent."""
        return self._get_knowledge_setting("rag", "use_structured_graph", default=False)

    @property
    def agentic_pipeline(self) -> str:
        """Name of the registered agentic pipeline to use.

        Precedence (highest to lowest):
          1. Session override
          2. TOML / env var ``RAG_AGENTIC_PIPELINE``
          3. Backward compat: ``"default"`` when ``use_structured_graph=True``
          4. ``"legacy_react"`` (default)
        """
        override = self._get_session_override("rag", "agentic_pipeline")
        if override:
            return str(override)
        val = self._get_knowledge_setting("rag", "agentic_pipeline", default=None)
        if val:
            return str(val)
        if self.use_structured_graph:
            return "default"
        return "legacy_react"

    @property
    def enable_hallucination_check(self) -> bool:
        """Enable post-generation hallucination grounding check (ISSUP)."""
        return self._get_knowledge_setting("rag", "enable_hallucination_check", default=False)

    @property
    def enable_answer_quality_check(self) -> bool:
        """Enable post-generation answer quality check (ISUSE)."""
        return self._get_knowledge_setting("rag", "enable_answer_quality_check", default=False)

    @property
    def max_quality_check_attempts(self) -> int:
        """Maximum retry attempts for quality gate failures."""
        return self._get_knowledge_setting("rag", "max_quality_check_attempts", default=2)

    @property
    def enable_speculative_rag(self) -> bool:
        """Enable speculative RAG: parallel draft generation + verification."""
        return self._get_knowledge_setting("rag", "enable_speculative_rag", default=False)

    @property
    def speculative_n_drafts(self) -> int:
        """Number of parallel draft answers for speculative RAG."""
        return self._get_knowledge_setting("rag", "speculative_n_drafts", default=3)

    @property
    def speculative_subset_size(self) -> int:
        """Number of context chunks per draft in speculative RAG."""
        return self._get_knowledge_setting("rag", "speculative_subset_size", default=3)

    @property
    def self_ask_max_follow_ups(self) -> int:
        """Self-Ask: follow-up questions searched before the model must conclude."""
        return self._get_knowledge_setting(
            "rag", "self_ask_max_follow_ups", default=self._config["self_ask_max_follow_ups"]
        )

    @property
    def ircot_paragraphs_per_step(self) -> int:
        """IRCoT: paragraphs retrieved per step (K)."""
        return self._get_knowledge_setting(
            "rag", "ircot_paragraphs_per_step", default=self._config["ircot_paragraphs_per_step"]
        )

    @property
    def ircot_max_steps(self) -> int:
        """IRCoT: reasoning steps before the loop stops."""
        return self._get_knowledge_setting(
            "rag", "ircot_max_steps", default=self._config["ircot_max_steps"]
        )

    @property
    def ircot_max_paragraphs(self) -> int:
        """IRCoT: cap on the paragraphs collected across steps."""
        return self._get_knowledge_setting(
            "rag", "ircot_max_paragraphs", default=self._config["ircot_max_paragraphs"]
        )

    @property
    def flare_retrieval_threshold(self) -> float:
        """FLARE theta: retrieve when any token of the next sentence is below it."""
        return self._get_knowledge_setting(
            "rag", "flare_retrieval_threshold", default=self._config["flare_retrieval_threshold"]
        )

    @property
    def flare_mask_threshold(self) -> float:
        """FLARE beta: tokens below it are masked out of the retrieval query."""
        return self._get_knowledge_setting(
            "rag", "flare_mask_threshold", default=self._config["flare_mask_threshold"]
        )

    @property
    def flare_max_sentences(self) -> int:
        """FLARE: sentences generated before the loop stops."""
        return self._get_knowledge_setting(
            "rag", "flare_max_sentences", default=self._config["flare_max_sentences"]
        )

    @property
    def flare_docs_per_retrieval(self) -> int:
        """FLARE: documents retrieved per retrieval."""
        return self._get_knowledge_setting(
            "rag", "flare_docs_per_retrieval", default=self._config["flare_docs_per_retrieval"]
        )

    @property
    def flare_confidence_source(self) -> str:
        """FLARE: 'auto', 'logprobs' or 'self_reported' confidence."""
        return self._get_knowledge_setting(
            "rag", "flare_confidence_source", default=self._config["flare_confidence_source"]
        )

    @property
    def adaptive_rag_classifier_model(self) -> str:
        """Adaptive-RAG: model for the complexity classifier (empty = the loop's)."""
        return self._get_knowledge_setting(
            "rag",
            "adaptive_rag_classifier_model",
            default=self._config["adaptive_rag_classifier_model"],
        )

    @property
    def adaptive_rag_single_step_docs(self) -> int:
        """Adaptive-RAG: documents retrieved on the single-step route."""
        return self._get_knowledge_setting(
            "rag",
            "adaptive_rag_single_step_docs",
            default=self._config["adaptive_rag_single_step_docs"],
        )

    @property
    def self_rag_passages(self) -> int:
        """Self-RAG: passages retrieved, each answered from and critiqued."""
        return self._get_knowledge_setting(
            "rag", "self_rag_passages", default=self._config["self_rag_passages"]
        )

    @property
    def self_rag_weight_rel(self) -> float:
        """Self-RAG: weight of the relevance critique (ISREL)."""
        return self._get_knowledge_setting(
            "rag", "self_rag_weight_rel", default=self._config["self_rag_weight_rel"]
        )

    @property
    def self_rag_weight_sup(self) -> float:
        """Self-RAG: weight of the support critique (ISSUP)."""
        return self._get_knowledge_setting(
            "rag", "self_rag_weight_sup", default=self._config["self_rag_weight_sup"]
        )

    @property
    def self_rag_weight_use(self) -> float:
        """Self-RAG: weight of the utility critique (ISUSE)."""
        return self._get_knowledge_setting(
            "rag", "self_rag_weight_use", default=self._config["self_rag_weight_use"]
        )

    @property
    def self_rag_retrieval(self) -> str:
        """Self-RAG: 'adaptive' (the model judges) or 'always' retrieve."""
        return self._get_knowledge_setting(
            "rag", "self_rag_retrieval", default=self._config["self_rag_retrieval"]
        )

    @property
    def coverage_scorer_mode(self) -> str:
        """Coverage assessment mode: 'keyword' (fast) or 'llm' (accurate)."""
        return self._get_knowledge_setting("rag", "coverage_scorer_mode", default="keyword")

    @property
    def final_search_mode(self) -> str:
        """Final search behavior: 'merge' (combine hop + final) or 'replace' (legacy)."""
        return self._get_knowledge_setting("rag", "final_search_mode", default="merge")

    # =========================================================================
    # Document Awareness Properties
    # =========================================================================
    @property
    def document_awareness_enabled(self) -> bool:
        """Whether document awareness is enabled for agent prompts."""
        return self._get_knowledge_setting("rag", "document_awareness", "enabled", default=True)

    @property
    def enrichment_strategy(self) -> str:
        """Document enrichment strategy: first_tokens, llm_summary, or hybrid."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "enrichment_strategy", default="first_tokens"
        )

    @property
    def first_tokens_count(self) -> int:
        """Number of tokens to extract from document start."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "first_tokens_count", default=150
        )

    @property
    def include_filename(self) -> bool:
        """Include filename in document snippet."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "include_filename", default=True
        )

    @property
    def include_upload_date(self) -> bool:
        """Include upload date in document snippet."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "include_upload_date", default=True
        )

    @property
    def include_file_type(self) -> bool:
        """Include file type in document snippet."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "include_file_type", default=True
        )

    @property
    def summary_model(self) -> str:
        """Model for generating LLM summaries."""
        from lasagna.utils.model_utils import get_default_model_name

        return self._get_knowledge_setting(
            "rag", "document_awareness", "summary_model", default=get_default_model_name()
        )

    @property
    def summary_max_tokens(self) -> int:
        """Max tokens for document summaries."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "summary_max_tokens", default=100
        )

    @property
    def summary_prompt(self) -> str:
        """Prompt for generating document summaries."""
        default = "Summarize this document in 1-2 sentences, focusing on the main topic and key information:"
        return self._get_knowledge_setting(
            "rag", "document_awareness", "summary_prompt", default=default
        )

    @property
    def hybrid_llm_threshold_kb(self) -> int:
        """Documents larger than this (KB) get LLM summary in hybrid mode."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "hybrid_llm_threshold_kb", default=100
        )

    @property
    def max_documents_in_context(self) -> int:
        """Maximum documents to include in agent context."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "max_documents_in_context", default=10
        )

    @property
    def cache_snippets(self) -> bool:
        """Whether to cache document snippets."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "cache_snippets", default=True
        )

    @property
    def snippet_cache_ttl(self) -> int:
        """Cache lifetime for snippets in seconds."""
        return self._get_knowledge_setting(
            "rag", "document_awareness", "snippet_cache_ttl", default=3600
        )

    # =========================================================================
    # Fast Ingestion
    # =========================================================================
    @property
    def ingest_extract_workers(self) -> int:
        """Number of parallel workers for text extraction/chunking in fast ingest."""
        return int(
            self._get_knowledge_setting(
                "rag", "ingest_extract_workers", default=self._config["ingest_extract_workers"]
            )
        )

    @property
    def ingest_embed_batch_size(self) -> int:
        """Cross-document embedding batch size for fast ingest."""
        return int(
            self._get_knowledge_setting(
                "rag", "ingest_embed_batch_size", default=self._config["ingest_embed_batch_size"]
            )
        )

    @property
    def ingest_insert_batch_size(self) -> int:
        """Chunks per vector store insert call in fast ingest."""
        return int(
            self._get_knowledge_setting(
                "rag", "ingest_insert_batch_size", default=self._config["ingest_insert_batch_size"]
            )
        )

    # =========================================================================
    # Utility Methods
    # =========================================================================
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self._config.copy()

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        return self._config.get(key, default)

    def set_override(self, key: str, value: Any) -> None:
        """Set a runtime config override (same priority as RAG_* env vars).

        Writes into the env-override layer (priority 2 in
        ``_get_knowledge_setting``), so it beats TOML and defaults but is
        overridden by explicit session overrides.

        Note: ``with_session()`` deep-copies ``_env_overrides``, so this
        override is visible to session copies created *after* this call
        but does not propagate to copies that already exist.

        Args:
            key: Config key (e.g., ``'vector_backend'``, ``'enable_reranking'``)
            value: Config value
        """
        self._env_overrides[key] = value

    def clear_overrides(self) -> None:
        """Remove all runtime overrides, reload from ``RAG_*`` env vars only.

        Uses ``_config`` values as the defaults dict for ``_load_env_overrides``
        so that key-validation and type-coercion match the original load.
        This is safe because ``_config`` is never mutated after ``__init__``.
        """
        default_keys = {
            k for k in self._config if k not in ("upload_dir", "db_path", "citation_db_path")
        }
        self._env_overrides = self._load_env_overrides({k: self._config[k] for k in default_keys})


# ---------------------------------------------------------------------------
# Thread-safe lazy singleton
# ---------------------------------------------------------------------------
_rag_config: Optional[RAGConfig] = None
_rag_config_lock = Lock()


def get_rag_config() -> RAGConfig:
    """Get or create the global RAGConfig singleton.

    Uses double-check locking for thread-safe lazy initialization.
    Prefer this over the module-level ``rag_config`` name in new code.
    """
    global _rag_config
    if _rag_config is None:
        with _rag_config_lock:
            if _rag_config is None:
                _rag_config = RAGConfig()
    return _rag_config


def _reset_rag_config() -> None:
    """Reset the global RAGConfig singleton (for testing only)."""
    global _rag_config
    with _rag_config_lock:
        _rag_config = None


def reset_rag_config() -> None:
    """Reset the global RAGConfig singleton.

    After calling this, the next access to ``rag_config`` (or call to
    ``get_rag_config()``) will create a fresh instance that re-reads
    environment variables, TOML settings, and provider configuration.
    """
    _reset_rag_config()


def __getattr__(name: str):
    """Module-level __getattr__ for backward-compatible ``rag_config`` access.

    ``from lasagna.core.config import rag_config`` triggers
    this on first access, lazily creating the singleton.
    """
    if name == "rag_config":
        return get_rag_config()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
