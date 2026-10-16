"""Replay the LLM responses a sweep has already paid for.

Re-running a sweep asks every question it asked last time. At temperature 0 the
answer is the same request answered again, so a second run spent the same
tokens and the same hours to learn nothing new. This module keeps each response
on disk, in one SQLite file, and serves it the next time the exact same request
is made.

**Only deterministic requests are replayed** (``deterministic_only``, the
default). A response sampled at temperature 0.7 is one draw from a
distribution; replaying it would make repeated runs identical and their spread
zero, which reads as a stable system when it is a frozen sample. A call is
cached only when it asked for temperature 0. RAGAS asks its judge for 0.01, so
judge calls are not cached unless ``llm_cache: {deterministic_only: false}``.

**The run seed is part of the key.** Run *k* of an arm replays run *k*'s
responses, never run 0's, so ``n_runs > 1`` measures warm what it measured cold.

**Recorded tokens are those of the cold run.** A hit replays the response with
its usage, so token counts and LLM calls read the same cold or warm. What was
replayed is counted separately, as ``llm_cache_hits`` and ``llm_cache_misses``
on each task and a hit rate for the run, so a reader can tell.

**Active only while an experiment runs.** Every chat model carries the cache
below, but outside :func:`llm_cache_scope` it defers to LangChain's own global
cache (normally none). ``lasagna serve`` therefore never writes prompts, which
hold users' documents, to disk.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import json
import os
import sqlite3
import threading
import time
from contextlib import closing, contextmanager
from dataclasses import dataclass, field, replace
from typing import Any, Dict, Iterator, List, Optional, Set

from loguru import logger

#: Stored format. A row written in another format reads as a miss.
_FORMAT = 1

#: Values of ``LASAGNA_LLM_CACHE`` that turn the cache off.
_OFF = frozenset({"0", "false", "no", "off"})


def default_cache_path() -> str:
    """``<data dir>/cache/llm.sqlite`` (``~/.lasagna`` unless ``LASAGNA_DATA_DIR``
    says otherwise): the cache follows the data directory, beside
    ``research/results``."""
    from lasagna.core.paths import cache_dir

    return str(cache_dir() / "llm.sqlite")


@dataclass(frozen=True)
class LLMCacheSettings:
    """The sweep's ``llm_cache:`` key: ``true``, ``false`` or a mapping."""

    enabled: bool = True
    #: Empty means :func:`default_cache_path`.
    path: str = ""
    deterministic_only: bool = True

    @classmethod
    def parse(cls, value: Any) -> "LLMCacheSettings":
        """Read ``llm_cache:`` as written in a sweep file (absent means on)."""
        if isinstance(value, cls):
            return value
        if value is None:
            return cls()
        if isinstance(value, bool):
            return cls(enabled=value)
        if isinstance(value, dict):
            unknown = sorted(set(value) - {"enabled", "path", "deterministic_only"})
            if unknown:
                raise ValueError(
                    f"Unknown key(s) under `llm_cache:`: {unknown}. "
                    f"Known: enabled, path, deterministic_only."
                )
            for key in ("enabled", "deterministic_only"):
                if key in value and not isinstance(value[key], bool):
                    raise ValueError(f"`llm_cache.{key}` must be true or false, got {value[key]!r}")
            path = value.get("path") or ""
            if not isinstance(path, str):
                raise ValueError(f"`llm_cache.path` must be a file path, got {path!r}")
            return cls(
                enabled=value.get("enabled", True),
                path=path,
                deterministic_only=value.get("deterministic_only", True),
            )
        raise ValueError(
            f"`llm_cache:` must be true, false or a mapping "
            f"({{path: ..., deterministic_only: true}}), got {value!r}"
        )

    def resolved(self) -> "LLMCacheSettings":
        """These settings after ``LASAGNA_LLM_CACHE=0``, which turns the cache off."""
        if os.environ.get("LASAGNA_LLM_CACHE", "").strip().lower() in _OFF:
            return replace(self, enabled=False)
        return self

    def record(self, stats: "CacheStats") -> Dict[str, Any]:
        """What the run's metadata says about the cache.

        No path: bundles are published, and a path names the machine and the
        user it ran under.
        """
        record: Dict[str, Any] = {
            "enabled": self.enabled,
            "deterministic_only": self.deterministic_only,
            "hits": stats.hits,
            "misses": stats.misses,
            "hit_rate": stats.hit_rate,
        }
        if self.enabled and stats.not_applied:
            # Hits and misses of 0 would read as an empty cache.
            record["applied"] = False
            record["not_applied"] = "; ".join(sorted(stats.not_applied))
        return record


@dataclass
class CacheStats:
    """Hits and misses, for one task or a whole run."""

    hits: int = 0
    misses: int = 0
    #: Why some chat model could not carry the cache (see `note_not_applied`).
    not_applied: Set[str] = field(default_factory=set)

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> Optional[float]:
        """None when nothing was looked up: no rate, rather than a rate of 0."""
        return self.hits / self.lookups if self.lookups else None

    def add(self, other: "CacheStats") -> None:
        self.hits += other.hits
        self.misses += other.misses
        self.not_applied |= other.not_applied


class LLMCache:
    """Responses in one SQLite file, keyed by a digest of the request.

    Every operation opens its own connection: the harness reads and writes from
    worker threads, several tasks at once, and a connection per operation
    cannot be shared by two of them. WAL lets readers run beside the writer.
    The file is created on the first write, so a run that makes no LLM call
    leaves nothing behind.

    A storage error never fails the call it caches: a broken cache costs a
    miss, not a run.
    """

    def __init__(self, path: str = "") -> None:
        self.path = os.path.expanduser(path or default_cache_path())
        self._ready = False
        self._lock = threading.Lock()
        self._warned = False

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=30)

    def _ensure(self) -> None:
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with closing(self._connect()) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                with conn:
                    conn.execute(
                        "CREATE TABLE IF NOT EXISTS responses ("
                        "key TEXT PRIMARY KEY, value TEXT NOT NULL, created_at REAL NOT NULL)"
                    )
            self._ready = True

    def _failed(self, what: str, exc: Exception) -> None:
        if not self._warned:
            self._warned = True
            logger.warning(
                f"LLM cache {what} failed ({type(exc).__name__}: {exc}); calls go through"
            )

    def get(self, key: str) -> Optional[str]:
        if not self._ready and not os.path.exists(self.path):
            return None
        try:
            self._ensure()
            with closing(self._connect()) as conn:
                row = conn.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            self._failed("read", exc)
            return None
        return row[0] if row else None

    def put(self, key: str, value: str) -> None:
        try:
            self._ensure()
            with closing(self._connect()) as conn, conn:
                conn.execute(
                    "INSERT OR REPLACE INTO responses (key, value, created_at) VALUES (?, ?, ?)",
                    (key, value, time.time()),
                )
        except Exception as exc:  # noqa: BLE001 - see the class docstring
            self._failed("write", exc)

    def __len__(self) -> int:
        if not os.path.exists(self.path):
            return 0
        self._ensure()
        with closing(self._connect()) as conn:
            return int(conn.execute("SELECT COUNT(*) FROM responses").fetchone()[0])

    def clear(self) -> None:
        if not os.path.exists(self.path):
            return
        self._ensure()
        with closing(self._connect()) as conn, conn:
            conn.execute("DELETE FROM responses")


def request_key(platform: str, seed: Optional[int], llm_string: str, prompt: str) -> str:
    """sha256 over everything that decides the response."""
    blob = json.dumps([platform, seed, llm_string, prompt], separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()


# ---------------------------------------------------------------------------
# The scope: which cache is active, under which seed, and what it served
# ---------------------------------------------------------------------------


class CacheScope(CacheStats):
    """One task's view of the cache: the store, its seed, and its counts."""

    def __init__(self, store: Optional[LLMCache], seed: Optional[int], deterministic_only: bool):
        super().__init__()
        self.store = store
        self.seed = seed
        self.deterministic_only = deterministic_only
        self._lock = threading.Lock()

    def wants(self, deterministic: bool) -> bool:
        return self.store is not None and (deterministic or not self.deterministic_only)

    def count(self, hit: bool) -> None:
        with self._lock:
            if hit:
                self.hits += 1
            else:
                self.misses += 1


_active: contextvars.ContextVar[Optional[CacheScope]] = contextvars.ContextVar(
    "lasagna_llm_cache", default=None
)

#: One store per path, so concurrent tasks share one lock and one schema check.
_stores: Dict[str, LLMCache] = {}
_stores_lock = threading.Lock()


def _store_for(path: str) -> LLMCache:
    resolved = os.path.expanduser(path or default_cache_path())
    with _stores_lock:
        if resolved not in _stores:
            _stores[resolved] = LLMCache(resolved)
        return _stores[resolved]


@contextmanager
def llm_cache_scope(settings: LLMCacheSettings, seed: Optional[int]) -> Iterator[CacheScope]:
    """Serve and store responses for everything run inside, under ``seed``.

    A scope with the cache off still matters: inside it, a LangChain global
    cache an application set up is not consulted either, so ``--no-cache``
    means no replay at all.
    """
    store = _store_for(settings.path) if settings.enabled else None
    scope = CacheScope(store, seed, settings.deterministic_only)
    token = _active.set(scope)
    try:
        yield scope
    finally:
        _active.reset(token)


def active_scope() -> Optional[CacheScope]:
    return _active.get()


# ---------------------------------------------------------------------------
# LangChain chat models
# ---------------------------------------------------------------------------


def _dump_generations(generations: Any) -> Optional[str]:
    """Generations as JSON, usage included; None when one cannot be stored."""
    from langchain_core.messages import message_to_dict

    out: List[Dict[str, Any]] = []
    for gen in generations or []:
        message = getattr(gen, "message", None)
        if message is not None:
            # The message id names the run that made the call; a replay is not
            # that run.
            message = message.model_copy(update={"id": None})
            out.append({"message": message_to_dict(message), "info": gen.generation_info})
        else:
            out.append({"text": getattr(gen, "text", ""), "info": gen.generation_info})
    try:
        return json.dumps({"format": _FORMAT, "generations": out})
    except (TypeError, ValueError):
        return None


def _load_generations(raw: str) -> Optional[List[Any]]:
    from langchain_core.messages import messages_from_dict
    from langchain_core.outputs import ChatGeneration, Generation

    try:
        data = json.loads(raw)
        if data.get("format") != _FORMAT:
            return None
        gens: List[Any] = []
        for item in data["generations"]:
            if "message" in item:
                message = messages_from_dict([item["message"]])[0]
                gens.append(ChatGeneration(message=message, generation_info=item.get("info")))
            else:
                gens.append(Generation(text=item["text"], generation_info=item.get("info")))
        return gens or None
    except Exception:  # noqa: BLE001 - an unreadable row is a miss
        return None


_model_cache_class: Any = None


def _model_cache_type() -> Any:
    """Built lazily, so importing this module does not import langchain_core."""
    global _model_cache_class
    if _model_cache_class is not None:
        return _model_cache_class

    from langchain_core.caches import BaseCache
    from langchain_core.globals import get_llm_cache

    class _ModelCache(BaseCache):
        """The cache one chat model carries. It decides per call, from the
        active scope, whether to serve, store, or stay out of the way."""

        def __init__(self, platform: str, deterministic: bool) -> None:
            self.platform = platform
            self.deterministic = deterministic

        def _global(self) -> Any:
            cache = get_llm_cache()
            return None if cache is None or cache is self else cache

        def lookup(self, prompt: str, llm_string: str) -> Any:
            scope = _active.get()
            if scope is None:
                fallback = self._global()
                return fallback.lookup(prompt, llm_string) if fallback is not None else None
            if not scope.wants(self.deterministic):
                return None
            raw = scope.store.get(request_key(self.platform, scope.seed, llm_string, prompt))
            generations = _load_generations(raw) if raw is not None else None
            scope.count(hit=generations is not None)
            return generations

        def update(self, prompt: str, llm_string: str, return_val: Any) -> None:
            scope = _active.get()
            if scope is None:
                fallback = self._global()
                if fallback is not None:
                    fallback.update(prompt, llm_string, return_val)
                return
            if not scope.wants(self.deterministic):
                return
            value = _dump_generations(return_val)
            if value is not None:
                scope.store.put(request_key(self.platform, scope.seed, llm_string, prompt), value)

        def clear(self, **kwargs: Any) -> None:
            scope = _active.get()
            if scope is not None and scope.store is not None:
                scope.store.clear()

    _model_cache_class = _ModelCache
    return _ModelCache


def note_not_applied(reason: str) -> None:
    """Record, in the active scope, a chat model that cannot carry the cache."""
    scope = _active.get()
    if scope is not None and scope.store is not None:
        with scope._lock:
            scope.not_applied.add(reason)


def carries_llm_cache(model: Any) -> bool:
    """Whether ``model`` already has the experiment cache attached."""
    return _model_cache_class is not None and isinstance(
        getattr(model, "cache", None), _model_cache_class
    )


def attach_llm_cache(model: Any, platform: str, deterministic: bool) -> Any:
    """Give a chat model the experiment cache, in place, and return it.

    ``deterministic`` is whether the model was built to decode greedily
    (temperature 0). Failure is non-fatal: a model that cannot carry a cache is
    a slower run, not a crash at construction time.
    """
    try:
        model.cache = _model_cache_type()(platform, deterministic)
    except Exception as exc:  # noqa: BLE001 - see docstring
        logger.debug(f"Could not attach the LLM cache to {type(model).__name__}: {exc}")
    return model


# ---------------------------------------------------------------------------
# OpenAI SDK (the RAGAS judge is a raw client, not a LangChain model)
# ---------------------------------------------------------------------------

#: Marks a client whose completions are already cached, so caching the same
#: client twice adds one layer, not two.
_CACHED_MARKER = "_lasagna_llm_cached"

#: Request arguments that change how a call is made, not what it returns.
_TRANSPORT_KWARGS = frozenset({"timeout", "extra_headers", "extra_query"})


def _jsonable(value: Any) -> Any:
    schema = getattr(value, "model_json_schema", None)
    if isinstance(value, type) and callable(schema):
        return {"schema": schema()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump()
    # An object without a stable form keys by its repr; if that holds an
    # address, the call is never served from the cache, which is the safe way
    # to be wrong.
    return repr(value)


def cache_openai_client(client: Any) -> Any:
    """Serve an OpenAI client's ``chat.completions.create`` from the cache.

    Patched in place, like :func:`~lasagna.utils.token_usage.instrument_openai_client`
    and for the same reason: RAGAS hands the client to instructor, which
    dispatches on ``isinstance(client, OpenAI)``. Apply this first and the
    meter second, so a replayed response is still counted at its cold-run
    usage.

    Keyed on the request arguments (model, messages, temperature, tools,
    response_format, seed, ...) and the client's base URL. Returns the client
    untouched if it does not look like one.
    """
    try:
        completions = client.chat.completions
    except Exception as exc:  # noqa: BLE001 - never break the judge
        logger.debug(f"Could not reach chat.completions to cache it: {exc}")
        return client
    if _CACHED_MARKER in getattr(completions, "__dict__", {}):
        return client
    original = getattr(completions, "create", None)
    if original is None:
        return client
    platform = f"openai-client:{getattr(client, 'base_url', '')}"

    @functools.wraps(original)
    def cached(*args: Any, **kwargs: Any) -> Any:
        scope = _active.get()
        deterministic = kwargs.get("temperature") == 0
        if args or kwargs.get("stream") or scope is None or not scope.wants(deterministic):
            return original(*args, **kwargs)
        request = {k: v for k, v in kwargs.items() if k not in _TRANSPORT_KWARGS}
        key = request_key(
            platform, scope.seed, json.dumps(request, sort_keys=True, default=_jsonable), ""
        )
        raw = scope.store.get(key)
        if raw is not None:
            try:
                from openai.types.chat import ChatCompletion

                response = ChatCompletion.model_validate_json(raw)
            except Exception:  # noqa: BLE001 - an unreadable row is a miss
                response = None
            if response is not None:
                scope.count(hit=True)
                return response
        scope.count(hit=False)
        response = original(*args, **kwargs)
        dump = getattr(response, "model_dump_json", None)
        if callable(dump):
            try:
                scope.store.put(key, dump())
            except Exception as exc:  # noqa: BLE001 - storing must not fail the call
                logger.debug(f"Could not store a judge response: {exc}")
        return response

    try:
        setattr(completions, "create", cached)
        setattr(completions, _CACHED_MARKER, True)
    except Exception as exc:  # noqa: BLE001 - never break the judge
        logger.debug(f"Could not cache the judge client: {exc}")
    return client
