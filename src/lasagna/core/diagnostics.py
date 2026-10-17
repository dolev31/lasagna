"""What is wrong with this installation, in one place.

Most support conversations start by asking the same six questions: which
version, installed how, which backend, which model, is it reachable, and did
anything fail to load quietly. Every one of those is answerable locally, so
asking a human to answer them is asking them to do the machine's work.

Two audiences shape the output. A person needs to see what to fix. An **agent**
needs it as JSON, because the agent is the one that will paste it into an issue
— which is why every check carries a machine-readable ``status`` and an
``advice`` string rather than only prose.

Nothing here contacts the network except the checks that say they do, and
nothing emits a credential: ``provenance.capture()`` reports only whether
credentials are *present*, and every string is passed through ``redact()``
before it leaves this module.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Extras a user is most likely to be missing, and what breaks without them.
_OPTIONAL_IMPORTS = {
    "ragas": ("eval", "benchmark scoring (faithfulness, answer relevancy)"),
    "datasets": ("eval", "loading benchmark datasets"),
    # pymilvus is a CORE dependency; `vector-backends` is pinecone/qdrant/
    # weaviate and does not contain it. Telling a user to install that extra
    # sent them somewhere that could not help. `None` means "reinstall the
    # package", which is the honest answer when a core dependency is missing --
    # and `_check_vector_backend` below adds the Windows-specific advice, which
    # is the case where this genuinely happens.
    "pymilvus": (None, "the default vector store"),
    "sentence_transformers": (None, "embedding and reranking"),
    "torch": (None, "embedding and reranking"),
}

_OK = "ok"
_WARN = "warning"
_FAIL = "error"


@dataclass
class Check:
    """One diagnostic result.

    ``advice`` is the whole point: a check that reports a problem without
    saying what to do about it has moved the work rather than done it.
    """

    name: str
    status: str
    detail: str = ""
    advice: str = ""


@dataclass
class Diagnosis:
    checks: List[Check] = field(default_factory=list)
    environment: Dict[str, Any] = field(default_factory=dict)

    @property
    def worst(self) -> str:
        if any(c.status == _FAIL for c in self.checks):
            return _FAIL
        if any(c.status == _WARN for c in self.checks):
            return _WARN
        return _OK

    @property
    def problems(self) -> List[Check]:
        return [c for c in self.checks if c.status != _OK]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.worst,
            "environment": self.environment,
            "checks": [asdict(c) for c in self.checks],
        }


# Packages whose presence is not enough: they can be found on disk and still
# fail to import against a resolved dependency (ragas did, with one
# langchain-community release), so the check must actually import them.
_IMPORT_CHECKED = frozenset({"ragas", "datasets"})


def _installed(module: str) -> bool:
    try:
        if importlib.util.find_spec(module) is None:
            return False
    except (ImportError, ValueError):
        return False
    if module in _IMPORT_CHECKED:
        try:
            importlib.import_module(module)
        except Exception:  # any import failure means it cannot be used
            return False
    return True


def _check_optional_extras() -> List[Check]:
    checks = []
    for module, (extra, purpose) in _OPTIONAL_IMPORTS.items():
        if _installed(module):
            checks.append(Check(f"import:{module}", _OK, "installed"))
            continue
        install = f'pip install "lasagna-rag[{extra}]"' if extra else "pip install lasagna-rag"
        checks.append(
            Check(
                f"import:{module}",
                _WARN,
                f"not installed: {purpose} unavailable",
                f"{install}",
            )
        )
    return checks


def _check_llm_credentials(env: Dict[str, Any], session_id: str = "default") -> Check:
    """Presence only. The names of the variables are fine to print; values never are."""
    if env.get("llm_credentials_present"):
        return Check("llm_credentials", _OK, "an LLM API key is configured")
    from .setup_text import llm_key_hint

    return Check("llm_credentials", _WARN, "no LLM API key detected", llm_key_hint(session_id))


def _check_llm_sampling() -> Optional[Check]:
    """Whether the configured LLM takes a temperature, so answers can be pinned.

    Read from the environment, as provider auto-detection reads it, without
    building a client. None when no LLM is configured.
    """
    try:
        from lasagna.providers import _detect_llm_platform
        from lasagna.providers.llm_provider import SimpleLLMProvider
        from lasagna.providers.model_capabilities import unpinned_note
    except Exception:  # noqa: BLE001 - a broken install is reported elsewhere
        return None
    platform = _detect_llm_platform()
    if platform is None:
        return None
    model = (
        os.environ.get("RAG_EVAL_GENERATOR_MODEL")
        or os.environ.get("MODEL_NAME")
        or SimpleLLMProvider.SUPPORTED_PLATFORMS.get(platform, "")
    )
    note = unpinned_note(model, platform)
    if note is None:
        return Check("llm_sampling", _OK, f"{model} ({platform}) decodes at temperature 0")
    # One line: what is wrong, what it does, and what to do about it.
    return Check("llm_sampling", _WARN, note)


def _check_model_cache(session_id: str = "default") -> Check:
    """Whether the first search will trigger a multi-gigabyte download.

    This is the single most common 'it hangs' report. It is not a hang. The
    models checked are the session's own (its embedder, and its reranker when
    reranking is on): it counted the cache's folders and said "ok, 4 model(s)
    cached" while the profile's models were not among them.
    """
    from .config import rag_config
    from .model_downloads import fix_for_model, is_cached, offline
    from .setup_text import model_cache_hint

    config = rag_config.with_session(session_id)
    models = [config.embedding_model]
    if config.enable_reranking:
        models.append(config.reranker_model)
    models = [m for m in dict.fromkeys(models) if m]
    missing = [m for m in models if not is_cached(m)]
    if not missing:
        return Check("model_cache", _OK, f"cached for session {session_id!r}: {', '.join(models)}")
    if offline():
        # Nothing can download them, so the first upload or search fails:
        # this said "downloads 3.2 GB" and "0 blocking" with HF_HUB_OFFLINE=1.
        return Check(
            "model_cache",
            _FAIL,
            f"not downloaded for session {session_id!r}, and Hugging Face offline mode "
            f"is on (HF_HUB_OFFLINE), so its first upload or search fails: "
            f"{', '.join(missing)}",
            fix_for_model(session_id, missing[0]),
        )
    return Check(
        "model_cache",
        _WARN,
        f"not downloaded for session {session_id!r}: {', '.join(missing)}",
        model_cache_hint(missing, session_id),
    )


def _check_session_documents(session_id: str = "default") -> Optional[Check]:
    """Whether the session's documents are in the collection its model searches.

    A failed `configure --reindex`, or a switch with --force, left a session
    whose documents sat in the old model's collection while it searched the
    new one's: every search came back empty, or failed on a model that was not
    downloaded, and doctor said "0 blocking". Read from the metadata database
    only. None when the session has no documents.
    """
    from lasagna.api._profile_guard import _SEARCHED_STATUSES
    from lasagna.core.database import get_rag_database

    from .config import rag_config

    try:
        rows = [
            row
            for row in get_rag_database().get_documents(session_id)
            if str(row.get("status") or "").lower() in _SEARCHED_STATUSES
        ]
        config = rag_config.with_session(session_id)
        searched, model = config.collection_name, config.embedding_model
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        return Check("session_documents", _WARN, f"could not read session {session_id!r}: {exc}")
    if not rows:
        return None
    elsewhere = [r for r in rows if (r.get("collection_name") or searched) != searched]
    if not elsewhere:
        return Check(
            "session_documents",
            _OK,
            f"{len(rows)} in session {session_id!r}, all in {model}'s collection",
        )
    collections = sorted({str(r.get("collection_name")) for r in elsewhere})
    from lasagna.api._profile_guard import model_of_collection

    named = [model_of_collection(c) for c in collections]
    models = [m for m, _ in named]
    back = [p for _, p in named if p]
    profile = _active_profile(session_id)
    switch_back = (
        f"Or switch back to {back[0]}, whose model indexed them: lasagna configure "
        f"--profile {back[0]} -s {session_id}."
        if back
        else f"Or switch back to the profile that indexed them: lasagna configure "
        f"--profile <name> -s {session_id}."
    )
    return Check(
        "session_documents",
        _FAIL,
        f"{len(elsewhere)} of {len(rows)} documents in session {session_id!r} are in "
        f"{', '.join(models)}'s collection, but the session searches with {model}, so no "
        f"search finds them",
        f"Reindex them with {model}: lasagna configure --profile {profile} -s {session_id} "
        f"--reindex (it changes nothing if one fails). {switch_back}",
    )


def _active_profile(session_id: str) -> str:
    try:
        from .session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager(session_id=session_id)
        return mgr.get_active_profile(session_id=session_id) or "default"
    except Exception:  # noqa: BLE001
        return "default"


#: The vector stores Lasagna ships an adapter for (``vector_backends/``).
_BUILTIN_VECTOR_BACKENDS = ("milvus", "pgvector", "qdrant", "weaviate", "pinecone", "opensearch")


def vector_backend_source() -> str:
    """Where the configured vector backend's name was set, as a fix should name it.

    ``LASAGNA_VECTOR_BACKEND`` wins over ``RAG_VECTOR_BACKEND``, and either over
    the settings file. The advice said "set RAG_VECTOR_BACKEND" for a typo in
    the settings file, where setting the variable fixes nothing that is wrong.
    """
    for var in ("LASAGNA_VECTOR_BACKEND", "RAG_VECTOR_BACKEND"):
        if os.environ.get(var, "").strip():
            return var
    try:
        from .config import rag_config

        path = getattr(rag_config, "_settings_path", None)
        settings = getattr(rag_config, "_knowledge_settings", None) or {}
        rag = settings.get("rag") if isinstance(settings, dict) else getattr(settings, "rag", None)
        in_file = (rag or {}).get("vector_backend") if isinstance(rag, dict) else None
        if path and in_file is not None:
            return f"vector_backend in {path}"
    except Exception:  # noqa: BLE001 - the advice falls back to the variable
        pass
    return "RAG_VECTOR_BACKEND"


def unknown_vector_backend(name: str) -> Optional[Tuple[str, str]]:
    """``(what, advice)`` when ``name`` is no built-in or plugin vector backend, else None.

    Plugins (``lasagna_vector_*`` packages) are discovered only for a name that
    is not built in. The advice names where the value was set
    (:func:`vector_backend_source`).
    """
    if name in _BUILTIN_VECTOR_BACKENDS:
        return None
    known = set(_BUILTIN_VECTOR_BACKENDS)
    try:
        from lasagna.vector_backends.registry import discover_backends, get_registry

        discover_backends()
        known |= set(get_registry().list_backends())
    except Exception:  # noqa: BLE001 - a broken plugin must not hide the answer
        pass
    if name in known:
        return None
    from ._text import did_you_mean

    return (
        f"unknown backend {name!r}{did_you_mean(name, known)}",
        f"set {vector_backend_source()} to one of: {', '.join(sorted(known))}",
    )


def _check_vector_backend() -> Check:
    try:
        from .config import rag_config

        name = getattr(rag_config, "vector_backend", None) or "milvus"
    except Exception as exc:  # pragma: no cover - config import is exercised elsewhere
        return Check("vector_backend", _FAIL, f"could not read config: {exc}")

    # A typo ("milvuss") was reported as "ok", and the first upload failed.
    unknown = unknown_vector_backend(name)
    if unknown is not None:
        return Check("vector_backend", _FAIL, *unknown)

    module = {"milvus": "pymilvus", "qdrant": "qdrant_client", "chroma": "chromadb"}.get(name)
    if module and not _installed(module):
        # `vector-backends` is pinecone/qdrant/weaviate. It does not contain
        # pymilvus, which is a core dependency -- so for the DEFAULT backend the
        # advice pointed at an extra that could not fix it.
        if module == "pymilvus":
            advice = "pip install lasagna-rag  (pymilvus is a core dependency)"
        else:
            advice = 'pip install "lasagna-rag[vector-backends]"'
        return Check(
            "vector_backend",
            _FAIL,
            f"configured backend '{name}' needs {module}, which is not installed",
            advice,
        )

    # milvus-lite is the default store and pymilvus gates it on
    # `sys_platform != "win32"`, so on Windows pymilvus imports, this check
    # passes, and the failure surfaces at the first upload as a connection
    # error suggesting a package that will not install. Say so here instead.
    if name == "milvus" and sys.platform.startswith("win"):
        return Check(
            "vector_backend",
            _WARN,
            "milvus-lite has no Windows build, so the default local store is unavailable",
            "run under WSL, or point RAG_VECTOR_BACKEND at an external Milvus or another backend",
        )
    return Check("vector_backend", _OK, f"configured: {name}")


def _check_plugins() -> List[Check]:
    """Plugins that failed to load are the definition of a silent problem.

    Discovery logs a warning and continues, so a third-party component with a
    broken entry point simply never appears -- and the user's results quietly
    come from the default instead of the component they installed.
    """
    try:
        from .registry import discovery_log, registry

        # Discovery is lazy, so the log is empty until the registry is touched.
        registry.list_all()
        entries = discovery_log()
    except Exception as exc:
        return [Check("plugins", _WARN, f"discovery raised: {exc}")]

    if not entries:
        return [Check("plugins", _OK, "no third-party plugins installed")]

    loaded = [e for e in entries if e.get("ok")]
    noun = "entry point" if len(entries) == 1 else "entry points"
    checks = [Check("plugins", _OK, f"{len(loaded)} of {len(entries)} {noun} loaded")]
    for entry in entries:
        if entry.get("ok"):
            continue
        checks.append(
            Check(
                f"plugin:{entry.get('name', '?')}",
                _FAIL,
                f"failed to load: {entry.get('error')}",
                "Results will silently use the built-in default instead. Fix "
                "the entry point or uninstall the package.",
            )
        )
    return checks


def _check_writable_dirs() -> Check:
    from .config import rag_config

    for attr in ("upload_dir", "db_directory", "db_dir"):
        path = getattr(rag_config, attr, None)
        if not path:
            continue
        try:
            os.makedirs(path, exist_ok=True)
            if not os.access(path, os.W_OK):
                return Check("storage", _FAIL, f"{attr} is not writable", f"chmod u+w {path}")
        except OSError as exc:
            return Check("storage", _FAIL, f"cannot create {attr}: {exc}")
    return Check("storage", _OK, "data directories are writable")


def _check_accelerator(env: dict) -> Check:
    """Which device models run on, and what that means for a sweep."""
    gpu = env.get("gpu")
    from lasagna.core.config import FALSY

    if os.environ.get("RAG_USE_GPU", "").strip().lower() in FALSY:
        return Check("accelerator", _OK, "RAG_USE_GPU is off: models run on the CPU")
    if gpu == "mps":
        return Check(
            "accelerator",
            _OK,
            "Apple MPS: model calls from concurrent arms take turns on the device, "
            "since Metal aborts on overlapping ones",
            "If a sweep is slower than expected, compare with RAG_USE_GPU=false (CPU).",
        )
    return Check("accelerator", _OK, f"{gpu or 'CPU only'}")


def _check_gh_cli() -> Check:
    """Whether an agent can file an issue without a browser."""
    if shutil.which("gh"):
        return Check("gh_cli", _OK, "GitHub CLI available")
    return Check(
        "gh_cli",
        _WARN,
        "GitHub CLI not installed",
        "Needed only to report a problem non-interactively: "
        "https://cli.github.com. Otherwise open an issue in a browser.",
    )


def diagnose(*, check_gh: bool = True, session_id: str = "default") -> Diagnosis:
    """Run every check and return a redacted, serialisable result.

    ``session_id`` is the session whose models the cache check looks for.
    """
    from .provenance import capture, redact, redact_paths

    env = capture()
    checks: List[Check] = [
        _check_vector_backend(),
        _check_writable_dirs(),
        _check_llm_credentials(env, session_id),
        *filter(None, [_check_llm_sampling()]),
        _check_model_cache(session_id),
        *filter(None, [_check_session_documents(session_id)]),
        _check_accelerator(env),
        *_check_optional_extras(),
        *_check_plugins(),
    ]
    if check_gh:
        checks.append(_check_gh_cli())

    diagnosis = Diagnosis(checks=checks, environment=env)
    # Redact once, at the boundary. Paths appear in several checks and the
    # output is meant to be pasted into a public issue, so no absolute path
    # is left, inside the home directory or not.
    return Diagnosis(
        checks=[Check(**redact_paths(asdict(c))) for c in diagnosis.checks],
        environment=redact(env),
    )


def format_issue_body(
    diagnosis: Optional[Diagnosis] = None, context: Optional[List[Tuple[str, str]]] = None
) -> str:
    """A ready-to-paste diagnostics block for a bug report.

    ``context`` is the setup in force (session, profile, files), as
    ``setup_text.doctor_context`` gives it; already redacted.
    """
    d = diagnosis or diagnose()
    env = d.environment
    lines = [
        "<details><summary>lasagna doctor</summary>",
        "",
        "```",
        f"lasagna     {env.get('lasagna_version', '?')} ({env.get('installed_from', '?')})",
        f"python      {env.get('python', '?')}",
        f"platform    {env.get('platform', '?')} {env.get('machine', '')}".rstrip(),
        f"torch       {env.get('torch_version') or 'not installed'}",
        f"gpu         {env.get('gpu') or 'none'}",
        *(f"{name:<11} {value}" for name, value in context or []),
        "",
    ]
    for c in d.checks:
        mark = {"ok": "ok  ", "warning": "warn", "error": "FAIL"}[c.status]
        lines.append(f"{mark} {c.name}: {c.detail}")
    lines += ["```", "", "</details>"]
    return "\n".join(lines)
