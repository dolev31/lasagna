"""What a run was computed under, beyond the environment ``core.provenance`` captures.

``core.provenance.capture()`` records the machine and the package versions.
A number also depends on the metric definitions (``METRICS_VERSION``), the
settings the environment overrode, the judge, and the exact weights of the
models that ran, which a model *name* does not pin: ``all-MiniLM-L6-v2`` is
whatever commit the local Hugging Face cache holds. These are recorded in
``results.json``, and the resume key hashes the ones that can move a number, so
``--resume`` never restores a result computed under different ones.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

#: Setting names whose value is a credential, a host or a connection string.
_SENSITIVE_NAME = re.compile(
    r"(key|token|secret|password|passwd|dsn|uri|url|endpoint|host|broker|backend_config)",
    re.IGNORECASE,
)
_REDACTED = "<redacted>"

#: Evaluation variables that change a number: which models judge and write.
_EVAL_MODEL_ENV = ("RAG_EVAL_JUDGE_MODEL", "RAG_EVAL_GENERATOR_MODEL", "RAG_EVAL_SIMILARITY_MODEL")


def package_version() -> Optional[str]:
    try:
        from lasagna.sdk._version import __version__

        return __version__
    except ImportError:  # pragma: no cover - the version module always ships
        return None


def _setting_names() -> set:
    try:
        from lasagna.core.config import get_rag_config

        names = set(get_rag_config()._config)
    except Exception:  # noqa: BLE001 - provenance must never fail a run
        return set()
    return names - {"upload_dir", "db_path", "citation_db_path"}


def _redact_value(name: str, raw: str) -> Any:
    from lasagna.core.provenance import redact

    if _SENSITIVE_NAME.search(name) or "://" in raw:
        if raw.strip()[:1] == "{":
            try:
                parsed = json.loads(raw)
            except ValueError:
                return _REDACTED
            if isinstance(parsed, dict):
                return {
                    k: _REDACTED if _SENSITIVE_NAME.search(str(k)) else redact(v)
                    for k, v in parsed.items()
                }
        return _REDACTED
    return redact(raw)


def rag_overrides(environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The settings this run's environment overrode, with secrets and paths hidden.

    ``RAG_<SETTING>`` and ``LASAGNA_<SETTING>`` for every setting, and the
    evaluation model variables. Credentials, hosts and connection strings show
    as ``<redacted>``: a results file is meant to be published.
    """
    from lasagna.core.config import LASAGNA_RESERVED_ENV

    env = os.environ if environ is None else environ
    names = _setting_names()
    out: Dict[str, Any] = {}
    for var in sorted(env):
        if var in LASAGNA_RESERVED_ENV:
            continue
        for prefix in ("RAG_", "LASAGNA_"):
            if var.startswith(prefix) and var[len(prefix) :].lower() in names:
                out[var] = _redact_value(var, env[var])
        if var in _EVAL_MODEL_ENV:
            out[var] = _redact_value(var, env[var])
    return out


#: Libraries whose version can move a number: the models, the judge, the store.
_FINGERPRINT_LIBRARIES = ("torch", "transformers", "sentence-transformers", "ragas", "pymilvus")


def _library_version(distribution: str) -> Optional[str]:
    """The installed version of ``distribution``, read from its metadata (no import)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(distribution)
    except PackageNotFoundError:
        return None


def _settings_file_digest(path: Optional[str]) -> Optional[str]:
    """A hash of the settings file's contents (the packaged one when none is named)."""
    if path is None:
        try:
            from lasagna.core.config import get_rag_config

            path = getattr(get_rag_config(), "_settings_path", None)
        except Exception:  # noqa: BLE001 - provenance must never fail a run
            return None
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except (OSError, TypeError):
        return None


def resume_fingerprint(
    judge_model: Optional[str],
    environ: Optional[Mapping[str, str]] = None,
    metrics_version: Optional[str] = None,
    *,
    variance_source: Optional[str] = None,
    max_concurrency: Optional[int] = None,
    settings_file: Optional[str] = None,
    llms: Optional[Mapping[str, Any]] = None,
    model_revisions: Optional[Mapping[str, Any]] = None,
) -> str:
    """A hash of what a checkpointed result depends on beyond its task.

    The metric definitions, the package version, the judge, the overridden
    settings, the versions of the libraries that compute the numbers, the
    settings file's contents, ``variance_source`` and ``max_concurrency``;
    ``llms``, the provider and models the run resolved (``MODEL_NAME`` is not a
    ``RAG_`` variable, and the provider follows whichever API key is set); and
    ``model_revisions``, the local commit of each model the task loads. A
    checkpoint written under any other value of these is not restored; it
    would carry numbers computed some other way.
    """
    if metrics_version is None:
        from .bundle import METRICS_VERSION as metrics_version
    blob = json.dumps(
        {
            "metrics_version": metrics_version,
            "package_version": package_version(),
            "judge_model": judge_model or None,
            "overrides": rag_overrides(environ),
            "libraries": {d: _library_version(d) for d in _FINGERPRINT_LIBRARIES},
            "settings_file": _settings_file_digest(settings_file),
            "variance_source": variance_source,
            "max_concurrency": max_concurrency,
            "llms": dict(llms or {}),
            "model_revisions": dict(model_revisions or {}),
        },
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def model_revision(name: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """The commit of ``name`` in the local Hugging Face cache, or None.

    Reads the cache only, so it works offline and never downloads. A short
    sentence-transformers name (``all-MiniLM-L6-v2``) is also looked up under
    its ``sentence-transformers/`` repository, as the library resolves it.
    """
    if not name or ":" in name or os.path.isabs(name):
        return None
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    candidates = [name] if "/" in name else [name, f"sentence-transformers/{name}"]
    for repo in candidates:
        for filename in ("config.json", "modules.json", "tokenizer_config.json"):
            try:
                found = try_to_load_from_cache(repo, filename, cache_dir=cache_dir)
            except Exception:  # noqa: BLE001 - an odd name is "not cached", not an error
                found = None
            if isinstance(found, str):
                parts = found.replace(os.sep, "/").split("/")
                if "snapshots" in parts:
                    return parts[parts.index("snapshots") + 1]
    return None


def models_used(rows: Iterable[Mapping[str, Any]], run_meta: Mapping[str, Any]) -> List[str]:
    """The model names a run's tasks recorded running with.

    The embedder and reranker each task loaded (``TaskResult.system``) and the
    answer-similarity embedder. Component references (``module:function``) and
    API-served models have no Hugging Face commit and are left to the caller.
    """
    names = []
    for row in rows:
        system = row.get("system") or {}
        names += [system.get("embedder_model"), system.get("reranker_model")]
    names.append(run_meta.get("similarity_model"))
    return sorted({str(n) for n in names if n and ":" not in str(n)})


def component_provenance(rows: Iterable[Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """``{name: {"stage", "paper", "claims"}}`` for each component the tasks ran
    that declared a paper or claimed numbers, keyed by the name the run used.

    What lets a results table put the claimed number beside the measured one:
    ``@lasagna.reranker(paper="Doe et al. 2026", claims={"ndcg@10": 0.456})``.
    """
    from lasagna.core.registry import registry

    found: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        stages = (row.get("system") or {}).get("stages") or {}
        for stage, slot in stages.items():
            name = (slot or {}).get("component") if isinstance(slot, Mapping) else None
            if not name or name in found:
                continue
            try:
                declared = registry.provenance(stage, name)
            except ValueError:  # "other" and any slot that is no registry stage
                continue
            if declared.get("paper") or declared.get("claims"):
                found[name] = {
                    "stage": stage,
                    "paper": declared.get("paper"),
                    "claims": declared.get("claims") or {},
                }
    return found


def claimed(
    components: Optional[Mapping[str, Any]], system: Optional[Mapping[str, Any]], metric: str
) -> Optional[Dict[str, Any]]:
    """The number a task's component claims for ``metric``, with its paper.

    ``{"component", "value", "paper"}`` from the run's ``components`` record
    and the task's stage manifest, or None when no component in it claims
    that metric (names compared case-insensitively).
    """
    if not components or not system:
        return None
    for slot in ((system.get("stages") or {}) if isinstance(system, Mapping) else {}).values():
        name = (slot or {}).get("component") if isinstance(slot, Mapping) else None
        record = components.get(name) if name else None
        if not record:
            continue
        for key, value in (record.get("claims") or {}).items():
            if str(key).lower() == metric.lower() and isinstance(value, (int, float)):
                return {"component": name, "value": float(value), "paper": record.get("paper")}
    return None


def claim_text(claim: Optional[Mapping[str, Any]]) -> str:
    """``0.4560 (Doe et al. 2026)``, or "" when there is no claim."""
    if not claim:
        return ""
    paper = f" ({claim['paper']})" if claim.get("paper") else ""
    return f"{claim['value']:.4f}{paper}"


def model_revisions(names: Iterable[str], cache_dir: Optional[str] = None) -> Dict[str, Any]:
    """``{model: commit or None}``; None means the commit could not be read."""
    return {name: model_revision(name, cache_dir=cache_dir) for name in sorted(set(names))}


def determinism_label(concurrency: int) -> str:
    """What the seeding guarantees for this run.

    Each task seeds the process-wide generators before it runs. One task at a
    time, every task's random draws follow from its own seed; with several
    tasks interleaving on one event loop they share those generators, so a
    task's draws depend on what else was running.
    """
    return "seeded_per_task" if int(concurrency) <= 1 else "seeded_global_concurrent"
