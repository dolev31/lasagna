"""What a number was produced on, and by what.

A metric without its context is not comparable to anything. Two researchers
reporting `ndcg@10 = 0.71` on "the same benchmark" may differ in embedding
model, split, sample count, judge model, or library version, and nothing in a
results file said so — it recorded four fields: mode, n_runs, seed, and the
benchmark names.

The capture logic already existed twice, in places evaluation never reached:
`e2e/contract.py` (platform, git, hardware) and `core/pipeline.py`
(torch/CUDA/sentence-transformers). This module is the merge, made safe to put
in a published artifact:

* **No secret names.** The e2e version records *which* API-key variables were
  set. Useful in a private CI report, wrong in a file people upload — it
  discloses which providers a lab uses. This records only whether credentials
  were present.
* **No absolute paths, no hostnames.** Redacted, not trimmed, so a home
  directory or an unrelated project name cannot travel with a result.
* **`git_commit` only when it means something.** For a pip install, the user's
  working directory is unrelated to the code that ran; `installed_from`
  distinguishes the two.
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from typing import Any, Dict, Optional


def capture(*, include_git: Optional[bool] = None) -> Dict[str, Any]:
    """Capture environment, versions and install provenance.

    Safe to embed in a published results bundle: no paths, no hostnames, no
    credential names, no values.
    """
    env: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
    }

    try:
        env["cpu_count"] = os.cpu_count()
    except Exception:  # pragma: no cover - platform dependent
        env["cpu_count"] = None

    try:
        import psutil

        env["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except Exception:
        env["ram_gb"] = None

    env.update(_package_versions())
    env.update(_accelerator())
    env.update(_install_provenance(include_git=include_git))

    # Presence, never names or values. Which providers a lab has configured is
    # not something a published bundle should disclose.
    env["llm_credentials_present"] = _any_llm_credentials()
    env["active_fallbacks"] = _active_fallbacks()

    return env


def _package_versions() -> Dict[str, Any]:
    """Versions of the packages that can change a number.

    Recorded by import rather than from a lockfile, because what matters is
    what actually ran.
    """
    from importlib.metadata import PackageNotFoundError, version

    out: Dict[str, Any] = {}

    try:
        from lasagna.sdk._version import __version__ as lasagna_version

        out["lasagna_version"] = lasagna_version
    except ImportError:  # pragma: no cover - version module always ships
        out["lasagna_version"] = None

    # Anything here can move a metric: the embedder, the reranker, the judge
    # harness, the dataset loader, or the bootstrap.
    for dist in (
        "torch",
        "sentence-transformers",
        "transformers",
        "ragas",
        "datasets",
        "numpy",
        "scipy",
        "pymilvus",
        # Optional packages whose absence switches in a fallback that changes
        # chunks or BM25 scores; see active_fallbacks.
        "milvus-lite",
        "rank-bm25",
        "nltk",
        "docling",
    ):
        try:
            out[f"{dist.replace('-', '_')}_version"] = version(dist)
        except PackageNotFoundError:
            out[f"{dist.replace('-', '_')}_version"] = None

    return out


def _active_fallbacks() -> list:
    """Stand-ins that ran because an optional package is missing.

    The same named configuration chunks and scores differently on a core
    install and on one with the extras, and the two bundles used to be
    indistinguishable: a silent uncontrolled factor in a comparison whose point
    is holding everything else fixed.
    """
    from importlib.util import find_spec

    out = []
    if find_spec("nltk") is None:
        out += ["sentence_split_without_nltk", "bm25_without_stemming"]
    else:
        # nltk is installed but its tokenizer data is not: nothing downloads it
        # implicitly any more, so sentences are split with a regex until the
        # user runs the one-line downloader.
        try:
            from lasagna.processing.chunker.semantic import punkt_available

            if not punkt_available():
                out.append("sentence_split_without_punkt")
        except Exception:
            pass
    if find_spec("rank_bm25") is None:
        out.append("hybrid_without_bm25")
    try:
        from lasagna.core.config import rag_config

        if getattr(rag_config, "docling_enabled", False) and find_spec("docling") is None:
            out.append("docling_requested_but_missing")
    except Exception:
        pass
    return out


def _accelerator() -> Dict[str, Any]:
    """GPU details, which change float reduction order and therefore results."""
    out: Dict[str, Any] = {"gpu": None, "cuda_version": None}
    try:
        import torch

        if torch.cuda.is_available():  # pragma: no cover - no GPU in CI
            out["gpu"] = torch.cuda.get_device_name(0)
            out["cuda_version"] = torch.version.cuda
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            out["gpu"] = "mps"
    except Exception:
        pass
    return out


def _install_provenance(*, include_git: Optional[bool] = None) -> Dict[str, Any]:
    """Whether this ran from a checkout or an installed wheel.

    A git SHA from the current working directory is meaningless for a pip
    install -- it describes whatever repository the user happened to be
    standing in, not the code that produced the number.
    """
    out: Dict[str, Any] = {"installed_from": "unknown", "git_commit": None, "git_dirty": None}

    try:
        import lasagna

        pkg_dir = os.path.dirname(os.path.abspath(lasagna.__file__))
        out["installed_from"] = "source" if _is_git_checkout(pkg_dir) else "wheel"
    except Exception:  # pragma: no cover
        return out

    want_git = out["installed_from"] == "source" if include_git is None else include_git
    if not want_git:
        return out

    sha = _run_git(["rev-parse", "--short", "HEAD"], cwd=pkg_dir)
    if sha:
        out["git_commit"] = sha
        # A dirty tree means the recorded SHA does not describe what ran, which
        # is worth knowing before trusting a comparison against it.
        out["git_dirty"] = bool(_run_git(["status", "--porcelain"], cwd=pkg_dir))
    return out


def _is_git_checkout(path: str) -> bool:
    return _run_git(["rev-parse", "--is-inside-work-tree"], cwd=path) == "true"


def _run_git(args: list[str], cwd: str) -> Optional[str]:
    import shutil

    if not shutil.which("git"):
        return None
    try:
        proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10)
    except Exception:
        return None
    return proc.stdout.strip() or None if proc.returncode == 0 else None


#: Checked for presence only. Deliberately not recorded by name.
_LLM_CREDENTIAL_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "GROQ_API_KEY",
    "MISTRAL_API_KEY",
    "TOGETHER_API_KEY",
    "DEEPSEEK_API_KEY",
    "GOOGLE_API_KEY",
    "AZURE_OPENAI_API_KEY",
    "OPENROUTER_API_KEY",
    "FIREWORKS_API_KEY",
    "COHERE_API_KEY",
    "NVIDIA_API_KEY",
    "PERPLEXITY_API_KEY",
    "WATSONX_API_KEY",
    "WATSONX_APIKEY",
)


def _any_llm_credentials() -> bool:
    return any(os.environ.get(v) for v in _LLM_CREDENTIAL_VARS)


def redact_url(url: Any) -> str:
    """``url`` with its password replaced by ``***``, for a log line.

    Redis, broker and database URLs carry credentials in the authority
    (``redis://:secret@host:6379``) and sometimes in the query
    (``?password=...``). They were logged verbatim at startup. A PostgreSQL
    key=value DSN (``host=db user=rag password=...``) is not a URL; its
    secret values are replaced the same way.
    """
    import re
    from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

    secret_keys = {"password", "pass", "pwd", "token", "secret", "api_key", "apikey"}
    text = str(url or "")
    if "://" not in text and "=" in text:
        names = "|".join(sorted(secret_keys))
        return re.sub(rf"(?i)\b({names})\s*=\s*(?:'[^']*'|\S+)", r"\1=***", text)
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError:
        return "<unparseable url>"
    netloc = parts.netloc
    if parts.password is not None:
        host = parts.hostname or ""
        if ":" in host:
            host = f"[{host}]"
        if port is not None:
            host = f"{host}:{port}"
        netloc = f"{parts.username or ''}:***@{host}"
    query = parts.query
    if query:
        pairs = parse_qsl(query, keep_blank_values=True)
        if any(k.lower() in secret_keys for k, _ in pairs):
            query = urlencode(
                [(k, "***" if k.lower() in secret_keys else v) for k, v in pairs], safe="*"
            )
    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


#: An absolute POSIX path not already part of a URL or a ``<home>`` tag.
_ABSOLUTE_PATH = re.compile(r"(?<![\w:/>.~<-])/[^\s'\"`,;:()<>|]+")


def redact_paths(value: Any) -> Any:
    """:func:`redact`, and then no absolute path left at all.

    For what ``lasagna doctor`` prints, which promises "no absolute paths":
    a project outside the home directory printed its settings file's full
    path. A path under the working directory becomes ``./<relative>``, one
    :func:`redact` does not cover ``<path>/<last part>``.
    """
    if isinstance(value, dict):
        return {k: redact_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_paths(v) for v in value]
    if not isinstance(value, str):
        return redact(value)
    try:
        cwd = os.getcwd().rstrip("/") or "/"
    except OSError:  # the working directory was removed
        cwd = "/"

    def _relative(match: "re.Match[str]") -> str:
        path = match.group(0).rstrip(".")
        tail = match.group(0)[len(path) :]
        if path == cwd:
            return "." + tail
        if cwd != "/" and path.startswith(cwd + "/"):
            return "./" + path[len(cwd) + 1 :] + tail
        return match.group(0)

    def _anywhere(match: "re.Match[str]") -> str:
        path = match.group(0).rstrip(".")
        return f"<path>/{os.path.basename(path.rstrip('/'))}" + match.group(0)[len(path) :]

    return _ABSOLUTE_PATH.sub(_anywhere, redact(_ABSOLUTE_PATH.sub(_relative, value)))


def redact(value: Any) -> Any:
    """Strip machine-local paths from anything destined for a bundle.

    Provenance is only safe to publish if the things around it are too --
    subprocess output, tracebacks and error strings all carry absolute paths.
    """
    import re
    import tempfile

    if isinstance(value, dict):
        return {k: redact(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact(v) for v in value]
    if not isinstance(value, str):
        return value

    home = os.path.expanduser("~")
    tmp = tempfile.gettempdir().rstrip("/")
    out = value
    for path, tag in sorted(
        ((sys.executable, "python"), (home, "<home>"), (tmp, "<tmp>")),
        key=lambda kv: len(kv[0]),
        reverse=True,
    ):
        if not path:
            continue
        if tag == "<tmp>":
            # A whole path component only ("/tmp" does not start "/tmpab12"), and
            # with its /private spelling, so TMPDIR=/tmp leaves no "/private<tmp>".
            root = r"(?:/private)?/tmp" if path in ("/tmp", "/private/tmp") else re.escape(path)
            out = re.sub(rf"(?<![\w.~-]){root}(?![\w-])", tag, out)
        else:
            out = out.replace(path, tag)
    # macOS per-user temp roots embed a token that identifies the user even
    # when TMPDIR is unset.
    out = re.sub(r"(?:/private)?/var/folders/[^/\s]+/[^/\s]+/[A-Za-z]+", "<tmp>", out)
    for part in tmp.strip("/").split("/"):
        if len(part) >= 12:
            out = out.replace(part, "<tmp-id>")
    return out
