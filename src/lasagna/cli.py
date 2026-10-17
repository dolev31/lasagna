"""Lasagna CLI.

Standalone CLI for the knowledge package — no dependencies outside
``lasagna``.  Runnable as::

    python -m lasagna init
    python -m lasagna upload report.pdf
    python -m lasagna search "quarterly revenue"
    python -m lasagna serve
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.measure import Measurement
from rich.panel import Panel
from rich.table import Table

from .core._text import LOG_LEVELS

# No emoji codes: user-supplied names reach the console, and
# `component validate nope:x` printed "nope❌" (":x:" is an emoji code).
console = Console(emoji=False)
app = typer.Typer(
    name="lasagna",
    help="Lasagna: upload and search documents, benchmark pipeline changes, configure, and serve.",
    no_args_is_help=True,
)

#: Levels loguru accepts. An unknown value would raise from `logger.add`, which
#: would turn a typo in an environment variable into a crash on every command.
_LOG_LEVELS = frozenset(LOG_LEVELS)


def _configure_cli_logging(
    verbose: bool = False,
    default: str = "WARNING",
    hide: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> str:
    """Own logging for the CLI process, and return the level it chose.

    ``hide`` drops records that are noise for the command at hand (see
    :func:`_server_only_warning`); ``--verbose`` shows them anyway.

    Nothing configured it before, so every command inherited loguru's default
    DEBUG sink. That is three lines on `lasagna doctor` -- not the flood it
    might sound like, but two of them print absolute filesystem paths, and
    `doctor` is precisely the command that redacts paths from its own output
    because it is meant to be pasted into a public issue. It scrubbed stdout and
    leaked the same paths on stderr one line earlier.

    Deliberately called from a Typer callback rather than at module scope:
    `logger.remove()` drops every handler including a host application's, and
    `from lasagna.cli import app` happens in tests and in embedding code. This
    way it runs only when the CLI is genuinely being used.
    """
    import sys

    from loguru import logger

    from .core.paths import env_get

    # The log_level setting, by either name (LASAGNA_LOG_LEVEL wins over
    # RAG_LOG_LEVEL); a value it does not take is ignored, never fatal.
    level = "DEBUG" if verbose else (env_get("LOG_LEVEL") or "").strip().upper()
    if level not in _LOG_LEVELS:
        level = default

    logger.remove()
    if level == "DEBUG":
        # diagnose=False everywhere: loguru otherwise prints each frame's local
        # variables (queries, document text, settings) into the traceback.
        logger.add(sys.stderr, level=level, diagnose=False)
        return level
    threshold = logger.level(level).no

    def _shown(record) -> bool:
        if hide is not None and hide(record):
            return False
        # Progress lines (corpus ingestion) show at the warnings-only default:
        # without them a half-hour ingest looks like a hang.
        return record["level"].no >= threshold or bool(record["extra"].get("progress"))

    # "warning: ..." and the message, not loguru's timestamp, module and line:
    # those are for a developer, and -v still shows them.
    def _plain(record) -> str:
        if record["level"].no >= logger.level("WARNING").no:
            return "<level>{level}</level>: {message}\n{exception}"
        return "{message}\n{exception}"

    logger.add(
        sys.stderr,
        level=min(threshold, logger.level("INFO").no),
        filter=_shown,
        format=_plain,
        diagnose=False,
    )
    return level


def _server_only_warning(record: Dict[str, Any]) -> bool:
    """A warning about background work that only a long-lived process does.

    The backend schedules garbage collection when it starts, so every
    `upload`, `search` and `status` warned that APScheduler was missing and
    "GC will not run automatically". A one-shot command exits long before a
    cycle would run; `serve` and the MCP server still say it.
    """
    return record["name"] == "lasagna.lifecycle.scheduler" and (
        record["function"] == "_start_apscheduler"
    )


def _ephemeral_signing_secret_warning(record: Dict[str, Any]) -> bool:
    """The REST app's warning that RAG_JWT_SECRET is unset.

    On a loopback server with auth off, a per-process signing secret costs
    nothing anyone relies on: it only means session tokens end with the
    process. `serve` hides it there, and keeps it on any other address.
    """
    return record["name"] == "lasagna.api.rest.session_token" and (
        "RAG_JWT_SECRET" in record["message"]
    )


def _print_version(value: bool) -> None:
    if value:
        from lasagna import __version__

        typer.echo(f"lasagna {__version__}")
        raise typer.Exit()


@app.callback()
def _cli(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show INFO and DEBUG logs (default: warnings only)"
    ),
    version: bool = typer.Option(
        False,
        "--version",
        callback=_print_version,
        is_eager=True,
        help="Print the installed version and exit.",
    ),
):
    # First, before anything imports gRPC, tokenizers or tqdm.
    from ._quiet import quiet_third_party_output

    quiet_third_party_output()

    from loguru import logger

    logger.enable("lasagna")
    # A sweep whose every task failed warns from Python, where logs are off by
    # default; the CLI prints each failure itself, so the warning only repeated
    # the first one with a source line ("cli.py:1333: RuntimeWarning: ...").
    import warnings

    warnings.filterwarnings("ignore", message=r"All \d+ task\(s\) of ", category=RuntimeWarning)
    # Profiles and settings saved by one command must apply in the next process
    # (`configure` then `upload`, `search`, `serve`): keep sessions on disk.
    os.environ.setdefault("RAG_SESSION_BACKEND", "persistent")
    # This process's sink shows DEBUG when log_level asks for it; the
    # library's own DEBUG handler would print every line twice.
    os.environ.setdefault("LASAGNA_SKIP_LOGGING", "1")
    _settings_file_from_dotenv()
    _configure_cli_logging(verbose, hide=_server_only_warning)


def _settings_file_from_dotenv() -> None:
    """Use the settings file `lasagna init` pointed ./.env at.

    `init` writes ``RAG_SETTINGS_FILE=...`` to ./.env, and nothing read it:
    `lasagna serve` in that directory loaded the 3.2 GB default models the user
    had just declined. Only this key is taken, and only from the working
    directory, never over a value already in the environment: a .env holds
    other projects' credentials, which importing Lasagna must not pick up.
    """
    if os.environ.get("RAG_SETTINGS_FILE"):
        return
    env_file = Path.cwd() / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import dotenv_values

        value = dotenv_values(env_file).get("RAG_SETTINGS_FILE")
    except Exception:
        return
    if value:
        os.environ["RAG_SETTINGS_FILE"] = value


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(coro):
    """Run an async coroutine from sync CLI context."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor() as pool:
            return pool.submit(asyncio.run, coro).result()
    return asyncio.run(coro)


def _get_client(session_id: str = "default"):
    """Create a KnowledgeClient, loading the session's active profile config."""
    from lasagna.sdk import KnowledgeClient

    config = _load_profile_config(session_id)
    return KnowledgeClient(config=config, session_id=session_id)


def _client_for(session_id: str, url: Optional[str]):
    """A client for ``session_id``: through the running server at ``url``, if any.

    ``url`` comes from :func:`_serving_url`; the client then talks to that
    `lasagna serve` over its REST API instead of opening the store.
    """
    if url:
        from lasagna.sdk import KnowledgeClient

        return KnowledgeClient.from_url(url, session_id=session_id)
    return _get_client(session_id)


def _serving_url() -> Optional[str]:
    """The `lasagna serve` to go through when one holds the local store.

    Exits at once, naming the fix, when another process holds it: the command
    used to load its models for about 7 s and then fail.
    """
    from lasagna.core import local_server

    if not local_server.local_store_held():
        return None
    url = local_server.running_server()
    if url is None:
        _error(local_server.held_store_message("this command"))
    typer.echo(f"lasagna serve at {url} holds the local store; going through it.", err=True)
    return url


def _load_profile_config(session_id: str):
    """What the session chose (its profile's settings and its overrides) as a KnowledgeConfig.

    Only what the session chose: the ``default`` profile's stored config is a
    snapshot of the settings file, and passing it to the client as explicit
    fields made every one of them beat the environment, so
    ``RAG_ENABLE_RERANKING=false lasagna status`` still listed reranking and
    ``RAG_EMBEDDING_MODEL=...`` was ignored. Session settings still win over
    the environment, as docs/configuration.md says.

    Falls back to default KnowledgeConfig if profile loading fails.
    """
    from dataclasses import fields as dc_fields
    from lasagna.sdk.config import KnowledgeConfig

    try:
        from lasagna.core.session_config import (
            get_knowledge_config_manager,
        )

        mgr = get_knowledge_config_manager(session_id=session_id)
        # A built-in profile's own settings, and what the session set over it
        # (a Python client saves its settings there on its first upload).
        profile_config = dict(mgr.get_chosen_config(session_id) or {})

        if not profile_config:
            return KnowledgeConfig()

        # Filter to only keys accepted by KnowledgeConfig
        valid_keys = {f.name for f in dc_fields(KnowledgeConfig)}
        filtered = {k: v for k, v in profile_config.items() if k in valid_keys}
        return KnowledgeConfig.from_dict(filtered)

    except Exception:
        # If profile loading fails, fall back to defaults
        return KnowledgeConfig()


def _milliseconds(ms: float) -> str:
    """``0.42 ms``, ``12.3 ms``, ``under 0.01 ms``: a reading that is never "0.0 ms"."""
    if ms < 0.01:
        return "under 0.01 ms"
    return f"{ms:.2f} ms" if ms < 10 else f"{ms:.1f} ms"


def _count(n: int, noun: str) -> str:
    """``_count(1, "pair")`` -> ``"1 pair"``; ``_count(2, "pair")`` -> ``"2 pairs"``."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _output_file(output: str) -> Path:
    """``output`` as a path whose folder exists.

    ``-o newdir/run.json`` ended in a ``FileNotFoundError`` traceback when
    ``newdir`` did not exist; the folder is created, and one that cannot be
    is a one-line error.
    """
    path = Path(output)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _error(f"Cannot create the folder {path.parent} for {output}: {exc.strerror or exc}")
    return path


def _error(msg: str, exit_code: int = 1, exc: Optional[BaseException] = None):
    """Print error and exit.

    Pass the exception behind ``msg`` as ``exc`` when there is one: a failure
    with one known fix is then reported as that fix instead, and a bad setting
    exits 2, as :func:`main` does.
    """
    config_error = _in_cause_chain(exc, "ConfigError")
    if config_error is not None:
        console.print(f"[red]Error: {escape(str(config_error))}[/red]")
        raise typer.Exit(2)
    console.print(f"[red]Error: {escape(_known_failure(exc) or msg)}[/red]")
    raise typer.Exit(exit_code)


def _in_cause_chain(exc: Optional[BaseException], class_name: str) -> Optional[BaseException]:
    """The first exception named ``class_name`` in ``exc``'s cause chain.

    Matched by name, so the CLI does not import the module that defines it
    (the vector backend, the config) just to check.
    """
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if type(exc).__name__ == class_name:
            return exc
        exc = exc.__cause__ or exc.__context__
    return None


def main() -> None:
    """The `lasagna` command: the app, with a bad setting reported in one message.

    A ConfigError names the setting, its value and the fix. Raised from a
    command that does not catch it, it arrived inside a 40-line traceback.
    """
    try:
        app()
    except Exception as exc:
        config_error = _in_cause_chain(exc, "ConfigError")
        if config_error is None:
            raise
        console.print(f"[red]Error: {escape(str(config_error))}[/red]")
        raise SystemExit(2) from None


def _home(path: str) -> str:
    """``path`` with the home directory written as ``~``."""
    home = str(Path.home())
    path = str(path)
    if path == home or path.startswith(home + os.sep):
        return "~" + path[len(home) :]
    return path


def _known_failure(exc: Optional[BaseException]) -> Optional[str]:
    """The one-sentence explanation and fix for a failure we can name.

    Recognized by class name anywhere in the cause chain, so a wrapper that
    re-raises keeps the cause. A store held by `lasagna serve` was reported as
    "Failed to get status:" plus a 300-character message wrapped across five
    lines.
    """
    # Offline with a benchmark's data not cached: the message names the fix.
    not_cached = _in_cause_chain(exc, "DatasetNotCached")
    if not_cached is not None:
        return str(not_cached)
    locked = _in_cause_chain(exc, "DataDirLockedError")
    if locked is None:
        return None
    path = getattr(locked, "db_path", None)
    where = f" at {_home(path)}" if path else ""
    return (
        f"Another process is using the local store{where} (a running "
        f"`lasagna serve`, MCP server or benchmark). Stop it and run this "
        f"again, or give one of them its own store with "
        f"RAG_DB_DIR=<another directory>."
    )


def _require_known(kind: str, name: str, known, listing: str) -> None:
    """Exit with one sentence when ``name`` is not one of ``known``."""
    from .core._text import did_you_mean

    if name in known:
        return
    _error(f"Unknown {kind} {name!r}{did_you_mean(name, known)}. {listing}")


def _require_vector_backend() -> None:
    """Exit with one sentence when RAG_VECTOR_BACKEND names no backend, before anything loads.

    `RAG_VECTOR_BACKEND=milvuss lasagna benchmark run smoke` ran, failed its
    task, and advised pointing RAG_DB_DIR at a new directory.
    """
    from .core.config import rag_config
    from .core.diagnostics import unknown_vector_backend

    unknown = unknown_vector_backend(rag_config.vector_backend or "milvus")
    if unknown is not None:
        what, advice = unknown
        _error(f"Vector store: {what}. To fix it, {advice}.")


def _require_benchmark(name: str) -> None:
    from .knowledge_evaluation_sdk.benchmarks import benchmark_registry

    _require_known(
        "benchmark", name, benchmark_registry.names(), "`lasagna benchmark list` shows them all."
    )


def _require_profile(name: str, session_id: str = "default") -> None:
    from .core.session_config import get_knowledge_config_manager

    profiles = get_knowledge_config_manager(session_id=session_id).list_profiles(
        session_id=session_id
    )
    names = [p["name"] for p in profiles]
    _require_known("profile", name, names, f"Profiles: {', '.join(sorted(names))}.")


def _require_component(stage: str, name: str) -> None:
    """A registered name, or a ``module:attribute`` reference the sweep imports."""
    from . import components
    from .core.registry import registry

    if components.is_reference(name) or registry.has(stage, name):
        return
    listing = f"`lasagna component list {stage}` shows them."
    if "." in name or "/" in name:
        # Looks like an import path: the likely miss is how it was named.
        listing = (
            f"To use a {stage} from your own module, name it as module:attribute "
            f"(for example my_pkg:MyReranker), or install it as a "
            f"lasagna_components entry point. {listing}"
        )
    _require_known(stage.replace("_", " "), name, registry.list(stage)[stage], listing)


def _require_metric(results, metric: str, benchmark: str = "", where: str = "these results"):
    """Exit with the recorded metrics when ``results`` never recorded ``metric``.

    Every results command checks this before it computes anything: a typo in
    `-m` got a significance verdict and exit 0.
    """
    try:
        results.require_metric(metric, benchmark, where)
    except ValueError as exc:
        _error(str(exc))


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init():
    """Interactive setup wizard for RAG knowledge base.

    Asks for a vector backend and a quality profile, writes the backend to
    ./settings.lasagna.toml and applies the profile to the default session.
    The profile decides the embedding model. The LLM provider is not asked
    for: Lasagna picks it from the API key exported in the shell, so init
    names the provider that key selects, or which variable to export. API
    keys are not asked for or written anywhere.
    """
    try:
        import questionary
    except ImportError:
        console.print("[red]questionary is required for init wizard.[/red]")
        console.print("Install with: pip install questionary")
        raise typer.Exit(1)

    from lasagna.core.session_config._profiles import ProfileMixin
    from lasagna.core.setup_text import needs_llm_key, profile_download_text
    from lasagna.providers import detected_llm_key
    from lasagna.providers.llm_provider import SimpleLLMProvider
    from lasagna.utils.model_utils import get_default_model_name

    console.print()
    # The subtitle sat on the border, wider than the title: the box cut it off.
    console.print(Panel.fit("[bold cyan]Lasagna setup[/bold cyan]\nConfigure your knowledge base"))
    console.print()

    # 1. Vector backend. (There was an embedding question first, "Cloud API"
    # or local. Nothing read its answer: the profile chooses the model, and
    # the summary named text-embedding-3-small while `status` showed MiniLM.)
    vector_backend = questionary.select(
        "Vector database?",
        choices=[
            questionary.Choice("Milvus Lite (zero-config, embedded)", value="milvus"),
            questionary.Choice("pgvector (PostgreSQL extension)", value="pgvector"),
            questionary.Choice("Pinecone (managed cloud)", value="pinecone"),
            questionary.Choice("Weaviate (self-hosted or cloud)", value="weaviate"),
            questionary.Choice("Qdrant (self-hosted or cloud)", value="qdrant"),
        ],
    ).ask()
    if vector_backend is None:
        raise typer.Exit(0)

    # There was an "LLM provider?" question here too. Nothing read its answer:
    # the provider follows the API key exported in the shell, and the summary
    # named one nothing would call. The summary now says which key does.

    # 2. Quality profile
    profile = questionary.select(
        "Quality profile?",
        choices=[
            questionary.Choice(f"{label} ({profile_download_text(value)})", value=value)
            for label, value in (
                ("Speed: vector search only, fastest", "speed"),
                ("Standard: hybrid search and reranking, recommended", "standard"),
                ("Balanced: standard plus hierarchical chunking", "balanced"),
                ("Max quality: every retrieval feature on", "max_quality"),
            )
        ],
        default="standard",
    ).ask()
    if profile is None:
        raise typer.Exit(0)

    # Write settings override.
    #
    # This file used to be written and then never read: the loader looks in the
    # installed package directory or at $RAG_SETTINGS_FILE, never the working
    # directory. So a user picked the speed profile, saw "Setup complete!"
    # naming it, and downloaded 3.2 GB anyway. Pointing RAG_SETTINGS_FILE at the
    # file we just wrote is what makes the wizard's answers take effect.
    try:
        settings_path = Path("settings.lasagna.toml").resolve()
        settings_path.write_text(f'[rag]\nvector_backend = "{vector_backend}"\n')
        console.print(f"  [green]Created {settings_path.name}[/green]")
    except OSError as exc:
        _error(f"Could not write settings file: {exc}")

    try:
        env_path = Path(".env")
        existing = env_path.read_text() if env_path.exists() else ""
        if "RAG_SETTINGS_FILE" not in existing:
            with open(env_path, "a") as f:
                f.write(f"RAG_SETTINGS_FILE={settings_path}\n")
            console.print("  [green]Pointed RAG_SETTINGS_FILE at it in ./.env[/green]")
    except OSError as exc:
        console.print(
            f"  [yellow]Warning: could not set RAG_SETTINGS_FILE ({exc}). Set it "
            f"yourself, or these settings will not be used.[/yellow]"
        )

    # The profile was collected, displayed in the summary, and thrown away --
    # it was never part of config_data. Applying it here is what makes the
    # choice mean something.
    try:
        from lasagna.core.session_config import get_knowledge_config_manager

        get_knowledge_config_manager().set_active_profile(profile)
        console.print(f"  [green]Activated the {profile} profile[/green]")
    except Exception as exc:
        console.print(
            f"  [yellow]Warning: could not activate the {profile} profile ({exc}). "
            f"Run `lasagna configure --profile {profile}`.[/yellow]"
        )

    # What `lasagna status` will show: the profile's model.
    embedding = ProfileMixin._PROFILE_SPECS[profile]["embedding_model"]
    # What will answer an LLM call: the provider the exported key selects.
    found = detected_llm_key()
    if found is not None:
        key_name, platform = found
        model = get_default_model_name(SimpleLLMProvider.SUPPORTED_PLATFORMS.get(platform, ""))
        llm = f"{platform}, {model}" if model else platform
        key_line = f"({key_name} is set in this shell)"
    else:
        llm = "none"
        uses = (
            f"The {profile} profile sends complex queries to an LLM; without a key they "
            f"run its standard pipeline."
            if needs_llm_key(profile)
            else f"The {profile} profile searches without one."
        )
        key_line = (
            f"(no API key is set; the provider follows the key you export)\n{uses} "
            f"For LLM features (answer generation, HyDE, query expansion):\n"
            f"  export OPENAI_API_KEY=<your key>   (or ANTHROPIC_API_KEY, GROQ_API_KEY, ...)\n"
            f"and add that line to your shell profile (~/.zshrc or ~/.bashrc) to keep it."
        )

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Setup complete![/bold green]\n\n"
            f"Profile: [cyan]{profile}[/cyan]\n"
            f"Vector DB: [cyan]{vector_backend}[/cyan]\n"
            f"Embeddings: [cyan]{embedding}[/cyan] (the {profile} profile's model)\n"
            f"LLM: [cyan]{escape(llm)}[/cyan] {escape(key_line)}\n\n"
            f"[dim]Quick start (from this directory):[/dim]\n"
            f"  lasagna upload ./document.pdf\n"
            f'  lasagna search "your query"\n'
            f"  lasagna serve\n\n"
            f"[dim]`lasagna` commands run here read RAG_SETTINGS_FILE from ./.env.\n"
            f"Python code and other directories do not: there, first run\n"
            f"  export RAG_SETTINGS_FILE={settings_path}\n"
            f"Lasagna reads no other key from .env and never reads API keys from it.[/dim]",
        )
    )


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


def _reported_document_failure(record: Dict[str, Any]) -> bool:
    """The worker's log line for a document that failed.

    `upload` waits for the document and prints the failure, with its fix, as
    its error; the worker's ERROR line repeated it before that.
    """
    return "document_failure" in record["extra"]


def _upload_targets(patterns: List[str]) -> List[Path]:
    """The files ``patterns`` name: paths, or globs the shell left unexpanded.

    Exits naming the first one that matches no file.
    """
    import glob

    files: List[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_file():
            files.append(path)
            continue
        matched = sorted(Path(m) for m in glob.glob(pattern, recursive=True) if Path(m).is_file())
        if not matched:
            _error(
                f"File not found: {pattern}"
                if not glob.has_magic(pattern)
                else f"No file matches {pattern}"
            )
        files.extend(matched)
    return list(dict.fromkeys(files))


@app.command()
def upload(
    ctx: typer.Context,
    file_paths: List[str] = typer.Argument(
        ..., help="Files to upload: one or more paths, or glob patterns such as 'docs/*.md'"
    ),
    session_id: str = typer.Option("default", "--session", "-s", help="Session ID"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Upload the file again even when the session already holds the same content.",
    ),
):
    """Upload and index documents.

    Name several files, or a glob such as 'docs/*.md', in one command: they go
    through one client, so the models load once instead of once per file.
    """
    _configure_cli_logging(
        bool(ctx.find_root().params.get("verbose")),
        hide=lambda r: _server_only_warning(r) or _reported_document_failure(r),
    )
    paths = _upload_targets(file_paths)
    _require_vector_backend()

    url = _serving_url()
    names = paths[0].name if len(paths) == 1 else _count(len(paths), "file")
    console.print(f"Uploading [cyan]{escape(names)}[/cyan]...")

    async def _upload():
        client = _client_for(session_id, url)
        outcomes = []
        try:
            # All queued first, then waited for: processing runs in the
            # background, several files at a time.
            queued = []
            for path in paths:
                try:
                    queued.append((path, await client.upload(str(path), allow_duplicates=force)))
                except Exception as exc:  # noqa: BLE001 - reported per file below
                    queued.append((path, exc))
            for path, result in queued:
                if isinstance(result, Exception) or getattr(result, "duplicate_of", None):
                    outcomes.append((path, result, None))
                    continue
                # Closing the client before processing finishes tears down the
                # vector store under the worker, so wait for a terminal status.
                try:
                    doc = await client.documents.wait_for_processing(result.document_id)
                except Exception as exc:  # noqa: BLE001
                    outcomes.append((path, exc, None))
                    continue
                result.status = doc.status
                result.chunk_count = getattr(doc, "chunk_count", result.chunk_count)
                outcomes.append((path, result, doc))
        finally:
            await client.close()
        return outcomes

    try:
        outcomes = _run(_upload())
    except Exception as exc:
        _error(f"Upload failed: {exc}", exc=exc)

    if len(outcomes) == 1:
        _report_one_upload(*outcomes[0], session_id)
        return
    _report_uploads(outcomes, session_id)


def _already_uploaded(path: Path, result, session_id: str) -> str:
    return (
        f"Already uploaded: {escape(path.name)} has the same content as "
        f"{result.duplicate_of} in session {escape(session_id)}, so it was not uploaded "
        f"again. To upload it again: lasagna upload {shlex.quote(str(path))} "
        f"-s {shlex.quote(session_id)} --force"
    )


def _report_one_upload(path: Path, result, doc, session_id: str) -> None:
    from lasagna.sdk.types import DocumentStatus

    if isinstance(result, Exception):
        _error(f"Upload failed: {result}", exc=result)
    if doc is None:
        console.print(_already_uploaded(path, result, session_id))
        return
    if doc.status != DocumentStatus.READY:
        # A document that failed to index printed the green "Uploaded
        # successfully" panel with "Status: DocumentStatus.FAILED", exit 0.
        _error(
            f"Upload failed: {path.name} could not be indexed "
            f"({doc.error_message or 'no reason was recorded'}). "
            f"Document ID: {result.document_id}"
        )
    console.print(
        Panel.fit(
            f"[green]Uploaded successfully[/green]\n\n"
            f"Document ID: [cyan]{result.document_id}[/cyan]\n"
            f"Chunks: [cyan]{result.chunk_count}[/cyan]\n"
            f"Status: [cyan]{getattr(result.status, 'value', result.status)}[/cyan]",
        )
    )


def _report_uploads(outcomes, session_id: str) -> None:
    """One row per file, then one line per failure; exit 1 when any failed."""
    from lasagna.sdk.types import DocumentStatus

    table = Table(title=f"Uploaded to session {escape(session_id)}", show_lines=False)
    table.add_column("File", ratio=1)
    table.add_column("Document ID", style="cyan")
    table.add_column("Chunks", style="green", justify="right")
    table.add_column("Status")
    failures = []
    for path, result, doc in outcomes:
        if isinstance(result, Exception):
            table.add_row(escape(path.name), "", "", "[red]failed[/red]")
            failures.append(f"{path.name}: {result}")
        elif doc is None:
            table.add_row(escape(path.name), result.duplicate_of, "", "already uploaded")
        elif doc.status != DocumentStatus.READY:
            table.add_row(escape(path.name), result.document_id, "", "[red]failed[/red]")
            failures.append(
                f"{path.name} could not be indexed "
                f"({doc.error_message or 'no reason was recorded'})"
            )
        else:
            status = getattr(result.status, "value", result.status)
            table.add_row(escape(path.name), result.document_id, str(result.chunk_count), status)
    console.print(table)
    if failures:
        for line in failures:
            console.print(f"[red]Upload failed: {escape(line)}[/red]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
    session_id: str = typer.Option("default", "--session", "-s", help="Session ID"),
    context: bool = typer.Option(False, "--context", "-c", help="Show formatted LLM context"),
):
    """Search the knowledge base."""
    from lasagna.core.model_downloads import sessions_with_documents

    _require_vector_backend()

    # Before any model loads: searching a session with no documents loaded the
    # default model (or failed to, offline) to return nothing.
    populated = sessions_with_documents()
    if populated is not None and session_id not in populated:
        if populated:
            listed = ", ".join(f"{s} ({n})" for s, n in sorted(populated.items()))
            first = sorted(populated)[0]
            _error(
                f"Session {session_id!r} has no documents. Sessions with documents: "
                f"{listed}. Search one with: lasagna search {shlex.quote(query)} -s {first}"
            )
        _error(
            f"Session {session_id!r} has no documents, and no session has any. Add one "
            f"with: lasagna upload <file> -s {session_id}"
        )

    url = _serving_url()

    async def _search():
        client = _client_for(session_id, url)
        try:
            if context:
                return await client.get_context(query, max_chunks=top_k)
            return await client.search_documents(query, top_k=top_k)
        finally:
            await client.close()

    import warnings

    from lasagna.api._profile_guard import DocumentsElsewhereWarning

    try:
        with warnings.catch_warnings(record=True) as warned:
            warnings.simplefilter("always", DocumentsElsewhereWarning)
            result = _run(_search())
    except Exception as exc:
        # The SDK's SearchError already says "Search failed: ...".
        cause = exc.__cause__ if type(exc).__name__ == "SearchError" and exc.__cause__ else exc
        _error(f"Search failed: {cause}", exc=exc)
    elsewhere = any(issubclass(w.category, DocumentsElsewhereWarning) for w in warned)

    if context:
        # Chunk text is data, not markup: Rich deleted "[metrics](...)" and
        # "[copy.deepcopy(d) ...]" from what it printed.
        console.print(Panel(escape(str(result)), title="LLM Context", border_style="cyan"))
        if elsewhere:
            _warn_documents_elsewhere(session_id, warned)
        return

    table = Table(title=f'Search: "{escape(query)}"', show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Score", style="green", width=8)
    table.add_column("Content", ratio=1)
    table.add_column("Source", style="cyan", width=20)

    for i, r in enumerate(result, 1):
        content_text = r.chunk.content or ""
        content = content_text[:200].replace("\n", " ")
        if len(content_text) > 200:
            content += "..."
        source = getattr(getattr(r, "document", None), "filename", None) or (
            (r.chunk.document_id or "")[:12] + "..."
        )
        table.add_row(str(i), f"{r.score:.3f}", escape(content), escape(source))

    console.print(table)
    console.print(f"[dim]{_count(len(result), 'result')}[/dim]")
    if elsewhere:
        _warn_documents_elsewhere(session_id, warned)


def _warn_documents_elsewhere(session_id: str, warned: List[Any]) -> None:
    """Say why the search could not see some of the session's documents.

    Read from this machine's metadata database, which a local server shares;
    the SDK's own warning is the fallback.
    """
    from lasagna.api._profile_guard import DocumentsElsewhereWarning, documents_elsewhere

    message = None
    try:
        from lasagna.core.config import rag_config

        config = rag_config.with_session(session_id)
        message = documents_elsewhere(session_id, config.collection_name, config.embedding_model)
    except Exception:  # noqa: BLE001 - the SDK's warning says it too
        message = None
    if message is None:
        first = next(w for w in warned if issubclass(w.category, DocumentsElsewhereWarning))
        message = str(first.message)
    console.print(f"[yellow]Warning: {escape(message)}[/yellow]")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status(
    session_id: str = typer.Option("default", "--session", "-s", help="Session ID"),
):
    """Show knowledge base status."""
    url = _serving_url()
    if not url:
        # A typo ("milvuss") printed "No documents uploaded yet." and exit 0:
        # the local store is read only when this names a backend.
        _require_vector_backend()

    async def _served_status():
        client = _client_for(session_id, url)
        try:
            # The server's view of the session, not this process's defaults.
            return await client.list_documents(), await _served_config(url, session_id), []
        finally:
            await client.close()

    try:
        if url:
            docs, config, unused = _run(_served_status())
        else:
            # From the metadata database and the settings, with no backend and
            # no model: it took 7 s to list three documents.
            docs, config = _local_documents(session_id), _effective_config(session_id)
            unused = _unused_collections_if_any()
    except Exception as exc:
        _error(f"Failed to get status: {exc}", exc=exc)

    # Documents table
    if docs:
        table = Table(title="Documents", show_lines=True)
        table.add_column("ID", style="cyan", width=15)
        table.add_column("Filename", ratio=1)
        table.add_column("Chunks", style="green", width=8)
        table.add_column("Status", width=12)

        for doc in docs:
            doc_id = (doc.document_id or "")[:12] + "..."
            table.add_row(doc_id, doc.filename or "", str(doc.chunk_count), doc.status or "")

        console.print(table)
    else:
        from lasagna.core.model_downloads import sessions_with_documents

        others = {s: n for s, n in (sessions_with_documents() or {}).items() if s != session_id}
        if others:
            # It said "No documents uploaded yet." with my-corpus holding three.
            listed = ", ".join(f"{s} ({n})" for s, n in sorted(others.items()))
            console.print(
                f"[dim]No documents in session {escape(session_id)!r}. Sessions with "
                f"documents: {escape(listed)}. Show one with: lasagna status -s "
                f"{escape(shlex.quote(sorted(others)[0]))}[/dim]",
                soft_wrap=True,
            )
        else:
            console.print("[dim]No documents uploaded yet.[/dim]")
    for collection in unused:
        dim = collection.get("embedding_dim")
        size = f"{dim} dimensions" if dim else "an unknown size"
        console.print(
            f"[yellow]Unused: collection {escape(collection['name'])} holds "
            f"{collection['rows']:,} chunks at {size} from before 0.1.0, and no configured "
            f"model reads it. To search it again, configure the embedding model that "
            f"wrote it and set LASAGNA_LEGACY_COLLECTION_MODEL to that model's name.[/yellow]"
        )

    # Config summary
    console.print()
    config = config or {}
    features = []
    if config.get("enable_hybrid_search"):
        features.append("hybrid search")
    if config.get("enable_reranking"):
        features.append("reranking")
    if config.get("enable_mmr"):
        features.append("MMR diversity")
    if config.get("enable_hyde"):
        features.append("HyDE")
    if config.get("enable_hierarchical"):
        features.append("hierarchical chunking")

    console.print(
        Panel.fit(
            f"Session: [cyan]{session_id}[/cyan]\n"
            f"Documents: [cyan]{len(docs)}[/cyan]\n"
            f"Embedding: [cyan]{config.get('embedding_model', 'unknown')}[/cyan]\n"
            f"Features: [cyan]{', '.join(features) or 'none'}[/cyan]",
            title="Configuration",
        )
    )


#: What `status` and the `serve` banner report about a session's configuration.
_SHOWN_SETTINGS = (
    "embedding_model",
    "embedding_dim",
    "vector_backend",
    "enable_hybrid_search",
    "enable_reranking",
    "enable_mmr",
    "enable_hyde",
    "enable_hierarchical",
)


def _effective_config(session_id: str) -> Dict[str, Any]:
    """The settings a search on ``session_id`` runs with, every layer applied.

    Session over environment over settings file over defaults: the same
    resolution search uses, so what `status` prints is what runs.
    """
    from lasagna.core.config import rag_config

    config = rag_config.with_session(session_id)
    return {key: getattr(config, key) for key in _SHOWN_SETTINGS}


async def _served_config(url: str, session_id: str) -> Dict[str, Any]:
    """The session's configuration as a running server resolves it."""
    import httpx

    async with httpx.AsyncClient(timeout=10) as http:
        resp = await http.get(f"{url}/api/v1/config", headers={"X-Session-ID": session_id})
        resp.raise_for_status()
        return resp.json()


def _local_documents(session_id: str) -> List[Any]:
    """The session's documents, read from the metadata database."""
    from types import SimpleNamespace

    from lasagna.core.database import get_rag_database
    from lasagna.sdk.types import DocumentStatus

    def status(raw: Any) -> str:
        try:
            return DocumentStatus(raw).value  # "error" reads as "failed", as the SDK says
        except ValueError:
            return str(raw or "")

    return [
        SimpleNamespace(
            document_id=row.get("document_id"),
            filename=row.get("filename"),
            chunk_count=row.get("chunks_count", 0),
            status=status(row.get("status")),
        )
        for row in get_rag_database().get_documents(session_id)
    ]


def _may_hold_legacy_documents() -> bool:
    """Whether any document predates per-model collections (no collection, or the old one)."""
    from lasagna.core.config import LEGACY_COLLECTION_NAME
    from lasagna.core.database import get_rag_database
    from lasagna.core.model_downloads import sessions_with_documents

    db = get_rag_database()
    return any(
        row.get("collection_name") in (None, "", LEGACY_COLLECTION_NAME)
        for session in (sessions_with_documents() or {})
        for row in db.get_documents(session)
    )


def _unused_collections_if_any() -> List[Dict[str, Any]]:
    """Collections from before 0.1.0 that no configured model reads.

    The vector store is opened only when a document could live in one: it is
    the slow part of `status`. A listing: it never fails the command.
    """
    try:
        if not _may_hold_legacy_documents():
            return []
        from lasagna.api._collection_manager import legacy_collections
        from lasagna.api.rag_backend import get_rag_backend

        return legacy_collections(get_rag_backend()._vector_backend)
    except Exception as exc:  # noqa: BLE001
        from loguru import logger

        logger.debug(f"Could not list unused collections: {exc}")
        return []


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


#: loguru level -> uvicorn's --log-level, so one setting governs both.
_UVICORN_LEVELS = {"TRACE": "trace", "DEBUG": "debug", "INFO": "info", "SUCCESS": "info"}


def _session_summary(session_id: str = "default") -> List[str]:
    """What a request on ``session_id`` runs with, for the `serve` banner.

    The server's start-up log named the global default embedding model
    (BAAI/bge-m3, 1024 dimensions) while the default session, after
    `lasagna configure --profile speed`, searched with all-MiniLM-L6-v2 at 384.
    """
    try:
        from lasagna.core.config import rag_config
        from lasagna.core.session_config import get_knowledge_config_manager

        mgr = get_knowledge_config_manager(session_id=session_id)
        profile = mgr.get_active_profile(session_id=session_id)
        # Every layer applied, as search resolves it: the profile's stored
        # snapshot of the settings file ignored the environment.
        effective = _effective_config(session_id)
        model = effective["embedding_model"]
        dim = effective["embedding_dim"]
        backend = effective["vector_backend"]
        data = os.path.dirname(rag_config.db_path)
        if os.path.dirname(os.path.abspath(rag_config.upload_dir)) == os.path.dirname(data):
            data = os.path.dirname(data)  # dbs/ and uploads/ side by side
        others = _sessions_with_documents_line(mgr, exclude=session_id)
    except Exception as exc:  # the banner must never stop the server
        return [f"Session config: could not be read ({type(exc).__name__}: {exc})"]
    return [
        *others,
        f"Without X-Session-ID, requests use session '{session_id}':",
        f"Profile: {profile} (session '{session_id}')",
        f"Embedding: {model} ({dim} dimensions)",
        f"Vector store: {backend}",
        f"Data: {_home(data)}",
    ]


def _sessions_with_documents_line(mgr, exclude: str) -> List[str]:
    """``Sessions with documents: my-corpus (76; speed, all-MiniLM-L6-v2)``, or nothing.

    The banner described only session 'default', while the README configures
    and fills `my-corpus`: it named a model none of the user's requests ran.
    """
    from lasagna.core.config import rag_config
    from lasagna.core.model_downloads import sessions_with_documents

    populated = {s: n for s, n in (sessions_with_documents() or {}).items() if s != exclude}
    parts = []
    for sid, count in sorted(populated.items())[:5]:
        chosen = mgr.get_chosen_config(sid)
        model = chosen.get("embedding_model") or rag_config.embedding_model
        parts.append(f"{sid} ({count}; {mgr.get_active_profile(session_id=sid)}, {model})")
    if len(populated) > 5:
        parts.append(f"and {len(populated) - 5} more")
    return [f"Sessions with documents: {', '.join(parts)}"] if parts else []


def _port_problem(host: str, port: int) -> Optional[str]:
    """Why ``host:port`` cannot be bound (with uvicorn's socket options), or None."""
    import errno
    import socket

    family = socket.AF_INET6 if ":" in host else socket.AF_INET
    with socket.socket(family, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host or "0.0.0.0", port))
        except OSError as exc:
            if exc.errno == errno.EADDRINUSE:
                return f"Port {port} on {host} is already in use by another program."
            return f"Cannot listen on {host}:{port}: {exc.strerror or exc}."
    return None


def _configurator_line(base: str) -> str:
    """The banner's configurator line: the page, or how to build it.

    From a source checkout the page is not built, and the advertised
    /configurator answered 404.
    """
    from lasagna.configurator import FRONTEND_DIST

    if FRONTEND_DIST.is_dir():
        return f"Configurator: {base}/configurator"
    return (
        "Configurator: not built in this checkout "
        "(cd src/lasagna/configurator/frontend && npm ci && npm run build)"
    )


@app.command()
def serve(
    ctx: typer.Context,
    # No -p: everywhere else it is --profile.
    port: int = typer.Option(8000, "--port", help="Port to listen on."),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address. Default is loopback; --host 0.0.0.0 exposes the API "
        "on every interface, and it is UNAUTHENTICATED (see SECURITY.md).",
    ),
    configurator: bool = typer.Option(
        True,
        "--configurator/--no-configurator",
        help="Serve the configurator web page at /configurator.",
    ),
    open_browser: bool = typer.Option(
        False,
        "--open/--no-open",
        help="Open the configurator in a browser once the server starts.",
    ),
    serve_mcp: bool = typer.Option(
        False,
        "--mcp",
        help="Also serve the MCP server (SSE) at /mcp/sse, from this process. Milvus Lite "
        "admits one process per store, so this is how an agent and the REST API share one.",
    ),
    session: Optional[str] = typer.Option(
        None,
        "--session",
        "-s",
        help="The session for MCP tool calls and REST requests that name none "
        "(default: RAG_SESSION_ID, else 'default').",
    ),
):
    """Start the REST API server, with the configurator web page.

    While it runs it holds the local store; `lasagna status`, `search` and
    `upload` go through it.
    """

    def _hide(record) -> bool:
        if not _ephemeral_signing_secret_warning(record):
            return False
        # Decided when the warning arrives, from the module that sent it:
        # importing auth_mode any earlier builds the app, and with it the
        # warning, before this filter exists.
        from lasagna.api.rest.auth_mode import is_loopback, require_auth

        return is_loopback(host) and not require_auth()

    # Warnings only by default, as for every other command: the banner below
    # says what the server resolved. -v or LASAGNA_LOG_LEVEL=INFO shows more.
    level = _configure_cli_logging(bool(ctx.find_root().params.get("verbose")), hide=_hide)
    from lasagna.core import local_server

    _require_vector_backend()
    if local_server.local_store_held():
        running = local_server.running_server()
        _error(
            f"lasagna serve is already running at {running} and holds the local store."
            if running
            else local_server.held_store_message("lasagna serve")
        )
    problem = _port_problem(host, port)
    if problem:
        # Checked before anything loads: the banner used to print, and 6 s later
        # uvicorn's "[Errno 48] ... address already in use".
        _error(f"{problem} Start on another port: lasagna serve --port {port + 1}")
    if configurator:
        os.environ["RAG_CONFIGURATOR"] = "true"
    if serve_mcp:
        os.environ["RAG_EMBED_MCP"] = "true"
    # MCP tools and REST requests without a session use this one. Without it,
    # every MCP search over /mcp/sse failed: "Either session_id or
    # session_token is required".
    if session:
        os.environ["RAG_SESSION_ID"] = session
    default_session = os.environ.get("RAG_SESSION_ID", "").strip() or "default"

    # The address a browser can use: 0.0.0.0 and :: are bind-alls, not addresses.
    shown = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    shown = f"[{shown}]" if ":" in shown else shown
    base = f"http://{shown}:{port}"
    console.print(
        Panel.fit(
            f"[bold cyan]Lasagna API[/bold cyan]\n\n"
            f"API: {base}/api/v1\n"
            f"Health: {base}/api/v1/health\n"
            f"Docs: {base}/docs\n"
            + (f"{_configurator_line(base)}\n" if configurator else "")
            + (f"MCP (SSE): {base}/mcp/sse\n" if serve_mcp else "")
            + "\n"
            + "\n".join(escape(line) for line in _session_summary(default_session))
            + "\n\n[dim]Press Ctrl+C to stop.[/dim]",
        )
    )

    if open_browser and configurator:
        import threading

        def _open():
            import time

            time.sleep(2)
            try:
                import webbrowser

                webbrowser.open(f"{base}/configurator")
            except Exception:
                pass  # Silently skip on headless systems

        threading.Thread(target=_open, daemon=True).start()

    import uvicorn

    from lasagna.api.rest.auth_mode import (
        UnsafeBindError,
        require_auth_or_refuse_to_start,
        warn_once_if_unauthenticated,
    )

    # Refuse rather than warn. Every route here is reachable without a
    # credential when auth is off -- including irreversible document deletion --
    # and `mcp/server.py` already established that a startup-log warning is not
    # read by the person it affects.
    try:
        require_auth_or_refuse_to_start(host, surface="REST API")
    except UnsafeBindError as exc:
        _error(str(exc))
    warn_once_if_unauthenticated(host)

    # Forwarding headers are trusted only from a proxy the operator names.
    # Trusting 127.0.0.1 by default let any local client choose its own
    # X-Forwarded-For, and with it a fresh rate-limit bucket per request: 40
    # of 40 session mints went through where 35 should have been refused. A
    # proxy on this host and a client on it are indistinguishable by address.
    trusted_proxies = os.environ.get("RAG_FORWARDED_ALLOW_IPS", "").strip()
    # Other commands find this server here and go through it (core.local_server).
    local_server.announce(base)
    try:
        uvicorn.run(
            "lasagna.api.rest.app:app",
            host=host,
            port=port,
            reload=False,
            proxy_headers=bool(trusted_proxies),
            forwarded_allow_ips=trusted_proxies or None,
            log_level=_UVICORN_LEVELS.get(level, level.lower()),
        )
    finally:
        local_server.withdraw()


# ---------------------------------------------------------------------------
# configure
# ---------------------------------------------------------------------------


@app.command()
def configure(
    profile: Optional[str] = typer.Option(
        None,
        "--profile",
        "-p",
        help="Apply a preset profile (speed/standard/balanced/max_quality)",
    ),
    session_id: str = typer.Option("default", "--session", "-s", help="Session ID"),
    force: bool = typer.Option(
        False,
        "--force",
        help="Apply even if the session's stored documents would no longer be searched "
        "(they stay where they are until reindexed); the new models must load",
    ),
    reindex: bool = typer.Option(
        False,
        "--reindex",
        help="Reindex the session's documents with the new profile, then apply it; if one "
        "fails, the session keeps its profile and every document stays searchable",
    ),
):
    """Configure RAG settings.

    With --profile, applies a preset. Without arguments, lists available profiles.
    """
    from lasagna.core.session_config import get_knowledge_config_manager

    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
    except Exception as exc:
        _error(f"Failed to initialize config manager: {exc}")

    if profile:
        _require_profile(profile, session_id)
        from lasagna.core import local_server

        if local_server.local_store_held():
            # `lasagna serve` holds the store: it applies the switch, as it
            # serves status and search. Its readiness check tells the user to
            # run this command, which used to fail here.
            url = local_server.running_server()
            if url is not None:
                typer.echo(
                    f"lasagna serve at {url} holds the local store; going through it.", err=True
                )
                _configure_through_server(url, profile, session_id, force=force, reindex=reindex)
                return
            if reindex:
                _error(local_server.held_store_message("configure --reindex"))
        from lasagna.api._profile_guard import EmbeddingSizeConflict
        from lasagna.api._profile_switch import (
            ProfileCannotLoad,
            ReindexFailed,
            summary,
            switch_profile,
        )

        try:
            report = switch_profile(
                mgr, profile, session_id=session_id, force=force, reindex=reindex
            )
        except EmbeddingSizeConflict as exc:
            _error(
                f"{exc} To switch and reindex them now: lasagna configure --profile "
                f"{profile} -s {session_id} --reindex. To switch anyway: lasagna "
                f"configure --profile {profile} -s {session_id} --force."
            )
        except (ProfileCannotLoad, ReindexFailed, ValueError) as exc:
            _error(str(exc))
        finally:
            if reindex:
                from lasagna.api.rag_backend import close_backend

                close_backend()
        console.print(f"[green]Applied profile: {profile}[/green]")
        done = summary(report)
        if done:
            console.print(done)
        return

    # List profiles
    profiles = mgr.list_profiles(session_id=session_id)
    active = mgr.get_active_profile(session_id=session_id)

    table = Table(title="Available Profiles")
    table.add_column("Name", style="cyan")
    table.add_column("Description", ratio=1)
    table.add_column("Active", width=8)

    for p in profiles:
        marker = "[green]***[/green]" if p["name"] == active else ""
        table.add_row(p["name"], p.get("description", ""), marker)

    console.print(table)
    console.print("\n[dim]Use --profile NAME to apply a profile[/dim]")


# ---------------------------------------------------------------------------
# export / import
# ---------------------------------------------------------------------------


def _configure_through_server(
    url: str, profile: str, session_id: str, *, force: bool, reindex: bool
) -> None:
    """Switch ``session_id`` to ``profile`` through the running ``lasagna serve`` at ``url``."""
    import httpx

    body = {"active_profile": profile, "force": force, "reindex": reindex}
    try:
        # A reindex runs inside the request, document by document.
        resp = httpx.put(
            f"{url}/api/config/knowledge/active",
            json=body,
            headers={"X-Session-ID": session_id},
            timeout=None if reindex else 60,
        )
    except httpx.HTTPError as exc:
        _error(f"lasagna serve at {url} did not answer: {exc}")
    try:
        data = resp.json()
    except ValueError:
        data = {}
    if resp.status_code >= 400:
        detail = data.get("detail") if isinstance(data, dict) else None
        message = detail if isinstance(detail, str) else resp.text
        if resp.status_code == 409 and not (force or reindex) and "--reindex" not in message:
            message += (
                f" To switch and reindex them now: lasagna configure --profile {profile} "
                f"-s {session_id} --reindex. To switch anyway: lasagna configure --profile "
                f"{profile} -s {session_id} --force."
            )
        _error(message)
    console.print(f"[green]Applied profile: {profile}[/green]")
    report = data.get("reindex") if isinstance(data, dict) else None
    if report:
        from lasagna.api._profile_switch import SwitchReport, summary

        done = summary(
            SwitchReport(
                profile=profile,
                previous=report.get("previous_profile", ""),
                session_id=session_id,
                embedding_model=report.get("embedding_model", ""),
                reindexed=list(report.get("reindexed") or []),
                total=int(report.get("documents") or 0),
                reindex=True,
            )
        )
        if done:
            console.print(done)


@app.command(name="export")
def export_config(
    output: str = typer.Option("rag-config.json", "--output", "-o", help="Output file"),
    session_id: str = typer.Option("default", "--session", "-s", help="Session ID"),
):
    """Export current RAG configuration to JSON."""
    from lasagna.core.session_config import get_knowledge_config_manager

    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        data = mgr.export_config(session_id)
        with open(_output_file(output), "w") as f:
            json.dump(data, f, indent=2, default=str)
        console.print(f"[green]Configuration exported to {output}[/green]")
    except OSError as exc:
        _error(f"Failed to write file: {exc}")
    except Exception as exc:
        _error(f"Export failed: {exc}")


@app.command(name="import")
def import_config(
    input_file: str = typer.Argument(..., help="JSON config file to import"),
    session_id: str = typer.Option("default", "--session", "-s", help="Session ID"),
):
    """Import RAG configuration from JSON."""
    path = Path(input_file)
    if not path.exists():
        _error(f"File not found: {input_file}")

    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        _error(f"Invalid JSON: {exc}")
    except OSError as exc:
        _error(f"Cannot read file: {exc}")

    from lasagna.core.session_config import get_knowledge_config_manager

    try:
        mgr = get_knowledge_config_manager(session_id=session_id)
        mgr.import_config(data, session_id=session_id)
        console.print(f"[green]Configuration imported from {input_file}[/green]")
    except ValueError as exc:
        _error(f"Import validation failed: {exc}")
    except Exception as exc:
        _error(f"Import failed: {exc}")


# ---------------------------------------------------------------------------
# Benchmark sub-app
# ---------------------------------------------------------------------------

benchmark_app = typer.Typer(
    name="benchmark",
    help="Run RAG benchmark evaluations and experiments.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


def _retrieval_corpus(registry, name: str, info) -> str:
    """Where a benchmark's retrieval is measured, if anywhere.

    The mode column alone said "generation" for SQuAD v2 and MuSiQue, which
    measure retrieval live against a corpus pooled from the samples' gold
    passages: an easier task than a real corpus, and recorded as such.
    """
    try:
        adapter = registry.get(name)
    except Exception:
        return "?"
    if getattr(adapter, "pools_gold_context", False):
        return "pooled gold passages"
    if info.eval_mode.value in ("retrieval", "end_to_end"):
        return "its corpus"
    return "-"


@benchmark_app.command("list")
def benchmark_list(
    category: Optional[str] = typer.Option(
        None, help="Filter by category (single_hop, multi_hop, fact_verification, passage_ranking)"
    ),
):
    """List all available benchmarks."""
    from .knowledge_evaluation_sdk.benchmarks import benchmark_registry

    benchmarks = benchmark_registry.list_benchmarks()
    table = Table(title="Available Benchmarks")
    table.add_column("Name", style="cyan")
    table.add_column("Category", style="green")
    table.add_column("Mode", style="yellow")
    table.add_column("Retrieval measured on")
    table.add_column("Description")

    shown = 0
    for name, info in sorted(benchmarks.items()):
        if category and info.category.value != category:
            continue
        table.add_row(
            name,
            info.category.value,
            info.eval_mode.value,
            _retrieval_corpus(benchmark_registry, name, info),
            info.description,
        )
        shown += 1

    console.print(table)
    if category:
        console.print(
            f"\n[dim]{shown} of {len(benchmarks)} benchmarks shown (filtered by {category})[/dim]"
        )
    else:
        console.print(f"\n[dim]{_benchmark_count_line(benchmark_registry, benchmarks)}[/dim]")


def _benchmark_count_line(registry, benchmarks) -> str:
    """``31 benchmarks, plus smoke (a wiring check) and 1 from plugins``.

    The README counts the benchmarks that measure something: built in, not
    ``smoke``. This printed every registered name (32, or 33 with a plugin)
    while the README said 31.
    """
    builtin = [n for n in benchmarks if registry.is_builtin(n) and n != "smoke"]
    others = [n for n in benchmarks if not registry.is_builtin(n)]
    line = f"{_count(len(builtin), 'benchmark')}"
    if "smoke" in benchmarks:
        line += ", plus smoke (a wiring check, not a result)"
    if others:
        line += f"{' and' if 'smoke' in benchmarks else ', plus'} {len(others)} from plugins or your own files"
    return line


@benchmark_app.command("info")
def benchmark_info(
    name: str = typer.Argument(..., help="Benchmark name (e.g. beir/nfcorpus)"),
):
    """Show detailed information about a benchmark."""
    from .knowledge_evaluation_sdk.benchmarks import benchmark_registry

    _require_benchmark(name)
    adapter = benchmark_registry.get(name)

    info = adapter.info()
    splits = adapter.available_splits()
    filters = adapter.available_filters()

    panel_text = (
        f"[bold]{info.name}[/bold]\n\n"
        f"Description: {info.description}\n"
        f"Category: [green]{info.category.value}[/green]\n"
        f"Eval Mode: [yellow]{info.eval_mode.value}[/yellow]\n"
    )
    if info.num_queries:
        panel_text += f"Queries: {info.num_queries:,}\n"
    if info.num_documents:
        panel_text += f"Documents: {info.num_documents:,}\n"
    if info.paper:
        panel_text += f"Paper: {info.paper}\n"
    if info.license:
        panel_text += f"License: {info.license}\n"
    panel_text += f"Splits: {', '.join(splits)}\n"
    if filters:
        panel_text += f"Filters: {filters}\n"
    if info.tags:
        panel_text += f"Tags: {', '.join(info.tags)}"

    console.print(Panel(panel_text, title="Benchmark Info"))


@benchmark_app.command("run")
def benchmark_run(
    name: str = typer.Argument(
        ...,
        help="A benchmark name, as `lasagna benchmark list` shows it, or your own data: "
        "a .jsonl or .csv file or a BEIR folder (docs/own-data.md).",
    ),
    samples: Optional[int] = typer.Option(
        100, "--samples", "-n", help="Queries to evaluate, at most."
    ),
    runs: int = typer.Option(
        1, "--runs", "-r", help="Runs per arm. Repeat runs when an LLM stage is in the loop."
    ),
    mode: str = typer.Option(
        "auto",
        "--mode",
        help="auto, live (retrieve from the benchmark's corpus) or generation_only "
        "(answer from the gold passages).",
    ),
    split: str = typer.Option("test", "--split", help="Dataset split."),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output directory (default: a new folder under ~/.lasagna)."
    ),
    seed: int = typer.Option(42, "--seed", help="Random seed for sampling queries."),
    profile: str = typer.Option(
        "speed",
        "--profile",
        "-p",
        help="Profile every arm starts from. speed needs no API key and downloads 92 MB; "
        "standard downloads 3.2 GB and sends complex queries to an LLM.",
    ),
    tracks: Optional[str] = typer.Option(
        None,
        "--tracks",
        help="retrieval, generation, or both, comma-separated. Default: retrieval, or both "
        "with --generator. Generation needs an LLM API key and the answer-scoring models, "
        "so it runs only when asked for.",
        show_default=False,
    ),
    reranker: Optional[str] = typer.Option(
        None, "--reranker", help="Run with this registered reranker."
    ),
    query_enhancer: Optional[str] = typer.Option(
        None, "--query-enhancer", help="Run with this registered query enhancer."
    ),
    generator: Optional[str] = typer.Option(
        None, "--generator", help="Write answers with this registered generator."
    ),
):
    """Run one benchmark, or a data file of your own, and print its leaderboard."""
    from .knowledge_evaluation_sdk.benchmarks.adapters.local_files import is_local_data_path
    from .knowledge_evaluation_sdk.experiment import (
        Experiment,
        ExperimentConfig,
        BenchmarkSpec,
        PipelineSweepSpec,
        _parse_tracks,
    )

    if tracks is None:
        # Naming a generator is asking for answers; it ran on nothing otherwise.
        tracks = "retrieval,generation" if generator else "retrieval"
    # A .jsonl or .csv file or a BEIR folder of your own (docs/own-data.md).
    bench = {"path": name} if is_local_data_path(name) else {"name": name}

    try:
        track_list = _parse_tracks([t.strip() for t in tracks.split(",") if t.strip()])
    except ValueError as exc:
        _error(str(exc))

    if mode not in ("auto", "live", "generation_only", "offline"):
        _error(
            f"Invalid mode '{mode}'. Must be 'auto', 'live' or "
            f"'generation_only' ('offline' is a deprecated alias)."
        )

    # Before anything loads: a misspelled name ran for seconds, failed inside
    # its task, and printed every registered name without the one meant.
    if "name" in bench:
        _require_benchmark(name)
    _require_profile(profile)
    _require_vector_backend()
    for stage, component in (
        ("reranker", reranker),
        ("query_enhancer", query_enhancer),
        ("generator", generator),
    ):
        if component:
            _require_component(stage, component)
    if "generation" in track_list:
        from .knowledge_evaluation_sdk.synthesize import llm_configured, missing_llm_message

        if not llm_configured():
            _error(
                missing_llm_message(
                    "The generation track",
                    "To measure retrieval only, run with --tracks retrieval.",
                )
            )

    console.print(
        f"Running benchmark [cyan]{name}[/cyan] (up to {_count(samples, 'sample')}, "
        f"{_count(runs, 'run')}, mode={mode}, split={split}, profile={profile}, "
        f"tracks={','.join(track_list)})"
    )

    # Without a way to name a component, "check it runs" measured the stock
    # pipeline, whatever the user had written. A file of your own names its
    # results folder by its stem: ./my.jsonl gave benchmark_._my.jsonl_<time>.
    label = Path(bench["path"]).stem if "path" in bench else name.replace("/", "_")
    config = ExperimentConfig(
        name=f"benchmark_{label}",
        benchmarks=[BenchmarkSpec(**bench, n_samples=samples, split=split)],
        pipeline=PipelineSweepSpec(
            base_profile=profile,
            rerankers=[reranker] if reranker else [],
            query_enhancers=[query_enhancer] if query_enhancer else [],
            generators=[generator] if generator else [],
        ),
        mode=mode,
        n_runs=runs,
        seed=seed,
        output_dir=output or "",
        tracks=track_list,
    )
    exp = Experiment(config=config)

    async def _execute():
        return await exp.run()

    try:
        results = _run(_execute())
    except Exception as exc:
        _error(f"Benchmark failed: {exc}", exc=exc)

    _print_leaderboard(results)

    if results.failed_tasks:
        n_failed = len(results.failed_tasks)
        console.print(f"\n[red]{_count(n_failed, 'task')} failed[/red]")
    console.print(f"Output: {escape(config.effective_output_dir())}", soft_wrap=True)
    _release_backend()
    _exit_if_tasks_failed(results)


def _print_leaderboard(results) -> None:
    """The run's leaderboard by its primary metric, as `benchmark run` ends."""
    metric_names = results.get_metric_names()
    if metric_names:
        # A retrieval run has no generated answers, so leading with the first
        # metric printed "exact_match 0.0000" beside a real NDCG.
        primary = next(
            (m for m in ("ndcg@10", "recall@10", "mrr") if m in metric_names), metric_names[0]
        )
        console.print(f"\n[bold]Results ({primary}):[/bold]")
        console.print(results.leaderboard(primary))
    else:
        console.print("[yellow]No metrics computed[/yellow]")
    cache = _llm_cache_line(getattr(results, "experiment_config", None) or {})
    if cache:
        console.print(cache)


def _llm_cache_line(meta: Dict[str, Any]) -> str:
    """``LLM cache: 412 hits, 37 misses (92%)`` for a run that generated answers
    and looked responses up; "" otherwise, so a retrieval-only run prints nothing.

    A fast run with a high hit rate replayed answers from an earlier run, and
    says so where its numbers are printed.
    """
    cache = meta.get("llm_cache") or {}
    hits, misses = int(cache.get("hits") or 0), int(cache.get("misses") or 0)
    if "generation" not in (meta.get("tracks") or []) or not cache.get("enabled"):
        return ""
    if not hits + misses:
        return ""
    return (
        f"LLM cache: {hits} {'hit' if hits == 1 else 'hits'}, "
        f"{misses} {'miss' if misses == 1 else 'misses'} ({hits / (hits + misses):.0%})"
    )


@benchmark_app.command("experiment")
def benchmark_experiment(
    config_file: str = typer.Argument(..., help="YAML or TOML config file"),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoints"),
    concurrency: Optional[int] = typer.Option(
        None, "--concurrency", "-c", help="Max parallel tasks"
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Call the LLM for every request instead of replaying responses cached by "
        "an earlier run (same as llm_cache: false)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print the arms, tasks and models, and estimate tokens, time and cost from "
        "a 3-sample probe per benchmark, without running the sweep or writing results",
    ),
    no_probe: bool = typer.Option(
        False, "--no-probe", help="With --dry-run, skip the probe, so nothing runs at all"
    ),
):
    """Run a multi-benchmark experiment from a config file."""
    from .knowledge_evaluation_sdk.experiment import Experiment, runs_were_identical
    from .knowledge_evaluation_sdk.checkpoint import CheckpointManager

    if not Path(config_file).exists():
        _error(f"Config file not found: {config_file}")

    try:
        exp = Experiment.from_file(config_file)
    except (ValueError, TypeError, KeyError, ImportError, OSError) as exc:
        # A bad sweep file is the user's to fix: the message, not a traceback.
        _error(f"{config_file}: {exc}")
    if concurrency:
        exp.config.max_concurrency = concurrency
    if no_cache:
        from dataclasses import replace

        exp.config.llm_cache = replace(exp.config.llm_cache, enabled=False)
    if no_probe and not dry_run:
        _error("--no-probe only applies to a dry run. Add --dry-run, or drop --no-probe.")
    if dry_run:
        # Before anything below can clear checkpoints or make a folder.
        from .knowledge_evaluation_sdk.cost import dry_run_lines

        try:
            lines = dry_run_lines(exp, probe=not no_probe)
        except (ValueError, TypeError, KeyError, ImportError) as exc:
            _error(str(exc))
        for line in lines:
            console.print(escape(line), soft_wrap=True)
        _release_backend()
        return

    if resume and not (exp.config.output_dir or exp.config.checkpoint_dir):
        # Without output_dir each run gets a new timestamped folder; resume
        # the newest one of this experiment, or say there is none.
        latest = exp.config.latest_output_dir()
        if latest is None:
            from lasagna.core.paths import results_dir

            _error(
                f"Nothing to resume: no checkpoints for experiment {exp.config.name!r} under "
                f"{results_dir()}. Run without --resume, or set output_dir: in "
                f"the sweep to resume a specific folder."
            )
        exp.config._resolved_output_dir = latest
        console.print(f"Resuming from {escape(latest)}", soft_wrap=True)

    _require_vector_backend()
    try:
        tasks = exp.generate_tasks()
    except (ValueError, TypeError, KeyError, ImportError) as exc:
        # The pre-flight's refusals (unknown benchmark or component, a missing
        # data file, wrong signature, a contract failure) end in one actionable
        # sentence, before "Starting experiment"; a 40-line Rich traceback
        # buried it, and an unknown benchmark failed every task instead.
        _error(str(exc))
    cp = CheckpointManager(exp.config.checkpoint_dir or exp.config.effective_output_dir())
    if not resume:
        # When not resuming, clear any stale checkpoints from a previous run
        cleared = cp.clear()
        if cleared:
            console.print(f"  [dim]Cleared {_count(cleared, 'stale checkpoint')}[/dim]")

    console.print(f"Starting experiment [cyan]{exp.config.name}[/cyan]")
    console.print(f"  Benchmarks: {len(exp.config.benchmarks)}")
    console.print(f"  Runs: {exp.config.n_runs}")
    console.print(f"  Mode: {exp.config.mode}")
    console.print(f"  Total tasks: {len(tasks)}")
    if resume:
        done = cp.completed_keys() & {t.task_key for t in tasks}
        console.print(f"  Resumed {len(done)} of {len(tasks)} from checkpoints")

    def _progress(completed: int, total: int, task_key: str):
        # The arm, not the task key: its hashes wrapped every line in two.
        parts = task_key.split("|")
        label = (
            f"{parts[0]}  {parts[1]}  {parts[5].replace('_', ' ')}" if len(parts) > 5 else task_key
        )
        console.print(f"  [{completed}/{total}] {escape(label)}", soft_wrap=True)

    async def _execute():
        return await exp.run(progress_callback=_progress)

    try:
        results = _run(_execute())
    except Exception as exc:
        _error(f"Experiment failed: {exc}", exc=exc)

    output_dir = exp.config.effective_output_dir()
    # Every task failed: a green "Experiment Complete" box and a "Next" command
    # for results that do not exist read as a finished run.
    all_failed = bool(results.total_tasks) and len(results.failed_tasks) == results.total_tasks
    if not all_failed:
        _print_leaderboard(results)
        # Before the summary: printed after "Output" and "Next", a failed check
        # read as an afterthought to a run that had looked complete.
        _print_isolation(results)
    console.print(
        Panel.fit(
            (
                "[bold red]Experiment failed[/bold red]: every task failed\n\n"
                if all_failed
                else "[bold]Experiment Complete[/bold]\n\n"
            )
            + f"Tasks: [cyan]{results.total_tasks}[/cyan]\n"
            f"Failed: [{'red' if results.failed_tasks else 'green'}]{len(results.failed_tasks)}[/]",
            border_style="red" if all_failed else "none",
        )
    )
    # Outside the panel, unwrapped, so the path and the command can be copied.
    results_file = shlex.quote(str(Path(output_dir) / "results.json"))
    console.print(f"Output: {escape(output_dir)}", soft_wrap=True)
    if not all_failed:
        console.print(
            f"Next:   lasagna results significance {escape(results_file)} --all", soft_wrap=True
        )
    if exp.config.n_runs > 1 and runs_were_identical(results._results):
        console.print(
            f"[dim]Note: all {exp.config.n_runs} runs of every arm were identical (nothing "
            f"in this pipeline samples), so n_runs: 1 gives the same answer "
            f"{exp.config.n_runs}x faster. Repeat runs when an LLM or agentic stage "
            f"is in the loop.[/dim]"
        )
    _release_backend()
    _exit_if_tasks_failed(results)


def _print_isolation(results) -> None:
    """Say whether the arms differed only in the stages they named."""
    from .knowledge_evaluation_sdk.stages import compared_pairs, isolation_report

    rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results._results]
    report = isolation_report(rows)
    pairs = compared_pairs(rows)
    if report["undeclared"]:
        console.print(
            "[red]Stage isolation FAILED: the arms differ outside the stages they "
            "name, so the comparison is not attributable.[/red]"
        )
        for line in report["undeclared"]:
            console.print(f"  - {escape(line)}")
    elif pairs and report["inert"]:
        # Said, not left out: an inert arm printed only its own line, with no
        # verdict on the rest of the comparison.
        console.print(
            f"[yellow]Stage isolation: the configuration that reached each stage differs "
            f"only in the stages the arms name ({_count(pairs, 'pair')} compared), but "
            f"{_count(len(report['inert']), 'arm')} changed nothing there:[/yellow]"
        )
    elif pairs:
        console.print(
            f"[green]Stage isolation: the configuration that reached each stage differs "
            f"only in the stages the arms name ({_count(pairs, 'pair')} compared).[/green]"
        )
    elif not pairs and len(rows) > 1:
        console.print(
            "[yellow]Stage isolation not checked: fewer than two arms finished on a "
            "benchmark, so there was nothing to compare.[/yellow]"
        )
    for line in report["unverified"]:
        console.print(f"[yellow]Stage isolation unverified: {escape(line)}[/yellow]")
    for line in report["inert"]:
        console.print(f"[yellow]Inert arm: {escape(line)}[/yellow]")


def _release_backend() -> None:
    """Close the shared backend so the process can exit.

    Milvus Lite's gRPC threads are not daemons, and Python joins non-daemon
    threads before atexit handlers run, so a finished experiment printed its
    summary and then never exited -- which hangs CI jobs and batch scripts.
    """
    try:
        from .api.rag_backend import close_backend

        close_backend()
    except Exception as exc:  # never let cleanup mask the run's outcome
        from loguru import logger

        logger.debug(f"Backend close failed: {exc}")


def _exit_if_tasks_failed(results) -> None:
    """Name each failed task and exit non-zero, so a script or CI job notices.

    A failed task's numbers are left out of every aggregate and report; a run
    that printed "Failed: 3" and exited 0 was easy to mistake for a result.
    """
    # Retrieval scored, generation not: the retrieval numbers above stand.
    partial = [
        t
        for t in getattr(results, "_results", [])
        if not t.error and (t.system or {}).get("generation_failed")
    ]
    if not results.failed_tasks and not partial:
        return
    for task in results.failed_tasks:
        console.print(f"  [red]failed[/red] {escape(task.task_key)}: {escape(str(task.error))}")
    for task in partial:
        console.print(
            f"  [yellow]generation not scored[/yellow] {escape(task.task_key)} (its retrieval "
            f"metrics are above): {escape(task.system['generation_failed'])}"
        )
    raise typer.Exit(code=1)


def _deprecated_spelling(old: str, new: str) -> None:
    console.print(
        f"[yellow]`lasagna {old}` is deprecated and will be removed in 0.2.0; "
        f"use `lasagna {new}`.[/yellow]"
    )


# `benchmark compare` and `results compare` did different things under one
# name; the `results` commands are the ones that check comparability. These
# stay for one release, hidden.
@benchmark_app.command("compare", hidden=True)
def benchmark_compare(
    results_a: str = typer.Argument(..., help="First results JSON."),
    results_b: str = typer.Argument(..., help="Second results JSON."),
    metric: str = typer.Option("ndcg@10", "--metric", "-m", help="Metric to compare."),
    format_: str = typer.Option(
        "markdown", "--format", "-f", help="markdown, or latex for one table per run."
    ),
):
    """Old name of `lasagna results compare`."""
    _deprecated_spelling("benchmark compare", "results compare")
    if format_ != "latex":
        results_compare(results_a, results_b, metric=metric)
        return

    runs = [_load_results(path) for path in (results_a, results_b)]
    for path, run in zip((results_a, results_b), runs):
        _require_metric(run, metric, where=Path(path).name)
    console.print(runs[0].to_latex(metric))
    console.print()
    console.print(runs[1].to_latex(metric))


@benchmark_app.command("licenses")
def benchmark_licenses(
    output: str = typer.Option("", "--output", "-o", help="Write markdown to this path"),
):
    """List every benchmark with its license and citation.

    Generated from the benchmark adapters. Lasagna redistributes no benchmark
    data: each dataset is fetched from its own source under its own terms, and
    those terms are yours to honor.
    """
    from .knowledge_evaluation_sdk.benchmarks import benchmark_registry

    infos = {
        name: info
        for name, info in benchmark_registry.list_benchmarks().items()
        # Internal-only entries; they load nothing without a locally built set.
        if not name.startswith("private_eval/")
    }

    lines = [
        "# Benchmark datasets",
        "",
        "Generated by `lasagna benchmark licenses`. Do not edit by hand.",
        "",
        "Lasagna **redistributes no benchmark data**. Each dataset below is",
        "downloaded from its own upstream source on first use, under its own",
        "license and terms of use. Those terms are yours to honor, including",
        "for anything you publish from results produced with them.",
        "",
        "`Live` marks the benchmarks that ship a retrievable corpus. The rest",
        "can only score generation against gold context; see `mode` in the guide.",
        "",
        "| Benchmark | License | Live | Paper | Description |",
        "|---|---|:--:|---|---|",
    ]
    for name, info in sorted(infos.items()):
        adapter = benchmark_registry.get(name)
        from .knowledge_evaluation_sdk.experiment import _adapter_has_corpus

        live = "yes" if _adapter_has_corpus(adapter) else "-"
        paper = info.paper or "-"
        lic = info.license or "**unknown**"
        desc = (info.description or "").replace("|", "\\|")
        lines.append(f"| `{name}` | {lic} | {live} | {paper} | {desc} |")

    unknown = [n for n, i in infos.items() if not i.license]
    if unknown:
        lines += [
            "",
            "## Unknown license",
            "",
            "These declare no license. Check upstream before publishing results:",
            "",
            *(f"- `{n}`" for n in sorted(unknown)),
        ]

    lines += [
        "",
        "## Model licenses",
        "",
        "Dataset terms are separate from model terms. Several selectable models",
        "are non-commercial; see [model-licenses.md](../../../../docs/model-licenses.md).",
        "",
    ]

    markdown = "\n".join(lines)
    if output:
        _output_file(output).write_text(markdown)
        console.print(f"[green]Wrote[/green] {output} ({len(infos)} benchmarks)")
    else:
        console.print(markdown, markup=False)


@benchmark_app.command("leaderboard", hidden=True)
def benchmark_leaderboard(
    results_path: str = typer.Argument(..., help="A directory of bundles, or one results.json."),
    metric: str = typer.Option("ndcg@10", "--metric", "-m", help="Metric to rank by."),
):
    """Old name of `lasagna results leaderboard`."""
    _deprecated_spelling("benchmark leaderboard", "results leaderboard")
    # Every option by value: called directly, a typer function's defaults are
    # its OptionInfo objects, and `fmt` arrived as one ("Unknown format").
    results_leaderboard(results_path, metric=metric, output=None, fmt="markdown")


@benchmark_app.command("validate-data")
def benchmark_validate_data(
    path: str = typer.Argument(
        ...,
        help="A .jsonl or .csv file, or a BEIR folder (corpus.jsonl, queries.jsonl, "
        "qrels/test.tsv)",
    ),
):
    """Check your own benchmark data and say which tracks it supports.

    Exits 0 when the data can be run, 1 with every problem listed when not.
    The formats are in docs/own-data.md.
    """
    from .knowledge_evaluation_sdk.benchmarks.adapters.local_files import (
        DataFileError,
        read_local_data,
    )

    try:
        data = read_local_data(path)
    except DataFileError as exc:
        for line in str(exc).splitlines():
            console.print(f"[red]{escape(line)}[/red]", soft_wrap=True)
        them = "it" if len(exc.errors) == 1 else "them"
        console.print(f"{_count(len(exc.errors), 'problem')}; fix {them} and run this again.")
        raise typer.Exit(1)

    rows = data.rows()
    if not data.supported_tracks():
        # One verdict: it printed "valid jsonl data" and then exited 1 because
        # nothing could be measured.
        why = "; ".join(why for why in data.track_support().values() if why)
        console.print(
            f"[red]Nothing can be measured on {escape(data.label)} ({escape(why)}). "
            f'Add an "answer" or "contexts" to the rows.[/red]',
            soft_wrap=True,
        )
        console.print("1 problem; fix it and run this again.")
        raise typer.Exit(1)
    lines = [
        f"[green]{escape(data.label)}: valid {data.kind} data[/green]",
        f"  Questions:      {len(rows)}",
    ]
    if data.kind != "beir":
        lines += [
            f"  With answers:   {sum(1 for r in rows if r.answers)}",
            f"  With contexts:  {sum(1 for r in rows if r.contexts)}",
            f"  Unanswerable:   {sum(1 for r in rows if r.unanswerable)}",
        ]
    else:
        lines.append(
            "  Splits:         "
            + ", ".join(f"{s} ({len(r)} judged)" for s, r in sorted(data.splits.items()))
        )
    pooled = " (the rows' contexts, pooled)" if data.corpus and data.kind != "beir" else ""
    lines.append(f"  Documents:      {len(data.corpus)}{pooled}")
    lines.append(f"  Benchmark name: {escape(data.benchmark_name())}")
    for track, why in data.track_support().items():
        status = "[green]yes[/green]" if why is None else f"[yellow]no[/yellow], {escape(why)}"
        lines.append(f"  {track.capitalize() + ':':<15} {status}")
    for line in lines:
        console.print(line, soft_wrap=True)
    if data.corpus and data.kind != "beir":
        console.print(
            "  Retrieval searches a pool of gold passages only, an easier task than a real "
            "corpus; results are labeled pooled_gold.",
            soft_wrap=True,
        )
    if data.track_support()["generation"] is None and not data.corpus:
        console.print(
            "  With no contexts, generation is closed-book: use the closed_book generator.",
            soft_wrap=True,
        )
    tracks = ",".join(data.supported_tracks())
    from .knowledge_evaluation_sdk.synthesize import llm_configured

    if "generation" in data.supported_tracks() and not llm_configured():
        # Suggesting generation with no key ran retrieval, failed every
        # answer, and threw the retrieval numbers away.
        tracks = ",".join(t for t in data.supported_tracks() if t != "generation")
        console.print(
            "  Generation needs an LLM API key, and none is set (OPENAI_API_KEY, "
            "ANTHROPIC_API_KEY, ...). With one, add generation to --tracks.",
            soft_wrap=True,
        )
    if not tracks:
        return
    console.print(
        f"Run it:  lasagna benchmark run {escape(shlex.quote(path))} --profile speed "
        f"--tracks {tracks}",
        soft_wrap=True,
    )


@benchmark_app.command("synthesize")
def benchmark_synthesize(
    corpus: str = typer.Argument(
        ..., help="A folder of your documents (.md, .txt, .pdf, .docx, .html, ...), or one file"
    ),
    n: int = typer.Option(100, "--n", help="Number of questions to write"),
    out: str = typer.Option("synthetic_qa.jsonl", "--out", help="JSONL file to write"),
    mix: str = typer.Option(
        "factoid=0.6,multihop=0.3,unanswerable=0.1",
        "--mix",
        help="Share of each question type: factoid, multihop, unanswerable",
    ),
    seed: int = typer.Option(42, "--seed", help="Seed for choosing the source passages"),
    model: Optional[str] = typer.Option(
        None, "--model", help="LLM that writes and checks the questions (default: MODEL_NAME)"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the plan and a token estimate; call no LLM, write nothing"
    ),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace --out if it exists"),
):
    """Write a question set from your own documents with an LLM, for development.

    Each question is checked by a second call, and rows that fail are dropped.
    The output runs with `lasagna benchmark run <out>`. Synthetic questions favor
    the phrasing of the model that wrote them: report results on human-written
    benchmarks (docs/own-data.md).
    """
    from .knowledge_evaluation_sdk import synthesize as synth
    from .providers import LLMProviderNotConfigured

    try:
        plan = synth.plan_synthesis(corpus, n, mix, seed)
    except ValueError as exc:
        _error(str(exc))

    counts = ", ".join(f"{t} {plan.counts[t]}" for t in synth.QUESTION_TYPES)
    tokens_in, tokens_out = plan.token_estimate()
    console.print(f"Plan for {escape(corpus)}:", soft_wrap=True)
    console.print(f"  Documents:  {len(plan.documents)}")
    for rel, reason in plan.skipped:
        console.print(f"    skipped {escape(rel)}: {escape(reason)}", soft_wrap=True)
    console.print(
        f"  Passages:   {len(plan.passages)} (fixed chunker, {plan.chunk_words} words, "
        f"no overlap; shorter than {synth.MIN_PASSAGE_WORDS} words dropped)",
        soft_wrap=True,
    )
    console.print(f"  Questions:  {len(plan.items)} ({counts})")
    console.print(
        f"  LLM calls:  {plan.llm_calls} (one writes each question, one checks it)",
        soft_wrap=True,
    )
    console.print(
        f"  Tokens:     about {tokens_in:,} in and {tokens_out:,} out (an estimate at "
        f"4 characters per token)",
        soft_wrap=True,
    )
    if dry_run:
        console.print("Dry run: no LLM was called and nothing was written.")
        return

    if Path(out).exists() and not overwrite:
        _error(f"{out} exists. Choose another --out, or pass --overwrite to replace it.")
    try:
        llm = synth.default_llm(model)
    except LLMProviderNotConfigured as exc:
        _error(str(exc))

    step = max(1, len(plan.items) // 10)

    def _progress(done: int, total: int) -> None:
        if done and (done % step == 0 or done == total):
            console.print(f"  {done}/{total} written and checked")

    try:
        result = synth.run_synthesis(plan, llm, model=model, progress=_progress)
    except RuntimeError as exc:
        _error(str(exc))
    dropped = ", ".join(f"{k} {v}" for k, v in result.dropped.items() if v) or "none"
    if not result.rows:
        _error(f"No question passed the check (dropped: {dropped}); nothing was written.")
    synth.write_jsonl(result.rows, out)

    kept = ", ".join(f"{t} {result.kept[t]}" for t in synth.QUESTION_TYPES)
    console.print(
        f"[green]Wrote {_count(len(result.rows), 'question')}[/green] to {escape(out)} "
        f"({kept}; model {escape(result.model)})",
        soft_wrap=True,
    )
    console.print(f"  Dropped: {dropped}")
    console.print(
        "  Synthetic questions are for development: report results on human-written benchmarks.",
        soft_wrap=True,
    )
    console.print(
        f"Next:  lasagna benchmark validate-data {escape(shlex.quote(out))}", soft_wrap=True
    )


# ---------------------------------------------------------------------------
# Component / plugin sub-apps — for people extending Lasagna
# ---------------------------------------------------------------------------

component_app = typer.Typer(
    name="component",
    help="List components, and check components you have written.",
    no_args_is_help=True,
)
app.add_typer(component_app, name="component")


_FUNCTION_STAGES = (
    "reranker",
    "query_enhancer",
    "chunker",
    "embedder",
    "retriever",
    "document_processor",
    "generator",
)


@component_app.command("list")
def component_list(
    stage: Optional[str] = typer.Argument(
        None,
        help="Show one stage only: agentic_pipeline, chunker, document_processor, embedder, "
        "generator, query_enhancer, reranker or retriever.",
    ),
):
    """List the components a sweep can name, by stage.

    Built-in components and those of installed plugins, with the paper each
    one implements. A component of your own module is named as module:function
    and does not appear here.
    """
    from lasagna.core.registry import COMPONENT_TYPES, registry

    if stage is not None and stage != "enhancer":
        _require_known(
            "stage",
            stage,
            sorted(COMPONENT_TYPES),
            f"Stages: {', '.join(sorted(COMPONENT_TYPES))}.",
        )
    entries = registry.list_components(stage)

    by_stage: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        by_stage.setdefault(entry["stage"], []).append(entry)
    for stage_name, rows in by_stage.items():
        table = Table(
            title=f"{stage_name} ({len(rows)})",
            title_justify="left",
            title_style="bold cyan",
            show_lines=False,
        )
        table.add_column("Name", style="bold", no_wrap=True)
        table.add_column("Description", ratio=3)
        table.add_column("Paper", style="dim", ratio=1, overflow="fold")
        for entry in rows:
            description = next(
                (ln.strip() for ln in (entry.get("description") or "").splitlines() if ln.strip()),
                "",
            )
            name = entry["name"] if entry["builtin"] else f"{entry['name']} (plugin)"
            table.add_row(escape(name), escape(description), escape(entry.get("paper") or ""))
        console.print(table)
    console.print(
        f"[dim]{_count(len(entries), 'component')}. Name one in a sweep's pipeline "
        f"section, or try a reranker with `lasagna benchmark run smoke --reranker NAME`.[/dim]"
    )


def _registered_component(name: str, stage: Optional[str]) -> Any:
    """The class registered as ``name``, in ``stage`` or the one stage that has it."""
    return _registered_component_and_stage(name, stage)[0]


def _registered_component_and_stage(name: str, stage: Optional[str]) -> tuple:
    """``(class, stage)`` registered as ``name``, in ``stage`` or the one stage that has it."""
    from .core._text import did_you_mean
    from .core.registry import registry

    stages = [stage] if stage else list(_FUNCTION_STAGES)
    found = [s for s in stages if registry.has(s, name)]
    if not found:
        known = sorted({n for s in stages for n in registry.list(s).get(s, [])})
        _error(
            f"No component is registered as {name!r}{did_you_mean(name, known)}. Name a "
            f"registered one (`lasagna component list`), or your own as module:function."
        )
    if len(found) > 1:
        _error(f"{name!r} is registered for {', '.join(found)}; pick one with --stage.")
    return registry.get_class(found[0], name), found[0]


@contextmanager
def _validated_with_profile(stage: str, name: str, profile: str):
    """Build a registered component with ``profile``'s model, and say which before loading.

    ``component validate cross_encoder`` built it with the settings file's
    reranker, an 881 MB download, while the speed profile and the README's
    sweeps use a 92 MB one.
    """
    from lasagna.core.config import rag_config
    from lasagna.core.model_downloads import is_cached
    from lasagna.core.session_config._profiles import ProfileMixin
    from lasagna.core.setup_text import COMPONENT_MODEL_SETTINGS, MODEL_DOWNLOAD_BYTES, human_size

    setting = COMPONENT_MODEL_SETTINGS.get((stage, name))
    if setting is None:
        yield
        return
    spec = ProfileMixin._PROFILE_SPECS.get(profile, {})
    model = str(spec.get(setting) or getattr(rag_config, setting, "") or "")
    size = MODEL_DOWNLOAD_BYTES.get(model)
    where = "cached" if is_cached(model) else "downloaded on first use"
    console.print(
        f"Builds {escape(name)} with the {escape(profile)} profile's {setting}: "
        f"{escape(model)}" + (f" ({human_size(size)}, {where})." if size else f" ({where}).")
    )
    overrides = rag_config._env_overrides
    had, previous = setting in overrides, overrides.get(setting)
    overrides[setting] = model
    try:
        yield
    finally:
        if had:
            overrides[setting] = previous
        else:
            overrides.pop(setting, None)


@component_app.command("validate")
def component_validate(
    target: str = typer.Argument(
        ...,
        help="What a sweep would name: a registered component (cross_encoder), "
        "my_ideas:keyword_overlap (a decorated function), my_pkg.rerankers:MyReranker "
        "(a class) or path/to/file.py:name",
    ),
    stage: Optional[str] = typer.Option(
        None,
        "--stage",
        help="The stage a plain, undecorated function implements: " + ", ".join(_FUNCTION_STAGES),
    ),
    profile: str = typer.Option(
        "speed",
        "--profile",
        "-p",
        help="Profile whose model a registered component is built with (default speed: "
        "the 92 MB cross-encoder, not the settings file's 881 MB one)",
    ),
):
    """Check a component satisfies its stage contract.

    Checks what makes a component safe to swap: it returns the declared type,
    survives empty input, and does not change what it was handed. The last is
    what makes a comparison attributable: a component that edits its input
    changes the results of whatever runs after it.

    The target is resolved from the working directory, as a sweep resolves it,
    so the function you name in a sweep is the one checked here. A function
    decorated with @lasagna.reranker (or another stage decorator) is checked as
    the component the sweep would run.
    """
    import inspect

    from lasagna import components
    from lasagna.core.testing import ComponentTestHarness

    module_path, _, attr = target.rpartition(":")
    found_stage = None  # the stage of a registered component
    if stage is not None and stage not in _FUNCTION_STAGES:
        _error(f"--stage must be one of {', '.join(_FUNCTION_STAGES)}, got {stage!r}")
    if not module_path or not attr:
        # A registered name, as a sweep's arm names one: `component validate
        # cross_encoder` was refused as not being module:function.
        obj, found_stage = _registered_component_and_stage(target, stage)
        attr = name = target
    else:
        try:
            obj = components.import_reference(target)
        except Exception as exc:
            _error(f"Could not import {target}: {type(exc).__name__}: {exc}")
        # What a sweep names it: a file's module is its stem, so
        # path/to/my_ideas.py:keyword_overlap is my_ideas:keyword_overlap.
        name = f"{Path(module_path).stem}:{attr}" if module_path.endswith(".py") else target

    if not inspect.isclass(obj) and callable(obj):
        tagged = getattr(obj, "__lasagna_component__", None)
        chosen = stage or (tagged[0] if tagged else None)
        if chosen is None:
            _error(
                f"{target} is a plain function, so its stage is unknown. Decorate it "
                f"(@lasagna.reranker, @lasagna.query_enhancer, ...) or pass --stage."
            )
        try:
            obj = components.as_component_class(chosen, obj, name)
        except (TypeError, ValueError) as exc:
            _error(str(exc))

    from contextlib import nullcontext

    _require_profile(profile)
    built_with = (
        _validated_with_profile(found_stage, target, profile) if found_stage else nullcontext()
    )
    with built_with:
        try:
            instance = obj() if isinstance(obj, type) else obj
        except Exception as exc:
            _error(
                f"Could not instantiate {attr} with no arguments: {type(exc).__name__}: "
                f"{exc}. Components must be constructible without required arguments so "
                f"the pipeline can build them from a config name."
            )

        try:
            report = ComponentTestHarness().validate(instance)
        except Exception as exc:  # a model that cannot load, a missing key: no verdict
            _error(f"The checks could not run: {type(exc).__name__}: {exc}")

    console.print(f"\n[bold]{escape(name)}[/bold] ({report.component_type or 'unknown'})")
    for check in report.checks:
        # `checks` records every check as "PASS: name" or "FAIL: name"; the
        # failures, with their reasons, are printed below.
        if check.startswith("PASS: "):
            console.print(f"  [green]pass[/green]  {escape(check[len('PASS: ') :])}")
    for failure in report.failures:
        console.print(f"  [red]FAIL[/red]  {escape(failure)}")

    if report.latency_ms is not None:
        console.print(f"\n  latency: {_milliseconds(report.latency_ms)}")

    if report.passed:
        console.print("\n[green]Contract satisfied.[/green] This component is interchangeable.")
        # The README's flow: a sweep names the function by its import path, with
        # no registration call. "Next: register it" said otherwise.
        example = (
            f'rerankers: [cross_encoder, "{name}"]'
            if report.component_type == "reranker"
            else f'"{name}" in the stage\'s list'
        )
        console.print(
            f"Next: name it in a sweep to compare it against a control, as {escape(example)}: "
            f"https://github.com/dolev31/lasagna/blob/main/docs/guide.md"
            f"#6-compare-it-against-a-baseline",
            soft_wrap=True,
        )
    else:
        console.print(
            f"\n[red]{_count(len(report.failures), 'contract failure')}.[/red] "
            f"A component that does not satisfy its contract can still be registered, "
            f"but it is not interchangeable, and a comparison against one that is "
            f"would measure the interface difference too."
        )
        raise typer.Exit(1)


plugins_app = typer.Typer(
    name="plugins",
    help="Inspect third-party components and benchmarks.",
    no_args_is_help=True,
)
app.add_typer(plugins_app, name="plugins")


@plugins_app.command("list")
def plugins_list():
    """Show discovered plugins, including the ones that failed to load.

    One broken plugin does not stop the others loading, so a plugin that fails
    is simply absent from the registry. This lists it with its error. Set
    LASAGNA_STRICT_PLUGINS=1 to make a failure raise instead.
    """
    from lasagna.core.registry import discovery_log, registry
    from lasagna.knowledge_evaluation_sdk.benchmarks import benchmark_registry

    # Touch both registries so lazy discovery has actually run.
    registry.list_all()
    benchmark_registry.names()

    entries = discovery_log()
    if not entries:
        console.print("No third-party plugins installed.")
        # markup=False: rich reads [project.entry-points] as a style tag and
        # swallows the line, which would make the one piece of copy-pasteable
        # guidance here disappear.
        console.print(
            "\nTo publish one, declare an entry point in your pyproject.toml:\n"
            '  [project.entry-points."lasagna_components"]\n'
            '  reranker.my_reranker = "my_pkg:MyReranker"\n'
            "\n  [project.entry-points.lasagna_benchmarks]\n"
            '  my_dataset = "my_pkg:MyBenchmarkAdapter"\n'
            "\nSee examples/lasagna-plugin-template/ for a working package.",
            markup=False,
        )
        return

    table = Table(title="Discovered plugins")
    table.add_column("Group")
    table.add_column("Name")
    table.add_column("Target")
    table.add_column("Status")
    for e in entries:
        status = "[green]loaded[/green]" if e["ok"] else f"[red]FAILED[/red] {e['error']}"
        table.add_row(e["group"], e["name"], e["value"], status)
    console.print(table)

    failed = [e for e in entries if not e["ok"]]
    if failed:
        console.print(
            f"\n[yellow]{_count(len(failed), 'plugin')} failed to load and "
            f"{'is' if len(failed) == 1 else 'are'} silently absent from the "
            f"registry.[/yellow] Re-run with LASAGNA_STRICT_PLUGINS=1 to raise."
        )
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Results sub-app — publishing and comparing runs
# ---------------------------------------------------------------------------

results_app = typer.Typer(
    name="results",
    help="Package, validate and compare evaluation results.",
    no_args_is_help=True,
)
app.add_typer(results_app, name="results")


def _results_file(path: str) -> Path:
    """``path`` as a run's results.json: the file itself, or the run folder holding it.

    Every results command takes either. A folder (what a run prints after
    "Output:") ended in a 30-line IsADirectoryError traceback in all but view
    and leaderboard.
    """
    given = Path(path).expanduser()
    if given.is_dir():
        inside = given / "results.json"
        if inside.is_file():
            return inside
        _error(
            f"{path} is a folder with no results.json in it. Pass the folder a run printed "
            f'after "Output:", or the results.json inside it.'
        )
    if not given.exists():
        _error(f"No such file or folder: {path}")
    return given


def _load_results(path: str):
    """The run at ``path`` (a results.json or its folder), or a one-line error."""
    from .knowledge_evaluation_sdk.results import BenchmarkResults

    file = _results_file(path)
    try:
        return BenchmarkResults.load(str(file))
    except json.JSONDecodeError as exc:
        _error(f"{file} is not valid JSON ({exc}). Pass a results.json from a run.")
    except (OSError, UnicodeDecodeError) as exc:
        _error(f"Cannot read {file}: {exc}")
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        _error(f"{file} is not a results.json from a run ({type(exc).__name__}: {exc}).")


@results_app.command("bundle")
def results_bundle(
    results_path: str = typer.Argument(..., help="results.json from a run"),
    output: str = typer.Option("run.json", "--output", "-o", help="Bundle path to write"),
    name: str = typer.Option("", "--name", help="Submitter name"),
    affiliation: str = typer.Option("", "--affiliation", help="Submitter affiliation"),
):
    """Package a run into a self-contained, publishable bundle."""
    from .knowledge_evaluation_sdk.bundle import MetricsVersionError

    results = _load_results(results_path)
    submitter = {k: v for k, v in {"name": name, "affiliation": affiliation}.items() if v}
    try:
        bundle = results.to_bundle(submitter=submitter or None)
    except MetricsVersionError as exc:
        _error(str(exc))

    try:
        _output_file(output).write_text(json.dumps(bundle, indent=2, default=str))
    except OSError as exc:
        _error(f"Cannot write {output}: {exc.strerror or exc}")
    console.print(f"[green]Wrote[/green] {escape(output)}")
    console.print(f"  run_id: {bundle['run_id']}")
    benchmarks = {cfg.get("benchmark") for cfg in bundle["configs"]}
    for cfg in bundle["configs"]:
        label = str(cfg["name"])
        if len(benchmarks) > 1:
            label = f"{cfg.get('benchmark')}/{label}"
        console.print(f"  {escape(label)}: comparability key {cfg['comparability_key']}")
    warnings = bundle["integrity"]["warnings"]
    if warnings:
        console.print(f"[yellow]  integrity warnings: {', '.join(warnings)}[/yellow]")


@results_app.command("validate")
def results_validate(
    bundle_path: str = typer.Argument(..., help="run.json bundle to check"),
    strict: bool = typer.Option(False, "--strict", help="Treat integrity warnings as failures"),
):
    """Check a bundle is well-formed, complete, and internally consistent.

    Recomputes the comparability key, the run id and every aggregate from the
    recorded fields, so a number edited after the run no longer matches its
    bundle. It detects mistakes and accidental edits, not tampering: the run id
    is a content hash, not a signature, and anyone who edits a number can
    recompute it.
    """
    from .knowledge_evaluation_sdk.bundle import (
        _content_hash,
        aggregate_mismatches,
        comparability_key,
        config_label,
        metrics_version_problems,
        ranked_list_mismatches,
    )

    given = Path(bundle_path).expanduser()
    if given.is_dir():
        _error(
            f"{bundle_path} is a folder; results validate checks a bundle. Make one from "
            f"the run with: lasagna results bundle {shlex.quote(bundle_path)} -o run.json"
        )
    if not given.exists():
        _error(f"No such file: {bundle_path}")
    try:
        bundle = json.loads(given.read_text())
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        _error(f"{bundle_path} is not valid JSON ({exc}). Pass a bundle from results bundle.")
    problems: List[str] = []

    schema = str(bundle.get("schema_version", ""))
    if not schema.startswith("1."):
        problems.append(f"unsupported schema_version {schema!r} (this build reads 1.x)")

    for field in ("run_id", "experiment", "configs", "results", "integrity"):
        if field not in bundle:
            problems.append(f"missing required block: {field}")

    if "run_id" in bundle and _content_hash(bundle) != bundle["run_id"]:
        problems.append("run_id does not match the bundle contents: it was edited after generation")

    for cfg in bundle.get("configs", []):
        recomputed = comparability_key(cfg.get("held_fixed", {}))
        if recomputed != cfg.get("comparability_key"):
            problems.append(
                f"config {config_label(cfg)}: comparability key does not match its "
                f"held_fixed block ({cfg.get('comparability_key')} vs {recomputed})"
            )
        benchmark = cfg.get("benchmark")
        held_benchmark = (cfg.get("held_fixed") or {}).get("benchmark")
        if benchmark and held_benchmark != benchmark:
            problems.append(
                f"config {config_label(cfg)}: held_fixed.benchmark is {held_benchmark!r}, "
                f"not the entry's benchmark"
            )

    problems.extend(metrics_version_problems(bundle))
    problems.extend(
        f"aggregate disagrees with per-query scores: {m}"
        for m in aggregate_mismatches(bundle.get("results") or [])
    )
    problems.extend(
        f"per-query score above what its ranked list supports: {m}"
        for m in ranked_list_mismatches(bundle.get("results") or [])
    )

    env = (bundle.get("experiment") or {}).get("environment") or {}
    for required in ("lasagna_version", "python", "platform"):
        if not env.get(required):
            problems.append(f"provenance incomplete: environment.{required} is missing")

    blocking = (bundle.get("integrity") or {}).get("blocking") or []
    problems.extend(f"integrity: {code}" for code in blocking)
    for line in ((bundle.get("integrity") or {}).get("isolation") or {}).get("undeclared", []):
        problems.append(f"  {line}")

    warnings = (bundle.get("integrity") or {}).get("warnings") or []
    # What each active fallback swapped in and what to install, so the
    # ACTIVE_FALLBACKS code is actionable. Older bundles have no list.
    fallbacks = (
        [str(f) for f in (bundle.get("integrity") or {}).get("fallbacks", [])]
        if "ACTIVE_FALLBACKS" in warnings
        else []
    )
    if strict:
        problems.extend(f"integrity (strict): {code}" for code in warnings)

    if problems:
        console.print(f"[red]INVALID[/red], {_count(len(problems), 'problem')}:")
        # Problem lines quote arm names, which are free text: `a[/b]` is a
        # Rich closing tag and crashed the command instead of printing.
        for p in problems:
            console.print(f"  - {escape(p)}")
            if p == "integrity (strict): ACTIVE_FALLBACKS":
                for line in fallbacks:
                    console.print(f"      {escape(line)}")
        raise typer.Exit(1)

    console.print(f"[green]VALID[/green] (schema {escape(schema)}, run_id {bundle['run_id']})")
    if warnings:
        console.print(f"[yellow]  warnings: {escape(', '.join(warnings))}[/yellow]")
        for line in fallbacks:
            console.print(f"    {escape(line)}")


@results_app.command("compare")
def results_compare(
    a: str = typer.Argument(..., help="First bundle or results.json"),
    b: str = typer.Argument(..., help="Second bundle or results.json"),
    metric: str = typer.Option("ndcg@10", "--metric", "-m", help="Metric to compare."),
):
    """Compare two runs, reporting whether they are comparable at all."""

    first, second = _load_results(a), _load_results(b)
    for path, results in ((a, first), (b, second)):
        _require_metric(results, metric, where=Path(path).name)
    console.print(first.compare(second, metric), markup=False, highlight=False)


@results_app.command("power")
def results_power(
    results_path: str = typer.Argument(..., help="results.json from a run"),
    config_a: str = typer.Argument(..., help="Baseline config"),
    config_b: str = typer.Argument(..., help="Candidate config"),
    metric: str = typer.Option("ndcg@10", "--metric", "-m", help="Metric to test."),
    benchmark: str = typer.Option("", "--benchmark", help="Which benchmark"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level of the test."),
    power: float = typer.Option(
        0.80, "--power", help="Chance of detecting the effect, for the smallest one reported."
    ),
):
    """Could this run have detected the effect you care about?

    A non-significant result has two readings: the techniques are the same, or
    the run was too small to tell. They call for opposite next steps. Reports
    the smallest difference this run could have detected, and how many queries
    the observed difference would need.
    """

    results = _load_results(results_path)
    benchmarks = sorted({r.benchmark_name for r in results._results if not r.error})
    if not benchmarks:
        _error(
            f"No successful results in {results_path}. Every task either errored "
            f"or the file holds no results at all."
        )
    if not benchmark:
        if len(benchmarks) > 1:
            _error(
                f"This run covers {len(benchmarks)} benchmarks ({', '.join(benchmarks)}). "
                "Pass --benchmark: variance differs per benchmark, so one MDE across "
                "them would describe none of them."
            )
        benchmark = benchmarks[0]
    _require_metric(results, metric, benchmark)

    try:
        report = results.minimum_detectable_effect(
            benchmark, config_a, config_b, metric, alpha=alpha, power=power
        )
    except ValueError as exc:
        _error(str(exc))

    console.print(
        f"[bold]{escape(metric)}[/bold] on {escape(benchmark)}: "
        f"{escape(config_b)} vs {escape(config_a)}"
    )
    console.print(f"  paired queries       : {report['n_paired']}")
    console.print(f"  observed difference  : {report['observed_difference']:+.4f}")
    console.print(f"  sd of differences    : {report['sd_of_differences']:.4f}")
    console.print(
        f"  p, uncorrected       : {report['p_value']:.4f}"
        + ("  [green]significant[/green]" if report["significant"] else "")
    )
    # Yellow only when a miss is the likely story: a significant pair was
    # detected, whatever its size relative to the MDE.
    colour = "yellow" if report["underpowered"] and not report["significant"] else "green"
    console.print(
        f"  [bold]effect detected {int(report['power'] * 100)}% of the time[/bold]: "
        f"[{colour}]{report['minimum_detectable_effect']:.4f}[/{colour}]"
    )
    if report["n_needed_for_observed_effect"]:
        console.print(
            f"  queries needed for the observed effect: {report['n_needed_for_observed_effect']}"
        )
    console.print()
    console.print(f"  {escape(report['interpretation'])}")


def _format_effect_size(effect: Optional[float]) -> str:
    """Cohen's ``d_z``, with ``+∞``/``-∞`` for an effect with no spread
    (`results.format_effect_size`, shared with the HTML view).

    A standalone function rather than a branch inline, because as a branch only
    one of its two signs was reachable from the test fixtures and disabling the
    other changed nothing.
    """
    from .knowledge_evaluation_sdk.results import format_effect_size

    return format_effect_size(effect)


@results_app.command("significance")
def results_significance(
    results_path: str = typer.Argument(..., help="results.json from a run"),
    metric: str = typer.Option("ndcg@10", "--metric", "-m", help="Metric to test."),
    benchmark: str = typer.Option("", "--benchmark", help="Which benchmark"),
    correction: str = typer.Option(
        "holm", "--correction", help="'holm' (Holm-Bonferroni) or 'none'"
    ),
    alpha: float = typer.Option(0.05, "--alpha", help="Family-wise significance level"),
    show_all: bool = typer.Option(
        False, "--all", help="Include pairs that did not reach significance"
    ),
):
    """Test every pair of configs at once, correcting for having asked N times.

    Running `results compare` across a grid one pair at a time is the standard
    way to manufacture a false positive: at alpha=0.05, six configs make fifteen
    comparisons and roughly a 54% chance that at least one clears the bar by
    luck alone. Holm-Bonferroni controls the family-wise error rate over the
    whole set, so what comes out is a claim about the grid rather than about the
    pair that happened to look best. Each pair is a paired sign-flip test.
    """

    if correction not in ("holm", "none"):
        _error(f"Unknown correction {correction!r}. Use 'holm' or 'none'.")

    results = _load_results(results_path)
    benchmarks = sorted({r.benchmark_name for r in results._results if not r.error})
    if not benchmarks:
        _error(
            f"No successful results in {results_path}. Every task either errored "
            f"or the file holds no results at all."
        )
    if not benchmark:
        if len(benchmarks) > 1:
            _error(
                f"This run covers {len(benchmarks)} benchmarks ({', '.join(benchmarks)}). "
                "Pass --benchmark: a correction pooled across benchmarks would be "
                "correcting for comparisons nobody made."
            )
        benchmark = benchmarks[0]
    _require_metric(results, metric, benchmark)

    try:
        rows = results.pairwise_significance(metric, benchmark, correction=correction, alpha=alpha)
    except ValueError as exc:
        _error(str(exc))

    if not rows:
        _error(
            f"No comparable pairs for {metric!r} on {benchmark!r}. "
            "Significance needs per-query scores from at least two configs."
        )

    label = "Holm-Bonferroni" if correction == "holm" else "uncorrected"
    console.print(
        f"[bold]{escape(metric)}[/bold] on {escape(benchmark)}: {_count(len(rows), 'pair')}, {label}"
    )
    console.print()

    tested = [r for r in rows if r.get("p_value") is not None]
    untestable = [r for r in rows if r.get("p_value") is None]
    any_dropped = any(r.get("n_dropped_a") or r.get("n_dropped_b") for r in tested)

    shown_rows = [r for r in tested if show_all or r.get("significant")]

    def _table(label, between: str) -> Table:
        table = Table(box=None, pad_edge=False)
        table.add_column("pair", no_wrap=True)
        table.add_column("diff", justify="right", no_wrap=True)
        table.add_column("d_z", justify="right", no_wrap=True)
        table.add_column("p", justify="right", no_wrap=True)
        if correction == "holm":
            table.add_column("p (Holm)", justify="right", no_wrap=True)
        # How many queries the test actually paired, and how many each side
        # scored that the other did not.
        table.add_column("n", justify="right", no_wrap=True)
        if any_dropped:
            table.add_column("dropped", justify="right", no_wrap=True)
        table.add_column("", justify="left", no_wrap=True)
        for row in shown_rows:
            cells = [
                f"{label(row['config_a'])}{between}vs {label(row['config_b'])}",
                f"{row['difference']:+.4f}",
                _format_effect_size(row.get("effect_size")),
                _format_p(row["p_value"]),
            ]
            if correction == "holm":
                cells.append(_format_p(row.get("p_adjusted", row["p_value"])))
            cells.append(str(row.get("n_paired", "")))
            if any_dropped:
                cells.append(f"{row.get('n_dropped_a', 0)}/{row.get('n_dropped_b', 0)}")
            cells.append("[green]significant[/green]" if row.get("significant") else "[dim]-[/dim]")
            table.add_row(*cells)
        return table

    # Config names are what identifies a pair, so they are printed whole: one
    # per line when the table fits, numbered with a legend when it does not.
    # Folding them (`reranker=cross_e / ncoder`) or truncating them made two
    # arms impossible to tell apart.
    table = _table(lambda name: escape(str(name)), "\n")
    if not shown_rows:
        # A header over no rows read as a failure to compute anything.
        table = None
    elif _natural_width(table) > console.width:
        numbered = {
            name: f"[{i}]"
            for i, name in enumerate(
                sorted({str(r[k]) for r in shown_rows for k in ("config_a", "config_b")}), 1
            )
        }
        for name, tag in numbered.items():
            console.print(f"  {escape(tag)} {escape(name)}")
        console.print()
        table = _table(lambda name: escape(numbered[str(name)]), " ")
    if table is not None:
        console.print(table)

    survivors = sum(1 for r in rows if r.get("significant"))
    console.print()
    if correction == "holm":
        console.print(_significance_verdict(rows, metric, benchmark, alpha))
    else:
        console.print(
            f"  {survivors} of {_count(len(rows), 'pair')} "
            f"{'is' if survivors == 1 else 'are'} significant at alpha={alpha}, "
            f"[yellow]uncorrected[/yellow]: with {len(rows)} comparisons, expect "
            f"about {len(rows) * alpha:.1f} of these by chance."
        )
    if any_dropped:
        console.print(
            "  [yellow]dropped[/yellow] is queries only A / only B scored. Those pairs "
            "compare a shared subset, not each config's full run."
        )
    if untestable:
        console.print(
            f"  [yellow]{_count(len(untestable), 'pair')} could not be tested[/yellow] and count in "
            "the family as p = 1:"
        )
        for row in untestable:
            console.print(
                f"    {escape(str(row['config_a']))} vs {escape(str(row['config_b']))}: "
                f"{escape(str(row.get('untestable', '')))}"
            )
    hidden = len(tested) - len(shown_rows)
    if not show_all and hidden:
        pairs = _count(hidden, "non-significant pair")
        console.print(f"  {pairs} {'is' if hidden == 1 else 'are'} not shown; --all shows them.")


def _significance_verdict(rows: list, metric: str, benchmark: str, alpha: float) -> Panel:
    """The answer the table supports, in one box.

    A green box names each pair that survives Holm correction, which side won and
    by how much. When nothing survives, the box says so and points at
    `results power`, because "not significant" has two readings (no difference,
    or too few queries to see one) that call for opposite next steps. The words
    come from `results.verdict_lines`, which `results view` renders too.
    """
    from .knowledge_evaluation_sdk.results import verdict_lines

    verdict = verdict_lines(rows, metric, benchmark, alpha)
    markup = {"text": "{}", "winner": "[bold green]{}[/bold green]", "command": "[bold]{}[/bold]"}
    body = "\n".join(
        "".join(markup[role].format(escape(text)) for role, text in line.spans)
        for line in verdict.lines
    )
    significant = verdict.style == "significant"
    return Panel(
        body,
        title=f"[bold]{verdict.title}[/bold]" if significant else verdict.title,
        title_align="left",
        border_style="green" if significant else "yellow",
        expand=False,
    )


def _format_p(p: float) -> str:
    """Four decimals, or scientific notation below 0.0001 (`results.format_p`)."""
    from .knowledge_evaluation_sdk.results import format_p

    return format_p(p)


@results_app.command("per-query")
def results_per_query(
    results_path: str = typer.Argument(..., help="results.json from a run"),
    output: str = typer.Option("", "--output", "-o", help="Write to .csv or .jsonl"),
    benchmark: str = typer.Option("", "--benchmark", help="Restrict to one benchmark"),
    config: str = typer.Option("", "--config", help="Restrict to one config"),
    limit: int = typer.Option(20, "--limit", help="Rows to print when not writing a file"),
    sort_by: str = typer.Option("", "--sort-by", help="Metric to sort ascending (worst first)"),
):
    """One row per query, retrieval and generation metrics side by side.

    Turns "config A lost" into "config A lost on the multi-hop questions, at
    retrieval rather than generation".

    In the JSONL export, `_retrieved` lists each document's `score` and `rank`.
    The rank is authoritative: scores can tie, so sorting by score does not
    rebuild the ranking the metrics were computed on.
    """

    rows = _load_results(results_path).per_query(benchmark)
    if config:
        rows = [r for r in rows if r["config"] == config]
    if not rows:
        _error("No per-query scores in this run. Nothing scored, or an older results file.")

    if sort_by:
        if not any(sort_by in r for r in rows):
            from .core._text import did_you_mean

            columns = sorted(_metric_columns(rows))
            _error(
                f"No row carries {sort_by!r}{did_you_mean(sort_by, columns)}. "
                f"Available: {', '.join(columns)}."
            )
        rows = sorted(rows, key=lambda r: r.get(sort_by, float("inf")))

    if output:
        path = _output_file(output)
        if path.suffix == ".jsonl":
            path.write_text("\n".join(json.dumps(r, default=str) for r in rows) + "\n")
        else:
            import csv

            # Trajectories and retrieved-id lists do not belong in a CSV cell;
            # JSONL keeps them intact for anyone who wants them.
            columns = [c for c in _row_columns(rows) if not c.startswith("_")]
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
        console.print(f"[green]Wrote[/green] {escape(output)} ({len(rows)} rows)")
        return

    # The sort column first: `--sort-by ndcg@10` used to show the first six
    # metrics alphabetically, which can leave out ndcg@10 itself.
    available = sorted(_metric_columns(rows))
    if sort_by in available:
        available = [sort_by] + [m for m in available if m != sort_by]
    n_queries = len({(r["benchmark"], str(r["sample_id"])) for r in rows})
    configs = sorted({str(r["config"]) for r in rows})
    # Rows, not queries: each query appears once per config.
    title = f"{len(rows)} rows: {n_queries} queries x {len(configs)} configs" + (
        f" (worst {sort_by} first)" if sort_by else ""
    )

    def _table(label, metrics: List[str]) -> Table:
        table = Table(title=escape(title))
        table.add_column("config", no_wrap=True)
        table.add_column("sample", no_wrap=True)
        for name in metrics:
            table.add_column(escape(name), justify="right", no_wrap=True)
        for row in rows[:limit]:
            table.add_row(
                label(str(row["config"])),
                escape(str(row["sample_id"])[:24]),
                *[f"{row[m]:.3f}" if m in row else "-" for m in metrics],
            )
        return table

    # Config names are printed whole; they are what tells two arms apart, and
    # truncating them printed `reranke…` for both. Metric columns are dropped
    # from the right until the table fits, and names are numbered with a
    # legend only when not even one metric column fits beside them.
    numbered = {name: f"[{i}]" for i, name in enumerate(configs, 1)}
    table, legend, shown = None, False, 0
    for use_legend in (False, True):
        label = (lambda n: escape(numbered[n])) if use_legend else escape
        for shown in range(min(6, len(available)), 0, -1):
            candidate = _table(label, available[:shown])
            if _natural_width(candidate) <= console.width or shown == 1 and use_legend:
                table, legend = candidate, use_legend
                break
        if table is not None:
            break
    if legend:
        for name in configs:
            console.print(f"  {escape(numbered[name])} {escape(name)}")
    console.print(table)
    if len(available) > shown:
        console.print(
            f"[dim]{_count(len(available) - shown, 'more metric column')} not shown; "
            "-o rows.csv has all of them.[/dim]"
        )
    if len(rows) > limit:
        console.print(f"[dim]... {len(rows) - limit} more. Use -o rows.csv for all of them.[/dim]")


def _natural_width(renderable) -> int:
    """Width a renderable needs to print without folding or cropping anything."""
    return Measurement.get(console, console.options.update_width(10_000), renderable).maximum


def _row_columns(rows: List[dict]) -> List[str]:
    """Union of keys across rows, with the identifying ones first."""
    lead = ["benchmark", "config", "sample_id", "n_runs"]
    rest = sorted({k for r in rows for k in r} - set(lead))
    return lead + rest


def _metric_columns(rows: List[dict]) -> set:
    return {
        k
        for r in rows
        for k, v in r.items()
        if not k.startswith("_")
        and k not in {"benchmark", "config", "sample_id", "n_runs"}
        and isinstance(v, (int, float))
    }


@results_app.command("interaction")
def results_interaction(
    results_path: str = typer.Argument(..., help="results.json from a factorial run"),
    benchmark: str = typer.Option(
        "", "--benchmark", help="Which benchmark (default: the only one)"
    ),
    metric: str = typer.Option("ndcg@10", "--metric", "-m", help="Metric to test."),
    neither: str = typer.Option("neither", "--neither", help="Config with both techniques off"),
    only_a: str = typer.Option("", "--only-a", help="Config with technique A only"),
    only_b: str = typer.Option("", "--only-b", help="Config with technique B only"),
    both: str = typer.Option("", "--both", help="Config with both on"),
    alpha: float = typer.Option(0.05, "--alpha", help="Significance level of the test."),
    resamples: int = typer.Option(
        10_000,
        "--resamples",
        help="Random sign patterns for the sign-flip permutation test when more than "
        "30 queries differ; with 30 or fewer it is exact. The interval is a t interval.",
    ),
    bootstrap: Optional[int] = typer.Option(
        None, "--bootstrap", hidden=True, help="Old name of --resamples, removed in 0.2.0."
    ),
):
    """Do two techniques compose, or do they overlap?

    Estimates `(AB - neither) - (A - neither) - (B - neither)` from a 2x2 run.
    The p-value is a paired sign-flip permutation test on each query's
    interaction, and the confidence interval a t interval. A is reported against
    a baseline, B is reported against a baseline, and whether A+B is worth A
    alone cannot be recovered from the two, because the interaction is not a
    function of the main effects.
    """

    results = _load_results(results_path)

    benchmarks = sorted({r.benchmark_name for r in results._results if not r.error})
    if not benchmark:
        if len(benchmarks) != 1:
            _error(
                f"This run covers {len(benchmarks)} benchmarks ({', '.join(benchmarks)}). "
                "Pass --benchmark: an interaction is a property of one benchmark, and "
                "pooling across them mixes different corpora into one number."
            )
        benchmark = benchmarks[0]
    _require_metric(results, metric, benchmark)

    configs = sorted({r.config_name for r in results._results if not r.error})
    # A factorial run names its arms `neither`, `<a>`, `<b>`, `<a>+<b>`, so the
    # common case needs no flags at all.
    if not (only_a and only_b and both):
        guessed = [c for c in configs if "+" in c]
        if len(guessed) == 1 and neither in configs:
            left, right = guessed[0].split("+", 1)
            if left in configs and right in configs:
                only_a, only_b, both = only_a or left, only_b or right, both or guessed[0]
    if not (only_a and only_b and both):
        _error(
            "Could not identify the four cells. Pass --only-a/--only-b/--both "
            f"explicitly. Configs in this run: {', '.join(configs)}"
        )

    try:
        report = results.interaction_test(
            benchmark,
            metric,
            neither=neither,
            only_a=only_a,
            only_b=only_b,
            both=both,
            n_bootstrap=resamples if bootstrap is None else bootstrap,
            alpha=alpha,
        )
    except ValueError as exc:
        _error(str(exc))

    table = Table(
        title=f"{escape(metric)} on {escape(benchmark)} (n={report['n_paired']} paired queries)"
    )
    table.add_column("Cell")
    table.add_column("Mean", justify="right")
    table.add_column("vs neither", justify="right")
    base = report["means"][neither]
    for label, name in (
        ("neither", neither),
        ("A only", only_a),
        ("B only", only_b),
        ("both", both),
    ):
        mean = report["means"][name]
        delta = "-" if name == neither else f"{mean - base:+.4f}"
        table.add_row(f"{label}  [dim]{escape(name)}[/dim]", f"{mean:.4f}", delta)
    console.print(table)
    for line in _identical_cells(results, benchmark, [neither, only_a, only_b, both]):
        console.print(f"[yellow]{escape(line)}[/yellow]")

    console.print()
    console.print(f"  expected if additive : {report['expected_if_additive']:+.4f}")
    console.print(f"  actually observed    : {report['combined_effect']:+.4f}")
    colour = "green" if report["significant"] else "yellow"
    console.print(
        f"  [bold]interaction[/bold]          : [{colour}]{report['interaction']:+.4f}[/{colour}]"
        f"  {report['confidence_level']:.0%} CI [{report['ci_lower']:+.4f}, {report['ci_upper']:+.4f}]"
        f"  p={_format_p(report['p_value'])}"
    )
    if report["composition_efficiency"] is not None:
        console.print(
            f"  composition efficiency: {report['composition_efficiency'] * 100:.1f}% "
            "[dim](of the additive prediction actually delivered)[/dim]"
        )
    elif report.get("composition_efficiency_omitted"):
        console.print(
            "  composition efficiency: not reported "
            f"[dim]because {escape(report['composition_efficiency_omitted'])}[/dim]"
        )
    console.print()
    console.print(f"  [bold]{report['verdict']}[/bold]")


def _identical_cells(results, benchmark: str, cells: List[str]) -> List[str]:
    """Pairs of cells that returned the same ranked list on every shared query.

    A factor that never changed retrieval in a cell still yields an
    interaction estimate, and a confident one: in a rerank x MMR factorial on
    `speed` the MMR cell matched `neither` on 12 of 12 queries and the command
    reported a significant redundancy.
    """
    lists: Dict[str, Dict[Any, tuple]] = {}
    for name in cells:
        seen: Dict[Any, tuple] = {}
        for task in results._by_bench_config.get((benchmark, name), []):
            if task.error:
                continue
            for i, sample in enumerate(task.per_sample_scores or []):
                ids = sample.get("_retrieved_ids")
                if isinstance(ids, list):
                    seen.setdefault(sample.get("_sample_id", i), tuple(ids))
        lists[name] = seen

    lines = []
    for i, a in enumerate(cells):
        for b in cells[i + 1 :]:
            shared = set(lists[a]) & set(lists[b])
            if shared and all(lists[a][sid] == lists[b][sid] for sid in shared):
                lines.append(
                    f"Cells {a!r} and {b!r} returned identical ranked lists on all "
                    f"{len(shared)} shared queries. If the factor that separates them "
                    "acts on retrieval, it did nothing in that cell, and the interaction "
                    "below reflects that rather than an overlap between the techniques."
                )
    return lines


skills_app = typer.Typer(
    name="skills",
    help="Agent skills shipped with Lasagna.",
    no_args_is_help=True,
)
app.add_typer(skills_app, name="skills")


def _skills_dir() -> Path:
    return Path(__file__).parent / "skills"


def _available_skills() -> List[Path]:
    root = _skills_dir()
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir() if (p / "SKILL.md").is_file())


def _skill_description(path: Path) -> str:
    """The `description:` line from the skill's frontmatter."""
    text = (path / "SKILL.md").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("description:"):
            return line.split(":", 1)[1].strip()
    return ""


@skills_app.command("list")
def skills_list():
    """Show the agent skills this package ships."""
    skills = _available_skills()
    if not skills:
        console.print("[yellow]No skills bundled in this build.[/yellow]")
        raise typer.Exit(1)

    table = Table(show_lines=True)
    table.add_column("Skill", style="cyan", no_wrap=True)
    table.add_column("Use when", ratio=1)
    for path in skills:
        table.add_row(path.name, _skill_description(path))
    console.print(table)
    console.print()
    console.print("[dim]Install into an agent with `lasagna skills install`.[/dim]")


@skills_app.command("install")
def skills_install(
    target: str = typer.Option(
        ".claude/skills",
        "--target",
        "-t",
        help="Where to write them (Claude Code/Desktop use .claude/skills)",
    ),
    force: bool = typer.Option(False, "--force", help="Overwrite existing files"),
):
    """Copy the bundled skills where a coding agent will find them.

    Skills teach an agent how to use Lasagna: which mode to run a benchmark in,
    why the first search is slow, how to file a useful bug report.
    """
    import shutil

    skills = _available_skills()
    if not skills:
        _error("this build bundles no skills")

    dest = Path(target)
    dest.mkdir(parents=True, exist_ok=True)

    written, skipped = [], []
    for path in skills:
        out = dest / path.name
        if out.exists() and not force:
            skipped.append(path.name)
            continue
        shutil.copytree(path, out, dirs_exist_ok=True)
        written.append(path.name)

    for name in written:
        console.print(f"  [green]installed[/green] {name}")
    for name in skipped:
        console.print(f"  [yellow]exists[/yellow]    {name} [dim](--force to overwrite)[/dim]")

    console.print()
    console.print(f"{_count(len(written), 'skill')} written to {dest}/")
    if written:
        console.print("[dim]Restart the agent, or start a new session, to pick them up.[/dim]")


def _diagnose_or_report_the_setting(diagnose, session_id: str = "default"):
    """Run doctor's checks; a bad setting becomes a failed check, not a traceback.

    Every other command stops at an invalid setting, so this is where a user
    looks next, and it crashed on the same error.
    """
    from .core.diagnostics import Check, Diagnosis

    try:
        return diagnose(check_gh=False, session_id=session_id)
    except Exception as exc:
        config_error = _in_cause_chain(exc, "ConfigError")
        if config_error is None:
            raise
    from .core.provenance import redact

    try:
        from .core.provenance import capture

        environment = redact(capture())
    except Exception:  # noqa: BLE001 - the setting may break this too
        environment = {}
    return Diagnosis(
        checks=[Check("config", "error", redact(str(config_error)))], environment=environment
    )


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output"),
    issue: bool = typer.Option(False, "--issue", help="Print a block to paste into a bug report"),
    session_id: str = typer.Option(
        "default", "--session", "-s", help="Session whose profile to report."
    ),
):
    """Check this installation and say what to fix.

    Answers the questions every support thread opens with: version, install
    source, backend, credentials, model cache, and anything that failed to load
    quietly.

    Output is redacted: no absolute paths, no credential values. `--json` is
    for agents; `--issue` produces a collapsible block for a GitHub issue.
    """
    import json as _json

    from .core.diagnostics import diagnose, format_issue_body

    d = _diagnose_or_report_the_setting(diagnose, session_id)
    try:
        from .core.setup_text import doctor_context

        context = doctor_context(session_id)
    except Exception as exc:  # noqa: BLE001 - doctor reports, it does not fail
        context = [("session", session_id), ("setup", f"could not be read ({exc})")]

    if json_output:
        payload = {**d.to_dict(), "context": dict(context)}
        # soft_wrap: Rich folded long lines at the terminal width, newlines
        # inside JSON strings included, so a long advice line made it invalid.
        console.print(_json.dumps(payload, indent=2), markup=False, highlight=False, soft_wrap=True)
        raise typer.Exit(1 if d.worst == "error" else 0)

    if issue:
        console.print(format_issue_body(d, context), markup=False, highlight=False, soft_wrap=True)
        raise typer.Exit(1 if d.worst == "error" else 0)

    env = d.environment
    console.print(
        f"[bold]lasagna[/bold] {env.get('lasagna_version', '?')} "
        f"({env.get('installed_from', '?')}) · python {env.get('python', '?')} · "
        f"{env.get('platform', '?')}"
    )
    console.print(" · ".join(f"{name} {escape(value)}" for name, value in context))
    console.print()

    style = {"ok": "green", "warning": "yellow", "error": "red"}
    label = {"ok": "ok", "warning": "warn", "error": "FAIL"}
    for check in d.checks:
        colour = style[check.status]
        # `escape` on everything that came from a check rather than from this
        # function. Rich reads square brackets as style markup, and the advice
        # for a missing optional dependency is `pip install "lasagna-rag[eval]"`
        # -- so the extra name, which is the entire actionable content of the
        # message, was being parsed as a tag and dropped. Every affected user
        # was told to install the package they already had. Paths and detail
        # strings can contain brackets too.
        console.print(
            f"  [{colour}]{label[check.status]:<4}[/{colour}] "
            f"{escape(check.name)}: {escape(check.detail)}"
        )
        if check.advice:
            console.print(f"       [dim]{escape(check.advice)}[/dim]")

    console.print()
    problems = d.problems
    if not problems:
        console.print("[green]No problems found.[/green]")
    else:
        errors = sum(1 for c in problems if c.status == "error")
        console.print(
            f"{_count(len(problems), 'item')} {'needs' if len(problems) == 1 else 'need'} "
            f"attention ({errors} blocking). "
            "Run `lasagna doctor --issue` to attach this to a bug report."
        )
    raise typer.Exit(1 if d.worst == "error" else 0)


@results_app.command("leaderboard")
def results_leaderboard(
    directory: str = typer.Argument(
        ...,
        help="A directory of validated bundles, one bundle, or a run's results.json (or "
        "its folder)",
    ),
    metric: str = typer.Option("ndcg@10", "--metric", "-m", help="Metric to rank by."),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Folder to write one page per benchmark into (.md, or .tex with --format "
        "latex), plus index.json",
    ),
    fmt: str = typer.Option(
        "markdown",
        "--format",
        help="markdown, or latex: one booktabs table per comparability group, printed "
        "when no --output is given",
    ),
):
    """Render bundles, or one run's results.json, as leaderboards grouped by
    comparability key.

    Rows inside a group held the same configuration fixed, so they are
    comparable by construction. Rows in different groups are not, and are never
    put in the same table. A results.json with no --output prints the run's own
    leaderboard, the one `benchmark experiment` prints when it finishes.
    """
    from .knowledge_evaluation_sdk.bundle import MetricsVersionError
    from .knowledge_evaluation_sdk.leaderboard import (
        FORMATS,
        _is_bundle,
        build,
        grouped_rows,
        load_bundles,
        render_latex,
    )

    if fmt not in FORMATS:
        _error(f"Unknown format {fmt!r}. Use --format markdown or --format latex.")
    source = Path(directory)
    if not source.exists():
        _error(f"No such file or folder: {directory}")
    if source.is_dir() and (source / "results.json").is_file() and not load_bundles(source):
        # A run's output folder, as `benchmark experiment` names it at the
        # end: it holds results.json and no bundles, and read as a folder it
        # printed "0 row(s) from 0 bundle(s)".
        source = source / "results.json"
        directory = str(source)
    if source.is_file():
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            _error(f"{directory} is not valid JSON ({exc}). Pass a results.json or a bundle.")
        if not _is_bundle(data) and not (
            isinstance(data, dict) and isinstance(data.get("results"), list)
        ):
            _error(f"{source.name} is neither a bundle nor a results.json from a run.")
        if fmt == "markdown" and not output and not _is_bundle(data):
            from .knowledge_evaluation_sdk.results import BenchmarkResults

            run = BenchmarkResults.load(directory)
            _require_metric(run, metric)
            console.print(run.leaderboard(metric), markup=False)
            return
    try:
        if fmt == "latex" and not output:
            # The tables alone and unwrapped, so `> tables.tex` captures valid LaTeX.
            console.print(
                render_latex(grouped_rows(directory, metric), metric),
                markup=False,
                highlight=False,
                soft_wrap=True,
                end="",
            )
            return
        index = build(directory, metric=metric, output=output, fmt=fmt)
    except (MetricsVersionError, ValueError) as exc:
        _error(str(exc))
    console.print(
        f"[green]{_count(index['n_rows'], 'row')}[/green] from "
        f"{_count(index['n_bundles'], 'bundle')} across "
        f"{_count(len(index['benchmarks']), 'benchmark')}"
    )
    for name, info in index["benchmarks"].items():
        console.print(
            f"  {escape(name)}: {_count(info['n_rows'], 'row')} in "
            f"{_count(info['n_groups'], 'comparability group')}, {info['n_verified']} verified"
        )
    if output:
        console.print(f"[green]Wrote {escape(output)}[/green]")


@results_app.command("view")
def results_view(
    results_path: str = typer.Argument(
        ..., help="results.json from a run (or its folder), or a bundle from `results bundle`"
    ),
    metric: Optional[str] = typer.Option(
        None,
        "--metric",
        "-m",
        help="Metric to rank, test and compare queries by (default: ndcg@10 when recorded)",
    ),
    benchmark: str = typer.Option("", "--benchmark", help="Show one benchmark only"),
    diff: Optional[tuple[str, str]] = typer.Option(
        None,
        "--diff",
        metavar="A B",
        help="The two arms the per-query table compares (default: the top two)",
    ),
    output: str = typer.Option("view.html", "--output", "-o", help="HTML file to write"),
    open_page: bool = typer.Option(False, "--open", help="Open the page in your browser"),
):
    """Write one HTML page with everything a run measured.

    The header names the experiment, its date and versions; then the stage
    isolation check, and per benchmark the leaderboard, the Holm-corrected
    significance table with its verdict, every metric per arm, and a per-query
    diff between two arms with what each retrieved. The page is one file with
    no external requests, so it opens offline and can be attached anywhere.
    """
    from .knowledge_evaluation_sdk.viewer import render_html

    path = Path(results_path)
    if not path.exists():
        _error(f"No such file: {results_path}")
    try:
        page = render_html(path, metric=metric, benchmark=benchmark or None, diff=diff)
    except json.JSONDecodeError as exc:
        _error(f"{results_path} is not valid JSON ({exc}). Pass a results.json or a bundle.")
    except ValueError as exc:
        _error(str(exc))

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    console.print(f"[green]Wrote[/green] {escape(str(out))} ({len(page.encode()) / 1024:.0f} KB)")
    if open_page:
        import webbrowser

        webbrowser.open(out.resolve().as_uri())


@results_app.command("schema")
def results_schema(
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write to a file"),
):
    """Print the JSON Schema for a results bundle.

    Generated from the same constants the bundler uses, so a hub validating
    against it can never require a different set of fields from the one this
    package writes.
    """
    import json as _json

    from .knowledge_evaluation_sdk.bundle import SCHEMA_VERSION, json_schema

    text = _json.dumps(json_schema(), indent=2) + "\n"
    if output:
        _output_file(output).write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote {escape(output)}[/green] (schema {SCHEMA_VERSION})")
    else:
        console.print(text, markup=False, highlight=False)


# ---------------------------------------------------------------------------
# Pipelines sub-app
# ---------------------------------------------------------------------------

pipelines_app = typer.Typer(
    name="pipelines",
    help="Inspect registered agentic RAG pipelines.",
    no_args_is_help=True,
)
app.add_typer(pipelines_app, name="pipelines")


@pipelines_app.command("list")
def pipelines_list():
    """List all registered agentic pipelines.

    Shows name, description, implementing class, and which pipeline is
    active (from config). Use it to check that a pipeline you registered
    with @registry.register("agentic_pipeline", ...) is there.
    """
    try:
        from .core.registry import registry
        from .core.config import rag_config

        pipelines = registry.list_pipelines()
        active = rag_config.agentic_pipeline
    except Exception as exc:
        console.print(f"[red]Error loading pipeline registry: {exc}[/red]")
        raise typer.Exit(code=1)

    if not pipelines:
        console.print("[yellow]No agentic pipelines registered.[/yellow]")
        console.print(
            "[dim]Register a pipeline with "
            '@registry.register("agentic_pipeline", "name") '
            "or via entry-points.[/dim]"
        )
        raise typer.Exit()

    table = Table(title="Registered Agentic Pipelines")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Class", style="dim")
    table.add_column("Active", width=8)

    for p in pipelines:
        marker = "[green]\u2713[/green]" if p["name"] == active else ""
        table.add_row(p["name"], p.get("description", ""), p.get("class", "unknown"), marker)

    console.print(table)
    console.print(f"\n[dim]Active pipeline: [cyan]{active}[/cyan][/dim]")
    console.print(f"[dim]{_count(len(pipelines), 'pipeline')} registered[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
