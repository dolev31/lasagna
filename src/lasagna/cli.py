"""RAG Knowledge Base CLI.

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
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

console = Console()
app = typer.Typer(
    name="rag-knowledge",
    help="RAG Knowledge Base — upload, search, configure, and serve.",
    no_args_is_help=True,
)

#: Levels loguru accepts. An unknown value would raise from `logger.add`, which
#: would turn a typo in an environment variable into a crash on every command.
_LOG_LEVELS = frozenset({"TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"})


def _configure_cli_logging(verbose: bool = False, default: str = "WARNING") -> None:
    """Own logging for the CLI process.

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

    level = "DEBUG" if verbose else (os.environ.get("LASAGNA_LOG_LEVEL", "").strip().upper())
    if level not in _LOG_LEVELS:
        level = default

    logger.remove()
    if level == "DEBUG":
        logger.add(sys.stderr, level=level)
        return
    threshold = logger.level(level).no

    def _shown(record) -> bool:
        # Progress lines (corpus ingestion) show at the warnings-only default:
        # without them a half-hour ingest looks like a hang.
        return record["level"].no >= threshold or bool(record["extra"].get("progress"))

    logger.add(sys.stderr, level=min(threshold, logger.level("INFO").no), filter=_shown)


@app.callback()
def _cli(
    verbose: bool = typer.Option(
        False, "--verbose", "-v", help="Show INFO and DEBUG logs (default: warnings only)"
    ),
):
    from loguru import logger

    logger.enable("lasagna")
    # Profiles and settings saved by one command must apply in the next process
    # (`configure` then `upload`, `search`, `serve`): keep sessions on disk.
    os.environ.setdefault("RAG_SESSION_BACKEND", "persistent")
    _configure_cli_logging(verbose)


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


def _load_profile_config(session_id: str):
    """Load the active profile's config for a session as a KnowledgeConfig.

    Falls back to default KnowledgeConfig if profile loading fails.
    """
    from dataclasses import fields as dc_fields
    from lasagna.sdk.config import KnowledgeConfig

    try:
        from lasagna.core.session_config import (
            get_knowledge_config_manager,
        )

        mgr = get_knowledge_config_manager(session_id=session_id)
        profile_name = mgr.get_active_profile(session_id=session_id)

        # Find the active profile's full config dict
        state = mgr.get_session_state(session_id)
        profile_config: dict = {}
        for profile in state.profiles:
            if profile.name == profile_name:
                profile_config = profile.config or {}
                break

        if not profile_config:
            return KnowledgeConfig()

        # Filter to only keys accepted by KnowledgeConfig
        valid_keys = {f.name for f in dc_fields(KnowledgeConfig)}
        filtered = {k: v for k, v in profile_config.items() if k in valid_keys}
        return KnowledgeConfig.from_dict(filtered)

    except Exception:
        # If profile loading fails, fall back to defaults
        return KnowledgeConfig()


def _error(msg: str, exit_code: int = 1):
    """Print error and exit."""
    console.print(f"[red]Error: {escape(msg)}[/red]")
    raise typer.Exit(exit_code)


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------


@app.command()
def init():
    """Interactive setup wizard for RAG knowledge base.

    Guides you through choosing vector backend, embedding model,
    LLM provider, and quality profile. Writes a local configuration file.
    """
    try:
        import questionary
    except ImportError:
        console.print("[red]questionary is required for init wizard.[/red]")
        console.print("Install with: pip install questionary")
        raise typer.Exit(1)

    console.print()
    console.print(
        Panel.fit(
            "[bold cyan]RAG Knowledge Base Setup[/bold cyan]",
            subtitle="Configure your knowledge base",
        )
    )
    console.print()

    # 1. Embedding mode
    embedding_mode = questionary.select(
        "Where should embeddings run?",
        choices=[
            questionary.Choice(
                "Cloud API (OpenAI/Cohere) — lightweight, no GPU needed",
                value="cloud",
            ),
            questionary.Choice(
                "Local (sentence-transformers) — private, needs ~2GB disk",
                value="local",
            ),
        ],
    ).ask()
    if embedding_mode is None:
        raise typer.Exit(0)

    # 2. Vector backend
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

    # 3. LLM provider
    llm_provider = questionary.select(
        "LLM provider (for HyDE, query expansion, answer generation)?",
        choices=[
            questionary.Choice("OpenAI (GPT-4o)", value="openai"),
            questionary.Choice("Anthropic (Claude)", value="anthropic"),
            questionary.Choice("Groq (fast inference)", value="groq"),
            questionary.Choice("IBM watsonx", value="watsonx"),
            questionary.Choice("None — search only, no LLM features", value="none"),
        ],
    ).ask()
    if llm_provider is None:
        raise typer.Exit(0)

    # 4. Quality profile
    profile = questionary.select(
        "Quality profile?",
        choices=[
            questionary.Choice("Speed — pure vector search, fastest", value="speed"),
            questionary.Choice(
                "Standard — recommended (reranking + hybrid search)", value="standard"
            ),
            questionary.Choice("Balanced — best quality/speed tradeoff", value="balanced"),
            questionary.Choice("Max Quality — all features enabled", value="max_quality"),
        ],
        default="standard",
    ).ask()
    if profile is None:
        raise typer.Exit(0)

    # Build config
    if embedding_mode == "cloud":
        embedding_config = {
            "embedding_provider": "openai",
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
        }
    elif profile in ("balanced", "max_quality"):
        embedding_config = {
            "embedding_model": "BAAI/bge-large-en-v1.5",
            "embedding_dim": 1024,
        }
    else:
        embedding_config = {
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dim": 384,
        }

    # Determine which env vars to set
    env_vars = {}
    api_key_env = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
        "watsonx": "WATSONX_APIKEY",
    }
    if llm_provider in api_key_env:
        key_name = api_key_env[llm_provider]
        if not os.environ.get(key_name):
            api_key = questionary.password(f"Enter your {key_name}:").ask()
            if api_key:
                env_vars[key_name] = api_key

    # Write .env file
    try:
        env_path = Path(".env")
        if env_vars:
            existing = env_path.read_text() if env_path.exists() else ""
            with open(env_path, "a") as f:
                for k, v in env_vars.items():
                    if k not in existing:
                        f.write(f"{k}={v}\n")
            console.print(f"  [green]Updated {env_path}[/green]")
    except OSError as exc:
        console.print(f"  [yellow]Warning: could not write .env: {exc}[/yellow]")

    # Write settings override.
    #
    # This file used to be written and then never read: the loader looks in the
    # installed package directory or at $RAG_SETTINGS_FILE, never the working
    # directory. So a user picked the speed profile, saw "Setup complete!"
    # naming it, and downloaded 5.4 GB anyway. Pointing RAG_SETTINGS_FILE at the
    # file we just wrote is what makes the wizard's answers take effect.
    try:
        config_data = {"vector_backend": vector_backend, **embedding_config}
        settings_path = Path("settings.lasagna.toml").resolve()
        with open(settings_path, "w") as f:
            f.write("[rag]\n")
            for k, v in config_data.items():
                if isinstance(v, str):
                    f.write(f'{k} = "{v}"\n')
                else:
                    f.write(f"{k} = {v}\n")
        console.print(f"  [green]Created {settings_path.name}[/green]")
    except OSError as exc:
        _error(f"Could not write settings file: {exc}")

    try:
        env_path = Path(".env")
        existing = env_path.read_text() if env_path.exists() else ""
        if "RAG_SETTINGS_FILE" not in existing:
            with open(env_path, "a") as f:
                f.write(f"RAG_SETTINGS_FILE={settings_path}\n")
            console.print("  [green]Pointed RAG_SETTINGS_FILE at it[/green]")
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

    console.print()
    console.print(
        Panel.fit(
            f"[bold green]Setup complete![/bold green]\n\n"
            f"Profile: [cyan]{profile}[/cyan]\n"
            f"Vector DB: [cyan]{vector_backend}[/cyan]\n"
            f"Embeddings: [cyan]{embedding_config.get('embedding_model', 'cloud')}[/cyan]\n"
            f"LLM: [cyan]{llm_provider}[/cyan]\n\n"
            f"[dim]Quick start:[/dim]\n"
            f"  python -m lasagna upload ./document.pdf\n"
            f'  python -m lasagna search "your query"\n'
            f"  python -m lasagna serve",
        )
    )


# ---------------------------------------------------------------------------
# upload
# ---------------------------------------------------------------------------


@app.command()
def upload(
    file_path: str = typer.Argument(..., help="Path to document file"),
    session_id: str = typer.Option("default", "--session", "-s", help="Session ID"),
):
    """Upload and index a document."""
    path = Path(file_path)
    if not path.exists():
        _error(f"File not found: {file_path}")

    console.print(f"Uploading [cyan]{path.name}[/cyan]...")

    async def _upload():
        client = _get_client(session_id)
        try:
            result = await client.upload(str(path))
            # Processing runs in the background. Closing the client before it
            # finishes tears down the vector store under the worker and leaves
            # the document stuck in PROCESSING, so wait for a terminal status.
            doc = await client.documents.wait_for_processing(result.document_id)
            result.status = doc.status
            result.chunk_count = getattr(doc, "chunk_count", result.chunk_count)
            return result
        finally:
            await client.close()

    try:
        result = _run(_upload())
    except Exception as exc:
        _error(f"Upload failed: {exc}")

    console.print(
        Panel.fit(
            f"[green]Uploaded successfully[/green]\n\n"
            f"Document ID: [cyan]{result.document_id}[/cyan]\n"
            f"Chunks: [cyan]{result.chunk_count}[/cyan]\n"
            f"Status: [cyan]{result.status}[/cyan]",
        )
    )


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

    async def _search():
        client = _get_client(session_id)
        try:
            if context:
                return await client.get_context(query, max_chunks=top_k)
            return await client.search_documents(query, top_k=top_k)
        finally:
            await client.close()

    try:
        result = _run(_search())
    except Exception as exc:
        _error(f"Search failed: {exc}")

    if context:
        # Chunk text is data, not markup: Rich deleted "[metrics](...)" and
        # "[copy.deepcopy(d) ...]" from what it printed.
        console.print(Panel(escape(str(result)), title="LLM Context", border_style="cyan"))
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
    console.print(f"[dim]{len(result)} result(s)[/dim]")


# ---------------------------------------------------------------------------
# status
# ---------------------------------------------------------------------------


@app.command()
def status(
    session_id: str = typer.Option("default", "--session", "-s", help="Session ID"),
):
    """Show knowledge base status."""

    async def _status():
        client = _get_client(session_id)
        try:
            docs = await client.list_documents()
            config = client.get_config_dict()
            return docs, config
        finally:
            await client.close()

    try:
        docs, config = _run(_status())
    except Exception as exc:
        _error(f"Failed to get status: {exc}")

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
        console.print("[dim]No documents uploaded yet.[/dim]")

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


# ---------------------------------------------------------------------------
# serve
# ---------------------------------------------------------------------------


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="API port"),
    host: str = typer.Option(
        "127.0.0.1",
        "--host",
        help="Bind address. Default is loopback; --host 0.0.0.0 exposes the API "
        "on every interface, and it is UNAUTHENTICATED (see SECURITY.md).",
    ),
    configurator: bool = typer.Option(
        True, "--configurator/--no-configurator", help="Enable configurator UI"
    ),
    open_browser: bool = typer.Option(
        True, "--open/--no-open", help="Open browser to configurator"
    ),
):
    """Start the RAG REST API server with configurator UI."""
    # A long-lived server should say what it resolved. The CLI default of
    # WARNING is right for one-shot commands and wrong here.
    _configure_cli_logging(default="INFO")
    if configurator:
        os.environ["RAG_CONFIGURATOR"] = "true"

    console.print(
        Panel.fit(
            f"[bold cyan]RAG Knowledge API[/bold cyan]\n\n"
            f"API: http://{host}:{port}/api/v1\n"
            f"Docs: http://{host}:{port}/docs\n"
            + (f"Configurator: http://localhost:{port}/configurator\n" if configurator else ""),
        )
    )

    if open_browser and configurator:
        import threading

        def _open():
            import time

            time.sleep(2)
            try:
                import webbrowser

                webbrowser.open(f"http://localhost:{port}/configurator")
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

    uvicorn.run(
        "lasagna.api.rest.app:app",
        host=host,
        port=port,
        reload=False,
        # Trust the reverse proxy's forwarding headers when there is one.
        # Without these the app sees the proxy's address as the client for
        # every request, which makes any per-IP control meaningless and the
        # access log misleading.
        proxy_headers=True,
        forwarded_allow_ips=os.environ.get("RAG_FORWARDED_ALLOW_IPS", "127.0.0.1"),
    )


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
        help="Apply even if the profile's embedding size differs from stored documents",
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
        from lasagna.api._profile_guard import EmbeddingSizeConflict

        try:
            mgr.set_active_profile(profile, session_id=session_id, force=force)
        except EmbeddingSizeConflict as exc:
            _error(f"{exc} Or pass --force to switch anyway.")
        except ValueError as exc:
            _error(str(exc))
        console.print(f"[green]Applied profile: {profile}[/green]")
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
        with open(output, "w") as f:
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
# evaluate (researcher mode)
# ---------------------------------------------------------------------------


@app.command()
def evaluate(
    document: str = typer.Argument(..., help="Path to knowledge document"),
    ground_truth: str = typer.Argument(..., help="Path to ground truth JSONL"),
    output_dir: str = typer.Option("./eval_results", "--output", "-o", help="Output directory"),
    session_id: str = typer.Option("eval", "--session", "-s", help="Session ID"),
):
    """Run RAG evaluation benchmark.

    Uploads the document, runs queries from ground truth, scores with LLM judge.
    """
    doc_path = Path(document)
    gt_path = Path(ground_truth)

    if not doc_path.exists():
        _error(f"Document not found: {document}")
    if not gt_path.exists():
        _error(f"Ground truth not found: {ground_truth}")

    console.print("Running evaluation...")
    console.print(f"  Document: [cyan]{doc_path.name}[/cyan]")
    console.print(f"  Ground truth: [cyan]{gt_path.name}[/cyan]")

    from lasagna.knowledge_evaluation_sdk.models import EvalConfig
    from lasagna.knowledge_evaluation_sdk.evaluator import KnowledgeEvaluator

    config = EvalConfig(
        document_path=str(doc_path),
        ground_truth_path=str(gt_path),
        output_dir=output_dir,
    )

    evaluator = KnowledgeEvaluator(config)

    async def _run_eval():
        try:
            return await evaluator.run()
        finally:
            await evaluator.cleanup()

    try:
        metrics = _run(_run_eval())
    except Exception as exc:
        _error(f"Evaluation failed: {exc}")

    console.print()
    console.print(
        Panel.fit(
            f"[bold]Evaluation Complete[/bold]\n\n"
            f"Total cases: [cyan]{metrics.total_cases}[/cyan]\n"
            f"Average score: [cyan]{metrics.avg_score:.2f}[/cyan]\n"
            f"Pass rate: [green]{metrics.pass_rate:.1f}%[/green]\n"
            f"Results saved to: [cyan]{output_dir}[/cyan]",
        )
    )


# ---------------------------------------------------------------------------
# Benchmark sub-app
# ---------------------------------------------------------------------------

benchmark_app = typer.Typer(
    name="benchmark",
    help="Run RAG benchmark evaluations and experiments.",
    no_args_is_help=True,
)
app.add_typer(benchmark_app, name="benchmark")


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
    table.add_column("Description")

    shown = 0
    for name, info in sorted(benchmarks.items()):
        if category and info.category.value != category:
            continue
        table.add_row(
            name,
            info.category.value,
            info.eval_mode.value,
            info.description,
        )
        shown += 1

    console.print(table)
    if category:
        console.print(
            f"\n[dim]{shown} of {len(benchmarks)} benchmarks shown (filtered by {category})[/dim]"
        )
    else:
        console.print(f"\n[dim]{len(benchmarks)} benchmarks available[/dim]")


@benchmark_app.command("info")
def benchmark_info(
    name: str = typer.Argument(..., help="Benchmark name (e.g. beir/nfcorpus)"),
):
    """Show detailed information about a benchmark."""
    from .knowledge_evaluation_sdk.benchmarks import benchmark_registry

    try:
        adapter = benchmark_registry.get(name)
    except KeyError as exc:
        _error(str(exc))

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
    name: str = typer.Argument(..., help="Benchmark name"),
    samples: Optional[int] = typer.Option(100, "--samples", "-n", help="Number of samples"),
    runs: int = typer.Option(1, "--runs", "-r", help="Number of runs"),
    mode: str = typer.Option("auto", "--mode", "-m", help="auto | live | generation_only"),
    split: str = typer.Option("test", "--split", "-s", help="Dataset split"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output directory"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    profile: str = typer.Option("standard", "--profile", "-p", help="Base profile"),
    tracks: str = typer.Option(
        "retrieval,generation",
        "--tracks",
        help="retrieval, generation, or both (comma-separated)",
    ),
    reranker: Optional[str] = typer.Option(
        None, "--reranker", help="Run with this registered reranker"
    ),
    query_enhancer: Optional[str] = typer.Option(
        None, "--query-enhancer", help="Run with this registered query enhancer"
    ),
):
    """Run a single benchmark evaluation."""
    from .knowledge_evaluation_sdk.experiment import (
        Experiment,
        ExperimentConfig,
        BenchmarkSpec,
        PipelineSweepSpec,
        _parse_tracks,
    )

    try:
        track_list = _parse_tracks([t.strip() for t in tracks.split(",") if t.strip()])
    except ValueError as exc:
        _error(str(exc))

    if mode not in ("auto", "live", "generation_only", "offline"):
        _error(
            f"Invalid mode '{mode}'. Must be 'auto', 'live' or "
            f"'generation_only' ('offline' is a deprecated alias)."
        )

    console.print(
        f"Running benchmark [cyan]{name}[/cyan] (up to {samples} samples, {runs} run(s), "
        f"mode={mode}, split={split}, profile={profile}, tracks={','.join(track_list)})"
    )

    # Without a way to name a component, "check it runs" measured the stock
    # pipeline, whatever the user had written.
    config = ExperimentConfig(
        name=f"benchmark_{name.replace('/', '_')}",
        benchmarks=[BenchmarkSpec(name=name, n_samples=samples, split=split)],
        pipeline=PipelineSweepSpec(
            base_profile=profile,
            rerankers=[reranker] if reranker else [],
            query_enhancers=[query_enhancer] if query_enhancer else [],
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
        _error(f"Benchmark failed: {exc}")

    # Summary
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

    if results.failed_tasks:
        console.print(f"\n[red]{len(results.failed_tasks)} task(s) failed[/red]")
    _release_backend()
    _exit_if_tasks_failed(results)


@benchmark_app.command("experiment")
def benchmark_experiment(
    config_file: str = typer.Argument(..., help="YAML or TOML config file"),
    resume: bool = typer.Option(False, "--resume", help="Resume from checkpoints"),
    concurrency: Optional[int] = typer.Option(
        None, "--concurrency", "-c", help="Max parallel tasks"
    ),
):
    """Run a multi-benchmark experiment from a config file."""
    from .knowledge_evaluation_sdk.experiment import Experiment
    from .knowledge_evaluation_sdk.checkpoint import CheckpointManager

    if not Path(config_file).exists():
        _error(f"Config file not found: {config_file}")

    exp = Experiment.from_file(config_file)
    if concurrency:
        exp.config.max_concurrency = concurrency

    # When not resuming, clear any stale checkpoints from a previous run
    if not resume:
        cp_dir = exp.config.checkpoint_dir or exp.config.effective_output_dir()
        cp = CheckpointManager(cp_dir)
        cleared = cp.clear()
        if cleared:
            console.print(f"  [dim]Cleared {cleared} stale checkpoint(s)[/dim]")

    console.print(f"Starting experiment [cyan]{exp.config.name}[/cyan]")
    console.print(f"  Benchmarks: {len(exp.config.benchmarks)}")
    console.print(f"  Runs: {exp.config.n_runs}")
    console.print(f"  Mode: {exp.config.mode}")

    tasks = exp.generate_tasks()
    console.print(f"  Total tasks: {len(tasks)}")

    def _progress(completed: int, total: int, task_key: str):
        console.print(f"  [{completed}/{total}] {task_key}")

    async def _execute():
        return await exp.run(progress_callback=_progress)

    try:
        results = _run(_execute())
    except Exception as exc:
        _error(f"Experiment failed: {exc}")

    output_dir = exp.config.effective_output_dir()
    console.print(
        Panel.fit(
            f"[bold]Experiment Complete[/bold]\n\n"
            f"Tasks: [cyan]{results.total_tasks}[/cyan]\n"
            f"Failed: [{'red' if results.failed_tasks else 'green'}]{len(results.failed_tasks)}[/]\n"
            f"Output: [cyan]{output_dir}[/cyan]",
        )
    )
    _print_isolation(results)
    _release_backend()
    _exit_if_tasks_failed(results)


def _print_isolation(results) -> None:
    """Say whether the arms differed only in the stages they named."""
    from .knowledge_evaluation_sdk.stages import isolation_report

    rows = [r.to_dict() if hasattr(r, "to_dict") else dict(r) for r in results._results]
    report = isolation_report(rows)
    if report["undeclared"]:
        console.print(
            "[red]Stage isolation FAILED: the arms differ outside the stages they "
            "name, so the comparison is not attributable.[/red]"
        )
        for line in report["undeclared"]:
            console.print(f"  - {escape(line)}")
    elif rows and not report["unverified"]:
        console.print("[green]Stage isolation: arms differ only in the stages they name.[/green]")
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
    if not results.failed_tasks:
        return
    for task in results.failed_tasks:
        console.print(f"  [red]failed[/red] {task.task_key}: {task.error}")
    raise typer.Exit(code=1)


@benchmark_app.command("compare")
def benchmark_compare(
    results_a: str = typer.Argument(..., help="First results JSON"),
    results_b: str = typer.Argument(..., help="Second results JSON"),
    metric: str = typer.Option("ndcg@10", "--metric", "-m", help="Metric to compare"),
    format_: str = typer.Option(
        "markdown", "--format", "-f", help="Output format (markdown, latex)"
    ),
):
    """Compare two experiment results."""
    from .knowledge_evaluation_sdk.results import BenchmarkResults

    ra = BenchmarkResults.load(results_a)
    rb = BenchmarkResults.load(results_b)

    if format_ == "latex":
        console.print(ra.to_latex(metric))
        console.print()
        console.print(rb.to_latex(metric))
    else:
        console.print(ra.compare(rb, metric))


@benchmark_app.command("licenses")
def benchmark_licenses(
    output: str = typer.Option("", "--output", "-o", help="Write markdown to this path"),
):
    """List every benchmark with its license and citation.

    Generated rather than hand-written, so it cannot drift from the adapters.
    Lasagna redistributes no benchmark data — each dataset is fetched from its
    own source under its own terms, and those terms are the user's to honour.
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
        "license and terms of use. Those terms are yours to honour — including",
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

        live = "yes" if _adapter_has_corpus(adapter) else "—"
        paper = info.paper or "—"
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
        Path(output).write_text(markdown)
        console.print(f"[green]Wrote[/green] {output} ({len(infos)} benchmarks)")
    else:
        console.print(markdown, markup=False)


@benchmark_app.command("leaderboard")
def benchmark_leaderboard(
    results_path: str = typer.Argument(..., help="Results JSON file"),
    metric: str = typer.Option("ndcg@10", "--metric", "-m", help="Metric for ranking"),
):
    """Display a leaderboard from results."""
    from .knowledge_evaluation_sdk.results import BenchmarkResults

    results = BenchmarkResults.load(results_path)
    console.print(results.leaderboard(metric))


# ---------------------------------------------------------------------------
# Component / plugin sub-apps — for people extending Lasagna
# ---------------------------------------------------------------------------

component_app = typer.Typer(
    name="component",
    help="Validate components you have written.",
    no_args_is_help=True,
)
app.add_typer(component_app, name="component")


@component_app.command("validate")
def component_validate(
    target: str = typer.Argument(
        ..., help="Import path to your component class, e.g. my_pkg.rerankers:MyReranker"
    ),
):
    """Check a component satisfies its stage contract.

    Turns "did I implement this correctly?" from a code review into a command.
    The harness checks the things that make a component safe to swap: that it
    returns the declared type, survives empty input, and does not mutate what
    it was handed -- the last one being the property that makes a comparison
    attributable, since a component that edits its input changes results for
    whatever runs after it.
    """
    from lasagna.core.testing import ComponentTestHarness

    module_path, _, attr = target.rpartition(":")
    if not module_path or not attr:
        _error(f"Expected 'module.path:ClassName', got {target!r}")

    try:
        import importlib

        cls = getattr(importlib.import_module(module_path), attr)
    except Exception as exc:
        _error(f"Could not import {target}: {type(exc).__name__}: {exc}")

    try:
        instance = cls() if isinstance(cls, type) else cls
    except Exception as exc:
        _error(
            f"Could not instantiate {attr} with no arguments: {type(exc).__name__}: {exc}. "
            f"Components must be constructible without required arguments so the "
            f"pipeline can build them from a config name."
        )

    report = ComponentTestHarness().validate(instance)

    console.print(f"\n[bold]{report.component_class}[/bold] ({report.component_type or 'unknown'})")
    for check in report.checks:
        console.print(f"  [green]pass[/green]  {check}")
    for failure in report.failures:
        console.print(f"  [red]FAIL[/red]  {failure}")

    if report.latency_ms is not None:
        console.print(f"\n  latency: {report.latency_ms:.1f} ms")

    if report.passed:
        console.print("\n[green]Contract satisfied.[/green] This component is interchangeable.")
        console.print("Next: register it, then compare it against a control — see docs/guide.md")
    else:
        console.print(
            f"\n[red]{len(report.failures)} contract failure(s).[/red] "
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

    Discovery is best-effort: one broken plugin must not stop the registry
    loading. But a failure was previously only a log line, so an author whose
    entry point was malformed saw nothing — their component simply did not
    exist. Set ``LASAGNA_STRICT_PLUGINS=1`` to raise instead.
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
            f"\n[yellow]{len(failed)} plugin(s) failed to load and are silently absent "
            f"from the registry.[/yellow] Re-run with LASAGNA_STRICT_PLUGINS=1 to raise."
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


@results_app.command("bundle")
def results_bundle(
    results_path: str = typer.Argument(..., help="results.json from a run"),
    output: str = typer.Option("run.json", "--output", "-o", help="Bundle path to write"),
    name: str = typer.Option("", "--name", help="Submitter name"),
    affiliation: str = typer.Option("", "--affiliation", help="Submitter affiliation"),
):
    """Package a run into a self-contained, publishable bundle."""
    from .knowledge_evaluation_sdk.results import BenchmarkResults

    results = BenchmarkResults.load(results_path)
    submitter = {k: v for k, v in {"name": name, "affiliation": affiliation}.items() if v}
    bundle = results.to_bundle(submitter=submitter or None)

    Path(output).write_text(json.dumps(bundle, indent=2, default=str))
    console.print(f"[green]Wrote[/green] {output}")
    console.print(f"  run_id: {bundle['run_id']}")
    for cfg in bundle["configs"]:
        console.print(f"  {cfg['name']}: comparability key {cfg['comparability_key']}")
    warnings = bundle["integrity"]["warnings"]
    if warnings:
        console.print(f"[yellow]  integrity warnings: {', '.join(warnings)}[/yellow]")


@results_app.command("validate")
def results_validate(
    bundle_path: str = typer.Argument(..., help="run.json bundle to check"),
    strict: bool = typer.Option(False, "--strict", help="Treat integrity warnings as failures"),
):
    """Check a bundle is well-formed, complete, and internally consistent.

    Recomputes the comparability key and the run id from the recorded fields.
    A bundle whose numbers were edited by hand no longer matches its own id,
    which is what lets a reviewer reject tampering without rerunning anything.
    """
    from .knowledge_evaluation_sdk.bundle import (
        _content_hash,
        aggregate_mismatches,
        comparability_key,
    )

    bundle = json.loads(Path(bundle_path).read_text())
    problems: List[str] = []

    schema = str(bundle.get("schema_version", ""))
    if not schema.startswith("1."):
        problems.append(f"unsupported schema_version {schema!r} (this build reads 1.x)")

    for field in ("run_id", "experiment", "configs", "results", "integrity"):
        if field not in bundle:
            problems.append(f"missing required block: {field}")

    if "run_id" in bundle and _content_hash(bundle) != bundle["run_id"]:
        problems.append(
            "run_id does not match the bundle contents — it was edited after generation"
        )

    for cfg in bundle.get("configs", []):
        recomputed = comparability_key(cfg.get("held_fixed", {}))
        if recomputed != cfg.get("comparability_key"):
            problems.append(
                f"config {cfg.get('name')!r}: comparability key does not match its "
                f"held_fixed block ({cfg.get('comparability_key')} vs {recomputed})"
            )

    problems.extend(
        f"aggregate disagrees with per-query scores: {m}"
        for m in aggregate_mismatches(bundle.get("results") or [])
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
    if strict:
        problems.extend(f"integrity (strict): {code}" for code in warnings)

    if problems:
        console.print(f"[red]INVALID[/red] — {len(problems)} problem(s):")
        for p in problems:
            console.print(f"  - {p}")
        raise typer.Exit(1)

    console.print(f"[green]VALID[/green] (schema {schema}, run_id {bundle['run_id']})")
    if warnings:
        console.print(f"[yellow]  warnings: {', '.join(warnings)}[/yellow]")


@results_app.command("compare")
def results_compare(
    a: str = typer.Argument(..., help="First bundle or results.json"),
    b: str = typer.Argument(..., help="Second bundle or results.json"),
    metric: str = typer.Option("ndcg@10", "--metric", "-m"),
):
    """Compare two runs, reporting whether they are comparable at all."""
    from .knowledge_evaluation_sdk.results import BenchmarkResults

    console.print(BenchmarkResults.load(a).compare(BenchmarkResults.load(b), metric))


@results_app.command("power")
def results_power(
    results_path: str = typer.Argument(..., help="results.json from a run"),
    config_a: str = typer.Argument(..., help="Baseline config"),
    config_b: str = typer.Argument(..., help="Candidate config"),
    metric: str = typer.Option("ndcg@10", "--metric", "-m"),
    benchmark: str = typer.Option("", "--benchmark", help="Which benchmark"),
    alpha: float = typer.Option(0.05, "--alpha"),
    power: float = typer.Option(0.80, "--power"),
):
    """Could this run have detected the effect you care about?

    A non-significant result has two readings — the techniques are the same, or
    the run was too small to tell — and they call for opposite next steps.
    Reports the smallest difference this run could have detected, and how many
    queries the observed difference would need.
    """
    from .knowledge_evaluation_sdk.results import BenchmarkResults

    results = BenchmarkResults.load(results_path)
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

    try:
        report = results.minimum_detectable_effect(
            benchmark, config_a, config_b, metric, alpha=alpha, power=power
        )
    except ValueError as exc:
        _error(str(exc))

    console.print(f"[bold]{metric}[/bold] on {benchmark}: {config_b} vs {config_a}")
    console.print(f"  paired queries       : {report['n_paired']}")
    console.print(f"  observed difference  : {report['observed_difference']:+.4f}")
    console.print(f"  sd of differences    : {report['sd_of_differences']:.4f}")
    colour = "yellow" if report["underpowered"] else "green"
    console.print(
        f"  [bold]detectable at {int(report['power'] * 100)}% power[/bold]: "
        f"[{colour}]{report['minimum_detectable_effect']:.4f}[/{colour}]"
    )
    if report["n_needed_for_observed_effect"]:
        console.print(
            f"  queries needed for the observed effect: {report['n_needed_for_observed_effect']}"
        )
    console.print()
    console.print(f"  {report['interpretation']}")


def _format_effect_size(effect: Optional[float]) -> str:
    """Render Cohen's ``d_z``, including the case where it has no finite value.

    An infinite ``d_z`` means every query moved by the same amount -- the most
    consistent effect there is. Printing that as a number is the problem this
    exists to avoid: ``0.00`` reads as "no effect", and the raw float reads as
    a measurement of twenty quadrillion standard deviations.

    A standalone function rather than a branch inline, because as a branch only
    one of its two signs was reachable from the test fixtures and disabling the
    other changed nothing.
    """
    import math

    if effect is None:
        return "—"
    if math.isinf(effect):
        return "+∞" if effect > 0 else "-∞"
    if math.isnan(effect):
        return "—"
    return f"{effect:+.2f}"


@results_app.command("significance")
def results_significance(
    results_path: str = typer.Argument(..., help="results.json from a run"),
    metric: str = typer.Option("ndcg@10", "--metric", "-m"),
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
    pair that happened to look best.

    `BenchmarkResults.pairwise_significance` has done this since the beginning
    and was reachable only from Python -- the CLI offered `compare`, `power` and
    `per-query` but not the test a paper actually reports.
    """
    from .knowledge_evaluation_sdk.results import BenchmarkResults

    if correction not in ("holm", "none"):
        _error(f"Unknown correction {correction!r}. Use 'holm' or 'none'.")

    results = BenchmarkResults.load(results_path)
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
    console.print(f"[bold]{metric}[/bold] on {benchmark} — {len(rows)} pairs, {label}")
    console.print()

    table = Table(box=None, pad_edge=False)
    # Config names are what identifies a pair; wrap them rather than cut them.
    table.add_column("A", overflow="fold")
    table.add_column("B", overflow="fold")
    table.add_column("diff", justify="right")
    table.add_column("d_z", justify="right")
    table.add_column("p", justify="right")
    if correction == "holm":
        table.add_column("p (Holm)", justify="right")
    table.add_column("", justify="left")

    shown = 0
    for row in rows:
        if not show_all and not row.get("significant"):
            continue
        shown += 1
        significant = row.get("significant")
        cells = [
            escape(str(row["config_a"])),
            escape(str(row["config_b"])),
            f"{row['difference']:+.4f}",
            _format_effect_size(row.get("effect_size")),
            f"{row['p_value']:.4f}",
        ]
        if correction == "holm":
            cells.append(f"{row.get('p_adjusted', row['p_value']):.4f}")
        cells.append("[green]significant[/green]" if significant else "[dim]—[/dim]")
        table.add_row(*cells)
    console.print(table)

    survivors = sum(1 for r in rows if r.get("significant"))
    console.print()
    if correction == "holm":
        console.print(
            f"  {survivors} of {len(rows)} pairs survive correction at family-wise alpha={alpha}."
        )
    else:
        console.print(
            f"  {survivors} of {len(rows)} pairs are significant at alpha={alpha}, "
            f"[yellow]uncorrected[/yellow] — with {len(rows)} comparisons, expect "
            f"about {len(rows) * alpha:.1f} of these by chance."
        )
    if not show_all and shown < len(rows):
        console.print(f"  {len(rows) - shown} non-significant pairs hidden; --all shows them.")


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
    retrieval rather than generation". The two eval paths already wrote
    per-query rows into the same list and nothing joined them, so the answer was
    in the results file and could not be read out of it.
    """
    from .knowledge_evaluation_sdk.results import BenchmarkResults

    rows = BenchmarkResults.load(results_path).per_query(benchmark)
    if config:
        rows = [r for r in rows if r["config"] == config]
    if not rows:
        _error("No per-query scores in this run. Nothing scored, or an older results file.")

    if sort_by:
        if not any(sort_by in r for r in rows):
            _error(f"No row carries {sort_by!r}. Available: {sorted(_metric_columns(rows))}")
        rows = sorted(rows, key=lambda r: r.get(sort_by, float("inf")))

    if output:
        path = Path(output)
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
        console.print(f"[green]Wrote[/green] {output} ({len(rows)} rows)")
        return

    metrics = sorted(_metric_columns(rows))[:6]
    table = Table(title=f"{len(rows)} queries" + (f" (worst {sort_by} first)" if sort_by else ""))
    table.add_column("config")
    table.add_column("sample")
    for name in metrics:
        table.add_column(name, justify="right")
    for row in rows[:limit]:
        table.add_row(
            row["config"],
            str(row["sample_id"])[:24],
            *[f"{row[m]:.3f}" if m in row else "—" for m in metrics],
        )
    console.print(table)
    if len(rows) > limit:
        console.print(f"[dim]... {len(rows) - limit} more. Use -o rows.csv for all of them.[/dim]")


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
    metric: str = typer.Option("ndcg@10", "--metric", "-m"),
    neither: str = typer.Option("neither", "--neither", help="Config with both techniques off"),
    only_a: str = typer.Option("", "--only-a", help="Config with technique A only"),
    only_b: str = typer.Option("", "--only-b", help="Config with technique B only"),
    both: str = typer.Option("", "--both", help="Config with both on"),
    alpha: float = typer.Option(0.05, "--alpha"),
    bootstrap: int = typer.Option(10_000, "--bootstrap"),
):
    """Do two techniques compose, or do they overlap?

    Estimates `(AB - neither) - (A - neither) - (B - neither)` from a 2x2 run,
    with a bootstrap confidence interval. The question every RAG paper raises
    and none answers: A is reported against a baseline, B is reported against a
    baseline, and whether A+B is worth A alone cannot be recovered from the two
    papers, because the interaction is not a function of the main effects.
    """
    from .knowledge_evaluation_sdk.results import BenchmarkResults

    results = BenchmarkResults.load(results_path)

    benchmarks = sorted({r.benchmark_name for r in results._results if not r.error})
    if not benchmark:
        if len(benchmarks) != 1:
            _error(
                f"This run covers {len(benchmarks)} benchmarks ({', '.join(benchmarks)}). "
                "Pass --benchmark: an interaction is a property of one benchmark, and "
                "pooling across them mixes different corpora into one number."
            )
        benchmark = benchmarks[0]

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
            n_bootstrap=bootstrap,
            alpha=alpha,
        )
    except ValueError as exc:
        _error(str(exc))

    table = Table(title=f"{metric} on {benchmark} (n={report['n_paired']} paired queries)")
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
        delta = "—" if name == neither else f"{mean - base:+.4f}"
        table.add_row(f"{label}  [dim]{name}[/dim]", f"{mean:.4f}", delta)
    console.print(table)

    console.print()
    console.print(f"  expected if additive : {report['expected_if_additive']:+.4f}")
    console.print(f"  actually observed    : {report['combined_effect']:+.4f}")
    colour = "green" if report["significant"] else "yellow"
    console.print(
        f"  [bold]interaction[/bold]          : [{colour}]{report['interaction']:+.4f}[/{colour}]"
        f"  95% CI [{report['ci_lower']:+.4f}, {report['ci_upper']:+.4f}]"
        f"  p={report['p_value']:.4f}"
    )
    if report["composition_efficiency"] is not None:
        console.print(
            f"  composition efficiency: {report['composition_efficiency'] * 100:.1f}% "
            "[dim](of the additive prediction actually delivered)[/dim]"
        )
    console.print()
    console.print(f"  [bold]{report['verdict']}[/bold]")


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

    Skills teach an agent how to use Lasagna correctly — which mode to run a
    benchmark in, why the first search is slow, how to file a useful bug report.
    An agent that has them stops guessing.
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
    console.print(f"{len(written)} skill(s) written to {dest}/")
    if written:
        console.print("[dim]Restart the agent, or start a new session, to pick them up.[/dim]")


@app.command("doctor")
def doctor(
    json_output: bool = typer.Option(False, "--json", help="Machine-readable output"),
    issue: bool = typer.Option(False, "--issue", help="Print a block to paste into a bug report"),
):
    """Check this installation and say what to fix.

    Answers the questions every support thread opens with — version, install
    source, backend, credentials, model cache, and anything that failed to load
    quietly — so nobody has to be asked them.

    Output is redacted: no absolute paths, no credential values. `--json` is
    for agents; `--issue` produces a collapsible block for a GitHub issue.
    """
    import json as _json

    from .core.diagnostics import diagnose, format_issue_body

    d = diagnose()

    if json_output:
        console.print(_json.dumps(d.to_dict(), indent=2), markup=False, highlight=False)
        raise typer.Exit(1 if d.worst == "error" else 0)

    if issue:
        console.print(format_issue_body(d), markup=False, highlight=False)
        raise typer.Exit(1 if d.worst == "error" else 0)

    env = d.environment
    console.print(
        f"[bold]lasagna[/bold] {env.get('lasagna_version', '?')} "
        f"({env.get('installed_from', '?')}) · python {env.get('python', '?')} · "
        f"{env.get('platform', '?')}"
    )
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
            f"{len(problems)} item(s) need attention ({errors} blocking). "
            "Run `lasagna doctor --issue` to attach this to a bug report."
        )
    raise typer.Exit(1 if d.worst == "error" else 0)


@results_app.command("leaderboard")
def results_leaderboard(
    directory: str = typer.Argument(..., help="Directory of validated bundles"),
    metric: str = typer.Option("ndcg@10", "--metric", "-m"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Write markdown here"),
):
    """Render bundles as leaderboards, grouped by comparability key.

    Rows inside a group held the same configuration fixed, so they are
    comparable by construction. Rows in different groups are not, and are never
    put in the same table.
    """
    from .knowledge_evaluation_sdk.leaderboard import build

    index = build(directory, metric=metric, output=output)
    console.print(
        f"[green]{index['n_rows']} row(s)[/green] from {index['n_bundles']} bundle(s) "
        f"across {len(index['benchmarks'])} benchmark(s)"
    )
    for name, info in index["benchmarks"].items():
        console.print(
            f"  {name}: {info['n_rows']} row(s) in {info['n_groups']} "
            f"comparability group(s), {info['n_verified']} verified"
        )
    if output:
        console.print(f"[green]Wrote {output}[/green]")


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
        Path(output).write_text(text, encoding="utf-8")
        console.print(f"[green]Wrote {output}[/green] (schema {SCHEMA_VERSION})")
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
    currently active (from config).  Useful for verifying custom pipeline
    registration after using ``@registry.register("agentic_pipeline", ...)``.
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
    console.print(f"[dim]{len(pipelines)} pipeline(s) registered[/dim]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app()
