"""Conftest for knowledge tests — data isolation and heavy-dependency probing.

Two responsibilities, both of which must happen *before* pytest imports any
test module:

1. **Data isolation.**  Redirect ``RAG_DB_DIR`` / ``RAG_UPLOAD_DIR`` and
   ``HOME`` at a throwaway directory so the suite never touches the developer's
   real ``~/.rag_knowledge`` store or anything else under their home, and turn
   off ragas usage telemetry.  See the notes below for why this is required.

2. **Dependency probing.**  With lazy imports in ``knowledge/__init__.py`` the
   package itself imports without heavy dependencies, but individual test
   files still need the full stack (langchain, sentence-transformers,
   pymilvus, etc.).  Probe those here and skip the whole directory when they
   are unavailable.
"""

import atexit
import os
import shutil
import sys
import tempfile

import pytest

# ---------------------------------------------------------------------------
# Data isolation — MUST run before any ``lasagna`` import.
#
# ``providers._auto_configure_from_env()`` reads RAG_DB_DIR/RAG_UPLOAD_DIR
# once, memoises the result in a module-level singleton, and otherwise
# defaults to ``~/.rag_knowledge``.  Profile and session persistence then goes
# through ``_get_db()`` -> ``get_rag_database()`` -> that directory.
#
# Without this block the suite reads and writes the real user database:
# ``KnowledgeConfigManager`` tests persist profiles such as "new_name" into
# ``~/.rag_knowledge/dbs/rag.db``, which both corrupts the developer's own
# configuration and makes the rename tests fail permanently on every
# subsequent run (the name they rename *to* already exists).
#
# Note the fixtures that patch ``_resolve_user_data_path`` do NOT provide this
# isolation: that seam only feeds the legacy JSON migration paths, not the
# SQLite store that persistence actually uses.
#
# Always override, never ``setdefault`` — a developer with RAG_DB_DIR exported
# for real use would otherwise still have their data written to by the suite.
# ---------------------------------------------------------------------------
_ISOLATED_DATA_DIR = tempfile.mkdtemp(prefix="lasagna-tests-")
os.environ["RAG_DB_DIR"] = os.path.join(_ISOLATED_DATA_DIR, "dbs")
os.environ["RAG_UPLOAD_DIR"] = os.path.join(_ISOLATED_DATA_DIR, "uploads")
os.makedirs(os.environ["RAG_DB_DIR"], exist_ok=True)
os.makedirs(os.environ["RAG_UPLOAD_DIR"], exist_ok=True)
atexit.register(shutil.rmtree, _ISOLATED_DATA_DIR, True)

# ---------------------------------------------------------------------------
# A throwaway HOME, and no usage telemetry.
#
# Libraries under test write under the home directory: ragas keeps a random
# user id in its data folder, matplotlib writes a config cache, and anything
# that defaults to ``~/.something`` lands there too. A plain ``pytest`` left
# those in the developer's real home. It also let a developer's own
# ``~/nltk_data`` change how sentences are split, so two machines ran
# different chunkers.
#
# The model cache is the one exception: HF_HOME is pinned to where it already
# was, so models a developer has downloaded are reused rather than fetched
# again (and stay found with HF_HUB_OFFLINE set).
#
# The package sets RAGAS_DO_NOT_TRACK on import; setting it here as well keeps
# the suite silent even for a test that builds a ragas judge before importing
# lasagna. tests/test_ragas_telemetry_is_off.py checks the package default.
# ---------------------------------------------------------------------------
_REAL_HOME = os.path.expanduser("~")
# For the few checks that look for the developer's real home path in files
# written outside the suite (e2e reports).
os.environ["LASAGNA_TESTS_REAL_HOME"] = _REAL_HOME
os.environ.setdefault(
    "HF_HOME",
    os.path.join(
        os.environ.get("XDG_CACHE_HOME") or os.path.join(_REAL_HOME, ".cache"), "huggingface"
    ),
)
os.environ["HOME"] = os.path.join(_ISOLATED_DATA_DIR, "home")
os.makedirs(os.environ["HOME"], exist_ok=True)
os.environ["RAGAS_DO_NOT_TRACK"] = "true"

# Starlette's TestClient sends ``Host: testserver``, and with auth off the API
# answers only to loopback names and those in RAG_ALLOWED_HOSTS
# (tests/test_loopback_servers_refuse_a_foreign_host.py). Name the test client
# here rather than in every test that builds one.
os.environ["RAG_ALLOWED_HOSTS"] = "testserver"

# gRPC's fork handlers. Once a test has opened a Milvus connection, every later
# subprocess the suite starts gets gRPC's "FD from fork parent still in poll
# list" lines on its stderr, written by this process's handlers between fork
# and exec. Tests that assert a child wrote nothing then failed in some test
# orders only. The suite never uses a gRPC channel in a forked child, so the
# handlers are not needed here. Must be set before grpc is first imported.
os.environ.setdefault("GRPC_ENABLE_FORK_SUPPORT", "false")

# Probe critical heavy dependencies at conftest-load time (before test
# modules are imported).  If any are missing, tell pytest to skip every
# file in this directory.
_REQUIRED_DEPS = [
    "langchain_core",
    "sentence_transformers",
]

_missing = []
for _dep in _REQUIRED_DEPS:
    if _dep not in sys.modules:
        try:
            __import__(_dep)
        except (ImportError, ModuleNotFoundError):
            _missing.append(_dep)

if _missing:
    # collect_ignore_glob is evaluated by pytest before it imports test files
    collect_ignore_glob = ["test_*.py"]


# ---------------------------------------------------------------------------
# Tests that load real embedding weights.
#
# A few tests construct the configured embedding model (BAAI/bge-m3 by
# default, BAAI/bge-large-en-v1.5 as fallback) instead of a stub, so on a cold
# cache they download it from the Hugging Face Hub. That is what CI does. With
# HF_HUB_OFFLINE (or TRANSFORMERS_OFFLINE) set and no cached copy they cannot
# pass, and a RuntimeError from the model loader says nothing about the code
# under test, so they are skipped with the reason instead. Online runs are
# unaffected.
# ---------------------------------------------------------------------------
_TRUTHY = {"1", "true", "yes", "on"}


def _hf_offline() -> bool:
    return any(
        os.environ.get(var, "").strip().lower() in _TRUTHY
        for var in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    )


def _embedding_weights_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        from lasagna.core.config import get_rag_config

        config = get_rag_config()
        names = [config.embedding_model, config.embedding_model_fallback]
    except Exception:  # noqa: BLE001 - any failure means "cannot confirm a cached copy"
        return False
    return any(
        isinstance(try_to_load_from_cache(name, "config.json"), str) for name in names if name
    )


def _models_cached(names) -> bool:
    """Whether every named model is in the Hugging Face cache (a bare name is sentence-transformers')."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except Exception:  # noqa: BLE001
        return False

    def cached(name: str) -> bool:
        candidates = [name] if "/" in name else [name, f"sentence-transformers/{name}"]
        return any(isinstance(try_to_load_from_cache(c, "config.json"), str) for c in candidates)

    return all(cached(n) for n in names)


def pytest_collection_modifyitems(config, items):
    """Skip ``needs_embedding_weights`` tests offline when their models are not cached.

    ``@pytest.mark.needs_embedding_weights("all-MiniLM-L6-v2", ...)`` names the
    models the test loads; without names it means the configured embedding
    model. Online (CI's weights job) nothing is skipped: a model that cannot
    load fails its test there.
    """
    if not _hf_offline():
        return
    import pytest

    configured = None
    for item in items:
        marker = item.get_closest_marker("needs_embedding_weights")
        if marker is None:
            continue
        if marker.args:
            if _models_cached(marker.args):
                continue
            reason = f"loads {', '.join(marker.args)}; Hugging Face offline mode is set and "
        else:
            if configured is None:
                configured = _embedding_weights_cached()
            if configured:
                continue
            reason = "loads the configured embedding model; Hugging Face offline mode is set and "
        item.add_marker(pytest.mark.skip(reason=reason + "no cached copy was found"))


@pytest.fixture(autouse=True)
def _process_limiters_restored():
    """Put the process-wide rate limiter and quota manager back after each test.

    A test that configured an enabled limiter (or a Redis one with no Redis)
    left it in place, so every REST test that ran after it in a shuffled order
    was throttled to ten requests and started failing with 429: three
    unrelated tests failed one run in five.
    """
    from lasagna.core import quota, rate_limiter

    saved = (rate_limiter._instance, quota._instance)
    yield
    rate_limiter._instance, quota._instance = saved


@pytest.fixture(autouse=True)
def _session_backend_restored():
    """Put the session backend and the providers back after each test.

    Running the CLI in process sets ``RAG_SESSION_BACKEND=persistent`` with
    ``os.environ.setdefault`` (as ``lasagna`` does for a real run) and leaves
    the persistent provider it configured, so every later test searched with
    the settings of the process-wide config manager. A test that wrote through
    a manager of its own then read another one's settings:
    ``test_configurator``'s save-as-active check failed whenever a CLI test ran
    before it, on public-main too.
    """
    import lasagna.providers as providers

    env = os.environ.get("RAG_SESSION_BACKEND")
    names = ("_config_provider", "_llm_provider", "_session_provider", "_auto_config_done")
    saved = {name: getattr(providers, name) for name in names}
    yield
    if env is None:
        os.environ.pop("RAG_SESSION_BACKEND", None)
    else:
        os.environ["RAG_SESSION_BACKEND"] = env
    with providers._provider_lock:
        for name, value in saved.items():
            setattr(providers, name, value)


@pytest.fixture(autouse=True)
def _lasagna_logs_enabled():
    """The package disables its own logs on import (library default); several
    tests capture them, so turn them back on before every test.

    Once per session was not enough: a test that ended with
    ``logger.disable("lasagna")`` silenced every test after it in a shuffled
    order, and one that waits for a warning failed one run in a few.
    """
    import lasagna  # noqa: F401  - make sure its disable() has already run
    from loguru import logger

    logger.enable("lasagna")
    yield


def pytest_sessionfinish(session, exitstatus):
    """Close every vector-store connection before the interpreter exits.

    Milvus Lite's gRPC threads are not daemons, and Python joins non-daemon
    threads before it exits: a test that left a client open kept the process
    alive after pytest printed its summary (seen intermittently in CI, with
    "too_many_pings" from the server), so a green run could still time out.
    """
    try:
        from lasagna.api.rag_backend import close_backend

        close_backend()
    except Exception:
        pass
    try:
        from pymilvus import connections

        for alias, _ in list(connections.list_connections()):
            try:
                connections.disconnect(alias)
            except Exception:
                pass
    except Exception:
        pass
    try:
        from milvus_lite.server_manager import server_manager_instance

        server_manager_instance.release_all()
    except Exception:
        pass
    _stop_idle_anyio_workers()
    _exit_even_if_a_thread_never_finishes(exitstatus)


def _stop_idle_anyio_workers():
    """Stop AnyIO worker threads that will never be told to stop.

    A worker stops when the task that owns its event loop finishes. The MCP
    tests run many short `asyncio.run` loops, and now and then one stops before
    the worker's stop callback runs: the thread then waits on its queue forever,
    and Python, which joins non-daemon threads at exit, never exits. Caught in
    5 of 12 local runs with a thread dump at exit.
    """
    import threading

    for thread in threading.enumerate():
        if thread.name == "AnyIO worker thread" and hasattr(thread, "stop"):
            try:
                thread.stop()
            except Exception:
                pass


def _exit_even_if_a_thread_never_finishes(exitstatus, grace_s: float = 120.0):
    """If something else keeps the process alive after the summary, say what and exit.

    A daemon thread keeps running while Python joins the others at exit. After
    `grace_s` it prints every thread's stack, so the culprit is named in the CI
    log, and exits with pytest's own status: a green run stays green instead of
    hanging until the job's timeout.
    """
    import faulthandler
    import threading
    import time

    def guard():
        time.sleep(grace_s)
        sys.stderr.write(
            f"\nThe test process did not exit {grace_s:.0f}s after the run ended; "
            "these threads kept it alive:\n"
        )
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        sys.stderr.flush()
        os._exit(int(exitstatus))

    threading.Thread(target=guard, name="exit-guard", daemon=True).start()
