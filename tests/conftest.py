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


def pytest_collection_modifyitems(config, items):
    if not _hf_offline():
        return
    marked = [item for item in items if item.get_closest_marker("needs_embedding_weights")]
    if not marked or _embedding_weights_cached():
        return
    import pytest

    skip = pytest.mark.skip(
        reason="loads the configured embedding model; Hugging Face offline mode is set "
        "and no cached copy was found"
    )
    for item in marked:
        item.add_marker(skip)


@pytest.fixture(autouse=True, scope="session")
def _lasagna_logs_enabled():
    """The package disables its own logs on import (library default); several
    tests capture them, so turn them back on once everything is imported."""
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
