"""Burst the OpenSearch container's file-descriptor budget with concurrent clients.

The Milvus analogue (``test_milvus_fd_burst.py``) is the same shape: eight
worker processes on one ``multiprocessing.Barrier`` so every worker checks
``has_collection`` before any of them creates, each building an
``OpenSearchBackend`` from ``OPENSEARCH_HOSTS`` and calling
``CollectionManager.initialize_collection()`` (create collection + index),
repeated for 50 rounds with a fresh collection name per round.

OpenSearch is the one store service whose pin was reverted: the ``nofile``
65536 and ``memlock`` unlimited values are what OpenSearch's own Docker
documentation requires, but the stack run that would confirm the node starts
and holds was never done (the image could not be pulled in the Codespace),
so ``backends/opensearch.yml`` carries no ``ulimits`` and the pair sits in
``PENDING_MEASUREMENT`` in ``tests/test_compose_nofile_policy.py``. This file
is the measurement that decides it.

Run against a real OpenSearch (CI does not run ``tests/stress``)::

    OPENSEARCH_HOSTS=http://127.0.0.1:9200 pytest tests/stress/test_opensearch_fd_burst.py -q

Skipped unless ``OPENSEARCH_HOSTS`` is set and ``opensearchpy`` imports.
``OPENSEARCH_RACE_TAG`` prefixes collection names (use a different tag per run);
``OPENSEARCH_RACE_ROUNDS`` overrides the round count for smoke tests (default 50).

No embedding model is downloaded: each worker passes a stub dimension,
the same bypass the unit tests use, because the burst under test is about
file descriptors, not embeddings.

The summary line reports the ``nofile`` limit pinned in
``backends/opensearch.yml`` (``unset`` while the measurement decides) alongside
the round counts, so the PR evidence can quote the container's budget and
whether it held::

    rounds=50 failed=0 pinned_nofile=unset index_path_failures=0 ...

If the burst exhausts the container's default descriptors, the failure *is*
the measurement: raise the limit the Qdrant way and pin it in
``tests/test_compose_nofile_policy.py`` (moving the pair out of
``PENDING_MEASUREMENT``), setting ``bootstrap.memory_lock=true`` as well, since
``memlock`` alone does nothing; if it holds, record the numbers instead.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_module
import time
import traceback
from types import SimpleNamespace

import pytest

OPENSEARCH_HOSTS = os.environ.get("OPENSEARCH_HOSTS")

try:
    import opensearchpy  # noqa: F401 - imported only to check availability

    OPENSEARCH_AVAILABLE = True
except ImportError:
    OPENSEARCH_AVAILABLE = False

WORKERS = 8
ROUNDS = int(os.environ.get("OPENSEARCH_RACE_ROUNDS", "50"))
WORKER_TIMEOUT_S = 180
EMBEDDING_DIM = 4


def _pinned_nofile() -> str:
    """The soft nofile pinned in backends/opensearch.yml, or 'unset'."""
    import yaml

    import tests as tests_pkg
    from pathlib import Path

    repo_root = Path(tests_pkg.__file__).resolve().parent.parent
    yml = repo_root / "src" / "lasagna" / "deploy" / "backends" / "opensearch.yml"
    try:
        services = yaml.safe_load(yml.read_text())["services"]
        ulimit = (services["opensearch"].get("ulimits") or {}).get("nofile")
        if ulimit is None:
            return "unset"
        if isinstance(ulimit, dict):
            return str(int(ulimit.get("soft", 0)))
        return str(int(ulimit or 0))
    except Exception:  # noqa: BLE001 - the burst is the measurement, not the parsing
        return "unknown"


pytestmark = pytest.mark.skipif(
    not (OPENSEARCH_HOSTS and OPENSEARCH_AVAILABLE),
    reason="OPENSEARCH_HOSTS not set or opensearchpy not installed",
)


def _round_collection_name(tag: str, round_no: int) -> str:
    return f"opensearch_race_{tag}_{round_no:03d}"


def _worker_main(barrier, collection_name, opensearch_hosts, result_queue):
    """One racer: wait on the barrier, then initialize the collection."""
    import lasagna.api._collection_manager as cm
    from lasagna.vector_backends.opensearch_backend import OpenSearchBackend
    from loguru import logger

    warnings: list = []
    logger.remove()
    handler_id = logger.add(lambda m: warnings.append(m.record["message"]), level="WARNING")
    # Importing the lasagna package disables its own loguru records
    # (src/lasagna/__init__.py); the CLIs re-enable them, and so does this
    # harness -- otherwise index-path warnings would be silently dropped.
    logger.enable("lasagna")

    outcome = {"raised": None, "warnings": []}
    try:
        # Fresh worker process: rebind module state directly (no monkeypatch
        # fixture out here). Stamps stay out of the stress run.
        cm._stamp_store = lambda: None  # noqa: E731 - rebind module state in a fresh worker process
        cm.rag_config = SimpleNamespace(
            collection_name=collection_name,
            index_type="AUTOINDEX",
            metric_type="COSINE",
            index_params={},
            embedding_dim=EMBEDDING_DIM,
            embedding_model="stress-test-stub",
        )
        backend = OpenSearchBackend(
            hosts=[h.strip() for h in opensearch_hosts.split(",") if h.strip()]
        )
        backend.connect()
        manager = cm.CollectionManager(
            backend, "opensearch", {}, SimpleNamespace(dimension=EMBEDDING_DIM)
        )
        barrier.wait(timeout=60)
        manager.initialize_collection()
    except BaseException as exc:  # noqa: BLE001 - the traceback is the evidence
        outcome["raised"] = {
            "type": type(exc).__name__,
            "message": str(exc)[:500],
            "traceback": traceback.format_exc(),
        }
    finally:
        outcome["warnings"] = list(warnings)
        logger.remove(handler_id)
        try:
            result_queue.put(outcome)
        except Exception:  # noqa: BLE001 - never lose a worker outcome silently
            pass


def _classify_failure(raised) -> str:
    """Which code path raised: the index path or the collection path."""
    tb = raised["traceback"]
    if "ensure_index_exists" in tb or "create_index_on_backend" in tb:
        return "index"
    return "collection"


def _run_round(tag: str, round_no: int, opensearch_hosts: str) -> dict:
    name = _round_collection_name(tag, round_no)
    # Spawn, not fork: forking a process that already runs threads (pytest,
    # the logger, client libraries) can deadlock the child. _worker_main is
    # module-level and does its own imports, so spawn works as is.
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(WORKERS)
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker_main, args=(barrier, name, opensearch_hosts, result_queue))
        for _ in range(WORKERS)
    ]
    for proc in procs:
        proc.start()
    deadline = time.monotonic() + WORKER_TIMEOUT_S
    for proc in procs:
        proc.join(timeout=max(0, deadline - time.monotonic()))
    results = []
    while True:
        try:
            results.append(result_queue.get_nowait())
        except queue_module.Empty:
            break
    stuck = [proc for proc in procs if proc.is_alive()]
    for proc in stuck:
        proc.terminate()
    for proc in procs:
        proc.join()

    # Verify afterwards with a fresh client, like a second deploy would see it.
    from lasagna.vector_backends.opensearch_backend import OpenSearchBackend

    verifier = OpenSearchBackend(
        hosts=[h.strip() for h in opensearch_hosts.split(",") if h.strip()]
    )
    verifier.connect()
    try:
        collection_exists = verifier.has_collection(name)
        index_exists = collection_exists and "embedding" in verifier.list_indexes(name)
    finally:
        verifier.disconnect()

    failures = []
    if stuck:
        failures.append({"kind": "timeout", "detail": f"{len(stuck)} workers stuck"})
    if len(results) < WORKERS:
        failures.append({"kind": "missing", "detail": f"only {len(results)}/{WORKERS} outcomes"})
    for outcome in results:
        if outcome["raised"] is not None:
            failures.append(
                {
                    "kind": _classify_failure(outcome["raised"]),
                    "detail": (
                        f"{outcome['raised']['type']}: {outcome['raised']['message']}\n"
                        + outcome["raised"]["traceback"]
                    ),
                }
            )
    if not collection_exists:
        failures.append({"kind": "collection", "detail": "collection missing afterwards"})
    if not index_exists:
        failures.append({"kind": "index", "detail": "embedding index missing afterwards"})
    index_warnings = sum(1 for outcome in results for w in outcome["warnings"] if "ndex" in w)
    return {"name": name, "failures": failures, "index_warnings": index_warnings}


def test_opensearch_fd_burst_against_real_opensearch():
    tag = os.environ.get("OPENSEARCH_RACE_TAG", "t")
    pinned = _pinned_nofile()
    failed_rounds = []
    index_path_failures = 0
    collection_path_failures = 0
    index_warnings = 0
    for round_no in range(ROUNDS):
        summary = _run_round(tag, round_no, OPENSEARCH_HOSTS)
        index_warnings += summary["index_warnings"]
        for failure in summary["failures"]:
            if failure["kind"] == "index":
                index_path_failures += 1
            else:
                collection_path_failures += 1
        if summary["failures"]:
            failed_rounds.append(summary)
    print(
        f"\nrounds={ROUNDS} failed={len(failed_rounds)} pinned_nofile={pinned} "
        f"index_path_failures={index_path_failures} "
        f"collection_path_failures={collection_path_failures} "
        f"index_warnings={index_warnings}"
    )
    assert not failed_rounds, f"{len(failed_rounds)}/{ROUNDS} rounds failed: " + str(
        [(r["name"], [f["kind"] for f in r["failures"]]) for r in failed_rounds]
    )
