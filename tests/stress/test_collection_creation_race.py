"""Stress the collection-creation race against a real Qdrant server.

The fake backend in ``tests/test_collection_creation_race.py`` proves the
manager handles the error; it does not prove the race is gone. This file
races eight worker processes on one ``multiprocessing.Barrier`` so every
worker checks ``has_collection`` before any of them creates, then repeats
50 rounds with a fresh collection name per round -- the scenario CI's
"Live compose stack" hit on 2026-10-28.

Run against a real Qdrant (CI does not run ``tests/stress``)::

    QDRANT_URL=http://127.0.0.1:6333 pytest tests/stress/test_collection_creation_race.py -q

Skipped unless ``QDRANT_URL`` is set and ``qdrant_client`` imports.
``RACE_TAG`` prefixes collection names (use a different tag per run);
``RACE_ROUNDS`` overrides the round count for smoke tests (default 50).

No embedding model is downloaded: each worker passes a stub dimension,
the same bypass the unit tests use, because the race under test is about
collection creation, not embeddings.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue as queue_module
import time
import traceback
from types import SimpleNamespace

import pytest

QDRANT_URL = os.environ.get("QDRANT_URL")

try:
    import qdrant_client  # noqa: F401 - imported only to check availability

    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False

WORKERS = 8
ROUNDS = int(os.environ.get("RACE_ROUNDS", "50"))
WORKER_TIMEOUT_S = 180
EMBEDDING_DIM = 4


def _sanitize_no_proxy() -> None:
    """Drop NO_PROXY spellings that httpx 0.28.1 cannot parse.

    Some sandboxes export bracketed IPv6 literals (``[::1]``) in NO_PROXY.
    httpx only recognises the bare form (``::1``), so it synthesises a
    wildcard key (``all://*[::1]``) whose port fails to parse and every
    ``httpx.Client(trust_env=True)`` -- including the one qdrant-client
    builds -- raises ``InvalidURL`` before any request is made. Rewriting
    the bracketed form to the bare form keeps the bypass entries working.
    """
    import re

    for var in ("NO_PROXY", "no_proxy"):
        val = os.environ.get(var)
        if val:
            os.environ[var] = ",".join(
                re.sub(r"^\[(.*)\]$", r"\1", entry.strip()) for entry in val.split(",")
            )


_sanitize_no_proxy()

pytestmark = pytest.mark.skipif(
    not (QDRANT_URL and QDRANT_AVAILABLE),
    reason="QDRANT_URL not set or qdrant_client not installed",
)


def _round_collection_name(tag: str, round_no: int) -> str:
    return f"race_{tag}_{round_no:03d}"


def _worker_main(barrier, collection_name, qdrant_url, result_queue):
    """One racer: wait on the barrier, then initialize the collection."""
    import lasagna.api._collection_manager as cm
    from lasagna.vector_backends.qdrant_backend import QdrantBackend
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
        backend = QdrantBackend(url=qdrant_url)
        backend.connect()
        manager = cm.CollectionManager(
            backend, "qdrant", {}, SimpleNamespace(dimension=EMBEDDING_DIM)
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


def _run_round(tag: str, round_no: int, qdrant_url: str) -> dict:
    name = _round_collection_name(tag, round_no)
    ctx = mp.get_context("fork")
    barrier = ctx.Barrier(WORKERS)
    result_queue = ctx.Queue()
    procs = [
        ctx.Process(target=_worker_main, args=(barrier, name, qdrant_url, result_queue))
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
    from lasagna.vector_backends.qdrant_backend import QdrantBackend

    verifier = QdrantBackend(url=qdrant_url)
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


def test_collection_creation_race_against_real_qdrant():
    tag = os.environ.get("RACE_TAG", "t")
    failed_rounds = []
    index_path_failures = 0
    collection_path_failures = 0
    index_warnings = 0
    for round_no in range(ROUNDS):
        summary = _run_round(tag, round_no, QDRANT_URL)
        index_warnings += summary["index_warnings"]
        for failure in summary["failures"]:
            if failure["kind"] == "index":
                index_path_failures += 1
            else:
                collection_path_failures += 1
        if summary["failures"]:
            failed_rounds.append(summary)
    print(
        f"\nrounds={ROUNDS} failed={len(failed_rounds)} "
        f"index_path_failures={index_path_failures} "
        f"collection_path_failures={collection_path_failures} "
        f"index_warnings={index_warnings}"
    )
    assert not failed_rounds, f"{len(failed_rounds)}/{ROUNDS} rounds failed: " + str(
        [(r["name"], [f["kind"] for f in r["failures"]]) for r in failed_rounds]
    )
