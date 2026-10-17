"""Stress test fixtures and marks.

All tests in this package are opt-in via ``@pytest.mark.stress``.
Run with::

    pytest tests/stress/ -m stress -v
"""

from __future__ import annotations

import os
import platform
import sys
from typing import List

import pytest
import pytest_asyncio
from loguru import logger

# ---------------------------------------------------------------------------
# Graceful skip when heavy deps are missing (Medium fix)
# ---------------------------------------------------------------------------

_STRESS_DEPS = ["aiohttp"]
_missing = []
for _dep in _STRESS_DEPS:
    if _dep not in sys.modules:
        try:
            __import__(_dep)
        except (ImportError, ModuleNotFoundError):
            _missing.append(_dep)

if _missing:
    collect_ignore_glob = ["test_*.py"]


from .config import StressConfig  # noqa: E402 (after dep check)


# ---------------------------------------------------------------------------
# Register custom marks
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "stress: mark test as a stress test (opt-in, not in normal CI)"
    )


# `pytestmark` in a conftest does nothing -- pytest only honours it in test
# modules and classes -- so the line that used to sit here marked no tests at
# all. It happened not to matter because every module in this package declares
# `pytestmark = pytest.mark.stress` for itself, which is what actually makes
# `-m stress` select them. Keeping the dead line would suggest a new module
# could rely on it.


# ---------------------------------------------------------------------------
# Architecture helpers (Medium fix: handle amd64 as x86)
# ---------------------------------------------------------------------------


def _is_x86() -> bool:
    """Detect x86/amd64 architecture (handles common platform.machine() variants)."""
    arch = platform.machine().lower()
    return arch in ("x86_64", "amd64", "x86-64")


# ---------------------------------------------------------------------------
# Backend discovery
# ---------------------------------------------------------------------------


def get_available_backends() -> List[str]:
    """Return vector backends available on the current architecture.

    Auto-skip rules:
    - milvus: on x86_64/amd64 hosts (the suite's choice; the standalone image is
      published for amd64 and arm64)
    - milvus-lite: on other hosts (single-process embedded)
    - pinecone: requires PINECONE_API_KEY
    """
    backends = ["pgvector", "qdrant", "weaviate", "opensearch"]

    if _is_x86():
        backends.append("milvus")
    else:
        backends.append("milvus-lite")

    if os.environ.get("PINECONE_API_KEY"):
        backends.append("pinecone")

    # Allow override via env to test a single backend
    only = os.environ.get("STRESS_BACKEND_ONLY")
    if only:
        return [only] if only in backends else []

    return backends


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", params=get_available_backends())
def backend(request):
    """Module-scoped backend parameter.

    Each test module runs once per available backend.
    Controlled by STRESS_BACKEND_ONLY env var for single-backend runs.
    """
    return request.param


@pytest.fixture(scope="session")
def stress_config():
    """Session-scoped base stress configuration from environment.

    NOTE: This provides default values only. Per-backend overrides
    (celery, worker mode, profiles) are applied by docker_manager.start()
    which calls StressConfig.for_backend().
    """
    return StressConfig.from_env()


@pytest_asyncio.fixture(scope="session")
async def docker_manager(stress_config):
    """Session-scoped DockerComposeManager — starts Docker once, reuses across modules.

    Avoids 8× Docker restart overhead. The manager skips restart when the
    same backend is already running and healthy.
    """
    from .helpers.docker_manager import DockerComposeManager

    mgr = DockerComposeManager(stress_config)
    yield mgr
    await mgr.stop(keep_volumes=False)


@pytest_asyncio.fixture(scope="module")
async def loaded_backend(docker_manager, backend, stress_config):
    """Start backend, upload test corpus, return test context.

    1. Start compose stack for the backend (tears down any prior backend first)
    2. Upload document fixtures across tenants
    3. Wait for all documents to reach "ready" status
    4. Yield context dict with clients, backend_info, document_map
    5. Teardown handled by docker_manager fixture
    """
    from .helpers.api_client import StressAPIClient
    from .helpers.document_fixtures import generate_test_corpus
    import uuid
    import asyncio

    # Start the stack (manager tears down prior backend if any)
    backend_info = await docker_manager.start(backend)
    # Get the backend-specific config from the manager
    backend_config = docker_manager.backend_config

    # Generate test corpus
    corpus = generate_test_corpus(num_docs=min(backend_config.docs_per_tenant, 20))

    # Create tenant clients and upload documents
    # Use backend_config (not base stress_config) so api_base_url reflects any overrides
    clients = {}
    document_map = {}  # tenant_id -> [(doc_id, test_doc), ...]

    for i in range(min(backend_config.num_tenants, 10)):  # Use subset for corpus loading
        session_id = f"stress-tenant-{i}-{uuid.uuid4().hex[:8]}"
        client = StressAPIClient(backend_config.api_base_url, session_id)
        clients[session_id] = client
        document_map[session_id] = []

        for doc in corpus.documents:
            # Retry transient 500s (cold start race conditions)
            result = None
            for attempt in range(3):
                result = await client.upload_document(doc.path, doc.filename)
                if result.success or (result.status_code and result.status_code != 500):
                    break
                if attempt < 2:
                    await asyncio.sleep(2 * (attempt + 1))
            if result and result.success:
                document_map[session_id].append((result.document_id, doc))
            else:
                logger.warning(
                    f"Upload failed for tenant {i}, doc={doc.filename}: "
                    f"status={result.status_code} error={result.error}"
                )

    # Gate: fail early if uploads are too sparse for meaningful tests
    total_uploaded = sum(len(docs) for docs in document_map.values())
    total_attempted = min(backend_config.num_tenants, 10) * len(corpus.documents)
    if total_uploaded == 0:
        # Collect sample errors for debugging
        sample_errors = []
        for sid, client in list(clients.items())[:2]:
            for doc in corpus.documents[:1]:
                r = await client.upload_document(doc.path, doc.filename)
                sample_errors.append(f"status={r.status_code} err={r.error}")
        pytest.fail(
            f"All {total_attempted} document uploads failed. "
            f"Backend may not be accepting uploads. "
            f"Retry samples: {sample_errors}"
        )
    upload_success_rate = total_uploaded / total_attempted if total_attempted > 0 else 0
    if upload_success_rate < 0.5:
        pytest.fail(
            f"Upload success rate too low: {total_uploaded}/{total_attempted} "
            f"({upload_success_rate:.0%}). Corpus too sparse for valid stress tests."
        )

    # Wait for all documents to be ready
    ready_tasks = []
    for session_id, docs in document_map.items():
        client = clients[session_id]
        for doc_id, _ in docs:
            ready_tasks.append(client.wait_for_ready(doc_id, timeout=backend_config.upload_timeout))

    ready_results = await asyncio.gather(*ready_tasks, return_exceptions=True)
    failures = [r for r in ready_results if isinstance(r, Exception)]
    total_docs = sum(len(docs) for docs in document_map.values())
    ready_count = len(ready_results) - len(failures)

    if ready_count == 0 and total_docs > 0:
        pytest.fail(
            f"All {total_docs} document readiness checks failed. "
            f"First error: {failures[0] if failures else 'unknown'}"
        )

    if failures:
        failure_rate = len(failures) / len(ready_results) if ready_results else 0
        import logging

        log = logging.getLogger(__name__)
        if failure_rate > 0.5:
            pytest.fail(
                f"{len(failures)}/{len(ready_results)} documents failed readiness "
                f"({failure_rate:.0%}). Too many failures to produce valid test results. "
                f"First error: {failures[0]}"
            )
        else:
            log.warning(
                f"{len(failures)}/{len(ready_results)} documents failed readiness "
                f"({failure_rate:.0%}): {failures[0]}"
            )

    yield {
        "backend_info": backend_info,
        "clients": clients,
        "document_map": document_map,
        "corpus": corpus,
        "config": backend_config,
    }

    # Cleanup: close all client sessions
    for client in clients.values():
        await client.close()
