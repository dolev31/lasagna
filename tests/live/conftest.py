"""Shared fixtures for live integration tests.

These 100-odd tests have never run. The fixtures they gate on were inherited
from CUGA, the project this package was extracted from, and describe services
Lasagna does not have:

    - "Registry on :8001"  -- no such service; the string 8001 appears nowhere
                              outside this file and a sample document
    - "Demo server on :7860" -- likewise; that is Gradio's default port

so `registry_server` and `demo_server` skipped unconditionally, and every test
in ``test_smoke.py`` with them. Worse, the surviving `mcp_server` fixture was
one name doing the work of three different services:

    {mcp_server}/health, /docs, /api/v1/...  -> the REST API, port 8000
    {mcp_server}/sse                         -> the standalone MCP server, 8005
    {mcp_server}/mcp/sse                     -> MCP mounted inside the REST app

No single port satisfies all three, so some of these could not have passed in
any configuration. It probed :8005 for ``/api/v1/health``, which the MCP server
does not serve, so in practice everything skipped and the suite was quietly
inert rather than failing.

What actually exists:

    lasagna serve                  REST API + configurator on 127.0.0.1:8000
    python -m lasagna.mcp          MCP over SSE on 127.0.0.1:8005
    RAG_EMBED_MCP=true             also mounts MCP at /mcp inside the REST app

Each is a separate fixture below, each probing an endpoint that server really
has, so a test skips only when the thing it needs is absent.
"""

import os
import time
import uuid
from pathlib import Path

import pytest
import pytest_asyncio
import requests

# ---------------------------------------------------------------------------
# These tests can spend money. They are opt-in.
# ---------------------------------------------------------------------------
#
# Not obvious from reading them: a plain `search` reaches the agentic pipeline
# whenever query routing picks the comprehensive tier, so tests that look like
# pure retrieval issue LLM calls too. A single run of this directory produced 52
# POSTs to the configured chat-completions endpoint, from tests whose names say
# nothing about generation.
#
# The stress suite already gates itself this way with `-m stress`. This one
# needs it more, because its cost is charged to whoever configured the model.

_LIVE_OPT_IN = "LASAGNA_RUN_LIVE_TESTS"

_HERE = Path(__file__).resolve().parent


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "live: needs a running server and may issue billable LLM calls (opt-in)",
    )


def pytest_collection_modifyitems(config, items):
    """Skip everything *in this directory* unless the caller opted in.

    Checked at collection so the reason is reported once per test rather than
    discovered halfway through a run that is already spending money.

    The path filter is load-bearing and not decoration. `pytest_collection_
    modifyitems` is a session hook: defining it in a subdirectory conftest does
    NOT scope it to that subdirectory, and `items` is every test pytest
    collected anywhere. The first version of this had no filter and skipped the
    entire suite -- 2,617 skipped, 0 run -- while still exiting 0, so it looked
    like a pass.

    (For the same family of reasons there is no `pytestmark` here: pytest
    honours that in test modules and classes, not in conftest. `tests/stress/
    conftest.py` carries a note about the identical mistake.)
    """
    if os.environ.get(_LIVE_OPT_IN, "").strip().lower() in ("1", "true", "yes", "on"):
        return
    skip = pytest.mark.skip(
        reason=(
            f"live tests are opt-in: they need a running server and can issue "
            f"billable LLM calls. Set {_LIVE_OPT_IN}=1 to run them."
        )
    )
    for item in items:
        if _HERE in Path(str(item.fspath)).resolve().parents:
            item.add_marker(pytest.mark.live)
            item.add_marker(skip)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_PORT = int(os.environ.get("LASAGNA_API_PORT", "8000"))
MCP_PORT = int(os.environ.get("RAG_MCP_PORT", "8005"))

API_URL = f"http://127.0.0.1:{API_PORT}"
MCP_URL = f"http://127.0.0.1:{MCP_PORT}"

HEALTH_TIMEOUT = 10  # seconds to wait for service
POLL_INTERVAL = 0.5

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_health(url: str, timeout: float = HEALTH_TIMEOUT) -> bool:
    """Poll a URL until it returns 200 or timeout expires."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200:
                return True
        except (requests.ConnectionError, requests.Timeout):
            pass
        time.sleep(POLL_INTERVAL)
    return False


def _is_service_up(url: str) -> bool:
    """Quick single-shot check."""
    try:
        r = requests.get(url, timeout=3)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def unique_session_id(prefix: str = "live_test") -> str:
    """Generate a unique session ID for test isolation."""
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------------------
# Service availability fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_server():
    """The REST API — what ``lasagna serve`` starts.

    This is the fixture most tests in this directory want. It was called
    ``mcp_server`` and pointed at the MCP server's port, which is why they all
    skipped.
    """
    if not _wait_for_health(f"{API_URL}/api/v1/health"):
        pytest.skip(f"REST API not reachable at {API_URL}. Start it with: lasagna serve")
    return API_URL


@pytest.fixture(scope="session")
def mcp_server():
    """The standalone MCP server, for tests that connect to ``/sse``.

    Not started by ``lasagna serve``. Run it with ``python -m lasagna.mcp``, or
    use `embedded_mcp_url` instead if you would rather have one process.
    """
    if not _wait_for_health(f"{MCP_URL}/health"):
        pytest.skip(
            f"MCP server not reachable at {MCP_URL}. Start it with: "
            f"python -m lasagna.mcp  (or set RAG_EMBED_MCP=true and use "
            f"the embedded_mcp_url fixture)"
        )
    return MCP_URL


@pytest.fixture(scope="session")
def embedded_mcp_url(api_server):
    """MCP mounted inside the REST app at ``/mcp``, under ``RAG_EMBED_MCP``.

    A different URL from `mcp_server`, on a different port, reached only when
    the API was started with that variable set — which is why collapsing the
    two into one fixture could not work.
    """
    url = f"{api_server}/mcp/sse"
    try:
        # SSE holds the connection open, so a 200 never arrives; reaching the
        # handler at all is the signal. A 404 means it was not mounted.
        response = requests.get(url, timeout=3, stream=True)
        response.close()
        if response.status_code == 404:
            raise requests.ConnectionError
    except (requests.ConnectionError, requests.Timeout):
        pytest.skip(f"MCP is not embedded at {url}. Start the API with RAG_EMBED_MCP=true.")
    return url


# ---------------------------------------------------------------------------
# Agent fixtures
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture
async def agent(embedded_mcp_url):
    """Create a live CugaAgent with a unique session."""
    from langchain_core.tools import tool

    @tool
    def calculate(expression: str) -> str:
        """Evaluate a mathematical expression and return the result."""
        try:
            result = eval(expression)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    CugaAgent = pytest.importorskip(
        "cuga.sdk",
        reason="optional 'cuga' agent integration is not installed",
    ).CugaAgent

    session_id = unique_session_id()
    a = CugaAgent(tools=[calculate])
    a.rag.configure_mcp(embedded_mcp_url)
    a.rag._session_id = session_id
    yield a
    # Cleanup: clear all documents uploaded during the test
    try:
        await a.rag.clear(confirm=True)
    except Exception:
        pass
    try:
        await a.rag.disconnect()
    except Exception:
        pass


@pytest_asyncio.fixture
async def agent_with_doc(agent):
    """Agent with a sample product doc already uploaded."""
    doc_path = str(SAMPLE_DOCS_DIR / "product_overview.txt")
    result = await agent.rag.upload_document(doc_path, filename="product_overview.txt")
    yield agent, result


# ---------------------------------------------------------------------------
# Sample document fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def product_doc_path():
    """Path to the sample product overview document."""
    return str(SAMPLE_DOCS_DIR / "product_overview.txt")


@pytest.fixture
def tech_spec_path():
    """Path to the sample technical spec document."""
    return str(SAMPLE_DOCS_DIR / "technical_spec.md")


@pytest.fixture
def tmp_text_doc(tmp_path):
    """Create a temporary text document with unique content."""
    content = f"""Test Document {uuid.uuid4().hex[:8]}

This is a test document for live integration testing.
It contains information about quantum computing and its applications.

Key Topics:
- Quantum entanglement enables secure communication
- Quantum supremacy was demonstrated by Google's Sycamore processor in 2019
- Quantum error correction is essential for practical quantum computers
- The qubit is the fundamental unit of quantum information
"""
    doc = tmp_path / "test_doc.txt"
    doc.write_text(content)
    return str(doc)


# ---------------------------------------------------------------------------
# Session management
# ---------------------------------------------------------------------------


@pytest.fixture
def session_id():
    """Generate a unique session ID."""
    return unique_session_id()
