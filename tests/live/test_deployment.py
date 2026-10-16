"""End-to-end integration tests for the Lasagna Docker deployment.

These tests run against a live Docker Compose stack (API + Celery + Redis +
PostgreSQL + a vector store + MCP). The API is expected at
``LASAGNA_LIVE_BASE_URL`` (default http://localhost:8000/api/v1) and the MCP
server at ``LASAGNA_LIVE_MCP_URL`` (default http://localhost:8005). CI runs them
in the ``live-compose`` job of ``.github/workflows/tests.yml``.

Prerequisites (MiniLM, 384 dimensions, which these tests assume):
    cd src/lasagna/deploy
    RAG_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2 RAG_EMBEDDING_DIM=384 \
      docker compose -f docker-compose.yml -f backends/qdrant.yml \
      --profile celery --profile mcp up -d --wait

Run with:
    LASAGNA_RUN_LIVE_TESTS=1 pytest tests/live/test_deployment.py -v

This module used to live in ``tests/`` alongside the hermetic suite, where it
had two failure modes and no successes:

* In CI nothing listens on :8000, so all 33 tests skipped. The suite reported
  them as passing runs of a deployment that was never deployed.
* On a developer's machine with any server on :8000 -- including a plain
  ``lasagna serve`` -- it attached to THAT server, whose configuration it does
  not control, and uploaded and deleted documents in it. Against a default
  ``lasagna serve`` (bge-m3, 1024-dim) it produced 14 errors reading
  ``EMBEDDING DIMENSION MISMATCH``, because it assumes the 384-dim speed
  profile.

So it is a live test, and it belongs here: gated behind
``LASAGNA_RUN_LIVE_TESTS`` like its neighbours, and refusing to run against a
server whose embedding dimension it would fight with rather than reporting that
as a test failure.
"""

import io
import os
import time
import uuid

import pytest
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = os.environ.get("LASAGNA_LIVE_BASE_URL", "http://localhost:8000/api/v1").rstrip("/")
#: Set explicitly, an unreachable MCP server is a failure rather than a skip.
MCP_URL_ENV = "LASAGNA_LIVE_MCP_URL"
MCP_BASE_URL = os.environ.get(MCP_URL_ENV, "http://localhost:8005").rstrip("/")


def _make_session() -> requests.Session:
    """Return a ``requests.Session`` that auto-retries on 429 (rate-limit)."""
    s = requests.Session()
    retries = Retry(
        total=5,
        backoff_factor=1.0,
        status_forcelist=[429],
        allowed_methods=None,
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# Module-level session reused by all tests
http = _make_session()

# Timeouts (seconds)
HEALTH_TIMEOUT = 10
UPLOAD_TIMEOUT = 60
SEARCH_TIMEOUT = 30
DEFAULT_TIMEOUT = 15

# Polling for async document processing
POLL_INTERVAL = 1.0
POLL_MAX_WAIT = 30

# Sample content with distinct topics for search verification
SAMPLE_TXT_CONTENT = (
    "Quantum entanglement is a phenomenon in quantum mechanics where two "
    "particles become interconnected and the quantum state of one particle "
    "instantly influences the state of the other, regardless of the distance "
    "between them. Albert Einstein famously referred to this as 'spooky action "
    "at a distance'. Entanglement is fundamental to quantum computing and "
    "quantum cryptography. When particles are entangled, measuring the spin "
    "of one particle immediately determines the spin of its partner. This "
    "property enables quantum teleportation and quantum key distribution "
    "protocols like BB84 and E91."
)

SAMPLE_MD_CONTENT = (
    "# Machine Learning Overview\n\n"
    "## Supervised Learning\n\n"
    "Supervised learning algorithms learn from labeled training data. "
    "Common approaches include decision trees, support vector machines, "
    "and neural networks. The goal is to learn a mapping from inputs to "
    "outputs based on example input-output pairs.\n\n"
    "## Unsupervised Learning\n\n"
    "Unsupervised learning finds hidden patterns in data without labels. "
    "Clustering algorithms like K-means and DBSCAN group similar data points. "
    "Dimensionality reduction techniques like PCA and t-SNE simplify complex "
    "datasets while preserving important structure.\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _wait_for_document_ready(document_id: str, headers: dict) -> dict:
    """Poll GET /documents/{id}/status until status is 'ready' or timeout.

    Returns the document as it reads once processed. The upload response is
    written before processing (``status: processing``, ``chunks_count: 0``),
    so a test that checked the chunk count on it failed against any stack
    that processes uploads in the background, which compose always does.
    """
    url = f"{BASE_URL}/documents/{document_id}/status"
    deadline = time.time() + POLL_MAX_WAIT
    while time.time() < deadline:
        resp = http.get(url, headers=headers, timeout=DEFAULT_TIMEOUT)
        if resp.status_code == 200:
            status = resp.json().get("status")
            if status == "ready":
                doc = http.get(
                    f"{BASE_URL}/documents/{document_id}", headers=headers, timeout=DEFAULT_TIMEOUT
                )
                doc.raise_for_status()
                return doc.json()
            if status == "error":
                raise AssertionError(
                    f"Document {document_id} entered error state: "
                    f"{resp.json().get('error_message')}"
                )
        time.sleep(POLL_INTERVAL)
    raise AssertionError(f"Document {document_id} did not reach 'ready' within {POLL_MAX_WAIT}s")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


#: The embedding width this module's fixtures assume. Uploading into a
#: collection built at a different width is refused by the API with a 400, which
#: is correct behaviour on its part and not something these tests should report
#: as their own failure.
EXPECTED_EMBEDDING_DIM = 384


@pytest.fixture(scope="module")
def docker_service_available():
    """Verify a *suitable* deployment is reachable. Skip the module if not.

    Reachability alone is not enough. This used to accept any server on :8000,
    so a plain ``lasagna serve`` -- which defaults to bge-m3 at 1024 dimensions
    -- was treated as the Docker stack, and every upload came back 400 with
    EMBEDDING DIMENSION MISMATCH. Fourteen errors that said nothing about the
    deployment and everything about which server answered the phone.
    """
    try:
        resp = http.get(f"{BASE_URL}/health/live", timeout=HEALTH_TIMEOUT)
        if resp.status_code != 200:
            pytest.skip(f"Docker service returned {resp.status_code} on liveness check")
    except requests.ConnectionError:
        pytest.skip(f"Docker service not reachable at {BASE_URL}")
    except requests.Timeout:
        pytest.skip("Docker service timed out on liveness check")

    # Ask the server what it is before writing documents into it.
    try:
        stats = http.get(f"{BASE_URL}/stats", timeout=HEALTH_TIMEOUT).json()
    except (requests.RequestException, ValueError):
        stats = {}
    dim = stats.get("embedding_dim")
    if dim is not None and dim != EXPECTED_EMBEDDING_DIM:
        pytest.skip(
            f"the server on :8000 embeds at {dim} dimensions and this module "
            f"assumes {EXPECTED_EMBEDDING_DIM} ({stats.get('embedding_model')}). "
            f"Start the deployment stack rather than a default `lasagna serve`: "
            f"cd src/lasagna/deploy && VECTOR_BACKEND=milvus-lite make up"
        )
    return BASE_URL


@pytest.fixture(scope="module")
def session_id():
    """Generate a unique session ID for this test run."""
    return f"test-deploy-{uuid.uuid4().hex[:8]}"


@pytest.fixture(scope="module")
def session_headers(session_id):
    """Return headers dict with X-Session-ID.

    The module's shared session holds the two fixture documents with the
    default settings, and nothing else. A test that changes settings or adds
    documents uses ``own_session_headers``: pytest-randomly shuffles the order,
    and a profile change or a stray document in the shared session changed what
    every later search there returned.
    """
    return {"X-Session-ID": session_id}


@pytest.fixture
def own_session_headers():
    """Headers naming a session of this test's own."""
    return {"X-Session-ID": f"test-deploy-{uuid.uuid4().hex[:8]}"}


@pytest.fixture(scope="module")
def sample_txt_file(tmp_path_factory):
    """Create a .txt file with quantum entanglement content."""
    tmp_dir = tmp_path_factory.mktemp("deployment_test_files")
    path = tmp_dir / "quantum_entanglement.txt"
    path.write_text(SAMPLE_TXT_CONTENT, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def sample_md_file(tmp_path_factory):
    """Create a .md file with machine learning content."""
    tmp_dir = tmp_path_factory.mktemp("deployment_test_files")
    path = tmp_dir / "machine_learning.md"
    path.write_text(SAMPLE_MD_CONTENT, encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def uploaded_txt_doc(docker_service_available, session_headers, sample_txt_file):
    """Upload .txt file, wait for processing, yield response, delete on teardown."""
    with open(sample_txt_file, "rb") as f:
        resp = http.post(
            f"{BASE_URL}/documents",
            files={"file": ("quantum_entanglement.txt", f, "text/plain")},
            headers=session_headers,
            timeout=UPLOAD_TIMEOUT,
        )
    assert resp.status_code == 201, f"Upload failed: {resp.status_code} {resp.text}"
    data = resp.json()
    doc_id = data["document_id"]

    yield {**data, **_wait_for_document_ready(doc_id, session_headers)}

    http.delete(
        f"{BASE_URL}/documents/{doc_id}",
        headers=session_headers,
        timeout=DEFAULT_TIMEOUT,
    )


@pytest.fixture(scope="module")
def uploaded_md_doc(docker_service_available, session_headers, sample_md_file):
    """Upload .md file, wait for processing, yield response, delete on teardown."""
    with open(sample_md_file, "rb") as f:
        resp = http.post(
            f"{BASE_URL}/documents",
            files={"file": ("machine_learning.md", f, "text/markdown")},
            headers=session_headers,
            timeout=UPLOAD_TIMEOUT,
        )
    assert resp.status_code == 201, f"Upload failed: {resp.status_code} {resp.text}"
    data = resp.json()
    doc_id = data["document_id"]

    yield {**data, **_wait_for_document_ready(doc_id, session_headers)}

    http.delete(
        f"{BASE_URL}/documents/{doc_id}",
        headers=session_headers,
        timeout=DEFAULT_TIMEOUT,
    )


# ===========================================================================
# Test Classes
# ===========================================================================


@pytest.mark.usefixtures("docker_service_available")
class TestServiceHealth:
    """Verify health, liveness, and readiness probes."""

    def test_liveness_probe(self):
        resp = http.get(f"{BASE_URL}/health/live", timeout=HEALTH_TIMEOUT)
        assert resp.status_code == 200
        assert resp.json()["status"] == "alive"

    def test_readiness_probe_healthy(self):
        resp = http.get(f"{BASE_URL}/health/ready", timeout=HEALTH_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        for check_name, detail in data["checks"].items():
            assert detail["status"] == "up", f"Health check '{check_name}' is not up: {detail}"

    def test_full_health_check_structure(self):
        resp = http.get(f"{BASE_URL}/health", timeout=HEALTH_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        assert "status" in data
        assert "version" in data
        expected_checks = {"embedding_model", "vector_backend", "database"}
        actual_checks = set(data["checks"].keys())
        assert expected_checks.issubset(actual_checks), (
            f"Missing health checks: {expected_checks - actual_checks}"
        )

    def test_stats_endpoint(self):
        resp = http.get(f"{BASE_URL}/stats", timeout=HEALTH_TIMEOUT)
        # Stats may return 503 if backend dependency is transiently unavailable
        assert resp.status_code in (200, 503)
        if resp.status_code == 200:
            data = resp.json()
            assert "features" in data
            assert isinstance(data["features"], dict)


@pytest.mark.usefixtures("docker_service_available")
class TestConfiguration:
    """Verify config retrieval, profiles, and patching."""

    def test_get_default_config(self, session_headers):
        resp = http.get(f"{BASE_URL}/config", headers=session_headers, timeout=DEFAULT_TIMEOUT)
        assert resp.status_code == 200
        data = resp.json()
        expected_fields = {
            "embedding_model",
            "embedding_dim",
            "chunking_strategy",
            "enable_hybrid_search",
            "enable_reranking",
            "enable_mmr",
            "enable_hyde",
            "enable_hierarchical",
            "default_top_k",
            "rerank_top_k",
            "similarity_threshold",
            "hybrid_alpha",
            "mmr_lambda",
            "use_agentic_rag",
            "enable_prometheus",
            "enable_tracing",
        }
        missing = expected_fields - set(data.keys())
        assert not missing, f"Missing config fields: {missing}"

    def test_list_profiles(self, session_headers):
        resp = http.get(
            f"{BASE_URL}/config/profiles",
            headers=session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 200
        profile_names = {p["name"] for p in resp.json()["profiles"]}
        expected = {"speed", "standard", "balanced", "max_quality"}
        assert expected.issubset(profile_names), f"Missing profiles: {expected - profile_names}"

    def test_apply_speed_profile(self, own_session_headers):
        resp = http.put(
            f"{BASE_URL}/config/profiles/speed",
            headers=own_session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Verify response has valid ConfigResponse structure
        assert "embedding_model" in data
        assert "enable_reranking" in data

    def test_apply_nonexistent_profile_returns_404(self, session_headers):
        resp = http.put(
            f"{BASE_URL}/config/profiles/nonexistent_profile_xyz",
            headers=session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 404

    def test_patch_config(self, own_session_headers):
        resp = http.patch(
            f"{BASE_URL}/config",
            json={"similarity_threshold": 0.42, "default_top_k": 7},
            headers=own_session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        # Verify response has valid ConfigResponse structure
        assert "similarity_threshold" in data
        assert "default_top_k" in data


@pytest.mark.usefixtures("docker_service_available")
class TestDocumentUpload:
    """Test document upload, listing, status, and chunks."""

    def test_upload_txt_returns_201(self, uploaded_txt_doc):
        assert uploaded_txt_doc["document_id"]
        assert uploaded_txt_doc["filename"] == "quantum_entanglement.txt"
        assert uploaded_txt_doc["file_size"] > 0
        assert uploaded_txt_doc["chunks_count"] > 0

    def test_upload_md_returns_201(self, uploaded_md_doc):
        assert uploaded_md_doc["document_id"]
        assert uploaded_md_doc["filename"] == "machine_learning.md"
        assert uploaded_md_doc["chunks_count"] > 0

    def test_upload_empty_file(self, own_session_headers, tmp_path_factory):
        tmp_dir = tmp_path_factory.mktemp("empty_file_test")
        empty_file = tmp_dir / "empty.txt"
        empty_file.write_text("", encoding="utf-8")

        with open(empty_file, "rb") as f:
            resp = http.post(
                f"{BASE_URL}/documents",
                files={"file": ("empty.txt", f, "text/plain")},
                headers=own_session_headers,
                timeout=UPLOAD_TIMEOUT,
            )

        if resp.status_code == 201:
            # Cleanup
            http.delete(
                f"{BASE_URL}/documents/{resp.json()['document_id']}",
                headers=own_session_headers,
                timeout=DEFAULT_TIMEOUT,
            )
        else:
            assert resp.status_code in (400, 422, 500)

    def test_upload_unsupported_extension(self, session_headers):
        resp = http.post(
            f"{BASE_URL}/documents",
            files={"file": ("test.xyz", io.BytesIO(b"unsupported"), "application/octet-stream")},
            headers=session_headers,
            timeout=UPLOAD_TIMEOUT,
        )
        assert resp.status_code in (400, 422, 500), (
            f"Expected error for unsupported type, got {resp.status_code}"
        )

    def test_list_documents_contains_uploads(
        self, uploaded_txt_doc, uploaded_md_doc, session_headers
    ):
        resp = http.get(
            f"{BASE_URL}/documents",
            headers=session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        doc_ids = {d["document_id"] for d in data["documents"]}
        assert uploaded_txt_doc["document_id"] in doc_ids
        assert uploaded_md_doc["document_id"] in doc_ids

    def test_get_document_by_id(self, uploaded_txt_doc, session_headers):
        doc_id = uploaded_txt_doc["document_id"]
        resp = http.get(
            f"{BASE_URL}/documents/{doc_id}",
            headers=session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["document_id"] == doc_id
        assert data["filename"] == "quantum_entanglement.txt"
        assert data["status"] == "ready"

    def test_get_document_status_ready(self, uploaded_txt_doc, session_headers):
        doc_id = uploaded_txt_doc["document_id"]
        resp = http.get(
            f"{BASE_URL}/documents/{doc_id}/status",
            headers=session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"
        assert data["chunks_count"] > 0

    def test_get_document_chunks(self, uploaded_txt_doc, session_headers):
        doc_id = uploaded_txt_doc["document_id"]
        resp = http.get(
            f"{BASE_URL}/documents/{doc_id}/chunks",
            headers=session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 200
        chunks = resp.json()
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        first = chunks[0]
        assert "chunk_id" in first
        assert "content" in first
        assert first["document_id"] == doc_id

    def test_get_nonexistent_document_returns_404(self, session_headers):
        resp = http.get(
            f"{BASE_URL}/documents/doc_0000000000000000",
            headers=session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 404


@pytest.mark.usefixtures("docker_service_available")
class TestSearch:
    """Search tests — require prior document uploads."""

    def test_basic_search_returns_results(self, uploaded_txt_doc, session_headers):
        resp = http.post(
            f"{BASE_URL}/search",
            json={
                "query": "quantum entanglement spooky action",
                "top_k": 5,
                "use_hyde": False,
                "use_reranking": False,
            },
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] > 0
        assert data["query"] == "quantum entanglement spooky action"
        first = data["results"][0]
        assert "chunk" in first
        assert "score" in first
        assert first["chunk"]["content"]

    def test_search_top_k_1(self, uploaded_txt_doc, session_headers):
        resp = http.post(
            f"{BASE_URL}/search",
            json={
                "query": "quantum entanglement",
                "top_k": 1,
                "use_hyde": False,
                "use_reranking": False,
            },
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_results"] == 1
        assert len(data["results"]) == 1

    def test_search_features_disabled(self, uploaded_txt_doc, session_headers):
        resp = http.post(
            f"{BASE_URL}/search",
            json={
                "query": "BB84 protocol quantum key distribution",
                "top_k": 3,
                "use_hyde": False,
                "use_reranking": False,
            },
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        assert resp.status_code == 200
        assert resp.json()["total_results"] >= 1

    def test_search_nonexistent_content_low_relevance(self, uploaded_txt_doc, session_headers):
        resp = http.post(
            f"{BASE_URL}/search",
            json={
                "query": "xylophone manufacturing in medieval Polynesia",
                "top_k": 5,
                "use_hyde": False,
                "use_reranking": False,
                "min_score": 0.95,
            },
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        assert resp.status_code == 200
        # With a very high min_score, irrelevant queries should return few/no results
        assert resp.json()["total_results"] <= 1

    def test_search_with_citations(self, uploaded_txt_doc, session_headers):
        resp = http.post(
            f"{BASE_URL}/search/citations",
            json={
                "query": "quantum teleportation",
                "top_k": 3,
                "use_hyde": False,
                "use_reranking": False,
            },
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "results" in data
        assert "citations" in data
        assert "context" in data
        assert isinstance(data["context"], str)
        if data["total_results"] > 0:
            assert len(data["citations"]) > 0
            cit = data["citations"][0]
            assert "citation_id" in cit
            assert "content" in cit
            assert "score" in cit

    def test_search_filtered_by_document_id(
        self, uploaded_txt_doc, uploaded_md_doc, session_headers
    ):
        """Filter search results to a specific document.

        Uses a query relevant to the filtered document (quantum physics
        content in the txt doc) so the dense vector filter drives results.
        """
        txt_doc_id = uploaded_txt_doc["document_id"]
        resp = http.post(
            f"{BASE_URL}/search",
            json={
                "query": "quantum entanglement superposition",
                "top_k": 5,
                "use_hyde": False,
                "use_reranking": False,
                "document_ids": [txt_doc_id],
            },
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        assert resp.status_code == 200
        results = resp.json()["results"]
        assert len(results) > 0, "Expected at least one result from filtered doc"
        for result in results:
            assert result["document_id"] == txt_doc_id

    def test_search_cross_document_relevance(
        self, uploaded_txt_doc, uploaded_md_doc, session_headers
    ):
        """Quantum query should hit the txt doc; ML query should hit the md doc."""
        resp_q = http.post(
            f"{BASE_URL}/search",
            json={
                "query": "quantum entanglement Einstein",
                "top_k": 3,
                "use_hyde": False,
                "use_reranking": False,
            },
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        resp_ml = http.post(
            f"{BASE_URL}/search",
            json={
                "query": "supervised learning neural networks decision trees",
                "top_k": 3,
                "use_hyde": False,
                "use_reranking": False,
            },
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        assert resp_q.status_code == 200
        assert resp_ml.status_code == 200

        q_data = resp_q.json()
        ml_data = resp_ml.json()

        if q_data["total_results"] > 0:
            assert q_data["results"][0]["document_id"] == uploaded_txt_doc["document_id"]
        if ml_data["total_results"] > 0:
            assert ml_data["results"][0]["document_id"] == uploaded_md_doc["document_id"]


@pytest.mark.usefixtures("docker_service_available")
class TestDocumentLifecycle:
    """Full upload → verify → search → delete → verify-gone lifecycle."""

    def test_full_lifecycle(self, own_session_headers, tmp_path_factory):
        # Create a document with a unique term
        unique_term = f"xyzzy{uuid.uuid4().hex[:6]}"
        content = (
            f"The {unique_term} protocol is a novel approach to distributed "
            f"consensus in blockchain networks. The {unique_term} algorithm "
            f"achieves Byzantine fault tolerance with linear message complexity."
        )
        tmp_dir = tmp_path_factory.mktemp("lifecycle_test")
        doc_file = tmp_dir / "lifecycle_doc.txt"
        doc_file.write_text(content, encoding="utf-8")

        # Upload
        with open(doc_file, "rb") as f:
            resp = http.post(
                f"{BASE_URL}/documents",
                files={"file": ("lifecycle_doc.txt", f, "text/plain")},
                headers=own_session_headers,
                timeout=UPLOAD_TIMEOUT,
            )
        assert resp.status_code == 201
        doc_id = resp.json()["document_id"]

        try:
            _wait_for_document_ready(doc_id, own_session_headers)

            # Verify in list
            list_resp = http.get(
                f"{BASE_URL}/documents",
                headers=own_session_headers,
                timeout=DEFAULT_TIMEOUT,
            )
            assert doc_id in {d["document_id"] for d in list_resp.json()["documents"]}

            # Search finds the unique term
            search_resp = http.post(
                f"{BASE_URL}/search",
                json={
                    "query": f"{unique_term} consensus blockchain",
                    "top_k": 5,
                    "use_hyde": False,
                    "use_reranking": False,
                },
                headers=own_session_headers,
                timeout=SEARCH_TIMEOUT,
            )
            assert search_resp.status_code == 200
            assert search_resp.json()["total_results"] > 0

            # Delete
            del_resp = http.delete(
                f"{BASE_URL}/documents/{doc_id}",
                headers=own_session_headers,
                timeout=DEFAULT_TIMEOUT,
            )
            assert del_resp.status_code == 200
            assert del_resp.json()["success"] is True

            # Verify gone from list
            list_resp2 = http.get(
                f"{BASE_URL}/documents",
                headers=own_session_headers,
                timeout=DEFAULT_TIMEOUT,
            )
            assert doc_id not in {d["document_id"] for d in list_resp2.json()["documents"]}

            # Verify GET returns 404
            get_resp = http.get(
                f"{BASE_URL}/documents/{doc_id}",
                headers=own_session_headers,
                timeout=DEFAULT_TIMEOUT,
            )
            assert get_resp.status_code == 404

            # Search no longer returns results from deleted doc
            search_resp2 = http.post(
                f"{BASE_URL}/search",
                json={
                    "query": f"{unique_term} consensus",
                    "top_k": 5,
                    "use_hyde": False,
                    "use_reranking": False,
                    "min_score": 0.3,
                },
                headers=own_session_headers,
                timeout=SEARCH_TIMEOUT,
            )
            assert search_resp2.status_code == 200
            for result in search_resp2.json()["results"]:
                assert result["document_id"] != doc_id

        finally:
            # Safety cleanup
            http.delete(
                f"{BASE_URL}/documents/{doc_id}",
                headers=own_session_headers,
                timeout=DEFAULT_TIMEOUT,
            )


@pytest.mark.usefixtures("docker_service_available")
class TestMultiSessionIsolation:
    """Documents in one session must be invisible to another."""

    def test_session_isolation(self, tmp_path_factory):
        session_a = f"test-isolation-a-{uuid.uuid4().hex[:8]}"
        session_b = f"test-isolation-b-{uuid.uuid4().hex[:8]}"
        headers_a = {"X-Session-ID": session_a}
        headers_b = {"X-Session-ID": session_b}

        unique_word = f"isolationword{uuid.uuid4().hex[:6]}"
        tmp_dir = tmp_path_factory.mktemp("isolation_test")
        doc_file = tmp_dir / "isolation_doc.txt"
        doc_file.write_text(
            f"The {unique_word} concept describes cross-session data boundaries "
            f"in multi-tenant architectures. The {unique_word} principle ensures "
            f"that no data leaks between tenants.",
            encoding="utf-8",
        )

        # Upload to session A only
        with open(doc_file, "rb") as f:
            resp = http.post(
                f"{BASE_URL}/documents",
                files={"file": ("isolation.txt", f, "text/plain")},
                headers=headers_a,
                timeout=UPLOAD_TIMEOUT,
            )
        assert resp.status_code == 201
        doc_id = resp.json()["document_id"]

        try:
            _wait_for_document_ready(doc_id, headers_a)

            # Session A sees the document
            list_a = http.get(f"{BASE_URL}/documents", headers=headers_a, timeout=DEFAULT_TIMEOUT)
            assert doc_id in {d["document_id"] for d in list_a.json()["documents"]}

            # Session B does NOT see the document
            list_b = http.get(f"{BASE_URL}/documents", headers=headers_b, timeout=DEFAULT_TIMEOUT)
            assert doc_id not in {d["document_id"] for d in list_b.json()["documents"]}, (
                "Document leaked to another session"
            )

            # Session B search does NOT return session A's document
            search_b = http.post(
                f"{BASE_URL}/search",
                json={
                    "query": f"{unique_word} multi-tenant",
                    "top_k": 5,
                    "use_hyde": False,
                    "use_reranking": False,
                },
                headers=headers_b,
                timeout=SEARCH_TIMEOUT,
            )
            assert search_b.status_code == 200
            for result in search_b.json()["results"]:
                assert result["document_id"] != doc_id, (
                    "Search in session B returned document from session A"
                )

        finally:
            http.delete(
                f"{BASE_URL}/documents/{doc_id}",
                headers=headers_a,
                timeout=DEFAULT_TIMEOUT,
            )


@pytest.mark.usefixtures("docker_service_available")
class TestEdgeCases:
    """Edge case and input validation tests."""

    def test_delete_nonexistent_document_returns_404(self, session_headers):
        resp = http.delete(
            f"{BASE_URL}/documents/doc_ffffffffffffffff",
            headers=session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 404

    def test_search_empty_query_returns_422(self, session_headers):
        resp = http.post(
            f"{BASE_URL}/search",
            json={"query": "", "top_k": 5, "use_hyde": False, "use_reranking": False},
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        assert resp.status_code == 422

    def test_search_excessive_top_k_returns_422(self, session_headers):
        resp = http.post(
            f"{BASE_URL}/search",
            json={"query": "test", "top_k": 999, "use_hyde": False, "use_reranking": False},
            headers=session_headers,
            timeout=SEARCH_TIMEOUT,
        )
        assert resp.status_code == 422

    def test_chunks_pagination(self, uploaded_txt_doc, session_headers):
        doc_id = uploaded_txt_doc["document_id"]
        resp = http.get(
            f"{BASE_URL}/documents/{doc_id}/chunks",
            params={"offset": 0, "limit": 1},
            headers=session_headers,
            timeout=DEFAULT_TIMEOUT,
        )
        assert resp.status_code == 200
        assert len(resp.json()) <= 1

    def test_no_session_header_uses_default(self):
        resp = http.get(f"{BASE_URL}/documents", timeout=DEFAULT_TIMEOUT)
        assert resp.status_code == 200


@pytest.mark.usefixtures("docker_service_available")
class TestFullStack:
    """The services that run beside the API: the Celery worker and the MCP server.

    The api stored an upload at a path inside its own container and handed it to
    the worker, which had no such file, so every upload processed by Celery
    ended in "file not found". The shared uploads volume fixes that; this is
    the end-to-end check that it holds.
    """

    def test_an_upload_is_processed_off_the_request_and_becomes_searchable(
        self, own_session_headers, tmp_path_factory
    ):
        path = tmp_path_factory.mktemp("full_stack") / "tidal_locking.txt"
        path.write_text(
            "Tidal locking makes the Moon keep one face toward Earth: its rotation "
            "period equals its orbital period, about 27.3 days.",
            encoding="utf-8",
        )
        with open(path, "rb") as f:
            resp = http.post(
                f"{BASE_URL}/documents",
                files={"file": (path.name, f, "text/plain")},
                headers=own_session_headers,
                timeout=UPLOAD_TIMEOUT,
            )
        assert resp.status_code == 201, resp.text
        doc = resp.json()
        try:
            # Dispatched to a background worker, not processed in the request.
            assert doc["status"] == "processing", doc
            _wait_for_document_ready(doc["document_id"], own_session_headers)
            found = http.post(
                f"{BASE_URL}/search",
                json={
                    "query": "Why does the Moon show one face to Earth?",
                    "top_k": 3,
                    "use_hyde": False,
                    "use_reranking": False,
                },
                headers=own_session_headers,
                timeout=SEARCH_TIMEOUT,
            )
            assert found.status_code == 200, found.text
            ids = {r["document_id"] for r in found.json()["results"]}
            assert doc["document_id"] in ids, found.json()
        finally:
            http.delete(
                f"{BASE_URL}/documents/{doc['document_id']}",
                headers=own_session_headers,
                timeout=DEFAULT_TIMEOUT,
            )

    def test_a_batch_of_twenty_files_is_processed_in_full(
        self, own_session_headers, tmp_path_factory
    ):
        """A batch may hold 20 files; without Celery the pool has 5 threads."""
        folder = tmp_path_factory.mktemp("batch")
        files = []
        for i in range(20):
            path = folder / f"note_{i}.txt"
            path.write_text(f"Field note {i}: the heron count at pond {i} was {i * 3}.")
            files.append(("files", (path.name, path.read_bytes(), "text/plain")))
        resp = http.post(
            f"{BASE_URL}/documents/batch",
            files=files,
            headers=own_session_headers,
            timeout=UPLOAD_TIMEOUT * 4,
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()
        ids = [r["document_id"] for r in body["results"] if r["document_id"]]
        try:
            assert (body["succeeded"], body["failed"]) == (20, 0), body
            for doc_id in ids:
                _wait_for_document_ready(doc_id, own_session_headers)
        finally:
            for doc_id in ids:
                http.delete(
                    f"{BASE_URL}/documents/{doc_id}",
                    headers=own_session_headers,
                    timeout=DEFAULT_TIMEOUT,
                )

    def test_every_api_worker_follows_an_upload_and_a_delete(self, own_session_headers):
        """Each search on a new connection, so the gunicorn workers take turns.

        Each worker keeps its own keyword index. It was filled once per session,
        so a worker that had searched the session before missed a document
        processed elsewhere, and kept a deleted one.
        """
        term = f"quillwort{uuid.uuid4().hex[:6]}"
        text = f"The {term} survey counted aquatic ferns in three alpine lakes."

        def search():
            fresh = _make_session()
            reply = fresh.post(
                f"{BASE_URL}/search",
                json={"query": f"{term} survey", "top_k": 5, "use_hyde": False},
                headers={**own_session_headers, "Connection": "close"},
                timeout=SEARCH_TIMEOUT,
            )
            assert reply.status_code == 200, reply.text
            return {r["document_id"] for r in reply.json()["results"]}

        for _ in range(4):  # each worker reads the session before the upload
            search()
        resp = http.post(
            f"{BASE_URL}/documents",
            files={"file": ("ferns.txt", text.encode(), "text/plain")},
            headers=own_session_headers,
            timeout=UPLOAD_TIMEOUT,
        )
        assert resp.status_code == 201, resp.text
        doc_id = resp.json()["document_id"]
        _wait_for_document_ready(doc_id, own_session_headers)

        assert all(doc_id in search() for _ in range(8))
        deleted = http.delete(
            f"{BASE_URL}/documents/{doc_id}", headers=own_session_headers, timeout=DEFAULT_TIMEOUT
        )
        assert deleted.status_code == 200, deleted.text
        assert not any(doc_id in search() for _ in range(8))

    def test_the_mcp_server_is_healthy(self):
        try:
            resp = http.get(f"{MCP_BASE_URL}/health", timeout=HEALTH_TIMEOUT)
        except requests.ConnectionError:
            if os.environ.get(MCP_URL_ENV):
                raise
            pytest.skip(f"no MCP server at {MCP_BASE_URL}; set {MCP_URL_ENV} to require one")
        assert resp.status_code == 200, resp.text
        assert resp.json()["status"] == "healthy"
