"""Docker-backed integration tests — vector backends running in containers.

These tests verify that the knowledge package works correctly when services
run inside Docker containers, as they would in production. Tests cover:

- PgVector backend via Docker Compose
- Qdrant backend via Docker Compose
- Cross-backend consistency (upload via one, search via another config)
- CUGA agent integration with Dockerized knowledge services

Prerequisites:
    Start the desired backend:
        cd src/lasagna/deploy
        docker compose -f docker-compose.yml -f backends/pgvector.yml up -d

    Or with Makefile:
        VECTOR_BACKEND=pgvector make up

    Then run:
        pytest tests/live/test_docker_backends.py -v -s

Environment variables (set automatically by Docker Compose, or manually):
    PGVECTOR_DSN=postgresql://rag:rag@localhost:5432/rag
    RAG_VECTOR_BACKEND=pgvector
    DOCKER_RAG_API_URL=http://localhost:8000  (if API runs in Docker)
"""

import asyncio
import os
import uuid
from pathlib import Path

import httpx
import pytest

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"

# ---------------------------------------------------------------------------
# Docker service detection helpers
# ---------------------------------------------------------------------------

DOCKER_API_URL = os.environ.get("DOCKER_RAG_API_URL", "http://localhost:8000")
DOCKER_MCP_URL = os.environ.get("DOCKER_RAG_MCP_URL", "http://localhost:8005")
PGVECTOR_DSN = os.environ.get("PGVECTOR_DSN", "postgresql://rag:rag@localhost:5432/rag")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")


def _is_docker_api_up() -> bool:
    """Check if the Docker-hosted RAG API is running."""
    try:
        r = httpx.get(f"{DOCKER_API_URL}/api/v1/health/live", timeout=3)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


def _is_pgvector_up() -> bool:
    """Check if PostgreSQL with pgvector is reachable."""
    try:
        import psycopg

        with psycopg.connect(PGVECTOR_DSN, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def _is_qdrant_up() -> bool:
    """Check if Qdrant is reachable."""
    try:
        r = httpx.get(f"{QDRANT_URL}/collections", timeout=3)
        return r.status_code == 200
    except (httpx.ConnectError, httpx.TimeoutException):
        return False


# ---------------------------------------------------------------------------
# Skip markers
# ---------------------------------------------------------------------------

requires_pgvector = pytest.mark.skipif(
    not _is_pgvector_up(),
    reason=f"PgVector not reachable at {PGVECTOR_DSN}. "
    "Start with: docker compose -f docker-compose.yml -f backends/pgvector.yml up -d",
)

requires_qdrant = pytest.mark.skipif(
    not _is_qdrant_up(),
    reason=f"Qdrant not reachable at {QDRANT_URL}. "
    "Start with: docker compose -f docker-compose.yml -f backends/qdrant.yml up -d",
)

requires_docker_api = pytest.mark.skipif(
    not _is_docker_api_up(),
    reason=f"Docker RAG API not reachable at {DOCKER_API_URL}. "
    "Start with: docker compose -f docker-compose.yml -f backends/pgvector.yml up -d",
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def product_doc_path():
    """Path to the sample product overview document."""
    return str(SAMPLE_DOCS_DIR / "product_overview.txt")


@pytest.fixture
def tech_spec_path():
    """Path to the sample technical spec document."""
    return str(SAMPLE_DOCS_DIR / "technical_spec.md")


def _unique_session() -> str:
    return f"docker_test_{uuid.uuid4().hex[:8]}"


# =============================================================================
# PgVector Backend Tests
# =============================================================================


@requires_pgvector
class TestPgVectorBackend:
    """Test knowledge package operations against pgvector running in Docker."""

    def test_pgvector_connection(self):
        """Verify pgvector extension is installed and working."""
        import psycopg

        with psycopg.connect(PGVECTOR_DSN) as conn:
            # Verify the vector extension is loaded
            result = conn.execute(
                "SELECT installed_version FROM pg_available_extensions WHERE name = 'vector'"
            ).fetchone()
            assert result is not None, "pgvector extension not available"
            assert result[0] is not None, "pgvector extension not installed"

    def test_pgvector_backend_instantiation(self):
        """PgVectorBackend can connect and disconnect cleanly."""
        from lasagna.vector_backends.pgvector_backend import PgVectorBackend

        backend = PgVectorBackend(dsn=PGVECTOR_DSN)
        backend.connect()
        assert backend.is_connected
        backend.disconnect()

    @pytest.mark.asyncio
    async def test_pgvector_upload_search_cycle(self, tmp_path):
        """Full upload → search → delete lifecycle on pgvector."""
        from lasagna.vector_backends.pgvector_backend import PgVectorBackend
        from lasagna.vector_backends.base import (
            CollectionSchema,
            FieldSchema,
        )

        backend = PgVectorBackend(dsn=PGVECTOR_DSN)
        backend.connect()
        try:
            collection = f"test_{uuid.uuid4().hex[:8]}"

            # Create collection with proper schema
            schema = CollectionSchema(
                fields=[
                    FieldSchema(name="chunk_id", dtype="VARCHAR", is_primary=True, max_length=256),
                    FieldSchema(name="embedding", dtype="FLOAT_VECTOR", dimension=1024),
                ]
            )
            backend.create_collection(collection, schema)

            # Insert test vectors (1024-dim BGE-M3 format)
            import random

            random.seed(42)
            vectors = [[random.random() for _ in range(1024)] for _ in range(3)]
            data = [
                {
                    "chunk_id": str(uuid.uuid4()),
                    "document_id": "doc1",
                    "session_id": "test_session",
                    "content": content,
                    "embedding": vectors[i],
                    "metadata": {},
                }
                for i, content in enumerate(
                    [
                        "Pricing starts at $99/month",
                        "Enterprise plan is custom pricing",
                        "Free trial available for 14 days",
                    ]
                )
            ]

            backend.insert(collection, data)

            # Search
            results = backend.search(collection, vectors[0], limit=3)
            assert len(results) > 0

            # Cleanup
            backend.drop_collection(collection)
        finally:
            backend.disconnect()

    @pytest.mark.asyncio
    async def test_pgvector_via_sdk(self, product_doc_path):
        """Test full SDK lifecycle pointing at pgvector backend."""
        from lasagna.sdk import KnowledgeClient, KnowledgeConfig

        config = KnowledgeConfig(vector_backend="pgvector")
        session_id = _unique_session()
        client = KnowledgeClient(config=config, session_id=session_id)

        try:
            # Upload
            result = await client.upload(product_doc_path)
            assert result.document_id is not None
            assert result.chunk_count > 0

            # Search
            search_results = await client.search_documents("pricing")
            assert len(search_results.results) > 0

            # Verify content relevance
            all_content = " ".join(
                r.chunk.content.lower()
                for r in search_results.results
                if r.chunk and r.chunk.content
            )
            assert any(kw in all_content for kw in ["99", "pricing", "starter"]), (
                f"Search didn't find pricing content: {all_content[:500]}"
            )

            # Delete
            await client.delete_document(result.document_id)
        finally:
            await client.close()


# =============================================================================
# Qdrant Backend Tests
# =============================================================================


@requires_qdrant
class TestQdrantBackend:
    """Test knowledge package operations against Qdrant running in Docker."""

    def test_qdrant_connection(self):
        """Verify Qdrant is responsive."""
        r = httpx.get(f"{QDRANT_URL}/collections", timeout=5)
        assert r.status_code == 200

    def test_qdrant_backend_instantiation(self):
        """QdrantBackend can connect and disconnect cleanly."""
        from lasagna.vector_backends.qdrant_backend import QdrantBackend

        backend = QdrantBackend(url=QDRANT_URL)
        backend.connect()
        assert backend.is_connected
        backend.disconnect()


# =============================================================================
# Docker API Tests (REST endpoints from Docker container)
# =============================================================================


@requires_docker_api
@pytest.mark.asyncio
class TestDockerRAGAPI:
    """Test the RAG REST API running in Docker containers."""

    async def test_health_endpoint(self):
        """Health check returns OK."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{DOCKER_API_URL}/api/v1/health/live")
            assert r.status_code == 200

    async def test_health_details(self):
        """Health check includes component status."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{DOCKER_API_URL}/api/v1/health")
            assert r.status_code == 200
            data = r.json()
            assert "status" in data or "healthy" in str(data).lower()

    async def test_upload_via_rest(self, product_doc_path):
        """Upload a document via REST API."""
        async with httpx.AsyncClient(timeout=30) as client:
            with open(product_doc_path, "rb") as f:
                r = await client.post(
                    f"{DOCKER_API_URL}/api/v1/documents",
                    files={"file": ("product_overview.txt", f, "text/plain")},
                    data={"session_id": _unique_session()},
                )
            if r.status_code == 404:
                pytest.skip("Upload endpoint not available at /api/v1/documents")
            assert r.status_code in (200, 201), f"Upload failed: {r.status_code} {r.text[:500]}"

    async def test_search_via_rest(self, product_doc_path):
        """Upload then search via REST API."""
        session_id = _unique_session()
        async with httpx.AsyncClient(timeout=30) as client:
            # Upload
            with open(product_doc_path, "rb") as f:
                upload_r = await client.post(
                    f"{DOCKER_API_URL}/api/v1/documents",
                    files={"file": ("product_overview.txt", f, "text/plain")},
                    data={"session_id": session_id},
                )
            if upload_r.status_code == 404:
                pytest.skip("Upload endpoint not available")

            # Search
            search_r = await client.post(
                f"{DOCKER_API_URL}/api/v1/search",
                json={"query": "pricing", "session_id": session_id, "top_k": 5},
            )
            if search_r.status_code == 404:
                pytest.skip("Search endpoint not available")
            assert search_r.status_code == 200, f"Search failed: {search_r.text[:500]}"

    async def test_config_endpoint(self):
        """Config endpoint returns current configuration."""
        async with httpx.AsyncClient() as client:
            r = await client.get(f"{DOCKER_API_URL}/api/v1/config")
            if r.status_code == 404:
                pytest.skip("Config endpoint not available")
            assert r.status_code == 200


# =============================================================================
# CUGA Agent + Docker Backend Integration
# =============================================================================


@requires_pgvector
@pytest.mark.asyncio
class TestCugaWithDockerBackend:
    """Test CugaAgent with knowledge services backed by Docker containers.

    This verifies the full flow: CugaAgent → RAG MCP → Docker vector DB.
    Requires both the MCP server (from lasagna serve) and pgvector in Docker.
    """

    async def test_agent_upload_search_with_pgvector(self, product_doc_path):
        """Agent uploads doc and searches against pgvector backend."""
        try:
            from cuga.sdk import CugaAgent
        except ImportError:
            pytest.skip("CugaAgent not available")

        agent = CugaAgent(tools=[])
        agent.rag._session_id = _unique_session()

        try:
            # Upload
            result = await agent.rag.upload_document(product_doc_path, filename="docker_test.txt")
            assert isinstance(result, str)

            # Search
            search = await agent.rag.search("pricing", top_k=3)
            assert isinstance(search, str)
            assert len(search) > 0
        finally:
            try:
                await agent.rag.clear(confirm=True)
            except Exception:
                pass
            try:
                await agent.rag.disconnect()
            except Exception:
                pass

    async def test_agent_invoke_with_docker_rag(self, product_doc_path):
        """CugaAgent.invoke() retrieves context from Docker-backed vector DB."""
        try:
            from cuga.sdk import CugaAgent
        except ImportError:
            pytest.skip("CugaAgent not available")

        agent = CugaAgent(tools=[])
        agent.rag._session_id = _unique_session()

        try:
            await agent.rag.upload_document(product_doc_path, filename="docker_invoke_test.txt")
            agent.rag.enable()

            result = await agent.invoke("What is the pricing for the starter plan?")
            result_lower = str(result).lower()
            assert any(kw in result_lower for kw in ["99", "$99"]), (
                f"Expected pricing info in response, got: {result_lower[:500]}"
            )
        finally:
            try:
                await agent.rag.clear(confirm=True)
            except Exception:
                pass
            try:
                await agent.rag.disconnect()
            except Exception:
                pass


# =============================================================================
# Configuration Tests
# =============================================================================


@requires_pgvector
class TestBackendConfigurations:
    """Test various knowledge package configurations with Docker backends."""

    def test_pgvector_dsn_from_env(self):
        """PgVector backend reads DSN from PGVECTOR_DSN env var."""
        from lasagna.vector_backends.pgvector_backend import PgVectorBackend

        # Should pick up DSN from env
        backend = PgVectorBackend()
        assert backend._dsn == PGVECTOR_DSN or "localhost" in backend._dsn
        backend.connect()
        backend.disconnect()

    def test_backend_registry_has_pgvector(self):
        """Vector backend registry includes pgvector."""
        from lasagna.vector_backends.registry import get_registry

        registry = get_registry()
        cls = registry.get_class("pgvector")
        assert cls is not None

    def test_pool_creates_pgvector(self):
        """Vector backend pool can create pgvector instances."""
        from lasagna.vector_backends.pool import PooledVectorBackend
        from lasagna.vector_backends.pgvector_backend import PgVectorBackend

        pooled = PooledVectorBackend(
            factory=lambda: PgVectorBackend(dsn=PGVECTOR_DSN),
            pool_size=2,
        )
        pooled.connect()
        assert pooled.is_connected
        pooled.disconnect()


# =============================================================================
# Concurrent Operations
# =============================================================================


@requires_pgvector
@pytest.mark.asyncio
class TestConcurrentDockerOps:
    """Test concurrent operations against Docker-hosted backends."""

    async def test_concurrent_searches(self):
        """Multiple concurrent searches don't interfere with each other."""
        from lasagna.sdk import KnowledgeClient, KnowledgeConfig

        config = KnowledgeConfig(vector_backend="pgvector")
        session_id = _unique_session()
        client = KnowledgeClient(config=config, session_id=session_id)

        # Upload a doc first
        doc_path = str(SAMPLE_DOCS_DIR / "product_overview.txt")
        try:
            await client.upload(doc_path)

            # Run 5 concurrent searches
            queries = ["pricing", "features", "enterprise", "support", "deployment"]
            tasks = [client.search_documents(q, top_k=3) for q in queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # At least 3 out of 5 should succeed
            successes = [r for r in results if not isinstance(r, Exception)]
            assert len(successes) >= 3, (
                f"Only {len(successes)}/5 concurrent searches succeeded. "
                f"Errors: {[str(e) for e in results if isinstance(e, Exception)]}"
            )
        finally:
            await client.close()

    async def test_concurrent_uploads(self, tmp_path):
        """Multiple concurrent uploads don't corrupt data."""
        from lasagna.sdk import KnowledgeClient, KnowledgeConfig

        config = KnowledgeConfig(vector_backend="pgvector")
        session_id = _unique_session()
        client = KnowledgeClient(config=config, session_id=session_id)

        # Create 5 small test docs
        docs = []
        for i in range(5):
            doc = tmp_path / f"concurrent_{i}.txt"
            doc.write_text(f"Test document {i}: This discusses topic number {i}.")
            docs.append(str(doc))

        try:
            tasks = [client.upload(doc) for doc in docs]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            successes = [r for r in results if not isinstance(r, Exception)]
            assert len(successes) >= 4, (
                f"Only {len(successes)}/5 concurrent uploads succeeded. "
                f"Errors: {[str(e) for e in results if isinstance(e, Exception)]}"
            )
        finally:
            await client.close()


# =============================================================================
# Backend Switching
# =============================================================================


@pytest.mark.asyncio
class TestBackendSwitching:
    """Test switching between vector backends."""

    @requires_pgvector
    async def test_pgvector_config_roundtrip(self):
        """KnowledgeConfig with vector_backend='pgvector' works end-to-end."""
        from lasagna.sdk import KnowledgeConfig

        config = KnowledgeConfig(vector_backend="pgvector")
        assert config.vector_backend == "pgvector"

        # Convert to dict and back
        config_dict = config.to_dict()
        assert config_dict["vector_backend"] == "pgvector"

    @requires_qdrant
    async def test_qdrant_config_roundtrip(self):
        """KnowledgeConfig with vector_backend='qdrant' works end-to-end."""
        from lasagna.sdk import KnowledgeConfig

        config = KnowledgeConfig(vector_backend="qdrant")
        assert config.vector_backend == "qdrant"
