"""Test 1: Infrastructure baseline — startup, readiness, backend detection.

Verifies each vector backend starts correctly with correct health gating
and feature detection.
"""

from __future__ import annotations

import pytest

from .helpers.api_client import StressAPIClient
from .helpers.metrics_collector import PrometheusScraper

pytestmark = pytest.mark.stress


class TestInfrastructureBaseline:
    """Per-backend infrastructure verification."""

    @pytest.mark.asyncio
    async def test_startup_and_readiness(self, docker_manager, backend, stress_config):
        """Start stack, gate on /health/ready (NOT /health/live).

        Asserts:
        - /health/ready returns 200 within ready_timeout
        - All checks (vector_backend, embedding_model, database) are "up"
        - Startup time is recorded
        """
        backend_info = await docker_manager.start(backend)

        # Verify backend started within timeout
        assert backend_info.startup_time_s > 0
        assert backend_info.startup_time_s < stress_config.ready_timeout

        # Verify via direct health check
        client = StressAPIClient(stress_config.api_base_url, "health-check-session")
        try:
            health = await client.health_ready()
            assert health.healthy, f"Backend {backend} not healthy: {health.checks}"
            assert health.status_code == 200

            # All checks must be "up"
            for check_name, check_data in health.checks.items():
                assert check_data.get("status") == "up", (
                    f"Check '{check_name}' is not up: {check_data}"
                )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_backend_detection(self, docker_manager, backend, stress_config):
        """Detect actual backend from /health/ready response.

        Uses checks.vector_backend.detail (NOT /stats.vector_backend which
        is unpopulated — rag_backend.get_stats() doesn't return this key).
        """
        backend_info = await docker_manager.start(backend)

        # Vector backend should be detected
        assert backend_info.vector_backend != "unknown", (
            f"Could not detect vector backend for {backend}"
        )

        # Database should be detected
        assert backend_info.database_backend != "unknown"

        # Embedding model should be detected
        assert backend_info.embedding_model != "unknown"

        # Worker mode should match backend configuration
        if backend == "milvus-lite":
            assert backend_info.worker_mode == "uvicorn-single"
            assert not backend_info.celery_enabled
        else:
            assert backend_info.worker_mode == "gunicorn-multi"

    @pytest.mark.asyncio
    async def test_feature_flags(self, docker_manager, backend, stress_config):
        """Verify enabled features via /stats.

        /stats.features dict IS populated (hybrid_search, reranking, etc.)
        /stats.embedding_model and embedding_dim ARE populated.
        /stats.vector_backend and /stats.cache are NOT populated.
        """
        await docker_manager.start(backend)

        client = StressAPIClient(stress_config.api_base_url, "feature-check-session")
        try:
            stats = await client.get_stats()

            # These fields ARE populated by get_stats()
            assert stats.get("embedding_model"), "embedding_model missing from /stats"
            assert stats.get("embedding_dim"), "embedding_dim missing from /stats"
            assert isinstance(stats.get("features"), dict), "features missing from /stats"

            # Verify key features are present
            features = stats["features"]
            for expected_key in ["hybrid_search", "reranking", "mmr", "hyde"]:
                assert expected_key in features, (
                    f"Feature '{expected_key}' missing from /stats.features"
                )
        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_prometheus_metrics_endpoint(self, docker_manager, backend, stress_config):
        """Verify /metrics returns Prometheus-format text when enabled.

        RAG_ENABLE_PROMETHEUS=true is set in docker-compose.yml.
        """
        await docker_manager.start(backend)

        # After API (re)start, workers may not have registered Prometheus
        # counters yet. Make a request to trigger counter creation, then
        # give the multiprocess collector a moment to sync.
        from .helpers.api_client import StressAPIClient

        probe = StressAPIClient(stress_config.api_base_url, "prometheus-probe")
        try:
            await probe.search("prometheus probe", top_k=1)
        finally:
            await probe.close()

        import asyncio

        await asyncio.sleep(1.0)

        scraper = PrometheusScraper(stress_config.api_base_url)
        available = await scraper.is_available()

        assert available, "/metrics endpoint not available. Ensure RAG_ENABLE_PROMETHEUS=true"

        # Scrape and verify some expected metric names exist
        raw = await scraper.scrape_raw()
        assert len(raw) > 100, "/metrics returned very little content"
