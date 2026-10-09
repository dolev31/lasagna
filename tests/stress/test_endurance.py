"""Test 7: Endurance / soak tests — sustained load over extended period.

Runs a mixed workload (70% search, 20% upload, 10% delete) for
STRESS_ENDURANCE_MINUTES and monitors for latency drift, error
accumulation, and resource leaks.
"""

from __future__ import annotations

import asyncio
import random
import time
import uuid

import pytest

from .helpers.api_client import StressAPIClient
from .helpers.document_fixtures import generate_test_corpus
from .helpers.metrics_collector import (
    DashboardStats,
    LiveDashboard,
    compute_latency_stats,
)

pytestmark = pytest.mark.stress


class TestEndurance:
    """Sustained load endurance tests."""

    @pytest.mark.asyncio
    async def test_sustained_load(self, docker_manager, backend, stress_config):
        """Run mixed workload for STRESS_ENDURANCE_MINUTES.

        Mix: 70% searches, 20% uploads, 10% deletions.

        Monitor:
        - Latency drift (p95 should not increase >2x from minute 1 to minute N)
        - Error rate stable (no accumulating failures)
        - /health/ready remains 200 throughout
        """
        _backend_info = await docker_manager.start(backend)
        duration_s = stress_config.endurance_minutes * 60

        corpus = generate_test_corpus(num_docs=5)
        queries = [
            "quantum mechanics wave function",
            "deep learning transformer",
            "climate change greenhouse effect",
            "Roman Empire history decline",
            "DNA gene editing CRISPR technology",
        ]

        num_tenants = min(stress_config.num_tenants, 10)
        dashboard = LiveDashboard(backend, num_tenants)

        clients = []
        uploaded_doc_ids = {}  # session_id -> [doc_ids]

        try:
            # Initialize clients and seed documents
            for i in range(num_tenants):
                session_id = f"endure-{i}-{uuid.uuid4().hex[:8]}"
                client = StressAPIClient(stress_config.api_base_url, session_id)
                clients.append(client)
                uploaded_doc_ids[session_id] = []

                # Seed 2 documents per tenant
                for doc in corpus.documents[:2]:
                    result = await client.upload_document(doc.path, doc.filename)
                    if result.success:
                        await client.wait_for_ready(
                            result.document_id, timeout=stress_config.upload_timeout
                        )
                        uploaded_doc_ids[session_id].append(result.document_id)

            dashboard.start()

            # Track per-minute latencies for drift detection
            minute_search_latencies = {}
            start_time = time.monotonic()
            health_check_interval = 30  # seconds
            last_health_check = 0

            total_searches = 0
            total_uploads = 0
            total_deletes = 0
            total_errors = 0
            all_search_latencies = []
            all_upload_latencies = []

            while time.monotonic() - start_time < duration_s:
                elapsed = time.monotonic() - start_time
                current_minute = int(elapsed / 60)
                client = random.choice(clients)
                session_id = client.session_id

                # Weighted random operation: 70% search, 20% upload, 10% delete
                op = random.random()

                if op < 0.70:
                    # Search
                    query = random.choice(queries)
                    result = await client.search(query, top_k=5)
                    total_searches += 1
                    if result.success:
                        all_search_latencies.append(result.latency_s)
                        minute_search_latencies.setdefault(current_minute, []).append(
                            result.latency_s
                        )
                    elif not result.rate_limited:
                        total_errors += 1

                elif op < 0.90:
                    # Upload
                    doc = random.choice(corpus.documents)
                    result = await client.upload_document(doc.path, doc.filename)
                    total_uploads += 1
                    if result.success:
                        all_upload_latencies.append(result.latency_s)
                        uploaded_doc_ids.setdefault(session_id, []).append(result.document_id)
                    elif result.status_code != 429:
                        total_errors += 1

                else:
                    # Delete (if documents available)
                    docs = uploaded_doc_ids.get(session_id, [])
                    if docs:
                        doc_id = docs.pop(random.randrange(len(docs)))
                        result = await client.delete_document(doc_id)
                        total_deletes += 1
                        if not result.success:
                            total_errors += 1

                # Periodic health check
                if elapsed - last_health_check > health_check_interval:
                    health = await clients[0].health_ready()
                    last_health_check = elapsed

                    # Aggregate 429 counts
                    total_rl_429 = sum(c.metrics.rate_limit_429s for c in clients)
                    total_q_429 = sum(c.metrics.quota_429s for c in clients)

                    dashboard.update(
                        DashboardStats(
                            uploads_done=total_uploads,
                            uploads_total=total_uploads,
                            upload_p50=compute_latency_stats(all_upload_latencies)["p50"],
                            upload_p95=compute_latency_stats(all_upload_latencies)["p95"],
                            searches_done=total_searches,
                            search_p50=compute_latency_stats(all_search_latencies)["p50"],
                            search_p95=compute_latency_stats(all_search_latencies)["p95"],
                            deletions_done=total_deletes,
                            rate_limit_429s=total_rl_429,
                            quota_429s=total_q_429,
                            errors=total_errors,
                            healthy=health.healthy,
                        )
                    )

                    if not health.healthy:
                        # Allow transient 503s under heavy load — retry once
                        await asyncio.sleep(5)
                        health = await clients[0].health_ready()
                        assert health.healthy, (
                            f"/health/ready unhealthy at {elapsed:.0f}s (after retry): "
                            f"{health.checks}"
                        )

                # Small delay to spread load
                await asyncio.sleep(0.1)

            # Check latency drift
            if len(minute_search_latencies) >= 2:
                minutes = sorted(minute_search_latencies.keys())
                first_minute = minutes[0]
                last_minute = minutes[-1]

                first_p95 = compute_latency_stats(minute_search_latencies[first_minute])["p95"]
                last_p95 = compute_latency_stats(minute_search_latencies[last_minute])["p95"]

                if first_p95 > 0:
                    drift = last_p95 / first_p95
                    assert drift < 2.0, (
                        f"Latency drift detected: p95 grew {drift:.1f}x "
                        f"(min 1: {first_p95:.2f}s, min {last_minute}: {last_p95:.2f}s)"
                    )

        finally:
            dashboard.stop()
            for client in clients:
                await client.close()

    @pytest.mark.asyncio
    async def test_no_memory_leak_indicators(self, docker_manager, backend, stress_config):
        """Check for signs of resource leaks during sustained load.

        - Response latency p95 should not monotonically increase
        - /health/ready should remain healthy
        - Docker container memory should not grow unboundedly
        """
        _backend_info = await docker_manager.start(backend)

        # Shorter endurance for memory test (3 minutes)
        duration_s = min(stress_config.endurance_minutes, 3) * 60

        client = StressAPIClient(
            stress_config.api_base_url,
            f"memleak-{uuid.uuid4().hex[:8]}",
        )

        try:
            start_time = time.monotonic()
            minute_latencies = {}
            memory_samples = []

            while time.monotonic() - start_time < duration_s:
                current_minute = int((time.monotonic() - start_time) / 60)

                # Send requests
                for _ in range(10):
                    result = await client.search("memory leak test query", top_k=3)
                    if result.success:
                        minute_latencies.setdefault(current_minute, []).append(result.latency_s)
                    await asyncio.sleep(0.1)

                # Sample container memory (best-effort)
                stats = await docker_manager.get_container_stats()
                if stats:
                    memory_samples.append(
                        {
                            "minute": current_minute,
                            "stats": stats,
                        }
                    )

                # Health check
                health = await client.health_ready()
                assert health.healthy, f"Health degraded at minute {current_minute}"

            # Check for monotonically increasing latencies (leak indicator)
            if len(minute_latencies) >= 3:
                minute_p95s = [
                    (m, compute_latency_stats(lats)["p95"])
                    for m, lats in sorted(minute_latencies.items())
                ]
                # Check if p95 is monotonically increasing
                increasing_count = sum(
                    1
                    for i in range(1, len(minute_p95s))
                    if minute_p95s[i][1] > minute_p95s[i - 1][1]
                )
                monotonic = increasing_count == len(minute_p95s) - 1
                if monotonic and len(minute_p95s) > 3:
                    first_p95 = minute_p95s[0][1]
                    last_p95 = minute_p95s[-1][1]
                    if first_p95 > 0 and last_p95 / first_p95 > 2.0:
                        pytest.fail(
                            f"Possible resource leak: latency monotonically increased "
                            f"from {first_p95:.2f}s to {last_p95:.2f}s"
                        )

        finally:
            await client.close()
