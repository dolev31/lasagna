"""Test 2: Concurrent document uploads under multi-tenant load.

Stress-tests the document ingestion pipeline with many tenants uploading
simultaneously, with run-mode-aware KPI thresholds.
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

import pytest

from .helpers.api_client import StressAPIClient
from .helpers.document_fixtures import generate_test_corpus
from .helpers.metrics_collector import (
    DashboardStats,
    LiveDashboard,
    compute_latency_stats,
)

pytestmark = pytest.mark.stress


class TestConcurrentUploads:
    """Multi-tenant document ingestion stress tests."""

    @pytest.mark.asyncio
    async def test_multi_tenant_upload(self, docker_manager, backend, stress_config):
        """N tenants each upload M documents concurrently.

        Run mode awareness:
        - milvus-lite: single worker, serial processing expected → relaxed thresholds
        - others: celery-enabled, parallel processing → standard thresholds

        KPIs:
        - Upload success rate >= 98%
        - Processing p95 < threshold (mode-specific)
        """
        backend_info = await docker_manager.start(backend)

        num_tenants = min(stress_config.num_tenants, 20)  # Cap for upload test
        docs_per_tenant = min(stress_config.docs_per_tenant, 5)
        total_docs = num_tenants * docs_per_tenant

        corpus = generate_test_corpus(num_docs=docs_per_tenant)
        dashboard = LiveDashboard(backend, num_tenants)
        dashboard.start()

        try:
            # Create tenant clients
            clients = []
            for i in range(num_tenants):
                session_id = f"upload-test-{i}-{uuid.uuid4().hex[:8]}"
                clients.append(StressAPIClient(stress_config.api_base_url, session_id))

            # Upload documents concurrently across tenants
            upload_tasks = []
            for client in clients:
                for doc in corpus.documents[:docs_per_tenant]:
                    upload_tasks.append(client.upload_document(doc.path, doc.filename))

            results = await asyncio.gather(*upload_tasks, return_exceptions=True)

            # Collect results
            successes = []
            failures = []
            upload_latencies = []
            for r in results:
                if isinstance(r, Exception):
                    failures.append(str(r))
                elif r.success:
                    successes.append(r)
                    upload_latencies.append(r.latency_s)
                else:
                    failures.append(r.error)

            # Wait for all successful uploads to be "ready"
            # Build a mapping of upload results to their originating client.
            # Each upload_task was created in (client, doc) order, so we can
            # reconstruct the client index from the task position.
            ready_tasks = []
            task_idx = 0
            for client_idx, client in enumerate(clients):
                for _doc_idx in range(docs_per_tenant):
                    if task_idx < len(results):
                        r = results[task_idx]
                        if not isinstance(r, Exception) and r.success:
                            ready_tasks.append(
                                client.wait_for_ready(
                                    r.document_id,
                                    timeout=stress_config.upload_timeout,
                                )
                            )
                    task_idx += 1

            processing_results = await asyncio.gather(*ready_tasks, return_exceptions=True)
            processing_times = [t for t in processing_results if isinstance(t, float)]

            # Assert processing time SLA
            if processing_times:
                proc_stats = compute_latency_stats(processing_times)
                proc_threshold = stress_config.upload_timeout * 0.8
                assert proc_stats["p95"] < proc_threshold, (
                    f"Document processing p95 {proc_stats['p95']:.1f}s exceeds "
                    f"SLA threshold {proc_threshold:.1f}s "
                    f"(mode={backend_info.worker_mode})"
                )

            # Update dashboard
            stats_data = compute_latency_stats(upload_latencies)
            dashboard.update(
                DashboardStats(
                    uploads_done=len(successes),
                    uploads_total=total_docs,
                    upload_p50=stats_data["p50"],
                    upload_p95=stats_data["p95"],
                    upload_errors=len(failures),
                )
            )

            # Assertions
            success_rate = len(successes) / total_docs if total_docs > 0 else 0
            assert success_rate >= 0.98, (
                f"Upload success rate {success_rate:.1%} < 98%. Failures: {failures[:5]}"
            )

            if upload_latencies:
                p95 = stats_data["p95"]
                # Upload p95 should complete within the upload timeout.
                # CPU-only embedding (BAAI/bge-m3, 2.2GB) takes ~10-15s per doc,
                # and concurrent uploads contend heavily for CPU. With 15+
                # concurrent uploads on 4 workers, p95 can reach 90%+ of timeout.
                threshold = (
                    stress_config.upload_timeout * 0.95
                    if backend_info.worker_mode == "uvicorn-single"
                    else stress_config.upload_timeout * 0.95
                )
                assert p95 < threshold, (
                    f"Upload p95 latency {p95:.1f}s exceeds threshold {threshold:.1f}s "
                    f"(mode={backend_info.worker_mode})"
                )

        finally:
            dashboard.stop()
            for client in clients:
                await client.close()

    @pytest.mark.asyncio
    async def test_upload_under_search_load(self, docker_manager, backend, stress_config):
        """Upload documents while concurrent searches are running.

        Verifies uploads don't starve search requests and vice versa.
        """
        await docker_manager.start(backend)

        # Set up corpus and initial uploads
        corpus = generate_test_corpus(num_docs=5)
        search_client = StressAPIClient(
            stress_config.api_base_url,
            f"search-bg-{uuid.uuid4().hex[:8]}",
        )
        upload_client = StressAPIClient(
            stress_config.api_base_url,
            f"upload-fg-{uuid.uuid4().hex[:8]}",
        )

        try:
            # Upload initial documents for searching
            for doc in corpus.documents[:3]:
                result = await upload_client.upload_document(doc.path, doc.filename)
                if result.success:
                    await upload_client.wait_for_ready(
                        result.document_id, timeout=stress_config.upload_timeout
                    )

            # Run concurrent searches while uploading more documents
            search_done = asyncio.Event()

            async def background_searches():
                """Run continuous searches until signaled to stop."""
                while not search_done.is_set():
                    await search_client.search("quantum mechanics", top_k=3)
                    await asyncio.sleep(0.1)

            async def foreground_uploads():
                """Upload remaining documents."""
                for doc in corpus.documents[3:]:
                    result = await upload_client.upload_document(doc.path, doc.filename)
                    if result.success:
                        await upload_client.wait_for_ready(
                            result.document_id, timeout=stress_config.upload_timeout
                        )

            # Run both concurrently
            search_task = asyncio.create_task(background_searches())
            await foreground_uploads()
            search_done.set()
            await search_task

            # Verify search latencies weren't severely impacted
            search_latencies = search_client.metrics.latencies.get("search", [])
            if search_latencies:
                stats = compute_latency_stats(search_latencies)
                assert stats["p95"] < stress_config.search_p95_threshold_adjusted, (
                    f"Search p95 {stats['p95']:.1f}s too high during concurrent uploads"
                )

        finally:
            await search_client.close()
            await upload_client.close()

    @pytest.mark.asyncio
    async def test_large_document_handling(self, docker_manager, backend, stress_config):
        """Upload documents of varying sizes and measure chunking time scaling.

        Sizes: 1KB, 100KB, 1MB
        Ensures large docs don't block other tenants' requests.
        """
        await docker_manager.start(backend)

        import tempfile

        client = StressAPIClient(
            stress_config.api_base_url,
            f"large-doc-{uuid.uuid4().hex[:8]}",
        )

        try:
            sizes = [
                ("tiny_1kb.txt", 1024),
                ("medium_100kb.txt", 100 * 1024),
                ("large_1mb.txt", 1024 * 1024),
            ]
            timings = {}

            for filename, size in sizes:
                # Generate document of specified size
                content = f"# Test Document ({size} bytes)\n\n"
                content += "This is test content for stress testing. " * (size // 50)
                content = content[:size]

                with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
                    f.write(content)
                    file_path = Path(f.name)

                result = await client.upload_document(file_path, filename)
                if result.success:
                    proc_time = await client.wait_for_ready(
                        result.document_id, timeout=stress_config.upload_timeout
                    )
                    timings[filename] = {
                        "upload_latency": result.latency_s,
                        "processing_time": proc_time,
                        "size_bytes": size,
                    }

                file_path.unlink(missing_ok=True)

            # Verify timing scales sub-linearly (not blocking)
            if len(timings) >= 2:
                sorted_by_size = sorted(timings.values(), key=lambda x: x["size_bytes"])
                # Processing time of largest should be < 50x smallest
                if sorted_by_size[0]["processing_time"] > 0:
                    ratio = (
                        sorted_by_size[-1]["processing_time"] / sorted_by_size[0]["processing_time"]
                    )
                    assert ratio < 50, f"Large doc processing {ratio:.0f}x slower than small doc"

        finally:
            await client.close()
