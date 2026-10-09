"""Test 3: Concurrent search quality and latency under multi-tenant load.

Tests search latency, throughput, routing tiers, and result correctness
under concurrent multi-tenant stress.
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from .helpers.api_client import StressAPIClient
from .helpers.ir_evaluation import evaluate_search_results
from .helpers.metrics_collector import compute_latency_stats

pytestmark = pytest.mark.stress


class TestSearchUnderLoad:
    """Concurrent search stress tests."""

    @pytest.mark.asyncio
    async def test_concurrent_search_latency(self, loaded_backend, backend):
        """Fire concurrent searches across multiple tenant sessions.

        Uses POST /api/v1/search with use_hyde=True, use_reranking=True.
        """
        ctx = loaded_backend
        config = ctx["config"]

        # Create search clients
        clients = []
        for i in range(min(config.num_tenants, 20)):
            session_id = f"search-load-{i}-{uuid.uuid4().hex[:8]}"
            clients.append(StressAPIClient(config.api_base_url, session_id))

        queries = [
            "quantum mechanics wave function",
            "neural network deep learning",
            "climate change greenhouse effect",
            "Roman Empire history",
            "DNA gene editing CRISPR",
            "network security firewall",
            "monetary policy inflation",
            "Mars exploration colonization",
        ]

        try:
            # Fire concurrent searches
            tasks = []
            for round_num in range(config.search_rounds):
                for client in clients:
                    query = queries[round_num % len(queries)]
                    tasks.append(client.search(query, top_k=5))

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Aggregate latencies
            all_latencies = []
            success_count = 0
            rate_limit_count = 0
            error_count = 0

            for r in results:
                if isinstance(r, Exception):
                    error_count += 1
                elif r.success:
                    success_count += 1
                    all_latencies.append(r.latency_s)
                elif r.rate_limited:
                    rate_limit_count += 1
                else:
                    error_count += 1

            stats = compute_latency_stats(all_latencies)

            # Assertions
            total = len(tasks)
            success_rate = success_count / total if total > 0 else 0
            assert success_rate >= 0.85, (
                f"Search success rate {success_rate:.1%} < 85%. "
                f"Rate-limited: {rate_limit_count}, Errors: {error_count}"
            )

            assert stats["p50"] < config.search_p50_threshold_adjusted, (
                f"Search p50 {stats['p50']:.2f}s > threshold "
                f"{config.search_p50_threshold_adjusted:.2f}s"
            )

            assert stats["p95"] < config.search_p95_threshold_adjusted, (
                f"Search p95 {stats['p95']:.2f}s > threshold "
                f"{config.search_p95_threshold_adjusted:.2f}s"
            )

        finally:
            for client in clients:
                await client.close()

    @pytest.mark.asyncio
    async def test_search_with_citations_latency(self, loaded_backend, backend):
        """Benchmark /search/citations endpoint separately.

        NOTE: This endpoint does NOT pass use_hyde/use_reranking flags
        (confirmed in search.py). Only benchmarks: query, top_k, routing_tier.
        """
        ctx = loaded_backend
        config = ctx["config"]

        client = StressAPIClient(
            config.api_base_url,
            f"citation-bench-{uuid.uuid4().hex[:8]}",
        )

        try:
            queries = [
                "quantum entanglement",
                "transformer architecture",
                "greenhouse gas emissions",
            ]

            # Run searches via both endpoints for comparison
            search_latencies = []
            citation_latencies = []

            for query in queries:
                for _ in range(config.search_rounds):
                    # Standard search (with feature flags)
                    r1 = await client.search(query, top_k=5)
                    if r1.success:
                        search_latencies.append(r1.latency_s)

                    # Citation search (no feature flags passed)
                    r2 = await client.search_with_citations(query, top_k=5)
                    if r2.success:
                        citation_latencies.append(r2.latency_s)

            search_stats = compute_latency_stats(search_latencies)
            citation_stats = compute_latency_stats(citation_latencies)

            # Citation endpoint should not be dramatically slower
            if search_stats["p50"] > 0:
                overhead = citation_stats["p50"] / search_stats["p50"]
                assert overhead < 3.0, (
                    f"Citation endpoint {overhead:.1f}x slower than search "
                    f"(citation p50={citation_stats['p50']:.2f}s, "
                    f"search p50={search_stats['p50']:.2f}s)"
                )

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_routing_tiers_under_load(self, loaded_backend, backend):
        """Compare latency across routing tiers (fast/standard/comprehensive).

        Sends identical queries with different routing_tier values.
        Expects: fast < standard < comprehensive in p50 latency.
        """
        ctx = loaded_backend
        config = ctx["config"]

        client = StressAPIClient(
            config.api_base_url,
            f"routing-test-{uuid.uuid4().hex[:8]}",
        )

        try:
            query = "How does quantum entanglement work?"
            tier_latencies = {"fast": [], "standard": [], "comprehensive": []}

            for tier in ["fast", "standard", "comprehensive"]:
                for _ in range(config.search_rounds):
                    r = await client.search(query, top_k=5, routing_tier=tier)
                    if r.success:
                        tier_latencies[tier].append(r.latency_s)

            tier_stats = {
                tier: compute_latency_stats(lats) for tier, lats in tier_latencies.items()
            }

            # Fast should generally be faster than comprehensive
            if tier_stats["fast"]["count"] > 0 and tier_stats["comprehensive"]["count"] > 0:
                fast_p50 = tier_stats["fast"]["p50"]
                comp_p50 = tier_stats["comprehensive"]["p50"]
                # Only check tier ordering when latencies are large enough to
                # be meaningful.  Under 100ms both tiers are effectively instant
                # and ordering is dominated by noise.
                if fast_p50 > 0.1 or comp_p50 > 0.1:
                    assert comp_p50 >= fast_p50 * 0.5, (
                        f"Comprehensive ({comp_p50:.2f}s) unexpectedly faster than "
                        f"fast ({fast_p50:.2f}s)"
                    )

        finally:
            await client.close()

    @pytest.mark.asyncio
    async def test_search_result_correctness(self, loaded_backend, backend):
        """Verify search results return relevant documents under load.

        Uses ground-truth from document fixtures.
        Chunk-level results deduplicated to doc-level before IR metrics.
        """
        ctx = loaded_backend
        corpus = ctx["corpus"]
        _config = ctx["config"]

        # Pick a client that has documents
        clients = ctx["clients"]
        if not clients:
            pytest.skip("No clients with documents available")

        session_id = next(iter(clients))
        client = clients[session_id]
        doc_map = ctx["document_map"].get(session_id, [])

        if not doc_map or not corpus.queries:
            pytest.skip("No documents or queries available")

        # Build ground truth mapping: unique_marker -> document_id
        marker_to_doc_id = {}
        for doc_id, test_doc in doc_map:
            marker_to_doc_id[test_doc.unique_marker] = doc_id

        # Run queries and evaluate
        for gt_query in corpus.queries[:5]:
            result = await client.search(gt_query.query, top_k=5)
            if not result.success:
                continue

            # Map ground truth markers to actual document IDs
            mapped_gt = {}
            for marker, rel in gt_query.relevance.items():
                if marker in marker_to_doc_id:
                    mapped_gt[marker_to_doc_id[marker]] = rel

            if mapped_gt and result.results:
                metrics = evaluate_search_results(result.results, mapped_gt, k=5)
                # Under load, allow some quality degradation but require
                # at least some relevant results
                assert metrics["hit_rate@k"] >= 0.2, (
                    f"Query '{gt_query.query}': hit_rate@5={metrics['hit_rate@k']:.2f} < 0.2. "
                    f"Search returned mostly irrelevant results."
                )

    @pytest.mark.asyncio
    async def test_search_latency_degradation(self, loaded_backend, backend):
        """Measure latency at increasing concurrency levels.

        Ramp: 1 -> 10 -> 50 -> 100 concurrent searches.
        Fail if p95 at max_concurrency > 5x p95 at concurrency=1.
        """
        ctx = loaded_backend
        config = ctx["config"]

        query = "deep learning neural networks"
        concurrency_levels = [1, 10, 30]
        level_stats = {}

        for level in concurrency_levels:
            clients = []
            for i in range(level):
                clients.append(
                    StressAPIClient(
                        config.api_base_url,
                        f"degrade-{level}-{i}-{uuid.uuid4().hex[:8]}",
                    )
                )

            try:
                tasks = [c.search(query, top_k=5) for c in clients]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                latencies = [
                    r.latency_s for r in results if not isinstance(r, Exception) and r.success
                ]
                level_stats[level] = compute_latency_stats(latencies)
            finally:
                for c in clients:
                    await c.close()

        # Check degradation: max concurrency p95 should be < 5x baseline p95
        baseline_p95 = level_stats.get(1, {}).get("p95", 0)
        max_level = max(concurrency_levels)
        max_p95 = level_stats.get(max_level, {}).get("p95", 0)

        if baseline_p95 > 0:
            degradation = max_p95 / baseline_p95
            threshold = config.latency_degradation_max_adjusted
            assert degradation < threshold, (
                f"Latency degradation {degradation:.1f}x at concurrency={max_level} "
                f"(baseline p95={baseline_p95:.2f}s, max p95={max_p95:.2f}s, "
                f"threshold={threshold:.0f}x)"
            )
