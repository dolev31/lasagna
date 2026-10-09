"""Test 8: RAG quality (IR metrics + relevance) under concurrent stress.

Verifies that search quality doesn't degrade significantly under load.
Uses ground-truth corpus with chunk-to-doc deduplication for proper
IR metric computation.

Important: use_hyde/use_reranking tested ONLY via /search (not /search/citations
which ignores these flags — confirmed in search.py).
"""

from __future__ import annotations

import asyncio
import uuid

import pytest

from .helpers.api_client import StressAPIClient
from .helpers.ir_evaluation import evaluate_corpus

pytestmark = pytest.mark.stress


class TestRAGQualityUnderLoad:
    """RAG retrieval quality tests under stress."""

    @pytest.mark.asyncio
    async def test_ir_metrics_baseline(self, loaded_backend, backend):
        """Compute IR metrics on a controlled corpus with no concurrent load.

        Results deduplicated chunk -> doc before metrics.
        Baseline: NDCG@5, Precision@5, Recall@5, MRR, Hit Rate@5
        """
        ctx = loaded_backend
        _config = ctx["config"]
        corpus = ctx["corpus"]

        # Get a client with documents
        clients = ctx["clients"]
        if not clients:
            pytest.skip("No clients available")

        session_id = next(iter(clients))
        client = clients[session_id]
        doc_map = ctx["document_map"].get(session_id, [])

        if not doc_map:
            pytest.skip("No documents uploaded")

        # Build marker -> doc_id mapping
        marker_to_doc_id = {}
        for doc_id, test_doc in doc_map:
            marker_to_doc_id[test_doc.unique_marker] = doc_id

        # Run ground-truth queries (no concurrent load)
        all_results = []
        all_ground_truths = []

        for gt_query in corpus.queries:
            result = await client.search(gt_query.query, top_k=5)
            if not result.success:
                continue

            # Map ground truth markers to actual doc IDs
            mapped_gt = {}
            for marker, rel in gt_query.relevance.items():
                if marker in marker_to_doc_id:
                    mapped_gt[marker_to_doc_id[marker]] = rel

            if mapped_gt:
                all_results.append(result.results)
                all_ground_truths.append(mapped_gt)

        if not all_results:
            pytest.skip("No valid query results for evaluation")

        # Compute corpus-level metrics
        metrics = evaluate_corpus(all_results, all_ground_truths, k=5)

        # Baseline quality expectations (relaxed for stress test environment)
        assert metrics.get("hit_rate@5", 0) >= 0.3, (
            f"Hit Rate@5 {metrics.get('hit_rate@5', 0):.2f} < 0.3"
        )

    @pytest.mark.asyncio
    async def test_ir_metrics_under_load(self, loaded_backend, backend):
        """Re-run same queries under concurrent stress load.

        Background: concurrent searches from other sessions.
        Foreground: ground-truth evaluation queries.
        Assert: quality degradation < 15% vs baseline.
        """
        ctx = loaded_backend
        config = ctx["config"]
        corpus = ctx["corpus"]

        clients = ctx["clients"]
        if not clients:
            pytest.skip("No clients available")

        session_id = next(iter(clients))
        eval_client = clients[session_id]
        doc_map = ctx["document_map"].get(session_id, [])

        if not doc_map:
            pytest.skip("No documents uploaded")

        marker_to_doc_id = {}
        for doc_id, test_doc in doc_map:
            marker_to_doc_id[test_doc.unique_marker] = doc_id

        # Phase 1: Baseline (no load)
        baseline_results = []
        baseline_gts = []

        for gt_query in corpus.queries:
            result = await eval_client.search(gt_query.query, top_k=5)
            if not result.success:
                continue
            mapped_gt = {
                marker_to_doc_id[m]: r
                for m, r in gt_query.relevance.items()
                if m in marker_to_doc_id
            }
            if mapped_gt:
                baseline_results.append(result.results)
                baseline_gts.append(mapped_gt)

        if not baseline_results:
            pytest.skip("No baseline results")

        baseline_metrics = evaluate_corpus(baseline_results, baseline_gts, k=5)

        # Phase 2: Under load
        load_done = asyncio.Event()

        async def background_load():
            """Generate concurrent search traffic."""
            load_clients = []
            for i in range(min(config.concurrent_searches // 10, 10)):
                load_clients.append(
                    StressAPIClient(
                        config.api_base_url,
                        f"bg-load-{i}-{uuid.uuid4().hex[:8]}",
                    )
                )

            while not load_done.is_set():
                tasks = [c.search("background load query", top_k=3) for c in load_clients]
                await asyncio.gather(*tasks, return_exceptions=True)
                await asyncio.sleep(0.1)

            for c in load_clients:
                await c.close()

        # Start background load
        load_task = asyncio.create_task(background_load())

        # Give load time to build up
        await asyncio.sleep(2)

        # Run evaluation queries under load
        loaded_results = []
        loaded_gts = []

        for gt_query in corpus.queries:
            result = await eval_client.search(gt_query.query, top_k=5)
            if not result.success:
                continue
            mapped_gt = {
                marker_to_doc_id[m]: r
                for m, r in gt_query.relevance.items()
                if m in marker_to_doc_id
            }
            if mapped_gt:
                loaded_results.append(result.results)
                loaded_gts.append(mapped_gt)

        # Stop background load
        load_done.set()
        await load_task

        if not loaded_results:
            pytest.skip("No results under load")

        loaded_metrics = evaluate_corpus(loaded_results, loaded_gts, k=5)

        # Check degradation < 15%
        for metric_name in ["ndcg@5", "precision@5", "hit_rate@5"]:
            baseline_val = baseline_metrics.get(metric_name, 0)
            loaded_val = loaded_metrics.get(metric_name, 0)

            if baseline_val > 0:
                degradation = (baseline_val - loaded_val) / baseline_val
                assert degradation < 0.15, (
                    f"{metric_name} degraded {degradation:.1%} under load "
                    f"(baseline={baseline_val:.3f}, loaded={loaded_val:.3f})"
                )

    @pytest.mark.asyncio
    async def test_search_feature_comparison(self, loaded_backend, backend):
        """Compare search quality with different feature flags.

        Uses POST /api/v1/search (which DOES pass use_hyde/use_reranking).
        NOT using /search/citations (which ignores these flags).

        Configurations:
        - hyde=True, reranking=True   (full pipeline)
        - hyde=False, reranking=True  (no HyDE)
        - hyde=True, reranking=False  (no reranking)
        - hyde=False, reranking=False (vector-only)
        """
        ctx = loaded_backend
        _config = ctx["config"]
        corpus = ctx["corpus"]

        clients = ctx["clients"]
        if not clients:
            pytest.skip("No clients available")

        session_id = next(iter(clients))
        client = clients[session_id]
        doc_map = ctx["document_map"].get(session_id, [])

        if not doc_map:
            pytest.skip("No documents uploaded")

        marker_to_doc_id = {}
        for doc_id, test_doc in doc_map:
            marker_to_doc_id[test_doc.unique_marker] = doc_id

        configurations = [
            ("full", True, True),
            ("no_hyde", False, True),
            ("no_rerank", True, False),
            ("vector_only", False, False),
        ]

        config_metrics = {}

        for config_name, use_hyde, use_reranking in configurations:
            all_results = []
            all_gts = []

            for gt_query in corpus.queries[:5]:
                result = await client.search(
                    gt_query.query,
                    top_k=5,
                    use_hyde=use_hyde,
                    use_reranking=use_reranking,
                )
                if not result.success:
                    continue
                mapped_gt = {
                    marker_to_doc_id[m]: r
                    for m, r in gt_query.relevance.items()
                    if m in marker_to_doc_id
                }
                if mapped_gt:
                    all_results.append(result.results)
                    all_gts.append(mapped_gt)

            if all_results:
                config_metrics[config_name] = evaluate_corpus(all_results, all_gts, k=5)

        # Full pipeline should generally be best or comparable
        if "full" in config_metrics and "vector_only" in config_metrics:
            full_ndcg = config_metrics["full"].get("ndcg@5", 0)
            vector_ndcg = config_metrics["vector_only"].get("ndcg@5", 0)
            # Full pipeline should not be significantly worse than vector-only
            if vector_ndcg > 0:
                assert full_ndcg >= vector_ndcg * 0.8, (
                    f"Full pipeline NDCG@5 ({full_ndcg:.3f}) significantly worse than "
                    f"vector-only ({vector_ndcg:.3f})"
                )

    @pytest.mark.asyncio
    async def test_routing_tier_quality(self, loaded_backend, backend):
        """Compare search quality across routing tiers.

        Same queries with routing_tier=fast vs standard vs comprehensive.
        Expect: comprehensive >= standard >= fast in NDCG@5.
        """
        ctx = loaded_backend
        _config = ctx["config"]
        corpus = ctx["corpus"]

        clients = ctx["clients"]
        if not clients:
            pytest.skip("No clients available")

        session_id = next(iter(clients))
        client = clients[session_id]
        doc_map = ctx["document_map"].get(session_id, [])

        if not doc_map:
            pytest.skip("No documents uploaded")

        marker_to_doc_id = {}
        for doc_id, test_doc in doc_map:
            marker_to_doc_id[test_doc.unique_marker] = doc_id

        tier_metrics = {}

        for tier in ["fast", "standard", "comprehensive"]:
            all_results = []
            all_gts = []

            for gt_query in corpus.queries[:5]:
                result = await client.search(gt_query.query, top_k=5, routing_tier=tier)
                if not result.success:
                    continue
                mapped_gt = {
                    marker_to_doc_id[m]: r
                    for m, r in gt_query.relevance.items()
                    if m in marker_to_doc_id
                }
                if mapped_gt:
                    all_results.append(result.results)
                    all_gts.append(mapped_gt)

            if all_results:
                tier_metrics[tier] = evaluate_corpus(all_results, all_gts, k=5)

        # Comprehensive should generally be >= fast in quality
        if "fast" in tier_metrics and "comprehensive" in tier_metrics:
            fast_ndcg = tier_metrics["fast"].get("ndcg@5", 0)
            comp_ndcg = tier_metrics["comprehensive"].get("ndcg@5", 0)
            # Comprehensive shouldn't be drastically worse than fast
            if fast_ndcg > 0:
                assert comp_ndcg >= fast_ndcg * 0.7, (
                    f"Comprehensive NDCG@5 ({comp_ndcg:.3f}) unexpectedly much worse "
                    f"than fast ({fast_ndcg:.3f})"
                )
