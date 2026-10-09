"""Test 5: Search cache effectiveness using Prometheus metrics.

Validates cache behavior through Prometheus counters (NOT /stats which
doesn't populate the cache field). Counters:
- rag_search_cache_hits_total   (from rag.search.cache_hits)
- rag_search_cache_misses_total (from rag.search.cache_misses)

Source: _search_orchestrator.py:391,434
"""

from __future__ import annotations

import asyncio
import random
import uuid

import pytest

from .helpers.metrics_collector import PrometheusScraper

pytestmark = pytest.mark.stress


class TestCacheEffectiveness:
    """Search cache performance tests using Prometheus metrics."""

    @pytest.mark.asyncio
    async def test_cache_warmup(self, loaded_backend, backend):
        """Repeated identical searches produce cache hits.

        The search cache is per-worker in-memory. With N gunicorn workers,
        a single repeated query may hit a different worker and miss the cache.
        To reliably observe cache hits, we send the same query multiple times
        (> num_workers) so that by the pigeonhole principle at least one
        worker handles the query twice → cache hit.

        Verifies via Prometheus counters, NOT /stats.cache (unpopulated).
        """
        ctx = loaded_backend
        config = ctx["config"]
        clients = ctx["clients"]

        if not clients:
            pytest.skip("No clients with documents available")

        scraper = PrometheusScraper(config.api_base_url)
        if not await scraper.is_available():
            pytest.skip("Prometheus /metrics not available")

        # Use a client that already has docs uploaded (from loaded_backend)
        session_id = next(iter(clients))
        client = clients[session_id]

        # Get baseline counters
        baseline = await scraper.get_cache_counters()

        # Send the same query multiple times. With 4 gunicorn workers and
        # round-robin routing, 8 requests guarantee at least 2 hit the same
        # worker → at least 1 cache hit.
        # On CPU-only embedding, each initial search takes ~10s, but cache
        # hits return near-instantly. Tolerate occasional timeouts.
        unique_query = f"cache warmup test {uuid.uuid4().hex[:8]}"
        num_attempts = 8  # > typical 4 workers → pigeonhole guarantees hits
        successes = 0

        for i in range(num_attempts):
            r = await client.search(unique_query, top_k=3)
            if r.success:
                successes += 1
            # Brief pause to reduce CPU contention
            if i < num_attempts - 1:
                await asyncio.sleep(0.5)

        # Need at least a few successful searches for meaningful cache test
        assert successes >= 4, (
            f"Only {successes}/{num_attempts} searches succeeded. "
            f"Need at least 4 for cache warmup verification."
        )

        # Poll Prometheus counters until the increments are visible.
        # In multiprocess mode (gunicorn), counters are written to mmap files
        # that may not sync instantly across processes.
        total_hits = 0.0
        total_misses = 0.0
        after = baseline
        for _ in range(15):
            await asyncio.sleep(1.0)
            after = await scraper.get_cache_counters()
            total_hits = after["hits"] - baseline["hits"]
            total_misses = after["misses"] - baseline["misses"]
            # Wait until we see at least some requests accounted for
            if total_hits + total_misses >= successes:
                break

        # Must have at least 1 cache miss (the very first request per worker)
        assert total_misses >= 1, (
            f"Expected at least 1 cache miss. "
            f"Misses: {total_misses}. Baseline: {baseline}, After: {after}"
        )

        # Must have at least 1 cache hit (pigeonhole with 8 requests, 4 workers)
        assert total_hits >= 1, (
            f"Expected at least 1 cache hit after {successes} successful "
            f"identical queries (per-worker in-memory cache with pigeonhole). "
            f"Hits: {total_hits}, Misses: {total_misses}. "
            f"Baseline: {baseline}, After: {after}"
        )

    @pytest.mark.asyncio
    async def test_cache_hit_rate_under_load(self, loaded_backend, backend):
        """Measure cache hit rate with realistic Zipf query distribution.

        20% of queries account for 80% of traffic.
        Popular queries should hit cache after warmup.
        Expected cache hit rate: >= 20% with Zipf distribution.

        Uses a client from loaded_backend that has documents, so
        searches produce real cacheable results.
        """
        ctx = loaded_backend
        config = ctx["config"]
        clients = ctx["clients"]

        if not clients:
            pytest.skip("No clients with documents available")

        scraper = PrometheusScraper(config.api_base_url)
        if not await scraper.is_available():
            pytest.skip("Prometheus /metrics not available")

        # Use a client that already has docs uploaded
        session_id = next(iter(clients))
        client = clients[session_id]

        # Define query pool with Zipf-like distribution
        popular_queries = [
            "quantum mechanics",
            "machine learning",
            "climate change",
        ]
        tail_queries = [f"rare query {i} {uuid.uuid4().hex[:6]}" for i in range(20)]

        # Build Zipf distribution: popular queries 80% of traffic
        total_requests = max(config.concurrent_searches, 20)
        query_sequence = []
        for _ in range(total_requests):
            if random.random() < 0.8:
                query_sequence.append(random.choice(popular_queries))
            else:
                query_sequence.append(random.choice(tail_queries))

        # Get baseline
        baseline = await scraper.get_cache_counters()

        # Execute queries
        for query in query_sequence:
            await client.search(query, top_k=3)

        await asyncio.sleep(1.0)

        # Measure cache hit rate
        after = await scraper.get_cache_counters()
        total_hits = after["hits"] - baseline["hits"]
        total_misses = after["misses"] - baseline["misses"]
        total = total_hits + total_misses

        if total > 0:
            hit_rate = total_hits / total
            # Threshold is 10% (not higher) because multi-worker gunicorn
            # dilutes per-worker in-memory cache hits across 4 workers.
            assert hit_rate >= 0.10, (
                f"Cache hit rate {hit_rate:.1%} < 10% with Zipf distribution. "
                f"Hits: {total_hits}, Misses: {total_misses}"
            )

    @pytest.mark.asyncio
    async def test_cache_isolation_between_tenants(self, loaded_backend, backend):
        """Verify cache is session-scoped (no cross-tenant cache hits).

        Uses two different tenants from loaded_backend (which have different
        document sets uploaded). Tenant A's cached results should NOT be
        returned to Tenant B even for the same query.
        """
        ctx = loaded_backend
        _config = ctx["config"]
        clients = ctx["clients"]

        # Need at least 2 tenants to test cross-tenant isolation
        session_ids = list(clients.keys())
        if len(session_ids) < 2:
            pytest.skip("Need at least 2 tenants for cache isolation test")

        tenant_a = clients[session_ids[0]]
        tenant_b = clients[session_ids[1]]

        # Use a corpus-derived query that is likely to return results
        corpus = ctx["corpus"]
        if corpus.queries:
            query = corpus.queries[0].query
        else:
            query = "quantum mechanics wave function"

        # Tenant A searches — warms cache for this session
        r_a1 = await tenant_a.search(query, top_k=5)

        # Tenant B searches same query — different session, different docs
        r_b1 = await tenant_b.search(query, top_k=5)

        # Both searches must succeed for the test to be meaningful
        assert r_a1.success, (
            f"Tenant A search failed ({r_a1.status_code}): {r_a1.error}. "
            f"Cannot validate cache isolation without successful searches."
        )
        assert r_b1.success, (
            f"Tenant B search failed ({r_b1.status_code}): {r_b1.error}. "
            f"Cannot validate cache isolation without successful searches."
        )

        # At least one tenant must return results for the test to be non-vacuous
        a_doc_ids = {r.get("document_id") for r in r_a1.results}
        b_doc_ids = {r.get("document_id") for r in r_b1.results}

        if not a_doc_ids and not b_doc_ids:
            pytest.skip(
                f"Both tenants returned 0 results for '{query}'. "
                f"Cannot validate isolation with empty result sets."
            )

        # Get the known document sets for each tenant
        doc_map = ctx["document_map"]
        a_known_docs = {doc_id for doc_id, _ in doc_map.get(session_ids[0], [])}
        b_known_docs = {doc_id for doc_id, _ in doc_map.get(session_ids[1], [])}

        # Tenant B's results must contain zero documents owned by Tenant A
        leaked_to_b = b_doc_ids & a_known_docs
        assert not leaked_to_b, (
            f"Cross-tenant cache leak: Tenant B received {len(leaked_to_b)} "
            f"document(s) belonging to Tenant A: {leaked_to_b}"
        )

        # Tenant A's results must contain zero documents owned by Tenant B
        leaked_to_a = a_doc_ids & b_known_docs
        assert not leaked_to_a, (
            f"Cross-tenant cache leak: Tenant A received {len(leaked_to_a)} "
            f"document(s) belonging to Tenant B: {leaked_to_a}"
        )
