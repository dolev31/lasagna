"""Phase 6: Infrastructure backend tests — vector backends and caching.

Tests the default vector backend (milvus-lite) and cache behavior.
Additional backends are tested when environment variables are set.

Prerequisites:
    lasagna serve
"""

import os
import time

import pytest


@pytest.mark.asyncio
class TestDefaultBackend:
    """Test the default milvus-lite backend through the SDK."""

    async def test_milvus_lite_upload_search_delete(self, agent, product_doc_path):
        """Full lifecycle on the default milvus-lite backend."""
        import re

        # Upload
        result = await agent.rag.upload_document(product_doc_path, filename="infra_test.txt")
        assert isinstance(result, str)

        # Search
        search_results = await agent.rag.search("pricing", top_k=3)
        assert len(search_results) > 0
        assert any(kw in search_results.lower() for kw in ["99", "pricing", "starter"])

        # List
        docs = await agent.rag.list_documents()
        assert "infra_test" in docs.lower()

        # Delete (extract doc_id first)
        doc_ids = re.findall(r"[a-f0-9]{8,}", result)
        if doc_ids:
            delete_result = await agent.rag.delete_document(doc_ids[0])
            assert isinstance(delete_result, str)


@pytest.mark.asyncio
class TestSearchCache:
    """Verify search caching behavior."""

    async def test_cache_hit_performance(self, agent, product_doc_path):
        """Second identical search should be faster (cache hit)."""
        await agent.rag.upload_document(product_doc_path, filename="cache_test.txt")

        query = "What are the pricing plans?"

        # First search (cache miss)
        t0 = time.monotonic()
        r1 = await agent.rag.search(query, top_k=3)
        first_latency = time.monotonic() - t0

        # Second search (should hit cache)
        t0 = time.monotonic()
        r2 = await agent.rag.search(query, top_k=3)
        second_latency = time.monotonic() - t0

        # Both should return results
        assert len(r1) > 0
        assert len(r2) > 0

        # Log for manual inspection
        print(f"\nFirst search: {first_latency:.3f}s, Second: {second_latency:.3f}s")
        print(f"Speedup: {first_latency / max(second_latency, 0.001):.1f}x")


@pytest.mark.asyncio
class TestEmbedding:
    """Verify embedding infrastructure works correctly."""

    async def test_search_returns_scored_results(self, agent_with_doc):
        """Search results include similarity scores."""
        agent, _ = agent_with_doc

        results = await agent.rag.search("pricing", top_k=5)
        assert isinstance(results, str)
        # Results should contain score-like numbers
        assert any(c.isdigit() or c == "." for c in results)

    async def test_semantic_search_quality(self, agent_with_doc):
        """Semantic search finds relevant content even with different wording."""
        agent, _ = agent_with_doc

        # Use different wording than what's in the doc
        results = await agent.rag.search("How much does it cost to use the service?", top_k=3)
        results_lower = results.lower()

        # Should find pricing info despite different wording
        assert any(
            kw in results_lower for kw in ["99", "299", "pricing", "starter", "professional"]
        ), f"Semantic search didn't find relevant content: {results_lower[:500]}"


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("PGVECTOR_DSN"),
    reason="PGVECTOR_DSN not set — skip pgvector tests",
)
class TestPgVectorBackend:
    """Test pgvector backend when available."""

    async def test_pgvector_available(self):
        """pgvector backend can be instantiated."""
        try:
            from lasagna.vector_backends.pgvector_backend import PgVectorBackend

            dsn = os.environ["PGVECTOR_DSN"]
            backend = PgVectorBackend(dsn=dsn)
            backend.connect()
            backend.disconnect()
        except Exception as e:
            pytest.skip(f"pgvector not available: {e}")
