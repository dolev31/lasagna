"""Phase 7: Edge case and stress tests.

Tests boundary conditions, bad inputs, concurrent operations,
and security-related input validation.

Prerequisites:
    lasagna serve
"""

import asyncio

import pytest

from .conftest import unique_session_id


@pytest.mark.asyncio
class TestLargeDocuments:
    """Test handling of large documents."""

    async def test_large_text_document(self, agent, tmp_path):
        """Upload a large (5MB) text file and search it."""
        large_doc = tmp_path / "large_doc.txt"
        with open(large_doc, "w") as f:
            for i in range(50000):
                f.write(
                    f"Line {i}: This is content about topic_{i % 100}. "
                    f"Quantum computing uses qubits for parallel processing.\n"
                )

        result = await agent.rag.upload_document(str(large_doc), filename="large.txt")
        assert isinstance(result, str)
        assert len(result) > 0

        # Search should still work
        search_results = await agent.rag.search("quantum computing qubits", top_k=3)
        assert len(search_results) > 0


@pytest.mark.asyncio
class TestConcurrentOperations:
    """Test concurrent document uploads."""

    async def test_concurrent_uploads(self, embedded_mcp_url, tmp_path):
        """Upload 5 documents concurrently and verify all succeed."""
        CugaAgent = pytest.importorskip(
            "cuga.sdk",
            reason="optional 'cuga' agent integration is not installed",
        ).CugaAgent

        agent = CugaAgent(tools=[])
        # Without this the SDK falls back to its own default MCP URL, and all
        # five uploads failed with ConnectionError while the test reported
        # "Too many failures: 5/5" as though the server had rejected them.
        agent.rag.configure_mcp(embedded_mcp_url)
        agent.rag._session_id = unique_session_id("concurrent")

        num_docs = 5
        paths = []
        for i in range(num_docs):
            doc = tmp_path / f"concurrent_{i}.txt"
            doc.write_text(
                f"Concurrent document number {i}. "
                f"Topic: {'physics' if i % 2 == 0 else 'chemistry'}. "
                f"Content about subject {i}."
            )
            paths.append(str(doc))

        # Upload all concurrently
        tasks = [
            agent.rag.upload_document(p, filename=f"concurrent_{i}.txt")
            for i, p in enumerate(paths)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        successes = [r for r in results if not isinstance(r, Exception)]
        failures = [r for r in results if isinstance(r, Exception)]

        print(f"\nConcurrent uploads: {len(successes)}/{num_docs} succeeded")
        for f in failures:
            print(f"  Failure: {f}")

        # At least 80% should succeed
        assert len(successes) >= num_docs * 0.8, f"Too many failures: {len(failures)}/{num_docs}"

        # Cleanup
        await agent.rag.clear(confirm=True)
        await agent.rag.disconnect()


@pytest.mark.asyncio
class TestBadInputs:
    """Test handling of bad/malicious inputs."""

    async def test_empty_query(self, agent_with_doc):
        """Empty query should be handled gracefully."""
        agent, _ = agent_with_doc
        # Should either raise or return empty results, not crash
        try:
            result = await agent.rag.search("", top_k=3)
            # If it doesn't raise, result should be valid
            assert isinstance(result, str)
        except (ValueError, Exception):
            pass  # Raising is also acceptable

    async def test_xss_injection_query(self, agent_with_doc):
        """XSS attempt in query should be sanitized or return safely."""
        agent, _ = agent_with_doc
        result = await agent.rag.search('<script>alert("xss")</script> pricing plans', top_k=3)
        assert isinstance(result, str)
        # Should NOT contain raw script tags in output
        assert "<script>" not in result

    async def test_sql_injection_query(self, agent_with_doc):
        """SQL injection attempt should be harmless."""
        agent, _ = agent_with_doc
        result = await agent.rag.search("'; DROP TABLE documents; --", top_k=3)
        assert isinstance(result, str)
        # Should return results normally (no crash)

    async def test_very_long_query(self, agent_with_doc):
        """Very long query should be handled (truncated or rejected)."""
        agent, _ = agent_with_doc
        long_query = "test " * 2000  # ~10000 chars
        try:
            result = await agent.rag.search(long_query, top_k=3)
            assert isinstance(result, str)
        except (ValueError, Exception):
            pass  # Rejection is acceptable

    async def test_special_characters_query(self, agent_with_doc):
        """Special characters in query should not break search."""
        agent, _ = agent_with_doc
        special_queries = [
            "price (in USD) for 'enterprise'",
            "features & benefits >> details",
            "cost = $99 | $299 ?? help",
            "query with\nnewlines\tand\ttabs",
        ]
        for q in special_queries:
            result = await agent.rag.search(q, top_k=1)
            assert isinstance(result, str)


@pytest.mark.asyncio
class TestSearchEmptyState:
    """Test search behavior with no documents."""

    async def test_search_empty_session(self, api_server):
        """Search in a session with no documents returns gracefully."""
        CugaAgent = pytest.importorskip(
            "cuga.sdk",
            reason="optional 'cuga' agent integration is not installed",
        ).CugaAgent

        agent = CugaAgent(tools=[])
        agent.rag._session_id = unique_session_id("empty")

        try:
            result = await agent.rag.search("anything", top_k=5)
            assert isinstance(result, str)
            # Should indicate no results rather than error
        finally:
            await agent.rag.disconnect()

    async def test_search_empty_session_default_top_k(self, api_server):
        """search in empty session returns gracefully, without an explicit top_k.

        Renamed: this shared a name with the test above, so it silently
        replaced it and the top_k=5 case never ran.
        """
        CugaAgent = pytest.importorskip(
            "cuga.sdk",
            reason="optional 'cuga' agent integration is not installed",
        ).CugaAgent

        agent = CugaAgent(tools=[])
        agent.rag._session_id = unique_session_id("empty_ctx")

        try:
            result = await agent.rag.search("anything")
            assert isinstance(result, str)
        finally:
            await agent.rag.disconnect()


@pytest.mark.asyncio
class TestUnsupportedFileTypes:
    """Test upload of unsupported file types."""

    async def test_upload_binary_file(self, agent, tmp_path):
        """Uploading an unsupported binary file should fail gracefully."""
        binary_file = tmp_path / "test.xyz"
        binary_file.write_bytes(b"\x00\x01\x02\x03" * 100)

        try:
            result = await agent.rag.upload_document(str(binary_file), filename="test.xyz")
            # Some backends may accept it (best effort), others may reject
            assert isinstance(result, str)
        except (ValueError, Exception) as e:
            # Rejection is expected
            assert "unsupported" in str(e).lower() or "error" in str(e).lower()


@pytest.mark.asyncio
class TestReindex:
    """Test document reindexing."""

    async def test_reindex_document(self, agent, product_doc_path):
        """Reindex preserves document ID and refreshes chunks."""
        import re

        result = await agent.rag.upload_document(product_doc_path, filename="reindex_test.txt")

        doc_ids = re.findall(r"[a-f0-9]{8,}", result)
        if not doc_ids:
            pytest.skip("Could not extract doc_id from upload result")

        doc_id = doc_ids[0]

        # Reindex
        reindex_result = await agent.rag.reindex_document(doc_id)
        assert isinstance(reindex_result, str)
        assert len(reindex_result) > 0

        # Document should still be searchable
        search_results = await agent.rag.search("pricing", top_k=3)
        assert len(search_results) > 0


class TestSessionIDValidation:
    """Test session ID validation edge cases."""

    def test_invalid_session_ids_rejected(self):
        """Invalid session IDs are rejected by RAGManager."""
        CugaAgent = pytest.importorskip(
            "cuga.sdk",
            reason="optional 'cuga' agent integration is not installed",
        ).CugaAgent

        for invalid in ("None", "null", "undefined", ""):
            with pytest.raises(ValueError, match="Invalid session_id"):
                agent = CugaAgent(tools=[])
                agent.rag._session_id = "valid"  # reset first
                agent.rag.session_id = invalid  # should raise

    def test_auto_generated_session_id(self):
        """None session_id auto-generates a unique ID."""
        CugaAgent = pytest.importorskip(
            "cuga.sdk",
            reason="optional 'cuga' agent integration is not installed",
        ).CugaAgent

        agent = CugaAgent(tools=[])
        assert agent.rag.session_id.startswith("sdk_")
        assert len(agent.rag.session_id) > 4
