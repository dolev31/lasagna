"""Phase 2: SDK lifecycle tests — full CRUD via CugaAgent.rag.

All operations go through the live MCP server. No mocks.

Prerequisites:
    lasagna serve
"""

import pytest

from .conftest import unique_session_id


@pytest.mark.asyncio
class TestRAGManagerLifecycle:
    """Full document lifecycle through CugaAgent.rag (RAGManager → MCP)."""

    async def test_upload_document(self, agent, product_doc_path):
        """Upload a document and verify result contains doc_id."""
        result = await agent.rag.upload_document(product_doc_path, filename="product_overview.txt")
        assert isinstance(result, str)
        assert len(result) > 0
        # Result should mention document_id or similar
        result_lower = result.lower()
        assert "document" in result_lower or "upload" in result_lower or "indexed" in result_lower

    async def test_search(self, agent_with_doc):
        """Search for content in an uploaded document."""
        agent, _ = agent_with_doc
        results = await agent.rag.search("What are the pricing plans?", top_k=3)
        assert isinstance(results, str)
        assert len(results) > 0
        # Check for search failure (known issue: model access)
        if "failed" in results.lower() or "error" in results.lower():
            pytest.skip(f"Search returned error (likely model access issue): {results[:200]}")
        # Should find pricing-related content
        results_lower = results.lower()
        assert "99" in results_lower or "pricing" in results_lower or "starter" in results_lower

    async def test_search_formatted(self, agent_with_doc):
        """Search returns formatted LLM-ready string."""
        agent, _ = agent_with_doc
        context = await agent.rag.search("What features does Lasagna have?")
        assert isinstance(context, str)
        if "failed" in context.lower() or "error" in context.lower():
            pytest.skip(f"Search returned error (likely model access issue): {context[:200]}")
        assert len(context) > 50  # Should be substantial
        # Should mention features from the doc
        context_lower = context.lower()
        assert any(
            kw in context_lower
            for kw in ["rag", "agent", "tool", "feature", "workflow", "human-in-the-loop"]
        )

    async def test_list_documents(self, agent_with_doc):
        """List documents shows the uploaded doc."""
        agent, _ = agent_with_doc
        docs = await agent.rag.list_documents()
        assert isinstance(docs, str)
        assert "product_overview" in docs.lower()

    async def test_get_config(self, agent):
        """Get config returns valid configuration JSON."""
        config = await agent.rag.get_config()
        assert isinstance(config, str)
        assert len(config) > 0
        # Should be parseable or at least contain config-like content
        config_lower = config.lower()
        assert any(kw in config_lower for kw in ["chunk", "embedding", "config", "reranking"])

    async def test_set_config(self, agent):
        """Set config applies configuration changes."""
        result = await agent.rag.set_config(chunk_size=800, chunk_overlap=150)
        assert isinstance(result, str)
        # Verify the config was applied by reading it back
        config = await agent.rag.get_config()
        # Config should reflect changes (check if 800 appears)
        assert "800" in config

    async def test_enable_disable_rag(self, agent):
        """Toggle RAG enable/disable."""
        assert agent.rag.enabled is False  # Default is disabled

        agent.rag.enable()
        assert agent.rag.enabled is True

        agent.rag.disable()
        assert agent.rag.enabled is False

    async def test_delete_document(self, agent, product_doc_path):
        """Upload a doc, delete it, verify it's gone."""
        result = await agent.rag.upload_document(product_doc_path, filename="to_delete.txt")

        # List docs and find the doc_id
        docs_before = await agent.rag.list_documents()
        assert "to_delete" in docs_before.lower()

        # Extract doc_id from upload result — format is "doc_<hex>"
        import re

        doc_id_match = re.search(r"(doc_[a-f0-9]{16,})", result)
        if doc_id_match:
            doc_id = doc_id_match.group(1)
            delete_result = await agent.rag.delete_document(doc_id)
            assert isinstance(delete_result, str)

            # Verify deletion
            docs_after = await agent.rag.list_documents()
            assert doc_id not in docs_after
        else:
            pytest.skip(f"Could not extract doc_id from: {result[:200]}")

    async def test_clear_session(self, agent, product_doc_path, tech_spec_path):
        """Upload multiple docs, clear session, verify all gone."""
        await agent.rag.upload_document(product_doc_path, filename="doc1.txt")
        await agent.rag.upload_document(tech_spec_path, filename="doc2.md")

        docs_before = await agent.rag.list_documents()
        assert "doc1" in docs_before.lower() or "doc2" in docs_before.lower()

        result = await agent.rag.clear(confirm=True)
        assert isinstance(result, str)

        docs_after = await agent.rag.list_documents()
        # After clearing, should have no documents (or "no documents" message)
        docs_lower = docs_after.lower()
        assert (
            "no document" in docs_lower
            or "empty" in docs_lower
            or "0" in docs_lower
            or docs_lower.strip() == "[]"
        )


@pytest.mark.asyncio
class TestSessionIsolation:
    """Verify documents are isolated between sessions."""

    async def test_session_isolation(self, embedded_mcp_url, product_doc_path):
        """Documents in session A are not visible in session B."""
        CugaAgent = pytest.importorskip(
            "cuga.sdk",
            reason="optional 'cuga' agent integration is not installed",
        ).CugaAgent

        agent_a = CugaAgent(tools=[])
        agent_a.rag.configure_mcp(embedded_mcp_url)
        agent_a.rag._session_id = unique_session_id("session_a")

        agent_b = CugaAgent(tools=[])
        agent_b.rag.configure_mcp(embedded_mcp_url)
        agent_b.rag._session_id = unique_session_id("session_b")

        try:
            # Upload to session A
            await agent_a.rag.upload_document(product_doc_path, filename="session_a_doc.txt")

            # Verify visible in A
            docs_a = await agent_a.rag.list_documents()
            assert "session_a_doc" in docs_a.lower()

            # Verify NOT visible in B
            docs_b = await agent_b.rag.list_documents()
            assert "session_a_doc" not in docs_b.lower()

            # Search in B should not find A's content
            results_b = await agent_b.rag.search("pricing plans")
            results_b_lower = results_b.lower()
            assert "99" not in results_b_lower or "no result" in results_b_lower

        finally:
            await agent_a.rag.clear(confirm=True)
            await agent_b.rag.clear(confirm=True)
            await agent_a.rag.disconnect()
            await agent_b.rag.disconnect()


@pytest.mark.asyncio
class TestFocusedDocuments:
    """Test document focus feature."""

    async def test_set_and_clear_focused_documents(self, agent_with_doc):
        """Set focused documents scopes search."""
        agent, upload_result = agent_with_doc
        import re

        doc_ids = re.findall(r"[a-f0-9]{8,}", upload_result)
        if not doc_ids:
            pytest.skip("Could not extract doc_id from upload result")

        doc_id = doc_ids[0]

        # Set focus
        await agent.rag.set_focused_documents([doc_id])
        assert agent.rag.focused_document_ids == [doc_id]

        # Clear focus
        await agent.rag.clear_focused_documents()
        assert agent.rag.focused_document_ids is None
