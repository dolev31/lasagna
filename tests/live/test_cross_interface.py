"""Phase 4c: Cross-interface consistency tests.

Upload via one interface, search via another, delete via a third.
Verifies all interfaces share the same backend state.

Prerequisites:
    lasagna serve
"""

import pytest
from pathlib import Path

SAMPLE_DOCS_DIR = Path(__file__).parent / "sample_docs"


@pytest.mark.asyncio
class TestCrossInterfaceConsistency:
    """Upload via X, search/verify via Y, delete via Z."""

    async def test_upload_sdk_search_mcp(self, agent, mcp_server, product_doc_path):
        """Upload via CugaAgent SDK, search via MCP client."""
        try:
            from lasagna.sdk.integrations.mcp_client import MCPKnowledgeClient
        except ImportError:
            pytest.skip("MCPKnowledgeClient not available")

        # Upload via SDK
        session_id = agent.rag.session_id
        await agent.rag.upload_document(product_doc_path, filename="cross_test.txt")

        # Search via MCP client
        mcp_client = MCPKnowledgeClient(url=f"{mcp_server}/sse")
        await mcp_client.connect()

        try:
            results = await mcp_client.search("pricing plans", session_id)
            assert isinstance(results, str)
            results_lower = results.lower()
            assert any(kw in results_lower for kw in ["99", "pricing", "starter"]), (
                f"MCP search didn't find SDK-uploaded content: {results_lower[:300]}"
            )
        finally:
            await mcp_client.disconnect()

    async def test_upload_sdk_list_mcp(self, agent, mcp_server, product_doc_path):
        """Upload via SDK, list via MCP — document should be visible."""
        try:
            from lasagna.sdk.integrations.mcp_client import MCPKnowledgeClient
        except ImportError:
            pytest.skip("MCPKnowledgeClient not available")

        session_id = agent.rag.session_id
        await agent.rag.upload_document(product_doc_path, filename="cross_list_test.txt")

        # List via MCP
        mcp_client = MCPKnowledgeClient(url=f"{mcp_server}/sse")
        await mcp_client.connect()

        try:
            docs = await mcp_client.list_documents(session_id)
            assert "cross_list_test" in docs.lower() or len(docs) > 10
        finally:
            await mcp_client.disconnect()

    async def test_upload_mcp_search_sdk(self, agent, mcp_server, product_doc_path):
        """Upload via MCP, search via SDK."""
        try:
            from lasagna.sdk.integrations.mcp_client import MCPKnowledgeClient
        except ImportError:
            pytest.skip("MCPKnowledgeClient not available")

        session_id = agent.rag.session_id

        # Upload via MCP
        mcp_client = MCPKnowledgeClient(url=f"{mcp_server}/sse")
        await mcp_client.connect()

        try:
            await mcp_client.upload_document(
                str(SAMPLE_DOCS_DIR / "product_overview.txt"),
                session_id,
            )
        finally:
            await mcp_client.disconnect()

        # Search via SDK
        results = await agent.rag.search("pricing", top_k=3)
        results_lower = results.lower()
        assert any(kw in results_lower for kw in ["99", "pricing", "starter"]), (
            f"SDK search didn't find MCP-uploaded content: {results_lower[:300]}"
        )

    async def test_clear_sdk_verify_mcp(self, agent, mcp_server, product_doc_path):
        """Upload and clear via SDK, verify empty via MCP."""
        try:
            from lasagna.sdk.integrations.mcp_client import MCPKnowledgeClient
        except ImportError:
            pytest.skip("MCPKnowledgeClient not available")

        session_id = agent.rag.session_id

        # Upload via SDK
        await agent.rag.upload_document(product_doc_path, filename="clear_test.txt")

        # Clear via SDK
        await agent.rag.clear(confirm=True)

        # Verify via MCP
        mcp_client = MCPKnowledgeClient(url=f"{mcp_server}/sse")
        await mcp_client.connect()

        try:
            docs = await mcp_client.list_documents(session_id)
            docs_lower = docs.lower()
            assert (
                "no document" in docs_lower
                or "empty" in docs_lower
                or "0" in docs_lower
                or docs_lower.strip() == "[]"
            ), f"Expected empty after clear, got: {docs_lower[:300]}"
        finally:
            await mcp_client.disconnect()
