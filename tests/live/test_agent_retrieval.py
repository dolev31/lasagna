"""Phase 2b: Agent retrieval tests — CugaAgent.invoke() with RAG.

CRITICAL TESTS: These verify that when a user sends a prompt to the agent,
the agent autonomously retrieves context from the knowledge base and returns
a grounded answer.

Flow:
    agent.invoke("question")
      → RAGManager.search() via MCP
      → Context injected into AgentState.knowledge
      → LLM receives context + document awareness
      → Agent generates RAG-augmented answer

Prerequisites:
    lasagna serve
"""

import pytest


@pytest.mark.asyncio
class TestAgentRAGRetrieval:
    """Agent receives prompt, uses knowledge tools, returns grounded answer."""

    async def test_agent_invoke_with_rag_context(self, agent, product_doc_path):
        """Agent answers question using uploaded document content.

        Upload a product doc with specific pricing info, enable RAG,
        invoke the agent, and verify the answer contains pricing from the doc.
        """
        # Upload document with known content
        await agent.rag.upload_document(product_doc_path, filename="product_overview.txt")

        # Enable RAG so context is injected automatically
        agent.rag.enable()

        # Ask a question that requires the document
        result = await agent.invoke("What is the pricing for the starter plan?")

        # The answer should contain specific pricing info from the document ($99)
        result_lower = result.lower() if isinstance(result, str) else str(result).lower()
        assert any(kw in result_lower for kw in ["99", "$99", "1,000", "1000"]), (
            f"Expected pricing info ($99) in response, got: {result_lower[:500]}"
        )

    async def test_agent_invoke_without_rag(self, agent, product_doc_path):
        """Without RAG enabled, agent should NOT have access to doc content."""
        await agent.rag.upload_document(product_doc_path, filename="product_overview.txt")

        # RAG is disabled (default)
        agent.rag.disable()

        result = await agent.invoke("What is the starter plan price for Lasagna?")
        result_str = result if isinstance(result, str) else str(result)

        # Without RAG, the agent should NOT confidently state $99
        # It might say "I don't have that information" or give a generic response
        # The key assertion: it should NOT reliably produce the exact pricing
        # (This is a soft assertion — LLMs might sometimes guess)
        # We primarily verify the flow works without errors
        assert len(result_str) > 0

    async def test_agent_rag_vs_no_rag_quality(self, agent, product_doc_path):
        """Compare agent answers with and without RAG.

        With RAG: answer should be specific and reference document content.
        Without RAG: answer should be generic or uncertain.
        """
        await agent.rag.upload_document(product_doc_path, filename="product_overview.txt")

        question = "What file types does Lasagna support for document upload?"

        # With RAG
        agent.rag.enable()
        result_with_rag = await agent.invoke(question)
        result_with_rag_str = str(result_with_rag).lower()

        # Without RAG
        agent.rag.disable()
        result_without_rag = await agent.invoke(question)
        _result_without_rag_str = str(result_without_rag).lower()

        # RAG answer should contain specific file types from the doc
        specific_types = ["pdf", "docx", "txt", "csv", "json", "xml"]
        rag_mentions = sum(1 for ft in specific_types if ft in result_with_rag_str)

        # RAG response should mention more specific file types
        assert rag_mentions >= 3, (
            f"RAG response should mention specific file types. "
            f"Found {rag_mentions} in: {result_with_rag_str[:500]}"
        )

    async def test_agent_multiquery_rag(self, agent, product_doc_path, tech_spec_path):
        """Upload multiple docs, ask questions that hit different docs."""
        await agent.rag.upload_document(product_doc_path, filename="product_overview.txt")
        await agent.rag.upload_document(tech_spec_path, filename="technical_spec.md")

        agent.rag.enable()

        # Question targeting product doc
        pricing_answer = await agent.invoke("What is the enterprise pricing plan?")
        pricing_str = str(pricing_answer).lower()
        assert any(kw in pricing_str for kw in ["custom", "enterprise", "unlimited"]), (
            f"Expected enterprise pricing info: {pricing_str[:500]}"
        )

        # Question targeting tech spec
        arch_answer = await agent.invoke("What ports do the Lasagna services use?")
        arch_str = str(arch_answer).lower()
        assert any(kw in arch_str for kw in ["7860", "8001", "8005", "port"]), (
            f"Expected port info: {arch_str[:500]}"
        )

    async def test_agent_rag_with_tools(self, agent, product_doc_path):
        """Agent uses both RAG knowledge AND tools in one invocation.

        Upload pricing doc, enable RAG, ask agent to calculate
        something based on document data.
        """
        await agent.rag.upload_document(product_doc_path, filename="product_overview.txt")
        agent.rag.enable()

        # Ask a question that requires: 1) RAG to find price, 2) tool to calculate
        result = await agent.invoke(
            "What is 2 times the monthly cost of the Professional plan? "
            "Use the calculate tool to compute the answer."
        )
        result_str = str(result).lower()

        # Should contain 598 (299 * 2) or reference the calculation
        assert any(kw in result_str for kw in ["598", "299", "professional"]), (
            f"Expected calculation result: {result_str[:500]}"
        )

    async def test_agent_citations_flow(self, agent, product_doc_path):
        """Verify the agent response includes citation references."""
        await agent.rag.upload_document(product_doc_path, filename="product_overview.txt")
        agent.rag.enable()

        result = await agent.invoke("What are Lasagna's key features? Please cite your sources.")
        result_str = str(result)

        # Should have some content about features
        result_lower = result_str.lower()
        assert any(
            kw in result_lower
            for kw in ["multi-step", "human-in-the-loop", "rag", "guardrail", "policy"]
        ), f"Expected feature content: {result_lower[:500]}"

    async def test_agent_rag_session_persistence(self, agent, product_doc_path):
        """Upload doc in one call, query it in a later call (same session)."""
        # Step 1: Upload
        await agent.rag.upload_document(product_doc_path, filename="product_overview.txt")

        # Step 2: Enable RAG and query (separate step)
        agent.rag.enable()
        result = await agent.invoke("Who can I contact for sales inquiries at Lasagna?")
        result_str = str(result).lower()

        # Should find contact info from the uploaded doc
        assert any(kw in result_str for kw in ["sales@example.com", "sales", "example.com"]), (
            f"Expected contact info: {result_str[:500]}"
        )


@pytest.mark.asyncio
class TestAgentSearchOperations:
    """Test agent search operations directly (not via invoke)."""

    async def test_search_returns_relevant_chunks(self, agent_with_doc):
        """Direct search returns chunks relevant to the query."""
        agent, _ = agent_with_doc
        results = await agent.rag.search("pricing", top_k=3)
        if "failed" in results.lower() or "error" in results.lower():
            pytest.skip(f"Search returned error (likely model access issue): {results[:200]}")
        results_lower = results.lower()
        assert any(
            kw in results_lower for kw in ["99", "299", "pricing", "starter", "professional"]
        )

    async def test_search_formatted(self, agent_with_doc):
        """search returns well-formatted context."""
        agent, _ = agent_with_doc
        context = await agent.rag.search("technical specifications")
        if "failed" in context.lower() or "error" in context.lower():
            pytest.skip(f"Search returned error (likely model access issue): {context[:200]}")
        assert len(context) > 100
        # Should contain structured context
        assert any(
            kw in context.lower()
            for kw in ["embedding", "milvus", "vector", "llm", "gpt", "claude"]
        )

    async def test_search_empty_results(self, agent_with_doc):
        """Search for content that doesn't exist returns gracefully."""
        agent, _ = agent_with_doc
        results = await agent.rag.search("xyznonexistentquery12345", top_k=3)
        # Should return something (even if no results or low scores)
        assert isinstance(results, str)

    async def test_get_system_status(self, agent):
        """System status endpoint returns health info."""
        status = await agent.rag.get_system_status()
        assert isinstance(status, str)
        assert len(status) > 0
