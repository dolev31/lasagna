"""Optional adapter for evaluating a full CUGA agent through Lasagna.

Lasagna's evaluator normally measures the RAG pipeline on its own.  This
adapter lets it instead drive a complete `CUGA <https://github.com/cuga-project/cuga-agent>`_
agent, so retrieval quality can be scored end-to-end inside a real agent
loop rather than in isolation.

This module is the **only** place in Lasagna that imports a third-party
agent framework, and the import happens lazily inside each function.  The
rest of the package -- and the evaluator itself -- has zero static or
runtime coupling to it, which the conformance tests assert.

Requires the optional dependency::

    pip install cuga

Every factory here raises ``ImportError`` with an actionable message when
that package is absent.  Use this module as the template for adapting
Lasagna's evaluator to any other agent framework.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from loguru import logger


def create_agent() -> Any:
    """Create a CugaAgent configured for evaluation.

    Returns:
        A ``CugaAgent`` instance with RAG enabled and MCP cache disabled.

    Raises:
        ImportError: If the optional ``cuga`` package is not installed.
    """
    try:
        # Sole import site for the CUGA agent framework; kept lazy so that
        # importing Lasagna never requires it.
        from cuga.sdk import CugaAgent
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "Evaluating a full CUGA agent requires the optional 'cuga' "
            "package. Install it with: pip install cuga"
        ) from exc

    agent = CugaAgent(tools=[])
    agent.rag.configure_mcp(enable_cache=False)
    return agent


async def upload_document(agent: Any, doc_path: Path) -> Optional[str]:
    """Upload a document via the agent's RAG subsystem.

    Returns:
        The document ID (or ``None`` if it could not be parsed from the
        agent's upload result).
    """
    upload_result = await agent.rag.upload_document(str(doc_path))
    logger.info(f"Upload result: {upload_result}")
    document_id = _parse_document_id(upload_result)
    if document_id:
        await agent.rag.set_focused_documents([document_id])
    agent.rag.enable()
    return document_id


async def search(agent: Any, query: str, document_ids: Optional[list[str]] = None) -> Optional[str]:
    """Retrieve RAG context through the agent."""
    return await agent.rag.search(query, document_ids=document_ids)


async def invoke(agent: Any, query: str) -> str:
    """Run the full agent invoke flow and return the answer string."""
    result = await agent.invoke(query)
    if isinstance(result, dict):
        return result.get("final_answer", str(result))
    return str(result)


async def cleanup(agent: Any) -> None:
    """Disconnect the agent's RAG subsystem."""
    try:
        await agent.rag.disconnect()
    except Exception as e:
        logger.warning(f"Cleanup error: {e}")


def _parse_document_id(upload_result: str) -> Optional[str]:
    """Extract document ID from upload result string."""
    for line in upload_result.split("\n"):
        if "document_id" in line.lower():
            parts = line.split(":")
            if len(parts) >= 2:
                return parts[-1].strip()
    return None
