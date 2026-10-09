"""Framework Integrations.

Provides integrations with LangChain and MCP.

Example:
    ```python
    from lasagna.sdk.integrations import (
        KnowledgeRetriever,
        KnowledgeDocumentLoader,
        create_mcp_server,
    )

    # LangChain retriever
    retriever = KnowledgeRetriever(client)
    docs = await retriever.ainvoke("query")

    # MCP server
    server = create_mcp_server(client)
    ```
"""

from .langchain import KnowledgeRetriever, KnowledgeDocumentLoader
from .mcp import create_mcp_server, MCPKnowledgeServer
from .mcp_client import MCPKnowledgeClient

__all__ = [
    # LangChain
    "KnowledgeRetriever",
    "KnowledgeDocumentLoader",
    # MCP Server
    "create_mcp_server",
    "MCPKnowledgeServer",
    # MCP Client
    "MCPKnowledgeClient",
]
