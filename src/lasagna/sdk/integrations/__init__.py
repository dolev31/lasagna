"""Framework Integrations.

Provides integrations with LangChain, LlamaIndex and MCP.

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

    # The other direction: a LangChain or LlamaIndex component as a stage
    name = from_langchain(CohereRerank(top_n=10))  # "langchain.CohereRerank"
    ```
"""

from .inbound import from_langchain, from_llamaindex
from .langchain import KnowledgeRetriever, KnowledgeDocumentLoader
from .mcp import create_mcp_server, MCPKnowledgeServer
from .mcp_client import MCPKnowledgeClient

__all__ = [
    # LangChain
    "KnowledgeRetriever",
    "KnowledgeDocumentLoader",
    # Inbound: LangChain and LlamaIndex components as pipeline stages
    "from_langchain",
    "from_llamaindex",
    # MCP Server
    "create_mcp_server",
    "MCPKnowledgeServer",
    # MCP Client
    "MCPKnowledgeClient",
]
