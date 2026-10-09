"""MCP (Model Context Protocol) Integration.

Provides MCP server creation for exposing knowledge operations.

Example:
    ```python
    from lasagna.sdk.integrations import create_mcp_server

    server = create_mcp_server(client)
    await server.run()
    ```
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..client import KnowledgeClient


class MCPKnowledgeServer:
    """MCP server wrapper for knowledge operations.

    Provides MCP-compatible endpoints for knowledge base operations.
    This is a simplified wrapper - for production use, see the
    full MCP server at lasagna.mcp.server.

    Example:
        ```python
        server = MCPKnowledgeServer(client)
        await server.start(port=8005)
        ```
    """

    def __init__(
        self,
        client: "KnowledgeClient",
        name: str = "knowledge",
    ):
        """Initialize MCP server.

        Args:
            client: KnowledgeClient instance
            name: Server name
        """
        self._client = client
        self._name = name
        self._server = None

    async def start(
        self,
        port: int = 8005,
        host: str = "localhost",
    ) -> None:
        """Start the MCP server.

        Args:
            port: Server port
            host: Server host
        """
        try:
            from mcp.server.fastmcp import FastMCP
        except ImportError:
            raise ImportError(
                "mcp package is required for MCP server. Install with: pip install mcp"
            )

        self._server = FastMCP(self._name, host=host, port=port)

        # Register tools
        @self._server.tool()
        async def search(query: str, top_k: int = 5) -> str:
            """Search the knowledge base."""
            return await self._client.get_context(query, max_chunks=top_k)

        @self._server.tool()
        async def list_documents() -> str:
            """List all documents."""
            docs = await self._client.list_documents()
            lines = [f"{d.filename} (ID: {d.document_id})" for d in docs]
            return "\n".join(lines) if lines else "No documents."

        @self._server.tool()
        async def upload_document(file_path: str) -> str:
            """Upload a document."""
            result = await self._client.upload(file_path)
            return f"Uploaded: {result.document_id} ({result.chunk_count} chunks)"

        @self._server.tool()
        async def delete_document(document_id: str) -> str:
            """Delete a document."""
            success = await self._client.delete_document(document_id)
            return "Deleted" if success else "Not found"

        # Serve over streamable HTTP until cancelled (FastMCP.run is
        # synchronous and takes no host/port; those are server settings).
        await self._server.run_streamable_http_async()

    async def stop(self) -> None:
        """Stop the MCP server."""
        if self._server:
            # Server cleanup
            self._server = None


def create_mcp_server(
    client: "KnowledgeClient",
    name: str = "knowledge",
) -> MCPKnowledgeServer:
    """Create an MCP server for knowledge operations.

    Factory function to create an MCP server instance.

    Args:
        client: KnowledgeClient instance
        name: Server name

    Returns:
        MCPKnowledgeServer instance

    Example:
        ```python
        from lasagna.sdk import KnowledgeClient
        from lasagna.sdk.integrations import create_mcp_server

        client = KnowledgeClient()
        server = create_mcp_server(client)
        await server.start(port=8005)
        ```
    """
    return MCPKnowledgeServer(client, name)
