"""MCP server for RAG Knowledge Base.

Run standalone::

    python -m lasagna.mcp --host 0.0.0.0 --port 8005

Or via stdio for MCP clients that prefer it::

    python -m lasagna.mcp --transport stdio
"""

from .server import main, create_mcp_server

__all__ = ["main", "create_mcp_server"]
