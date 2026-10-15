"""MCP server for Lasagna.

Run it the way an MCP client launches it (stdio, the default)::

    python -m lasagna.mcp

Or serve it on a port::

    python -m lasagna.mcp --transport sse --host 127.0.0.1 --port 8005
"""

from .server import main, create_mcp_server

__all__ = ["main", "create_mcp_server"]
