# Integrations

## LangChain Integration

### KnowledgeRetriever

```python
from lasagna.sdk import KnowledgeClient
from lasagna.sdk.integrations import KnowledgeRetriever

# Create retriever
client = KnowledgeClient()
retriever = KnowledgeRetriever(
    client,
    top_k=5,
    min_score=0.3,
)

# Use directly
docs = await retriever.ainvoke("What is the revenue?")
for doc in docs:
    print(doc.page_content[:100])
    print(f"  Source: {doc.metadata['filename']}")
```

### In RAG Chain

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

# Setup
llm = ChatOpenAI(model="gpt-6-luna")
prompt = ChatPromptTemplate.from_template("""
Answer based on the context below:

{context}

Question: {question}
""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Create chain
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
)

# Run
answer = await chain.ainvoke("What were the Q3 results?")
print(answer.content)
```

### KnowledgeDocumentLoader

```python
from lasagna.sdk.integrations import KnowledgeDocumentLoader

# Load all chunks
loader = KnowledgeDocumentLoader(client)
docs = await loader.aload()

# Load specific documents
loader = KnowledgeDocumentLoader(
    client,
    document_ids=["doc_123"],
    include_chunks=True,
)
docs = await loader.aload()
```

---

## MCP Integration

### Standalone MCP Server (recommended)

The full-featured MCP server runs standalone with 23 tools covering search, documents, profiles, configuration, focus, and system management:

```bash
# Set any supported API key and run
export OPENAI_API_KEY=sk-...
python -m lasagna.mcp --transport sse --port 8005

# Or stdio, the default, for a client that launches the server itself
python -m lasagna.mcp
```

Any MCP client (Claude Desktop, Cursor, custom agents) connects to `http://localhost:8005/sse`.

### MCPKnowledgeClient (Programmatic Access)

For agents and applications that need programmatic access to the MCP server, use `MCPKnowledgeClient` — the SDK's persistent-connection client with auto-reconnection and optional query caching:

```python
from lasagna.sdk import MCPKnowledgeClient

async with MCPKnowledgeClient(url="http://localhost:8005/sse") as client:
    # Search with inline citations
    response = await client.search_with_context("quarterly results", "session-1")

    # Extract structured citations from the response
    # (search_with_context appends a <search_metadata> JSON block)
    metadata = MCPKnowledgeClient.parse_search_metadata(response)
    citations = metadata.get("citations", [])
    chunks = metadata.get("chunks", [])

    # Get document awareness context for agent prompts
    # Adapts guidance based on how many searches the agent has done this turn
    doc_context = await client.get_document_context("session-1", rag_search_count=0)

    # All other tools are also available
    await client.upload_document("/path/to/file.pdf", "session-1")
    docs = await client.list_documents("session-1")
    status = await client.get_system_status()
```

Key methods:

| Method | Description |
| ------ | ----------- |
| `search_knowledge()` | SOTA semantic search (plain text results) |
| `search_with_context()` | Search + formatted context with `<search_metadata>` citations |
| `get_document_context()` | Adaptive document awareness for agent prompts |
| `parse_search_metadata()` | Static utility to extract citations/chunks from responses |
| `upload_document()` | Upload and index a document |
| `list_documents()` | List all documents in a session |
| `get_config()` / `patch_config()` | Read/modify RAG configuration |
| `set_focused_documents()` | Scope searches to specific documents |

### Response Format: search_with_context

The `search_with_context` tool returns LLM-ready formatted context followed by a machine-parseable metadata block:

```text
<knowledge_base>
The following information was retrieved from the user's uploaded documents.
SOURCES: [1] report.pdf (p.3), [2] summary.pdf (p.1)
CONTENT: [Source 1] ... [Source 2] ...
</knowledge_base>
<search_metadata>{"citations": [...], "chunks": [...]}</search_metadata>
```

The `<knowledge_base>` section is designed for LLM consumption. The `<search_metadata>` block is for programmatic use — parse it with `MCPKnowledgeClient.parse_search_metadata(response)` to get structured citation and chunk data without a separate API call.

### Agent Integration

When running inside a host application, the MCP server module and transport are configurable via `settings.toml`:

```toml
[knowledge]
mcp_module = "lasagna.mcp"
mcp_transport = "sse"
mcp_host = "127.0.0.1"

[server_ports]
rag_knowledge_mcp = 8005
```

Lasagna's tool registry automatically prefixes MCP tools with `knowledge_` (e.g., `knowledge_search_with_context`, `knowledge_get_document_context`) and auto-injects `session_id` for seamless agent integration.

---

## See Also

- [User Guide](user_guide.md)
- [Getting Started](getting_started.md)
