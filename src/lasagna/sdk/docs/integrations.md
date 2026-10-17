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

`lasagna-mcp` serves the knowledge base as MCP tools: search, documents, profiles, configuration, focus and system status. It has 29 tools. By default it serves the `safe` set, 22 tools, which leaves out the seven that delete data, reset configuration or change the server-wide retention policy; `--tools all` (or `LASAGNA_MCP_TOOLS=all`) adds them. The tool list is in the [MCP server README](../../mcp/README.md).

```bash
# Set any supported API key, then run it the way an MCP client launches it (stdio, the default)
export OPENAI_API_KEY=sk-...
lasagna-mcp

# Or serve it on a port, for several clients
lasagna-mcp --transport sse --host 127.0.0.1 --port 8005
```

`python -m lasagna.mcp` runs the same server. Over SSE, an MCP client (Claude Desktop, Cursor, your own agent) connects to `http://localhost:8005/sse`.

### MCPKnowledgeClient (Programmatic Access)

For agents and applications that call the MCP server from Python, `MCPKnowledgeClient` keeps one connection open, reconnects when it drops, and can cache query results (`enable_query_cache=True`). It connects over SSE, so start the server with `--transport sse`:

```python
from lasagna.sdk import MCPKnowledgeClient

async with MCPKnowledgeClient(url="http://localhost:8005/sse") as client:
    # Search: formatted context with inline citations
    response = await client.search("quarterly results", "session-1")

    # Extract structured citations from the response
    # (search appends a <search_metadata> JSON block)
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
| `search()` | Search with the session's configuration; returns formatted context with a `<search_metadata>` block |
| `search_knowledge()`, `search_with_context()` | Deprecated aliases of `search()` |
| `get_document_context()` | Document awareness text for agent prompts |
| `parse_search_metadata()` | Static utility to extract citations/chunks from responses |
| `upload_document()` | Upload and index a document |
| `list_documents()` | List all documents in a session |
| `get_config()` / `patch_config()` | Read/modify RAG configuration |
| `set_focused_documents()` | Scope searches to specific documents |

### Response Format: search

The `search` tool returns LLM-ready formatted context followed by a machine-parseable metadata block:

```text
<knowledge_base>
The following information was retrieved from the user's uploaded documents.
SOURCES: [1] report.pdf (p.3), [2] summary.pdf (p.1)
CONTENT: [Source 1] ... [Source 2] ...
</knowledge_base>
<search_metadata>{"citations": [...], "chunks": [...]}</search_metadata>
```

The `<knowledge_base>` section is for the LLM. The `<search_metadata>` block is for your code: parse it with `MCPKnowledgeClient.parse_search_metadata(response)` to get structured citation and chunk data without a separate call.

### Giving an Agent the Tools

Register `lasagna-mcp` with your MCP client as a stdio server. The tools take a `session_id`, which selects the session whose documents and settings they use. The [MCP server README](../../mcp/README.md) has the client configuration and the full tool list.

---

## See Also

- [User Guide](user_guide.md)
- [Getting Started](getting_started.md)
