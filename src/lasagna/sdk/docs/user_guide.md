# User Guide

## Table of Contents

1. [KnowledgeClient API](#knowledgeclient-api)
2. [Document Management](#document-management)
3. [Session Management](#session-management)
4. [Search Operations](#search-operations)
5. [Error Handling](#error-handling)
6. [API Reference](#api-reference)

---

## KnowledgeClient API

### Creating a Client

```python
from lasagna.sdk import ChunkingStrategy, KnowledgeClient, KnowledgeConfig

# Default configuration
client = KnowledgeClient()

# Custom configuration
config = KnowledgeConfig(
    enable_reranking=True,
    enable_hyde=True,
    chunking_strategy=ChunkingStrategy.SEMANTIC,
    chunk_size=800,
)
client = KnowledgeClient(config=config)

# With session ID (for multi-tenancy)
client = KnowledgeClient(session_id="user_123")

# Using context manager
async with KnowledgeClient() as client:
    await client.upload("./doc.pdf")
    results = await client.search_documents("query")
```

### Remote Client (`from_url`)

Connect to a Lasagna REST server (`lasagna serve`) over HTTP. The remote client sends every operation to the server with `httpx` and loads no models in your process:

```python
from lasagna.sdk import KnowledgeClient

# Connect to remote server
client = KnowledgeClient.from_url(
    "https://rag.myapp.com",
    api_key="...",              # A key from the server's RAG_API_KEYS (auth on)
    session_id="my-session",    # Optional (auto-generated if omitted)
)

# Same API as local client
await client.upload("report.pdf")
results = await client.search_documents("quarterly revenue")
context = await client.get_context("key findings")

# Cleanup
await client.close()
```

The remote client delegates all operations to the REST API. It supports upload, search, search with citations, document listing, and deletion. Use it when Lasagna runs as a separate service and your application only calls it.

### Core Methods

| Method | Description |
|--------|-------------|
| `upload(path)` | Upload and index a document |
| `search_documents(query)` | Search for relevant chunks |
| `get_context(query)` | Get formatted context for LLM |
| `list_documents()` | List all documents |
| `delete_document(id)` | Delete a document |
| `agentic_search(query)` | Multi-hop retrieval |
| `from_url(url)` | Connect to a remote RAG server (classmethod) |

### Accessing Managers

```python
# Document operations
result = await client.documents.upload("./file.pdf")
docs = await client.documents.list()
chunks = await client.documents.get_chunks("doc_id")

# Search operations
results = await client.search.search("query")
context = await client.search.get_context("query")
results = await client.search.hybrid_search("query", alpha=0.7)

# Citation operations
citation = await client.citations.get("citation_id")
is_valid = await client.citations.verify("citation_id")
```

---

## Document Management

### Uploading Documents

```python
# Simple upload
result = await client.upload("./document.pdf")
print(f"ID: {result.document_id}")
print(f"Status: {result.status}")  # DocumentStatus.PROCESSING: upload is async

# chunk_count is 0 until processing completes. Wait for the real numbers:
await client.documents.wait_for_processing(result.document_id)
doc = (await client.list_documents())[0]
print(f"Chunks: {doc.chunk_count}, Status: {doc.status}")

# With a custom filename (a document keeps its file, filename and id;
# metadata= is not stored and raises ValueError)
result = await client.documents.upload(
    "./file.pdf",
    filename="Q3 2024 Report",
)

# Batch upload with parallel processing (default: up to 5 concurrent)
results = await client.documents.upload_batch([
    "./doc1.pdf",
    "./doc2.pdf",
    "./doc3.txt",
])
for r in results:
    print(f"{r.filename}: {r.status}")

# Sequential batch upload (max_concurrent=1)
results = await client.documents.upload_batch(
    ["./doc1.pdf", "./doc2.pdf"],
    max_concurrent=1  # Process one at a time
)

# High concurrency for many small documents
results = await client.documents.upload_batch(
    file_paths,
    max_concurrent=10  # Up to 10 concurrent uploads
)
```

### Batch Uploads

`upload_batch()` uploads files concurrently, up to `max_concurrent` at a time (default 5), and the backend inserts each document's chunks into the vector store in batches rather than one at a time.

### Supported File Types

Uploads accept the extensions in the `supported_extensions` setting. By default:

| Type | Extensions |
|------|------------|
| PDF | `.pdf` |
| Word | `.docx` |
| Text | `.txt`, `.md` |
| Data | `.csv`, `.json`, `.xml` |
| Web | `.html` |

With `docling_enabled` on and Docling installed (`pip install "lasagna-rag[docling]"`), PDF, Word and HTML files are converted with Docling's layout and table analysis instead of the plain extractors.

### Listing Documents

```python
# List all documents
docs = await client.list_documents()
for doc in docs:
    print(f"{doc.filename}: {doc.chunk_count} chunks ({doc.status})")

# Filter by status
from lasagna.sdk import DocumentStatus

ready_docs = await client.documents.list(status=DocumentStatus.READY)
```

### Getting Document Info

```python
# Get specific document
doc = await client.documents.get("doc_abc123")
if doc:
    print(f"Filename: {doc.filename}")
    print(f"Size: {doc.file_size} bytes")
    print(f"Chunks: {doc.chunk_count}")
    print(f"Created: {doc.created_at}")

# Get document chunks
chunks = await client.documents.get_chunks("doc_abc123")
for chunk in chunks:
    print(f"Chunk {chunk.chunk_index} (page {chunk.page_number}):")
    print(f"  {chunk.content[:100]}...")
```

### Deleting Documents

```python
# Delete single document
success = await client.delete_document("doc_abc123")

# Delete all documents (requires confirmation)
count = await client.documents.delete_all(confirm=True)
print(f"Deleted {count} documents")
```

### Reindexing

```python
# Reindex after config changes
await client.set_config(chunk_size=1000)
result = await client.documents.reindex("doc_abc123")
print(f"Reindexed: {result.chunk_count} new chunks")
```

It works on a local client and on one made with `KnowledgeClient.from_url`.
The document keeps answering searches from its old chunks until the new ones
are stored; after a model switch it moves to the new model's collection. A
reindex that fails raises `DocumentError` and leaves the document as it was;
an unknown id raises `DocumentNotFoundError`.

---

## Session Management

Sessions provide data isolation for multi-tenant applications. Each session has its own documents, citations, search cache, and configuration state.

### Session Basics

```python
# Create client with session ID
client = KnowledgeClient(session_id="user_123")

# Change session
client.session_id = "user_456"

# Documents are isolated per session
docs = await client.list_documents()  # Only shows user_456's docs
```

### Delete Session

Delete all data associated with a session:

```python
# Delete entire session (documents, citations, cache, config)
result = await client.delete_session(confirm=True)

print(f"Documents deleted: {result['documents_deleted']}")
print(f"Citations deleted: {result['citations_deleted']}")
print(f"Cache invalidated: {result['cache_invalidated']}")
print(f"Session cleared: {result['session_cleared']}")

# Client remains usable - can add new documents
await client.upload("./new_doc.pdf")
```

**What gets deleted:**
- All documents and their vector embeddings
- All document chunks from the vector database
- All citations in the citation database
- Search cache entries
- Session configuration state
- BM25 hybrid search indices

**Safety:** Requires `confirm=True` to prevent accidental deletion.

```python
# This will raise ValueError
await client.delete_session()  # Error: confirm=True required

await client.delete_session(confirm=False)  # Error: confirm=True required

# This works
await client.delete_session(confirm=True)  # Deletes all session data
```

### Session Isolation

Each session maintains complete data isolation:

```python
# Session 1
client1 = KnowledgeClient(session_id="session_1")
await client1.upload("./doc1.pdf")

# Session 2
client2 = KnowledgeClient(session_id="session_2")
await client2.upload("./doc2.pdf")

# Searches are isolated
results1 = await client1.search_documents("query")  # Only searches doc1.pdf
results2 = await client2.search_documents("query")  # Only searches doc2.pdf

# Deleting session 1 doesn't affect session 2
await client1.delete_session(confirm=True)
docs2 = await client2.list_documents()  # Still has doc2.pdf
```

### Multi-Tenancy Architecture

The Knowledge SDK provides built-in multi-tenancy support through session-based isolation:

| Layer | Isolation Mechanism |
| ----- | ------------------- |
| **SQLite (metadata)** | All queries filter by `session_id` |
| **Vector Database** | Chunks store `session_id` in a field of their own and in their metadata; search filters on the field |
| **API Layer** | Every operation validates and uses `session_id` |
| **BM25 Index** | Separate indices per session |

**One session per tenant:**

```python
# Each tenant/user gets their own session
async def create_tenant_client(tenant_id: str) -> KnowledgeClient:
    return KnowledgeClient(session_id=f"tenant_{tenant_id}")

# Upload a tenant's files in parallel
client = await create_tenant_client("acme_corp")
results = await client.documents.upload_batch(
    files,
    max_concurrent=5  # Parallel processing within tenant
)

# Cross-tenant data is never accessible
# tenant_a cannot see tenant_b's documents
```

Session isolation is a filter on the `session_id` field in each query. For hard isolation between tenants on one server, `tenant_isolation="partition"` (set by `config_for_production()`) keeps each tenant in its own vector store partition.

---

## Search Operations

### Basic Search

```python
# Simple search
results = await client.search_documents("What is the revenue?")

# With options
results = await client.search_documents(
    query="quarterly performance",
    top_k=10,              # Number of results
    min_score=0.5,         # Minimum similarity threshold
)

# Iterate results
for result in results:
    print(f"Score: {result.score:.2f}")
    print(f"Source: {result.document.filename}")
    print(f"Page: {result.chunk.page_number}")
    print(f"Content: {result.chunk.content[:200]}...")
    print()
```

### Getting Context for LLM

```python
# Get formatted context string
context = await client.get_context("key findings")

# Use in prompt
prompt = f"""Based on the following information:

{context}

Please answer: What are the main conclusions?"""

# Response will include citations like [Source 1: report.pdf (page 5)]
```

### Hybrid Search

Combines vector similarity with BM25 keyword matching:

```python
# Default alpha (0.7 = more semantic)
results = await client.search.hybrid_search("revenue growth", top_k=10)

# More keyword emphasis
results = await client.search.hybrid_search(
    "API endpoint documentation",
    alpha=0.3,  # 0 = pure keyword, 1 = pure semantic
)
```

### Multi-Query Search

Search with multiple queries and merge results:

```python
results = await client.search.multi_query_search(
    queries=[
        "quarterly revenue",
        "Q3 income",
        "financial performance",
    ],
    top_k=5,
    deduplicate=True,  # Remove duplicate chunks
)
```

### Streaming Search

Stream results as they're processed for responsive UIs:

```python
async for result in client.search.search_stream("revenue"):
    print(f"Found: {result.chunk.content[:50]}...")
    # Process immediately, don't wait for all results
```

### Batch Search

Search multiple queries with concurrency control:

```python
queries = ["revenue", "expenses", "profit margin", "market share"]

results = await client.search.search_batch(
    queries=queries,
    top_k=5,
    max_concurrent=3,  # Limit concurrent searches
)

for query, result in zip(queries, results):
    print(f"{query}: {len(result)} results")
```

### Query Routing (Adaptive-RAG)

Each query can be sent to one of three retrieval tiers. A router reads the
query (its type, length, and words that signal several hops) and picks the
tier; the rules are in `retrieval/routing/strategies.py`. Routing is on by
default (`enable_query_routing=True`).

```python
# Routing (default): the router picks the tier
results = await client.search_documents("What is X?")           # short factual: usually fast
results = await client.search_documents("Compare X with Y")     # comparative: standard
results = await client.search_documents(
    "Summarize all findings across documents"                    # multi-hop or complex: comprehensive
)

# Force a specific tier
results = await client.search_documents("query", routing_tier="fast")
results = await client.search_documents("query", routing_tier="comprehensive")

# Also works on get_context and SearchManager.search
context = await client.get_context("query", routing_tier="fast")
results = await client.search.search("query", routing_tier="standard")
```

Three tiers are available:

- **fast**: vector-only search. HyDE, reranking, hybrid search, MMR, context reordering and auto-merge are off for the query.
- **standard**: the pipeline as configured (by default hybrid search, reranking and MMR; HyDE only when `enable_hyde` is on).
- **comprehensive**: agentic multi-hop retrieval, which calls an LLM. For questions that need several documents or several steps.

Configure routing behavior via config:

```python
await client.set_config(
    enable_query_routing=True,         # Enable/disable routing (default: True)
    routing_strategy="adaptive",       # "rule_based" or "adaptive"
    routing_default_tier="standard",   # Fallback when routing is disabled
    routing_confidence_threshold=0.5,  # Adaptive strategy escalation threshold
)
```

### Search with Citations

```python
results, context = await client.search.search_with_citations(
    "What are the key risks?",
    top_k=5,
)

# Access structured results
for r in results:
    print(f"[{r.citation_id}] {r.score:.2f} - {r.document.filename}")

# Use formatted context
print(context)
```

### Document Focus (Scoped Search)

Limit searches to specific documents:

```python
# Set focus
client.set_focused_documents(["doc_123", "doc_456"])

# All searches now scoped to these documents
results = await client.search_documents("revenue")  # Only searches focused docs

# Clear focus
client.clear_focused_documents()

# Or pass document_ids directly
results = await client.search_documents(
    "revenue",
    document_ids=["doc_789"],  # Overrides focus
)
```

---

## Error Handling

Every exception the SDK raises on its own is a subclass of `KnowledgeError`.

### Exception Hierarchy

```python
from lasagna.sdk import (
    # Base exception (catch-all)
    KnowledgeError,

    # Document exceptions
    DocumentError,
    DocumentNotFoundError,
    UploadError,
    ChunkingError,
    ExtractionError,

    # Search exceptions
    SearchError,
    QueryError,
    EmbeddingError,
    RetrievalError,

    # Citation exceptions
    CitationError,
    CitationNotFoundError,

    # Configuration exceptions
    ConfigurationError,
    ProfileNotFoundError,

    # Backend exceptions
    BackendError,
    BackendConnectionError,
    BackendTimeoutError,

    # Agentic exceptions
    AgenticError,
    MaxIterationsError,

    # Evaluation exceptions
    EvaluationError,
    JudgeError,
)
```

### Basic Error Handling

```python
from lasagna.sdk import (
    KnowledgeClient,
    DocumentNotFoundError,
    QueryError,
    SearchError,
    KnowledgeError,
)

client = KnowledgeClient()

# Handle specific exceptions
try:
    await client.documents.delete("nonexistent_doc")
except DocumentNotFoundError as e:
    print(f"Document not found: {e.document_id}")
except KnowledgeError as e:
    print(f"SDK error: {e}")

# Handle search errors
try:
    results = await client.search_documents("")
except QueryError as e:
    print(f"Invalid query: {e.query}")
except SearchError as e:
    print(f"Search failed: {e}")
```

### Exception Attributes

Exceptions include contextual attributes:

```python
try:
    await client.upload("./report.pdf")
except FileNotFoundError as e:
    # A missing file is caught before upload starts, as a plain FileNotFoundError
    print(f"No such file: {e}")
except UploadError as e:
    # The backend refused the file (for example, an unsupported file type)
    print(f"Upload failed: {e}")
    print(f"File: {e.file_path}")
    print(f"Stage: {e.stage}")  # 'processing'

try:
    await client.citations.get("invalid_id")
except CitationNotFoundError as e:
    print(f"Citation {e.citation_id} not found")

try:
    await client.config_manager.set_profile("nonexistent")
except ProfileNotFoundError as e:
    print(f"No profile named {e.profile_name}")
except ConfigurationError as e:
    # The parent class: the profile exists but could not be applied
    print(f"Could not switch profile: {e}")
```

`upload()` returns as soon as the document is queued. Extraction, chunking and indexing run afterward, so a failure in those steps does not raise here. `client.documents.wait_for_processing(document_id)` raises `DocumentProcessingError` for it, with the reason (`e.reason`) and, when there is a known one, the fix; pass `raise_on_failure=False` to get the document back with `DocumentStatus.FAILED` and `error_message` instead.

### Catching All SDK Errors

```python
try:
    await client.search_documents("query")
    await client.agentic_search("complex query")
except KnowledgeError as e:
    # Catches all SDK-specific errors
    logger.error(f"Knowledge SDK error: {e}")
```

### Handling Rate Limits (HTTP 429)

A REST server with `rate_limiting_enabled` on answers 429 when a client sends too many
requests, with a `Retry-After` header giving the wait in seconds. A remote client
(`KnowledgeClient.from_url`) raises `SearchError` for it, with the `httpx` error as its cause:

```python
import asyncio

import httpx
from lasagna.sdk import SearchError

async def search_with_retry(client, query, max_retries=3):
    for attempt in range(max_retries):
        try:
            return await client.search_documents(query)
        except SearchError as e:
            cause = e.__cause__
            if isinstance(cause, httpx.HTTPStatusError) and cause.response.status_code == 429:
                wait = int(cause.response.headers.get("Retry-After", 2))
                print(f"Rate limited, retrying in {wait}s (attempt {attempt + 1})")
                await asyncio.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")
```

Rate limiting is applied by the REST server's middleware, so an in-process client is never
rate limited. For other transient failures, use the retry helpers in
[Retry and Resilience](configuration.md#retry-and-resilience).

---

## API Reference

### KnowledgeClient

| Method | Returns | Description |
|--------|---------|-------------|
| `upload(path, filename?)` | `UploadResult` | Upload document |
| `search_documents(query, top_k?, min_score?, document_ids?, routing_tier?)` | `SearchResults` | Search |
| `get_context(query, document_ids?, max_chunks?, routing_tier?)` | `str` | Get LLM context |
| `list_documents(status?)` | `List[Document]` | List documents |
| `delete_document(id)` | `bool` | Delete document |
| `agentic_search(query, max_iterations?, document_ids?)` | `AgenticSearchResults` | Multi-hop |
| `set_focused_documents(ids)` | `None` | Focus searches |
| `clear_focused_documents()` | `None` | Clear focus |
| `set_config(**kwargs)` | `None` | Update config |
| `use_preset(name)` | `None` | Apply preset |
| `delete_session(confirm)` | `Dict` | Delete all session data |
| `close()` | `None` | Cleanup |

### DocumentManager

| Method | Returns | Description |
|--------|---------|-------------|
| `upload(path, filename?)` | `UploadResult` | Upload document |
| `upload_batch(paths, max_concurrent=5)` | `List[UploadResult]` | Batch upload with parallel processing |
| `list(status?)` | `List[Document]` | List documents |
| `get(id)` | `Optional[Document]` | Get document |
| `delete(id)` | `bool` | Delete document |
| `delete_all(confirm)` | `int` | Delete all |
| `get_chunks(id, offset?, limit?)` | `List[DocumentChunk]` | Get chunks |
| `wait_for_processing(id, timeout?, poll_interval?, raise_on_failure?)` | `Document` | Wait until indexing finishes; raises `DocumentProcessingError` if it failed |
| `reindex(id)` | `UploadResult` | Reindex with current settings; raises `DocumentError` if it failed |

### SearchManager

| Method | Returns | Description |
|--------|---------|-------------|
| `search(query, top_k?, min_score?, document_ids?, use_reranking?, use_hyde?, routing_tier?)` | `SearchResults` | Search |
| `get_context(query, document_ids?, max_chunks?, routing_tier?)` | `str` | Get context |
| `search_with_citations(query, top_k?, document_ids?)` | `Tuple[SearchResults, str]` | Search + context |
| `hybrid_search(query, top_k?, alpha?, document_ids?)` | `SearchResults` | Hybrid search (thread-safe) |
| `multi_query_search(queries, top_k?, document_ids?, deduplicate?)` | `SearchResults` | Multi-query |
| `search_stream(query, top_k?, min_score?, document_ids?)` | `AsyncIterator[SearchResult]` | Stream results |
| `search_batch(queries, top_k?, max_concurrent?, document_ids?)` | `List[SearchResults]` | Batch search |

### CitationManager

| Method | Returns | Description |
|--------|---------|-------------|
| `get(id)` | `Citation` | Get citation (raises `CitationNotFoundError`) |
| `verify(id)` | `bool` | Verify citation (raises `CitationError`) |
| `list_recent(limit?, query_id?)` | `List[Citation]` | List recent |
| `list_for_document(document_id)` | `List[Citation]` | List for doc |
| `clear(confirm)` | `int` | Clear citations |
| `cleanup_expired()` | `int` | Remove expired |

### Exceptions

| Exception | Parent | Description |
| --------- | ------ | ----------- |
| `KnowledgeError` | `Exception` | Base exception for all SDK errors |
| `DocumentError` | `KnowledgeError` | Document operation failed |
| `DocumentNotFoundError` | `DocumentError` | Document not found |
| `UploadError` | `DocumentError` | Upload failed (attrs: `file_path`, `stage`) |
| `SearchError` | `KnowledgeError` | Search operation failed |
| `QueryError` | `SearchError` | Invalid query (attr: `query`) |
| `CitationError` | `KnowledgeError` | Citation operation failed |
| `CitationNotFoundError` | `CitationError` | Citation not found (attr: `citation_id`) |
| `ConfigurationError` | `KnowledgeError` | Configuration error (attr: `config_key`) |
| `ProfileNotFoundError` | `ConfigurationError` | Profile not found (attr: `profile_name`) |
| `BackendError` | `KnowledgeError` | Backend error (attrs: `backend_name`, `operation`) |
| `AgenticError` | `KnowledgeError` | Agentic RAG error (attr: `iteration`) |
| `MaxIterationsError` | `AgenticError` | Max iterations exceeded |
| `EvaluationError` | `KnowledgeError` | Evaluation error |
| `JudgeError` | `EvaluationError` | LLM judge error |

### Retry Utilities

| Function/Class | Description |
| -------------- | ----------- |
| `with_retry(max_attempts, delay, backoff, exceptions, on_retry)` | Async retry decorator |
| `with_retry_sync(...)` | Sync retry decorator |
| `RetryConfig` | Reusable retry configuration |
| `RETRY_BACKEND` | Preset for backend operations (3 attempts, 0.5s) |
| `RETRY_SEARCH` | Preset for search operations (2 attempts, 0.2s) |

---

## See Also

- [Getting Started](getting_started.md)
- [Configuration](configuration.md)
- [Features](features.md)
