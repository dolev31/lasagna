# Lasagna Technical Architecture

## System Overview

Lasagna runs as one process that can expose these interfaces:

### Interfaces
| Interface | Port | Description |
|-----------|------|-------------|
| REST API | 8000 | Upload, search and configuration endpoints |
| MCP server | 8005 | Knowledge base tools for AI agents |

### RAG Pipeline Architecture

The RAG pipeline processes documents through several stages:

1. **Document Ingestion**: Files are parsed using format-specific extractors
2. **Chunking**: Content is split into semantic chunks (default 1024 tokens)
3. **Embedding**: Chunks are embedded using BGE-M3 (1024-dimensional vectors)
4. **Indexing**: Embeddings stored in Milvus vector database
5. **Retrieval**: Hybrid search combining dense + BM25 sparse vectors
6. **Reranking**: Cross-encoder reranking for precision

### Quality Profiles

| Profile | Latency | Features |
|---------|---------|----------|
| Speed | ~100ms | Vector-only search, no reranking |
| Standard | ~300ms | Hybrid search + reranking |
| Balanced | ~500ms | All standard + MMR diversity |
| Max Quality | ~2s | Everything + HyDE + agentic RAG |

### API Rate Limits

- Default: 1000 requests per minute per session
- Burst: 50 requests per second
- Document upload: 100 per hour per session

### Security

- Session-based multi-tenancy with partition isolation
- JWT token authentication for REST API
- Rate limiting with token bucket algorithm
- Input sanitization for all user-provided content
