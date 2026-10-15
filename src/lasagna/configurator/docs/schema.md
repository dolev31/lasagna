# Parameter Reference

91 parameters across 4 tabs and 18 groups. All defaults are verified against `core/config.py` + `settings.lasagna.toml` effective values.

Legend: **R** = requires reindex, **H** = hidden from UI.

---

## Essential Tab (15 parameters)

### Features

| Key | Type | Default | Description | Flags |
|-----|------|---------|-------------|-------|
| `enable_hybrid_search` | boolean | `true` | Dense vector + BM25 keyword search | |
| `enable_reranking` | boolean | `true` | Cross-encoder reranking of top-k candidates | |
| `enable_mmr` | boolean | `true` | Maximal Marginal Relevance for diversity | |
| `enable_hyde` | boolean | `false` | Hypothetical Document Embeddings (HyDE) | |
| `enable_ocr` | boolean | `false` | OCR for scanned documents | R |
| `use_agentic_rag` | boolean | `true` | Multi-hop agentic RAG with ReAct | |
| `enable_hierarchical` | boolean | `true` | Hierarchical parent/child chunking | R |

### AI Understanding Model

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `embedding_model` | enum | `"BAAI/bge-m3"` | Options: `all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5`, `BAAI/bge-m3` | R |
| `embedding_dim` | integer | `1024` | min=128, max=4096 | R, H |
| `use_gpu` | boolean | `true` | — | |
| `batch_size` | integer | `32` | min=1, max=256, step=8 | H |
| `normalize_embeddings` | boolean | `true` | — | R, H |

### Quick Tuning

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `default_top_k` | integer | `20` | min=1, max=500, step=5 |
| `similarity_threshold` | float | `0.3` | min=0.0, max=1.0, step=0.05 |
| `hybrid_alpha` | float | `0.7` | min=0.0, max=1.0, step=0.05 |

---

## Search & Retrieval Tab (20 parameters)

### Search Tuning

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `rerank_top_k` | integer | `5` | min=1, max=100, step=1 |
| `mmr_lambda` | float | `0.5` | min=0.0, max=1.0, step=0.05 |
| `mmr_score_normalization` | enum | `"sigmoid"` | Options: `sigmoid`, `minmax`, `sigmoid_minmax`, `none` |

### Result Re-Ranking

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `reranker_model` | enum | `"mixedbread-ai/mxbai-rerank-large-v1"` | Options: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/ms-marco-MiniLM-L-12-v2`, `BAAI/bge-reranker-base`, `BAAI/bge-reranker-large`, `mixedbread-ai/mxbai-rerank-large-v1` | |
| `reranker_batch_size` | integer | `16` | min=1, max=128, step=4 | H |

### Smarter Searching

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `hyde_model` | string | `"default"` | — |
| `enable_query_expansion` | boolean | `false` | — |
| `max_query_variations` | integer | `3` | min=1, max=10, step=1 |
| `enable_query_decomposition` | boolean | `false` | — |

### Smart Routing

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `enable_query_routing` | boolean | `true` | — |
| `routing_strategy` | enum | `"rule_based"` | Options: `rule_based`, `adaptive` |
| `routing_default_tier` | enum | `"standard"` | Options: `fast`, `standard`, `comprehensive` |
| `routing_confidence_threshold` | float | `0.5` | min=0.0, max=1.0, step=0.05 |

### How Many Results to Use

All nested under `chunk_cite_retrieval` parent key.

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `chunk_cite_retrieval.strategy` | enum | `"adaptive"` | Options: `adaptive`, `fixed`, `above_threshold`, `agent_decides` | |
| `chunk_cite_retrieval.max_chunks` | integer | `20` | min=1, max=100, step=1 | |
| `chunk_cite_retrieval.min_chunks` | integer | `2` | min=1, max=20, step=1 | |
| `chunk_cite_retrieval.fixed_size` | integer | `5` | min=1, max=50, step=1 | |
| `chunk_cite_retrieval.threshold_score` | float | `0.5` | min=0.0, max=1.0, step=0.05 | |
| `chunk_cite_retrieval.score_gap_buffer` | integer | `2` | min=0, max=10, step=1 | H |
| `chunk_cite_retrieval.min_gap_ratio` | float | `0.15` | min=0.0, max=0.5, step=0.05 | H |

---

## Processing Tab (23 parameters)

### Document Splitting

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `chunking_strategy` | enum | `"semantic"` | Options: `semantic`, `fixed`, `recursive`, `sentence_window` | R |
| `chunk_size` | integer | `800` | min=100, max=4000, step=50 | R |
| `chunk_overlap` | integer | `0` | min=0, max=1000, step=25 | R |
| `semantic_threshold` | float | `0.75` | min=0.0, max=1.0, step=0.05 | R |
| `min_chunk_size` | integer | `200` | min=50, max=1000, step=50 | R |
| `max_chunk_size` | integer | `1500` | min=500, max=10000, step=100 | R |
| `preserve_tables` | boolean | `true` | — | R |
| `preserve_headers` | boolean | `true` | — | R |

### Context-Aware Splitting

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `parent_chunk_size` | integer | `2000` | min=500, max=10000, step=100 | R |
| `child_chunk_size` | integer | `500` | min=100, max=2000, step=50 | R |
| `auto_merge_threshold` | float | `0.5` | min=0.1, max=1.0, step=0.1 | |

### Scanned Document Reading

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `ocr_engine` | enum | `"easyocr"` | Options: `easyocr`, `tesseract`, `paddleocr` | R |
| `ocr_confidence_threshold` | float | `0.5` | min=0.0, max=1.0, step=0.05 | R |
| `ocr_languages` | object | `["en"]` | — | R, H |

### Document Awareness

All nested under `document_awareness` parent key.

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `document_awareness.enabled` | boolean | `true` | — |
| `document_awareness.enrichment_strategy` | enum | `"first_tokens"` | Options: `first_tokens`, `llm_summary`, `hybrid` |
| `document_awareness.first_tokens_count` | integer | `150` | min=50, max=500, step=25 |
| `document_awareness.include_filename` | boolean | `true` | — |
| `document_awareness.include_upload_date` | boolean | `true` | — |
| `document_awareness.include_file_type` | boolean | `true` | — |
| `document_awareness.max_documents_in_context` | integer | `10` | min=1, max=50, step=1 |
| `document_awareness.cache_snippets` | boolean | `true` | — |
| `document_awareness.snippet_cache_ttl` | integer | `3600` | min=60, max=86400, step=60 |

---

## Advanced Tab (33 parameters)

### Deep Search Agent

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `max_agent_iterations` | integer | `3` | min=1, max=10, step=1 |
| `agent_temperature` | float | `0.1` | min=0.0, max=1.0, step=0.05 |
| `chunk_refinement_method` | enum | `"auto"` | Options: `vanilla`, `refined`, `auto` |
| `refinement_model` | string | `"gpt-6-luna"` | — |
| `agentic_hyde_strategy` | enum | `"adaptive"` | Options: `off`, `first_hop`, `adaptive`, `always` |
| `use_final_search` | boolean | `false` | — |
| `agentic_pipeline` | string | `"legacy_react"` | Built-in: `legacy_react`, `default`. Custom registered names also accepted. |
| `use_structured_graph` | boolean | `false` | *Legacy alias* for `agentic_pipeline` |
| `enable_hallucination_check` | boolean | `false` | — |
| `enable_answer_quality_check` | boolean | `false` | — |
| `coverage_scorer_mode` | enum | `"keyword"` | Options: `keyword`, `llm` |
| `final_search_mode` | enum | `"merge"` | Options: `merge`, `replace` |

### Citations & References

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `citation_format` | enum | `"stable_id"` | Options: `stable_id`, `index` |
| `citation_id_length` | integer | `8` | min=4, max=16, step=1 |
| `verify_citations` | boolean | `true` | — |
| `track_citations_db` | boolean | `true` | — |
| `show_page_numbers` | boolean | `true` | — |

### Performance Cache

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `enable_embedding_cache` | boolean | `true` | — |
| `embedding_cache_size` | integer | `10000` | min=100, max=100000, step=1000 |
| `enable_search_cache` | boolean | `true` | — |
| `search_cache_ttl` | integer | `300` | min=10, max=3600, step=10 |

### Monitoring & Logging

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `log_level` | enum | `"INFO"` | Options: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `enable_prometheus` | boolean | `false` | — |
| `enable_tracing` | boolean | `false` | — |

### Vector Database

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `vector_backend` | enum | `"milvus"` | Options: `milvus`, `pinecone`, `qdrant`, `weaviate`, `pgvector` |

### File Storage

| Key | Type | Default | Constraints |
|-----|------|---------|-------------|
| `storage_backend` | enum | `"local"` | Options: `local`, `s3`, `fsspec` |
| `storage_s3_bucket` | string | `""` | — |
| `storage_s3_endpoint_url` | string | `""` | — |
| `storage_s3_region` | string | `"us-east-1"` | — |
| `storage_fsspec_protocol` | string | `""` | — |
| `storage_fsspec_base_path` | string | `""` | — |

---

## Summary Counts

| Metric | Count |
|--------|-------|
| Total parameters | 91 |
| Reindex-requiring | 18 |
| Hidden from UI | 7 |
| Essential tab | 15 |
| Search tab | 20 |
| Processing tab | 23 |
| Advanced tab | 33 |
| Groups | 18 |
| Enum parameters | 15 |
| Boolean parameters | 30 |
| Integer parameters | 25 |
| Float parameters | 13 |
| String parameters | 6 |
| Object parameters | 1 |
| Nested (parent_key) | 16 |
