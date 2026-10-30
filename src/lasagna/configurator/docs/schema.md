# Parameter Reference

100 parameters across 4 tabs and 18 groups, as `PARAMETERS`, `TABS` and `GROUPS` in `configurator/schema.py` define them. The defaults are the effective values from `core/config.py` and `settings.lasagna.toml`; where `schema.py` lists a different one, the row says so.

Legend: **R** = requires reindex, **H** = hidden from UI.

---

## Essential Tab (16 parameters)

### Features

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `enable_hybrid_search` | boolean | `true` |  |  |
| `enable_reranking` | boolean | `true` |  |  |
| `enable_mmr` | boolean | `true` |  |  |
| `enable_hyde` | boolean | `false` |  |  |
| `enable_ocr` | boolean | `false` |  | R |
| `docling_enabled` | boolean | `true` (`schema.py` lists `false`) |  | R |
| `use_agentic_rag` | boolean | `true` |  |  |
| `enable_hierarchical` | boolean | `true` |  | R |

### AI Understanding Model

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `embedding_model` | enum | `"BAAI/bge-m3"` | Options: `all-MiniLM-L6-v2`, `BAAI/bge-small-en-v1.5`, `BAAI/bge-base-en-v1.5`, `BAAI/bge-large-en-v1.5`, `BAAI/bge-m3`, `intfloat/e5-small-v2`, `intfloat/e5-base-v2`, `intfloat/e5-large-v2`, `intfloat/multilingual-e5-large` (other names accepted) | R |
| `embedding_dim` | integer | `1024` | min=1, max=16384 | R, H |
| `use_gpu` | boolean | `true` |  |  |
| `batch_size` | integer | `32` | min=1, max=256, step=8 | H |
| `normalize_embeddings` | boolean | `true` |  | R, H |

### Quick Tuning

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `default_top_k` | integer | `20` | min=1, max=10000, step=5 |  |
| `similarity_threshold` | float | `0.3` | min=0.0, max=1.0, step=0.05 |  |
| `hybrid_alpha` | float | `0.7` | min=0.0, max=1.0, step=0.05 |  |

---

## Search & Retrieval Tab (29 parameters)

### Search Tuning

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `rerank_top_k` | integer | `5` | min=1, max=1000, step=1 |  |
| `mmr_lambda` | float | `0.5` | min=0.0, max=1.0, step=0.05 |  |
| `mmr_score_normalization` | enum | `"sigmoid"` | Options: `sigmoid`, `minmax`, `sigmoid_minmax`, `none` |  |
| `enable_context_reordering` | boolean | `true` |  |  |
| `rrf_k` | integer | `60` | min=1, max=10000, step=10 |  |
| `sparse_method` | enum | `"bm25"` | Options: `bm25`, `splade` |  |
| `enable_context_compression` | boolean | `false` |  |  |
| `compression_method` | enum | `"extractive"` | Options: `extractive`, `abstractive` |  |
| `compression_ratio` | float | `0.3` | min=0.1, max=1.0, step=0.1 |  |

### Result Re-Ranking

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `reranker_model` | enum | `"mixedbread-ai/mxbai-rerank-large-v1"` | Options: `cross-encoder/ms-marco-MiniLM-L-6-v2`, `cross-encoder/ms-marco-MiniLM-L-12-v2`, `BAAI/bge-reranker-base`, `BAAI/bge-reranker-large`, `BAAI/bge-reranker-v2-m3`, `BAAI/bge-reranker-v2-gemma`, `jinaai/jina-reranker-v2-base-multilingual`, `mixedbread-ai/mxbai-rerank-large-v1` (other names accepted) |  |
| `reranker_batch_size` | integer | `16` | min=1, max=128, step=4 | H |

### Smarter Searching

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `hyde_model` | string | `"default"` |  |  |
| `enable_query_expansion` | boolean | `false` |  |  |
| `max_query_variations` | integer | `3` | min=1, max=10, step=1 |  |
| `enable_query_decomposition` | boolean | `false` |  |  |
| `enable_rag_fusion` | boolean | `false` |  |  |
| `enable_crag` | boolean | `false` |  |  |

### How Many Results to Use

All nested under `chunk_cite_retrieval` parent key.

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `chunk_cite_retrieval.strategy` | enum | `"adaptive"` | Options: `adaptive`, `fixed`, `above_threshold`, `agent_decides` |  |
| `chunk_cite_retrieval.max_chunks` | integer | `20` | min=1, max=100, step=1 |  |
| `chunk_cite_retrieval.min_chunks` | integer | `2` | min=1, max=20, step=1 |  |
| `chunk_cite_retrieval.fixed_size` | integer | `5` | min=1, max=50, step=1 |  |
| `chunk_cite_retrieval.threshold_score` | float | `0.5` | min=0.0, max=1.0, step=0.05 |  |
| `chunk_cite_retrieval.score_gap_buffer` | integer | `2` | min=0, max=10, step=1 | H |
| `chunk_cite_retrieval.min_gap_ratio` | float | `0.15` | min=0.0, max=0.5, step=0.05 | H |

### Smart Routing

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `enable_query_classification` | boolean | `true` |  |  |
| `enable_query_routing` | boolean | `true` |  |  |
| `routing_strategy` | enum | `"rule_based"` | Options: `rule_based`, `adaptive` |  |
| `routing_default_tier` | enum | `"standard"` | Options: `fast`, `standard`, `comprehensive` |  |
| `routing_confidence_threshold` | float | `0.5` | min=0.0, max=1.0, step=0.05 |  |

No tab component in the frontend lists this group, so the UI does not show these parameters. They are in the schema response and accepted by the validate route; set them in a settings file or the environment instead.

---

## Processing Tab (24 parameters)

### Document Splitting

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `chunking_strategy` | enum | `"semantic"` | Options: `semantic`, `fixed`, `recursive`, `sentence_window`, `structure_aware`, `hierarchical` | R |
| `chunk_size` | integer | `800` | min=1, max=100000, step=50 | R |
| `chunk_overlap` | integer | `0` | min=0, max=100000, step=25 | R |
| `semantic_threshold` | float | `0.75` | min=0.0, max=1.0, step=0.05 | R |
| `min_chunk_size` | integer | `200` | min=0, max=100000, step=50 | R |
| `max_chunk_size` | integer | `1500` | min=1, max=100000, step=100 | R |
| `preserve_tables` | boolean | `true` |  | R |
| `preserve_headers` | boolean | `true` |  | R |
| `enable_contextual_retrieval` | boolean | `false` |  | R |

### Context-Aware Splitting

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `parent_chunk_size` | integer | `2000` | min=1, max=100000, step=100 | R |
| `child_chunk_size` | integer | `500` | min=1, max=100000, step=50 | R |
| `auto_merge_threshold` | float | `0.5` | min=0.1, max=1.0, step=0.1 |  |

### Scanned Document Reading

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `ocr_engine` | enum | `"easyocr"` | Options: `easyocr`, `tesseract`, `paddleocr` | R |
| `ocr_confidence_threshold` | float | `0.5` | min=0.0, max=1.0, step=0.05 | R |
| `ocr_languages` | object | `["en"]` |  | R, H |

### Document Awareness

All nested under `document_awareness` parent key.

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `document_awareness.enabled` | boolean | `true` |  |  |
| `document_awareness.enrichment_strategy` | enum | `"first_tokens"` | Options: `first_tokens`, `llm_summary`, `hybrid` |  |
| `document_awareness.first_tokens_count` | integer | `150` | min=1, max=10000, step=25 |  |
| `document_awareness.include_filename` | boolean | `true` |  |  |
| `document_awareness.include_upload_date` | boolean | `true` |  |  |
| `document_awareness.include_file_type` | boolean | `true` |  |  |
| `document_awareness.max_documents_in_context` | integer | `10` | min=1, max=50, step=1 |  |
| `document_awareness.cache_snippets` | boolean | `true` |  |  |
| `document_awareness.snippet_cache_ttl` | integer | `3600` | min=60, max=86400, step=60 |  |

---

## Advanced Tab (31 parameters)

### Deep Search Agent

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `max_agent_iterations` | integer | `3` | min=1, max=10, step=1 |  |
| `agent_temperature` | float | `0.1` | min=0.0, max=1.0, step=0.05 |  |
| `chunk_refinement_method` | enum | `"auto"` | Options: `vanilla`, `refined`, `auto` |  |
| `refinement_model` | string | `"gpt-6-luna"` | `MODEL_NAME` when set |  |
| `agentic_hyde_strategy` | enum | `"adaptive"` | Options: `off`, `first_hop`, `adaptive`, `always` |  |
| `use_final_search` | boolean | `false` |  |  |
| `use_structured_graph` | boolean | `false` | *Legacy alias* for `agentic_pipeline` |  |
| `agentic_pipeline` | string | `"legacy_react"` | Built-in: `legacy_react`, `default`, and the six published loops. Custom registered names also accepted. |  |
| `enable_hallucination_check` | boolean | `false` |  |  |
| `enable_answer_quality_check` | boolean | `false` |  |  |
| `coverage_scorer_mode` | enum | `"keyword"` | Options: `keyword`, `llm` |  |
| `final_search_mode` | enum | `"merge"` | Options: `merge`, `replace` |  |

### Citations & References

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `citation_format` | enum | `"stable_id"` | Options: `stable_id`, `index` |  |
| `citation_id_length` | integer | `8` | min=4, max=16, step=1 |  |
| `verify_citations` | boolean | `true` |  |  |
| `track_citations_db` | boolean | `true` |  |  |
| `show_page_numbers` | boolean | `true` |  |  |

### Performance Cache

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `enable_embedding_cache` | boolean | `true` |  |  |
| `embedding_cache_size` | integer | `10000` | min=100, max=100000, step=1000 |  |
| `enable_search_cache` | boolean | `true` |  |  |
| `search_cache_ttl` | integer | `300` | min=10, max=3600, step=10 |  |

### Monitoring & Logging

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `log_level` | enum | `"INFO"` | Options: `DEBUG`, `INFO`, `WARNING`, `ERROR` |  |
| `enable_prometheus` | boolean | `false` |  |  |
| `enable_tracing` | boolean | `false` |  |  |

### Vector Database

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `vector_backend` | enum | `"milvus"` | Options: `milvus`, `pinecone`, `qdrant`, `weaviate`, `pgvector`, `opensearch` |  |

### File Storage

| Key | Type | Default | Constraints | Flags |
|-----|------|---------|-------------|-------|
| `storage_backend` | enum | `"local"` | Options: `local`, `s3`, `fsspec` |  |
| `storage_s3_bucket` | string | `""` |  | H |
| `storage_s3_endpoint_url` | string | `""` |  | H |
| `storage_s3_region` | string | `"us-east-1"` |  | H |
| `storage_fsspec_protocol` | string | `""` |  | H |
| `storage_fsspec_base_path` | string | `""` |  | H |

No tab component in the frontend lists this group, so the UI does not show these parameters. They are in the schema response and accepted by the validate route; set them in a settings file or the environment instead. The `storage_s3_*` and `storage_fsspec_*` values are read by `storage/registry.py` from `RAG_STORAGE_S3_BUCKET`, `RAG_STORAGE_S3_ENDPOINT_URL`, `RAG_STORAGE_S3_REGION`, `RAG_STORAGE_FSSPEC_PROTOCOL` and `RAG_STORAGE_FSSPEC_BASE_PATH`, or from the `storage_backend_config` setting, which wins over them.

---

## Summary Counts

| Metric | Count |
|--------|-------|
| Total parameters | 100 |
| Reindex-requiring | 20 |
| Hidden from UI | 12 |
| Essential tab | 16 |
| Search & Retrieval tab | 29 |
| Processing tab | 24 |
| Advanced tab | 31 |
| Groups | 18 |
| Enum parameters | 19 |
| Boolean parameters | 37 |
| Integer parameters | 24 |
| Float parameters | 11 |
| String parameters | 8 |
| Object parameters | 1 |
| Nested (parent_key) | 16 |
