# Validation Rules Reference

`validation.py` provides three layers of validation, executed by `POST /configurator/api/validate`.

## Type & Range Checks

Every parameter in `schema.py` is automatically validated:

| Check | Condition | Severity |
|-------|-----------|----------|
| Boolean type | Value is not `bool` | Error |
| Numeric type | Value is not `int` or `float` | Error |
| Minimum bound | `value < min_value` | Error |
| Maximum bound | `value > max_value` | Error |
| Enum membership | Value not in `enum_options` | Error |

These are generated from `ParameterDef` metadata — no manual rules needed.

## Cross-Field Validation Rules (10 total)

| # | Key(s) | Condition | Severity | Rationale |
|---|--------|-----------|----------|-----------|
| 1 | `chunk_overlap`, `chunk_size` | `chunk_overlap >= chunk_size` | Warning | Overlap cannot exceed chunk size. Runtime auto-corrects to `chunk_size * 0.15`. |
| 2 | `min_chunk_size`, `max_chunk_size` | `min_chunk_size >= max_chunk_size` | Error | Invalid range — chunking would fail. |
| 3 | `child_chunk_size`, `parent_chunk_size` | `child_chunk_size >= parent_chunk_size` | Error | Hierarchical chunking requires child < parent. |
| 4 | `embedding_dim`, `embedding_model` | Dimension doesn't match model's known dimension | Warning | Auto-corrected at runtime. Known models: MiniLM=384, bge-small=384, bge-base=768, bge-large=1024, bge-m3=1024. |
| 5 | `chunk_cite_retrieval.min_chunks`, `chunk_cite_retrieval.max_chunks` | `min_chunks > max_chunks` | Error | Invalid range — retrieval would fail. |
| 6 | `routing_default_tier`, `use_agentic_rag` | `routing_default_tier="comprehensive"` AND `use_agentic_rag=False` | Warning | Comprehensive tier requires agentic RAG. Downgraded to standard at runtime. |
| 7 | `enable_hallucination_check` / `enable_answer_quality_check`, `agentic_pipeline` | Quality gate enabled but `agentic_pipeline` is not `"default"` (structured graph) | Warning | Quality gates only execute within the structured graph pipeline. Ignored at runtime when using `legacy_react` or custom pipelines. |
| 8 | `agentic_pipeline`, `use_agentic_rag` | `agentic_pipeline` is set (non-`legacy_react`) AND `use_agentic_rag=False` | Warning | Agentic pipelines require `use_agentic_rag=True`. Pipeline selection is ignored at runtime. |
| 9 | `agentic_pipeline` | Value is not a built-in name (`default`, `legacy_react`) and not confirmed registered | Warning | Non-builtin pipeline name. Suppressed if registry is available and confirms the name is registered. Otherwise warns (registry lookup is best-effort). If not registered before use, runtime will raise `KeyError` (fail-fast). |
| 10 | `use_structured_graph`, `agentic_pipeline` | Both `use_structured_graph` and `agentic_pipeline` are explicitly set with conflicting intent | Warning | `agentic_pipeline` takes precedence. `use_structured_graph` is a legacy alias. |

### Error vs Warning Classification

- **Error**: The configuration is invalid and would cause failures at runtime. The UI prevents saving.
- **Warning**: The configuration is saveable but suboptimal — runtime will auto-correct or silently disable the feature. The UI shows a yellow banner.

## Reindex Detection

When a `baseline` config is provided alongside the new config:

1. `get_reindex_keys()` returns the 18 parameter keys flagged with `requires_reindex=True` in `schema.py`
2. For each key, if `baseline[key] != config[key]` and `config[key]` is not `None`, the key is flagged
3. `ValidationResult.requires_reindex` is set to `True` and `reindex_keys` lists the changed keys
4. The frontend uses this to show a reindex prompt before saving

### Reindex Keys (18)

```
embedding_model, embedding_dim, normalize_embeddings,
chunking_strategy, chunk_size, chunk_overlap,
semantic_threshold, min_chunk_size, max_chunk_size,
preserve_tables, preserve_headers, enable_ocr,
ocr_engine, ocr_languages, ocr_confidence_threshold,
enable_hierarchical, parent_chunk_size, child_chunk_size
```

These keys are kept in sync with `frontend/src/schema/reindexKeys.ts`. A drift guard test in `test_configurator.py::TestReindexKeySync` enforces parity.
