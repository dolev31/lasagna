# Validation Rules Reference

`validate_config()` in `validation.py` checks a configuration in four steps, and `POST /configurator/api/validate` returns its result (`errors`, `warnings`, `requires_reindex`, `reindex_keys`).

## Type & Range Checks

Every parameter in `schema.py` is validated from its `ParameterDef`:

| Check | Condition | Severity |
|-------|-----------|----------|
| Boolean type | Value is not `bool` | Error |
| Numeric type | Value is not `int` or `float`, or is a `bool` | Error |
| Integer type | An `integer` parameter gets a non-integer (`800.5`) | Error |
| Minimum bound | `value < min_value` | Error |
| Maximum bound | `value > max_value` | Error |
| Enum membership | Value not in `enum_options` | Error |
| Open enum | An `open_enum` parameter (a model name) gets something other than a non-empty string | Error |
| String list | A `validate_as="string_list"` parameter (`ocr_languages`) is not a list of strings | Error |

These are generated from `ParameterDef` metadata; no manual rules are needed.

## Type Checks for the Other Keys

The config holds more keys than the schema's 100. Any other key in the submitted config is checked against the type of its default in `core/config.py` (nested dicts key by key), and a value of the wrong type is an error. A wrong-typed value would otherwise be stored and then break every later read of the session's config.

## Cross-Field Validation Rules (11 total)

| # | Key(s) | Condition | Severity | Rationale |
|---|--------|-----------|----------|-----------|
| 1 | `chunk_overlap`, `chunk_size` | `chunk_overlap >= chunk_size` | Warning | Overlap cannot exceed chunk size. Runtime auto-corrects to `chunk_size * 0.15`. |
| 2 | `min_chunk_size`, `max_chunk_size` | `min_chunk_size >= max_chunk_size` | Error | Invalid range: chunking would fail. |
| 3 | `child_chunk_size`, `parent_chunk_size` | `child_chunk_size >= parent_chunk_size` | Error | Hierarchical chunking requires child < parent. |
| 4 | `embedding_dim`, `embedding_model` | Dimension doesn't match model's known dimension | Warning | Auto-corrected at runtime. Known models: all-MiniLM-L6-v2=384, bge-small=384, bge-base=768, bge-large=1024, bge-m3=1024, e5-small-v2=384, e5-base-v2=768, e5-large-v2=1024, multilingual-e5-large=1024. |
| 5 | `chunk_cite_retrieval.min_chunks`, `chunk_cite_retrieval.max_chunks` | `min_chunks > max_chunks` | Error | Invalid range: retrieval would fail. |
| 6 | `routing_default_tier`, `use_agentic_rag` | `routing_default_tier="comprehensive"` AND `use_agentic_rag=False` | Warning | The `comprehensive` tier requires agentic RAG. Downgraded to `standard` at runtime. |
| 7 | `enable_hallucination_check` / `enable_answer_quality_check`, `agentic_pipeline` | Quality gate enabled, and neither `agentic_pipeline="default"` nor (`agentic_pipeline` unset and `use_structured_graph=True`) | Warning | Quality gates only execute within the structured graph pipeline. Ignored at runtime when using `legacy_react` or other pipelines. |
| 8 | `use_structured_graph`, `use_agentic_rag` | `use_structured_graph=True` AND `use_agentic_rag=False` | Warning | The structured graph is part of agentic RAG, so it is ignored. |
| 9 | `agentic_pipeline`, `use_agentic_rag` | `agentic_pipeline` is set (non-`legacy_react`) AND `use_agentic_rag=False` | Warning | Agentic pipelines require `use_agentic_rag=True`. Pipeline selection is ignored at runtime. |
| 10 | `agentic_pipeline` | Value is not a built-in name (`default`, `legacy_react`) and the component registry does not confirm it is registered | Warning | Suppressed when the registry confirms the name (the six published loops are registered, so they pass). If the name is not registered before use, runtime raises `KeyError` (fail-fast). |
| 11 | `use_structured_graph`, `agentic_pipeline` | `use_structured_graph=True` and `agentic_pipeline` is set to something other than `"default"` | Warning | `agentic_pipeline` takes precedence. `use_structured_graph` is a legacy alias. |

### Error vs Warning Classification

- **Error**: The configuration is invalid and would cause failures at runtime. The UI refuses to save.
- **Warning**: The configuration can be saved, but runtime will auto-correct it or ignore the feature.

## Reindex Detection

When a `baseline` config is provided alongside the new config:

1. `get_reindex_keys()` returns the 20 parameter keys flagged with `requires_reindex=True` in `schema.py`
2. For each key, if `baseline[key] != config[key]` and `config[key]` is not `None`, the key is flagged
3. `ValidationResult.requires_reindex` is set to `True` and `reindex_keys` lists the changed keys

The configurator UI calls the route without a baseline. It works out the reindex banner itself, by comparing the edit with the saved config over its own copy of the keys in `reindexKeys.ts`.

### Reindex Keys (20)

```
embedding_model, embedding_dim, normalize_embeddings,
chunking_strategy, chunk_size, chunk_overlap,
semantic_threshold, min_chunk_size, max_chunk_size,
preserve_tables, preserve_headers, docling_enabled,
enable_contextual_retrieval, enable_ocr,
ocr_engine, ocr_languages, ocr_confidence_threshold,
enable_hierarchical, parent_chunk_size, child_chunk_size
```

These keys are kept in sync with `frontend/src/schema/reindexKeys.ts`. A drift guard test in `test_configurator.py::TestReindexKeySync` enforces parity.
