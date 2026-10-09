/**
 * Parameters that require document re-indexing when changed.
 * Mirrored from Python schema.py — also fetched at runtime via /api/schema.
 */
export const REINDEX_KEYS: string[] = [
  "embedding_model",
  "embedding_dim",
  "normalize_embeddings",
  "chunking_strategy",
  "chunk_size",
  "chunk_overlap",
  "semantic_threshold",
  "min_chunk_size",
  "max_chunk_size",
  "preserve_tables",
  "preserve_headers",
  "docling_enabled",
  "enable_contextual_retrieval",
  "enable_ocr",
  "ocr_engine",
  "ocr_languages",
  "ocr_confidence_threshold",
  "enable_hierarchical",
  "parent_chunk_size",
  "child_chunk_size",
];

export const PROTECTED_PROFILES = new Set([
  "default",
  "speed",
  "standard",
  "balanced",
  "max_quality",
]);
