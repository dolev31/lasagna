# processing: extraction and chunking

The `document_processor` and `chunker` stages: turning a file into text (PDF, Office, HTML,
plain text, OCR for scans, Docling or Markdown converters) and text into chunks, plus the
optional LLM context prefix of Contextual Retrieval. The upload path in
`api/_document_pipeline.py` and the benchmark ingester in
`knowledge_evaluation_sdk/_fast_ingest.py` both call into this package.

`lasagna component list chunker` (or `document_processor`) lists the registered components and
the paper each implements; every setting and default is in
[docs/reference/settings.md](../../../docs/reference/settings.md).
Swapping a chunker in a sweep: [docs/guide.md](../../../docs/guide.md#3-the-stages-you-can-swap).

## Module map

| File | Responsibility |
| ---- | -------------- |
| `__init__.py` | Lazy exports (`DocumentProcessor`, `OCRProcessor`, `ContextualEnricher`, ...) via `__getattr__`, so heavy parsers load only when used |
| `document_processor.py` | `DocumentProcessor` (pypdf then pdfplumber, python-docx, BeautifulSoup, text with a UTF-8, Latin-1, CP1252 chain); `DocumentProcessorAdapter` (the `default` processor) |
| `ocr_processor.py` | `OCRProcessor`: EasyOCR, Tesseract, PaddleOCR engines, the configured one first and the others as fallbacks; page-at-a-time PDF OCR; `is_scanned_pdf` |
| `docling_processor.py` | `DoclingProcessor` (singleton: `get_instance`, `reset`), conversion plus Docling's `HybridChunker` on the upload path; `DoclingDocumentProcessor` (the `docling` component, extraction only) |
| `markdown_converters.py` | `MarkItDownProcessor`, `PyMuPDF4LLMProcessor` (the `markitdown` and `pymupdf4llm` components) |
| `chunker/semantic.py` | Built-in chunking: `SemanticChunker`, `HierarchicalChunker`, `StructureAwareChunker`, `resolve_effective_strategy`, `chunk_document()` (the upload-path entry point), and the adapters registered as `semantic`, `hierarchical`, `sentence_window`, `structure_aware` |
| `chunker/splitters.py` | `FixedSizeChunker` (`fixed`), `RecursiveChunker` (`recursive`); `chunk_unit` of chars, words or tokens |
| `chunker/markdown.py` | `MarkdownHeaderChunker` (`markdown_header`): one chunk per section, `metadata["header_path"]` |
| `chunker/llm_chunkers.py` | `PropositionChunker` (`proposition`) and `ContextualChunker` (`contextual`) |
| `contextual/contextual_enricher.py` | `ContextualEnricher`: LLM context prefix for the `enable_contextual_retrieval` flag, with an LRU cache (10,000 entries by default) |

## Design notes

**Structural flags outrank `chunking_strategy`.** With no chunker component named,
`resolve_effective_strategy` runs hierarchical chunking when `enable_hierarchical` is on, then
structure-aware chunking when `preserve_headers` or `preserve_tables` is on, and only then the
requested `chunking_strategy`. All three flags default to true, so the strategy setting alone
changes nothing; a warning names what ran instead. This is deliberate and loud because two
"different" arms would otherwise produce identical chunks. A named chunker component bypasses
this dispatch entirely.

**No silent fallback under a technique's name.** `semantic` raises when its embedding model
cannot load; `proposition` and `contextual` raise without an LLM. A fallback would score the
control under the arm's name. `contextual` combined with `enable_contextual_retrieval` is
refused, since the context would be prepended twice. The `enable_contextual_retrieval` flag
keeps a chunk bare when its LLM call fails, except under `strict_stages()`.

**Tokens mean tokens.** `chunk_unit = "tokens"` needs `chunk_tokenizer` (a Hugging Face
tokenizer name or `tiktoken:<encoding>`) and is refused without one rather than approximated.

**Models load once.** `semantic` and `structure_aware` embed through
`infrastructure.embedding.manager.shared_embedding_manager`, so a chunker naming the pipeline's
model reuses it and another model is loaded at most once per process. `DoclingProcessor` is a
double-checked singleton because its models take gigabytes.

**Docling on upload vs. as a component.** With `docling_enabled` (default true, skipped when
Docling is not installed) the upload path converts with Docling and chunks with its
`HybridChunker`. The `docling` component only extracts, so comparing processors does not also
compare chunkers.

**Stable chunk ids.** A chunk's id is the SHA-256 of its content (`Chunk` in
`chunker/semantic.py`), so citations survive re-indexing.

**Sentence splitting.** NLTK `punkt_tab` when its data is installed
(`python -m nltk.downloader punkt_tab`), a regex otherwise, noted once in the log.

**Not swept by the harness.** Benchmark corpora are loaded as text, so document processors are
compared on your own files, not in `lasagna benchmark experiment` arms.
