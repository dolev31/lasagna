# Document Processing Pipeline

> Transforms raw documents into structured, semantically coherent chunks ready for embedding and retrieval.

## Architecture

```
                        ┌──────────────────┐
                        │  Raw Document    │
                        │ (PDF/DOCX/HTML…) │
                        └────────┬─────────┘
                                 │
                    ┌────────────┼────────────┐
                    ▼            ▼             ▼
            ┌──────────┐ ┌────────────┐ ┌──────────┐
            │ Document │ │  Docling   │ │   OCR    │
            │Processor │ │ Processor  │ │Processor │
            │(text ext)│ │(structure) │ │(scanned) │
            └────┬─────┘ └─────┬──────┘ └────┬─────┘
                 │              │              │
                 └──────────────┼──────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Chunking Engine     │
                    │ (semantic/fixed/      │
                    │  hierarchical/        │
                    │  structure-aware)     │
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │ Contextual Enricher   │
                    │ (optional LLM context)│
                    └───────────┬───────────┘
                                ▼
                    ┌───────────────────────┐
                    │  Ready for Embedding  │
                    │  & Vector Storage     │
                    └───────────────────────┘
```

## Components

### Document Processor Components

The `document_processor` stage has four registered components. A pipeline picks one with `KnowledgePipeline(document_processor=...)`, and `ingest()` extracts with it:

| Name | What it does | Install |
|------|--------------|---------|
| `default` | The built-in extraction below: pypdf with a pdfplumber fallback, python-docx, BeautifulSoup, OCR when enabled | core |
| `docling` | Docling (Auer et al. 2024): layout analysis, TableFormer tables and OCR, exported as Markdown. Extraction only: the chunking is left to the arm's chunker | `pip install "lasagna-rag[docling]"` |
| `markitdown` | Microsoft MarkItDown: PDF, Office, HTML, CSV, JSON, XML and more, to Markdown, with plugins off | `pip install "lasagna-rag[markitdown]"` |
| `pymupdf4llm` | PyMuPDF4LLM: PDF to Markdown page by page, with headers and tables. AGPL-3.0, so it has its own extra and is not part of `all` | `pip install "lasagna-rag[pymupdf4llm]"` |

Each optional one is imported when a file is first extracted, and raises naming the install when it is missing. Markdown output keeps headings, lists and tables as structure the `markdown_header` chunker can split on.

Document processors do not sweep in the benchmark harness, because benchmark corpora are loaded as text. Compare them on your own files.

### Document Processor (`document_processor.py`)

The built-in `default` extraction: multi-format text extraction with fallback strategy.

| Format | Library | Fallback |
|--------|---------|----------|
| PDF | `pypdf` | `pdfplumber` |
| DOCX | `python-docx` | — |
| HTML | `beautifulsoup4` | regex tag stripping |
| TXT, MD, CSV, JSON, XML | built-in | encoding fallback chain (UTF-8 → Latin-1 → CP1252) |

**When to use vs Docling:** Use for simple text extraction without structure preservation. Docling is preferred for PDFs with tables, complex layouts, or when page/section metadata matters.

### OCR Processor (`ocr_processor.py`)

OCR engine abstraction with automatic fallback.

| Engine | Best for | GPU | Install |
|--------|----------|-----|---------|
| **EasyOCR** (default) | General use, high accuracy | Yes | `pip install easyocr` |
| **Tesseract** | Fast processing, wide availability | No | `pip install pytesseract` |
| **PaddleOCR** | Multilingual documents | Yes | `pip install paddleocr` |

Features:
- Confidence-based filtering (configurable threshold)
- Automatic engine selection with fallback chain
- Per-page incremental PDF processing (avoids OOM on large documents)
- Spatial sorting of results (top-to-bottom, left-to-right)
- Thread-safe singleton with double-check locking

### Docling Processor (`docling_processor.py`)

AI-powered document understanding via Docling (MIT-licensed). With `docling_enabled` on, the upload path converts with Docling and chunks with Docling's own HybridChunker. The `docling` component runs the same converter for extraction only, so comparing it with another processor does not also compare chunkers.

| Mode | Backend | RAM | GPU |
|------|---------|-----|-----|
| `local` | pypdfium2 | ~2.5 GB | No |
| `production` | docling_parse | ~6 GB | Yes (CUDA) |
| `auto` (default) | auto-detect | varies | if available |

Adds over basic extraction:
- **Layout analysis**: headers, sections, page structure
- **TableFormer**: table structure recognition
- **Formula detection**: mathematical expressions
- **Structure-aware chunking** via HybridChunker
- Falls back to legacy pipeline if Docling is not installed

### Chunker Components (`chunker/`)

The `chunker` stage has nine registered components. Each reads the arm's settings, so two arms that differ in a chunker setting get different chunks, and a sweep gives each chunker arm a corpus of its own.

| Name | Method | Settings (default) |
|------|--------|--------------------|
| `fixed` | Fixed-size windows with overlap. No boundary moves to a sentence or word end | `chunk_size` (800), `chunk_overlap` (0), `chunk_unit` (`chars`), `chunk_tokenizer` |
| `recursive` | LangChain's `RecursiveCharacterTextSplitter` algorithm: split on paragraphs, then lines, sentences, words and characters until each piece fits | the `fixed` settings, `recursive_separators`, `recursive_keep_separator` (`end`) |
| `sentence_window` | Sliding windows of whole sentences | `window_size` (3), `sentence_overlap` (1) |
| `markdown_header` | One chunk per Markdown section, with its header path in metadata (LangChain `MarkdownHeaderTextSplitter` semantics); long sections are split recursively | `markdown_header_levels` (6), `markdown_strip_headers` (true), `markdown_split_long_sections` (true) |
| `semantic` | Splits where adjacent sentences' embeddings diverge. `absolute` breaks where similarity falls below `semantic_threshold`; `percentile`, `standard_deviation`, `interquartile` and `gradient` are Kamradt's breakpoint methods | `semantic_breakpoint_method` (`absolute`), `semantic_breakpoint_amount`, `semantic_buffer_size` (1), `semantic_threshold` (0.75), `min_chunk_size` (200), `max_chunk_size` (1500) |
| `hierarchical` | Large parent chunks for context, small child chunks for search | `parent_chunk_size` (2000), `child_chunk_size` (500) |
| `structure_aware` | Tables become chunks of their own and headers mark sections; the text between is chunked semantically | `preserve_headers`, `preserve_tables`, and the `semantic` settings |
| `proposition` | Dense X Retrieval (Chen et al. 2023): an LLM rewrites each passage as atomic, self-contained propositions, one chunk each | `proposition_model` (the configured LLM), `proposition_passage_size` (100 words) |
| `contextual` | Contextual Retrieval (Anthropic 2024): chunks from a base chunker, each prefixed with LLM-written context that situates it in the document | `contextual_base_chunker` (`recursive`), `contextual_model` (the configured LLM), `contextual_max_document_chars` (50000) |

- `chunk_unit` is `chars`, `words`, or `tokens` of `chunk_tokenizer` (a Hugging Face tokenizer name, or `tiktoken:<encoding>`). `tokens` without a tokenizer is refused rather than approximated by words.
- `semantic` and `structure_aware` embed with the arm's `embedding_model`, loaded once per process. `semantic` raises when the model cannot be loaded; it no longer falls back to fixed-size chunking under its name.
- `proposition` and `contextual` need an LLM and raise without one; neither falls back to plain chunks.
- `contextual` cannot be combined with `enable_contextual_retrieval`: the context would be prepended twice, so the combination is refused.
- A named chunker component replaces the built-in strategies below outright. `enable_hierarchical` and `preserve_*` do not apply on top of it.

### Built-in Chunking Strategies (`chunker/semantic.py`)

With no chunker component named, `chunking_strategy` picks a built-in strategy: `semantic` (the default), `fixed`, `sentence_window` or `recursive` (the same algorithm as the `recursive` component). Two structural settings come first, though. `enable_hierarchical` (default true) runs hierarchical chunking, and `preserve_headers` or `preserve_tables` (both default true) run structure-aware chunking. So out of the box `chunking_strategy` does not take effect, and a warning names the strategy that ran instead. Set all three to false for `chunking_strategy` to apply.

| Strategy | When to use | Key param |
|----------|-------------|-----------|
| `semantic` | Default; splits at meaning boundaries | `semantic_threshold`, `semantic_breakpoint_method` |
| `fixed` | Fastest, no embedding needed | `chunk_size`, `chunk_overlap` |
| `sentence_window` | When sentence-level context matters | `window_size` |
| `recursive` | The common baseline: paragraph, line, sentence, word splitting | `chunk_size`, `recursive_separators` |
| `hierarchical` | Parent context + child search (`enable_hierarchical`) | `parent_chunk_size`, `child_chunk_size` |
| structure-aware | Documents with headers/tables/code | `preserve_tables`, `preserve_headers` |

### Late Chunking

Late chunking is an embedding-time setting, not a chunker, so it works with any of them. With `enable_late_chunking`, the whole document is embedded once and each chunk's vector is the mean of its own tokens' embeddings (Günther et al., arXiv:2409.04701); the chunk boundaries do not change. A document longer than the model's window is processed in overlapping windows (`late_chunking_overlap`, 512 tokens by default, at most half a window). Under evaluation, a document it cannot process raises instead of being embedded the standard way. Details: [embedding internals](../infrastructure/docs/embedding.md#late-chunking).

**Hierarchical chunking:** Creates large parent chunks (context) and small child chunks (search units). When enough child chunks from the same parent appear in results, they are merged back into the parent for richer LLM context (auto-merge retrieval, based on LlamaIndex).

**Stable IDs:** Every chunk gets a SHA-256 content-based hash for deterministic citation tracking across re-indexing.

### Contextual Enricher (`contextual/contextual_enricher.py`)

Implements Anthropic's [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) technique.

- Generates a short LLM context describing where each chunk fits in the document, with Anthropic's published prompt
- Prepends context to chunk before embedding
- Shows the LLM at most `contextual_max_document_chars` of the document (50,000 by default)
- LRU cache (default 10K entries) avoids re-computation
- Supports sync and async enrichment with concurrency control
- Used two ways. The `enable_contextual_retrieval` flag enriches the built-in chunks and keeps a chunk bare when its call fails, except under evaluation, where a failed enrichment fails the task. The `contextual` chunker component always raises instead.

## Configuration

Key processing parameters and where to set them:

| Parameter | Default | TOML | SDK | REST API |
|-----------|---------|------|-----|----------|
| `chunking_strategy` | `semantic` | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `chunk_size` | 800 | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `enable_hierarchical` | `true` | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `parent_chunk_size` | 2000 | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `child_chunk_size` | 500 | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `enable_ocr` | `false` | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `ocr_engine` | `easyocr` | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `docling_enabled` | `true` | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `docling_mode` | `auto` | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `docling_table_structure` | `true` | `[rag]` | `KnowledgeConfig` | `PATCH /config` |
| `docling_max_tokens` | 512 | `[rag]` | `KnowledgeConfig` | `PATCH /config` |

All parameters can also be set via `RAG_*` environment variables (e.g., `RAG_CHUNK_SIZE=1000`).

The component settings in the tables above (`chunk_unit`, `recursive_separators`, `markdown_*`, `semantic_breakpoint_*`, `proposition_*`, `contextual_*`, `late_chunking_overlap` and the rest) are read from the `[rag]` section of `settings.lasagna.toml`, and a sweep arm can set any of them. In a sweep, every chunking, embedding and document-processing setting gets its own corpus, so an arm that changes only one of them is measured on its own chunks.

## Design Decisions

- **Lazy loading in `__init__.py`:** Uses `__getattr__` to defer imports of heavy dependencies (PyPDF2, EasyOCR, Docling ~6 GB) until actually needed. When Docling is the active parser, legacy extractors are never imported.

- **ABC adapter pattern:** Static utility classes (`DocumentProcessor`, `SemanticChunker`) are wrapped as ABC-compliant adapters (`DocumentProcessorAdapter`, `SemanticChunkerAdapter`) for the component registry. This allows researchers to swap implementations via `registry.register()` without modifying pipeline code.

- **Content-hash chunk IDs:** SHA-256 hash of chunk content provides stable, deterministic citation IDs that survive re-indexing. Same content always produces the same citation reference.

- **Fallback chains:** OCR engines (EasyOCR → Tesseract → PaddleOCR), PDF extractors (pypdf → pdfplumber), and sentence tokenizers (NLTK → regex) all have automatic fallbacks. This maximizes compatibility across deployment environments.

- **Singleton for DoclingProcessor:** AI models (~2-6 GB) are loaded once and shared across all requests via thread-safe singleton with double-checked locking. The converter is thread-safe for inference.

- **Incremental PDF OCR:** Pages are converted to images one at a time instead of all at once, preventing OOM on large (1000+ page) documents.

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| `loguru` | Yes | Structured logging |
| `numpy` | Yes | Embedding array ops |
| `pypdf` | Yes | PDF text extraction (fallback: `pdfplumber`) |
| `python-docx` | Yes | DOCX extraction |
| `beautifulsoup4` | Yes | HTML extraction |
| `nltk` | Yes | Sentence tokenization; its `punkt_tab` data is a separate one-time download, and without it sentences are split with a regex |
| `easyocr` | Optional | GPU-accelerated OCR |
| `pytesseract` | Optional | Traditional OCR |
| `paddleocr` | Optional | Multilingual OCR |
| `pillow` | Optional | Image processing for OCR |
| `pdf2image` | Optional | PDF-to-image conversion |
| `docling` | Optional | AI document understanding (~2-6 GB models); the `docling` extra |
| `pdfplumber` | Yes | PDF fallback extraction |
| `markitdown` | Optional | The `markitdown` processor; the `markitdown` extra |
| `pymupdf4llm` | Optional | The `pymupdf4llm` processor (AGPL-3.0); the `pymupdf4llm` extra |
| `tiktoken` | Optional | `chunk_tokenizer = "tiktoken:<encoding>"` with `chunk_unit = "tokens"`; a Hugging Face tokenizer name needs nothing extra |
