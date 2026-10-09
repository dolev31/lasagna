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

### Document Processor (`document_processor.py`)

Multi-format text extraction with fallback strategy.

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

AI-powered document understanding via IBM Docling (MIT-licensed).

| Mode | Backend | RAM | GPU |
|------|---------|-----|-----|
| `local` | pypdfium2 | ~2.5 GB | No |
| `production` | docling_parse | ~6 GB | Yes (CUDA) |
| `auto` (default) | auto-detect | varies | if available |

Adds over basic extraction:
- **Layout analysis** — headers, sections, page structure
- **TableFormer** — 97.9% accuracy table extraction
- **Formula detection** — mathematical expressions
- **Structure-aware chunking** via HybridChunker
- Falls back to legacy pipeline if Docling is not installed

### Chunking Strategies (`chunker/semantic.py`)

| Strategy | When to use | Key param |
|----------|-------------|-----------|
| `semantic` | Default — splits at meaning boundaries | `semantic_threshold` |
| `fixed` | Fastest, no embedding needed | `chunk_size`, `chunk_overlap` |
| `sentence_window` | When sentence-level context matters | `window_size` |
| `hierarchical` | Best retrieval quality — parent context + child search | `parent_chunk_size`, `child_chunk_size` |
| structure-aware | Documents with headers/tables/code | `preserve_tables`, `preserve_headers` |

**Hierarchical chunking:** Creates large parent chunks (context) and small child chunks (search units). When enough child chunks from the same parent appear in results, they are merged back into the parent for richer LLM context (auto-merge retrieval, based on LlamaIndex).

**Stable IDs:** Every chunk gets a SHA-256 content-based hash for deterministic citation tracking across re-indexing.

### Contextual Enricher (`contextual/contextual_enricher.py`)

Implements Anthropic's [contextual retrieval](https://www.anthropic.com/news/contextual-retrieval) technique.

- Generates a short LLM context describing where each chunk fits in the document
- Prepends context to chunk before embedding
- Reduces retrieval failure by ~67% (Anthropic, Sep 2024)
- LRU cache (default 10K entries) avoids re-computation
- Supports sync and async enrichment with concurrency control
- Graceful degradation when no LLM is configured

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
| `pypdf` | Optional | PDF text extraction (fallback: `pdfplumber`) |
| `python-docx` | Optional | DOCX extraction |
| `beautifulsoup4` | Optional | HTML extraction |
| `nltk` | Optional | Sentence tokenization (fallback: regex) |
| `easyocr` | Optional | GPU-accelerated OCR |
| `pytesseract` | Optional | Traditional OCR |
| `paddleocr` | Optional | Multilingual OCR |
| `pillow` | Optional | Image processing for OCR |
| `pdf2image` | Optional | PDF-to-image conversion |
| `docling` | Optional | AI document understanding (~2-6 GB models) |
| `pdfplumber` | Optional | PDF fallback extraction |
