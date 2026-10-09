"""Processing layer for RAG Knowledge Base system.

This layer handles input processing including:
- Document extraction from various formats
- OCR for scanned documents
- Semantic text chunking
- Contextual retrieval enrichment
"""

# Chunking subsystem (always needed)
from .chunker import (
    SemanticChunker,
    ChunkingStrategy,
    Chunk,
    chunk_document,
)

# Legacy document/OCR processors and contextual enricher are lazy-imported
# to avoid triggering dependency probes (PyPDF2, easyocr, etc.) when Docling
# is the active document parser.
_LAZY_IMPORTS = {
    "DocumentProcessor": (".document_processor", "DocumentProcessor"),
    "OCRProcessor": (".ocr_processor", "OCRProcessor"),
    "get_ocr_processor": (".ocr_processor", "get_ocr_processor"),
    "is_scanned_pdf": (".ocr_processor", "is_scanned_pdf"),
    "ContextualEnricher": (".contextual.contextual_enricher", "ContextualEnricher"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_path, attr = _LAZY_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path, __name__)
        val = getattr(mod, attr)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Document processing
    "DocumentProcessor",
    # OCR
    "OCRProcessor",
    "get_ocr_processor",
    "is_scanned_pdf",
    # Chunking
    "SemanticChunker",
    "ChunkingStrategy",
    "Chunk",
    "chunk_document",
    # Contextual enrichment
    "ContextualEnricher",
]
