"""Document Processing Pipeline.

Provides document extraction, chunking, and OCR functionality.
Wraps the existing processing module.

Example:
    ```python
    from lasagna.sdk.processing import (
        SemanticChunker,
        DocumentExtractor,
        OCRProcessor,
    )

    # Extract text from document
    extractor = DocumentExtractor()
    text = extractor.extract("./document.pdf")

    # Chunk text
    chunker = SemanticChunker(chunk_size=800)
    chunks = chunker.chunk(text)

    # OCR for scanned documents
    ocr = OCRProcessor()
    text = ocr.process("./scanned.pdf")
    ```
"""

# Re-export from existing processing module
from ...processing.document_processor import DocumentProcessor as DocumentExtractor
from ...processing.ocr_processor import (
    OCRProcessor,
    get_ocr_processor,
    is_scanned_pdf,
)
from ...processing.chunker.semantic import (
    SemanticChunker,
    chunk_document,
)


class FixedChunker:
    """Simple fixed-size chunker with overlap.

    Splits text into chunks of fixed size with configurable overlap.

    Example:
        ```python
        chunker = FixedChunker(chunk_size=500, overlap=50)
        chunks = chunker.chunk(text)
        ```
    """

    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
    ):
        """Initialize fixed chunker.

        Args:
            chunk_size: Target chunk size in characters
            overlap: Overlap between chunks in characters
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[tuple[str, dict]]:
        """Chunk text into fixed-size pieces.

        Args:
            text: Text to chunk

        Returns:
            List of (chunk_text, metadata) tuples
        """
        if not text:
            return []

        chunks = []
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_size

            # Find a good break point (sentence/paragraph)
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self.chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", "! ", "? ", "\n"]:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + self.chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    (
                        chunk_text,
                        {
                            "chunk_index": chunk_index,
                            "char_start": start,
                            "char_end": end,
                        },
                    )
                )
                chunk_index += 1

            start = end - self.overlap

        return chunks


__all__ = [
    # Chunkers
    "SemanticChunker",
    "FixedChunker",
    "chunk_document",
    # Extraction
    "DocumentExtractor",
    # OCR
    "OCRProcessor",
    "get_ocr_processor",
    "is_scanned_pdf",
]
