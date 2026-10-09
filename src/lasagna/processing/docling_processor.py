"""Document processing via Docling — structure-aware extraction and chunking.

Provides a drop-in replacement for the legacy extract→chunk pipeline,
returning the same (chunks_with_metadata, parent_chunks_data) format
consumed by DocumentPipeline.process_document().

Docling is an optional dependency.  When not installed, is_docling_available()
returns False and the pipeline falls back to the legacy path transparently.
"""

import hashlib
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from lasagna.core.config import RAGConfig, rag_config

# ---------------------------------------------------------------------------
# Availability check (cached)
# ---------------------------------------------------------------------------

_docling_available: Optional[bool] = None


def _convert_with_timeout(converter: Any, file_path: str, timeout: float) -> Any:
    """``converter.convert(file_path)``, or TimeoutError after ``timeout`` seconds.

    Runs on a daemon thread, which cannot be stopped: a conversion that times
    out keeps running in the background, but ingestion moves on and the process
    can still exit.
    """
    if timeout <= 0:
        return converter.convert(file_path)
    box: Dict[str, Any] = {}

    def run() -> None:
        try:
            box["result"] = converter.convert(file_path)
        except BaseException as exc:  # handed to the caller below
            box["error"] = exc

    worker = threading.Thread(target=run, name="docling-convert", daemon=True)
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise TimeoutError(
            f"Docling did not convert {os.path.basename(file_path)} within "
            f"{timeout:.0f}s (docling_timeout)"
        )
    if "error" in box:
        raise box["error"]
    return box["result"]


def is_docling_available() -> bool:
    """Check whether the ``docling`` package is importable (result is cached)."""
    global _docling_available
    if _docling_available is None:
        try:
            import docling  # noqa: F401

            _docling_available = True
        except ImportError:
            _docling_available = False
    return _docling_available


def _reset_availability_cache() -> None:
    """Reset the cached import check (for testing only)."""
    global _docling_available
    _docling_available = None


# ---------------------------------------------------------------------------
# DoclingProcessor
# ---------------------------------------------------------------------------


class DoclingProcessor:
    """Structure-aware document processing via Docling.

    Singleton — AI models (~2-6 GB) are loaded once and shared across all
    requests.  The converter is thread-safe for inference.

    Deployment modes (``config.docling_mode``):
        ``"local"``      – pypdfium2 backend, CPU-only, conservative limits.
        ``"production"`` – docling_parse backend, GPU if available.
        ``"auto"``       – picks ``"production"`` when CUDA is detected,
                           otherwise falls back to ``"local"``.
    """

    _instance: Optional["DoclingProcessor"] = None
    _lock = threading.Lock()

    # -- singleton lifecycle -------------------------------------------------

    @classmethod
    def get_instance(cls, config: Optional[RAGConfig] = None) -> "DoclingProcessor":
        """Return the shared processor instance, creating it on first call."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(config)
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Discard the singleton (useful in tests)."""
        with cls._lock:
            cls._instance = None

    # Maximum distinct max_tokens values to cache chunkers for
    _MAX_CHUNKER_CACHE_SIZE = 16

    def __init__(self, config: Optional[RAGConfig] = None):
        self._config = config or rag_config
        self._converter = None  # lazy — built on first use
        self._chunker_cache: Dict[int, Any] = {}  # keyed by max_tokens
        self._init_lock = threading.Lock()  # guards lazy init of converter/chunker

    # -- converter (lazy) ----------------------------------------------------

    def _get_converter(self):
        if self._converter is None:
            with self._init_lock:
                if self._converter is None:
                    self._converter = self._build_converter()
        return self._converter

    def _build_converter(self):
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
        )

        mode = self._resolve_mode()
        logger.info(f"Building Docling converter (mode={mode})")

        pipeline_opts = PdfPipelineOptions()
        pipeline_opts.do_ocr = self._config.docling_ocr_enabled
        pipeline_opts.do_table_structure = self._config.docling_table_structure

        if self._config.docling_table_structure:
            pipeline_opts.table_structure_options = TableStructureOptions(
                do_cell_matching=True,
            )

        if mode == "production":
            from docling.backend.docling_parse_backend import (
                DoclingParseDocumentBackend,
            )

            backend_cls = DoclingParseDocumentBackend
            self._apply_gpu_opts(pipeline_opts)
        else:
            from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

            backend_cls = PyPdfiumDocumentBackend

        return DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_opts,
                    backend=backend_cls,
                ),
            }
        )

    def _resolve_mode(self) -> str:
        """Map ``docling_mode`` config to a concrete mode string."""
        mode = self._config.docling_mode
        if mode != "auto":
            return mode
        try:
            import torch

            return "production" if torch.cuda.is_available() else "local"
        except ImportError:
            return "local"

    @staticmethod
    def _apply_gpu_opts(pipeline_opts) -> None:
        """Attach CUDA accelerator options when a GPU is available."""
        try:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice,
                AcceleratorOptions,
            )
            import torch

            if torch.cuda.is_available():
                pipeline_opts.accelerator_options = AcceleratorOptions(
                    device=AcceleratorDevice.CUDA,
                )
                logger.info("Docling: CUDA GPU acceleration enabled")
        except (ImportError, Exception):
            pass  # CPU fallback — no error

    # -- chunker -------------------------------------------------------------

    def _get_chunker(self, max_tokens: int):
        if max_tokens not in self._chunker_cache:
            with self._init_lock:
                if max_tokens not in self._chunker_cache:
                    # Evict oldest entry if cache is full
                    if len(self._chunker_cache) >= self._MAX_CHUNKER_CACHE_SIZE:
                        oldest_key = next(iter(self._chunker_cache))
                        del self._chunker_cache[oldest_key]
                    from docling.chunking import HybridChunker
                    from docling_core.transforms.chunker.tokenizer.huggingface import (
                        HuggingFaceTokenizer,
                    )

                    tokenizer = HuggingFaceTokenizer.from_pretrained(
                        model_name="sentence-transformers/all-MiniLM-L6-v2",
                        max_tokens=max_tokens,
                    )
                    self._chunker_cache[max_tokens] = HybridChunker(
                        tokenizer=tokenizer,
                    )
        return self._chunker_cache[max_tokens]

    # -- public API ----------------------------------------------------------

    def process(
        self,
        file_path: str,
        document_id: str,
        session_id: str,
        filename: str,
        config: Optional[RAGConfig] = None,
    ) -> Tuple[
        List[Tuple[str, Dict[str, Any]]],
        Optional[List[Dict[str, Any]]],
        str,
    ]:
        """Process a document through Docling and return chunked results.

        Returns:
            A 3-tuple ``(chunks_with_metadata, parent_chunks_data, full_text)``
            matching the contract of ``chunk_document()`` in *semantic.py*:

            * ``chunks_with_metadata`` – ``[(chunk_text, metadata_dict), ...]``
            * ``parent_chunks_data`` – always ``None`` (Docling manages hierarchy
              internally via its HybridChunker).
            * ``full_text`` – Markdown export of the full document (used for
              preview generation upstream).
        """
        cfg = config or self._config
        converter = self._get_converter()
        max_tokens = cfg.docling_max_tokens

        # Convert document, bounded by docling_timeout: one pathological PDF
        # used to stall ingestion with no limit, and the setting was read by
        # nothing. On timeout the caller falls back to the legacy pipeline.
        result = _convert_with_timeout(
            converter, file_path, float(getattr(cfg, "docling_timeout", 0) or 0)
        )
        dl_doc = result.document

        # Export full text for preview / fallback
        full_text = dl_doc.export_to_markdown()
        if not full_text or not full_text.strip():
            raise ValueError("Docling extracted no content from document")

        # Chunk using structure-aware HybridChunker
        chunker = self._get_chunker(max_tokens)
        chunks = list(chunker.chunk(dl_doc=dl_doc))

        if not chunks:
            raise ValueError("Docling produced no chunks from document")

        # Build output in the same format as chunk_document()
        chunks_with_metadata: List[Tuple[str, Dict[str, Any]]] = []
        for idx, chunk in enumerate(chunks):
            text = chunker.contextualize(chunk)
            content_hash = hashlib.sha256(text.encode()).hexdigest()
            stable_id = content_hash[: cfg.citation_id_length]

            headings = getattr(chunk.meta, "headings", []) or []
            section_header = headings[-1] if headings else None

            # Extract provenance from Docling's native doc_items
            page_number, start_char, end_char = self._extract_provenance(chunk)

            metadata: Dict[str, Any] = {
                "document_id": document_id,
                "session_id": session_id,
                "filename": filename,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                "stable_id": stable_id,
                "chunk_type": self._detect_chunk_type(chunk),
                "start_char": start_char,
                "end_char": end_char,
                "line_start": 0,
                "line_end": 0,
                "processor": "docling",
            }

            if page_number is not None:
                metadata["page_number"] = page_number

            if section_header:
                metadata["section_header"] = section_header
            if headings:
                metadata["headings"] = headings

            chunks_with_metadata.append((text, metadata))

        logger.info(f"Docling processed {filename}: {len(chunks_with_metadata)} chunks")
        return chunks_with_metadata, None, full_text

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _extract_provenance(chunk) -> Tuple[Optional[int], int, int]:
        """Extract page number and character span from Docling's native provenance.

        Each ``doc_item`` carries a list of ``ProvenanceItem`` objects populated
        by Docling's document converter during AI layout analysis.  We read
        ``page_no`` and ``charspan`` directly — no heuristics.

        Returns:
            ``(page_number, start_char, end_char)`` from Docling's ``prov`` data.
        """
        doc_items = getattr(getattr(chunk, "meta", None), "doc_items", []) or []
        pages: set = set()
        min_char: float = float("inf")
        max_char: int = 0

        for item in doc_items:
            for prov in getattr(item, "prov", []):
                page_no = getattr(prov, "page_no", None)
                if page_no is not None:
                    pages.add(page_no)
                charspan = getattr(prov, "charspan", None)
                if charspan is None:
                    continue
                # Handle Docling's CharSpan Pydantic model (offset+length)
                # as well as plain tuples/lists
                if hasattr(charspan, "offset"):
                    start = charspan.offset
                    end = start + getattr(charspan, "length", 0)
                elif isinstance(charspan, (list, tuple)) and len(charspan) >= 2:
                    start, end = charspan[0], charspan[1]
                else:
                    continue
                min_char = min(min_char, start)
                max_char = max(max_char, end)

        page_number = min(pages) if pages else None
        start_char = int(min_char) if min_char != float("inf") else 0
        end_char = int(max_char) if max_char > 0 else 0

        return page_number, start_char, end_char

    @staticmethod
    def _detect_chunk_type(chunk) -> str:
        """Map a Docling chunk to a Lasagna chunk_type string."""
        doc_items = getattr(getattr(chunk, "meta", None), "doc_items", []) or []
        for item in doc_items:
            label = getattr(item, "label", None)
            if label is None:
                continue
            label_name = label.name if hasattr(label, "name") else str(label)
            label_lower = label_name.lower()
            if "table" in label_lower:
                return "table"
            if "code" in label_lower:
                return "code"
            if "picture" in label_lower or "figure" in label_lower:
                return "image"
            if "header" in label_lower or "title" in label_lower:
                return "header"
            if "formula" in label_lower:
                return "formula"
        return "text"


# ---------------------------------------------------------------------------
# ABC-compliant component
# ---------------------------------------------------------------------------

from lasagna.core.base_components import (  # noqa: E402
    BaseDocumentProcessor,
    ProcessedDocument,
)


class DoclingDocumentProcessor(BaseDocumentProcessor):
    """Docling: layout analysis, TableFormer tables and OCR, exported as Markdown.

    The same converter the pipeline uses when ``docling_enabled`` is on (the
    process-wide ``DoclingProcessor``, built from the first config it sees),
    run for extraction only: the Markdown it exports goes to the arm's chunker,
    so a comparison of processors is not also a comparison of chunkers. That
    is the difference from the built-in Docling path, which chunks with
    Docling's own HybridChunker.

    Settings: ``docling_mode``, ``docling_ocr_enabled``,
    ``docling_table_structure``, ``docling_timeout``. Needs
    ``pip install 'lasagna-rag[docling]'``. Reference: Auer et al., "Docling
    Technical Report", arXiv:2408.09869.
    """

    # Reads the arm's docling_* settings.
    _forwards_session_config = True

    #: Formats Docling converts.
    EXTENSIONS = [
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".html",
        ".htm",
        ".md",
        ".csv",
        ".adoc",
        ".png",
        ".jpg",
        ".jpeg",
        ".tif",
        ".tiff",
        ".bmp",
    ]

    def __init__(self, config: Optional[RAGConfig] = None, **kwargs):
        self._config = config

    @property
    def name(self) -> str:
        return "docling"

    @property
    def supported_extensions(self) -> List[str]:
        return list(self.EXTENSIONS)

    def extract(self, file_path: str, **kwargs) -> ProcessedDocument:
        if not is_docling_available():
            raise ImportError(
                "the docling processor needs Docling: pip install 'lasagna-rag[docling]'"
            )
        cfg = kwargs.get("config") or self._config or rag_config
        processor = DoclingProcessor.get_instance(cfg)
        result = _convert_with_timeout(
            processor._get_converter(), file_path, float(getattr(cfg, "docling_timeout", 0) or 0)
        )
        doc = result.document
        text = doc.export_to_markdown()
        tables = []
        for table in getattr(doc, "tables", None) or []:
            try:
                tables.append(table.export_to_markdown(doc=doc))
            except TypeError:  # docling-core before tables took the document
                tables.append(table.export_to_markdown())
        return ProcessedDocument(
            text=text,
            metadata={"source": file_path, "processor": "docling"},
            tables=tables or None,
        )
