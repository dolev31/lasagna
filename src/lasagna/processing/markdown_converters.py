"""Document processors that convert files to Markdown: MarkItDown and PyMuPDF4LLM.

Both are optional dependencies, imported when a file is first extracted; a
missing one raises naming the install. Markdown keeps headings, lists and
tables as structure the ``markdown_header`` chunker can split on.
"""

from __future__ import annotations

from typing import Any, List

from lasagna.core.base_components import BaseDocumentProcessor, ProcessedDocument


class MarkItDownProcessor(BaseDocumentProcessor):
    """Microsoft MarkItDown: PDF, Office, HTML, CSV, JSON, XML and more, to Markdown.

    Plugins stay off, so a result does not depend on what else is installed.
    ``metadata["title"]`` carries the document title when MarkItDown finds one.
    Needs ``pip install 'lasagna-rag[markitdown]'``.
    """

    #: Formats MarkItDown converts without an LLM or a speech model.
    EXTENSIONS = [
        ".pdf",
        ".docx",
        ".pptx",
        ".xlsx",
        ".xls",
        ".html",
        ".htm",
        ".csv",
        ".json",
        ".xml",
        ".txt",
        ".md",
        ".epub",
        ".ipynb",
    ]

    def __init__(self, **kwargs: Any):
        self._converter = None

    @property
    def name(self) -> str:
        return "markitdown"

    @property
    def supported_extensions(self) -> List[str]:
        return list(self.EXTENSIONS)

    def _markitdown(self) -> Any:
        if self._converter is None:
            try:
                from markitdown import MarkItDown
            except ImportError as exc:
                raise ImportError(
                    "the markitdown processor needs MarkItDown: "
                    "pip install 'lasagna-rag[markitdown]'"
                ) from exc
            try:
                self._converter = MarkItDown(enable_plugins=False)
            except TypeError:  # releases before plugins existed
                self._converter = MarkItDown()
        return self._converter

    def extract(self, file_path: str, **kwargs: Any) -> ProcessedDocument:
        result = self._markitdown().convert(file_path)
        text = getattr(result, "markdown", None) or result.text_content
        metadata = {"source": file_path, "processor": "markitdown"}
        if getattr(result, "title", None):
            metadata["title"] = result.title
        return ProcessedDocument(text=text, metadata=metadata)


class PyMuPDF4LLMProcessor(BaseDocumentProcessor):
    """PyMuPDF4LLM: PDF to Markdown with headers from font sizes, and tables.

    Converts page by page; ``pages`` holds each page's Markdown and ``text``
    their concatenation, which is what ``pymupdf4llm.to_markdown`` returns for
    the whole file. PyMuPDF and PyMuPDF4LLM are AGPL-3.0 (or commercially
    licensed), which is why they are an extra of their own and not part of
    ``all``. Needs ``pip install 'lasagna-rag[pymupdf4llm]'``.
    """

    EXTENSIONS = [".pdf"]

    def __init__(self, **kwargs: Any):
        pass

    @property
    def name(self) -> str:
        return "pymupdf4llm"

    @property
    def supported_extensions(self) -> List[str]:
        return list(self.EXTENSIONS)

    def extract(self, file_path: str, **kwargs: Any) -> ProcessedDocument:
        try:
            import pymupdf4llm
        except ImportError as exc:
            raise ImportError(
                "the pymupdf4llm processor needs PyMuPDF4LLM (AGPL-3.0): "
                "pip install 'lasagna-rag[pymupdf4llm]'"
            ) from exc
        pages = pymupdf4llm.to_markdown(file_path, page_chunks=True)
        texts = [page["text"] for page in pages]
        return ProcessedDocument(
            text="".join(texts),
            metadata={"source": file_path, "processor": "pymupdf4llm", "page_count": len(texts)},
            pages=texts,
        )
