"""An unsupported file is refused under its own extension, and only files that
something can extract are advertised as supported.

The MIME fallback in ``detect_file_type`` reused the name ``ext`` as its loop
variable, so every refusal named the last supported format: a ``.doc`` upload
failed with "Unsupported file type: .docx". The settings and the MCP server
also advertised ``.doc``, ``.rtf`` and ``.epub``, which no extractor handles,
and ``.pptx`` and ``.xlsx``, which the upload check refuses before Docling can
see them.
"""

from __future__ import annotations

import pytest

from lasagna.core.config import get_rag_config
from lasagna.processing.document_processor import DocumentProcessor


@pytest.mark.parametrize("name", ["a.doc", "notes.rtf", "book.epub"])
def test_a_refusal_names_the_files_own_extension(tmp_path, name):
    path = tmp_path / name
    path.write_text("hello")
    with pytest.raises(ValueError) as err:
        DocumentProcessor.detect_file_type(str(path))
    assert f"Unsupported file type: {path.suffix}" in str(err.value)


def test_every_advertised_extension_can_be_extracted():
    advertised = set(get_rag_config().supported_extensions)
    extractable = set(DocumentProcessor.SUPPORTED_FORMATS)
    assert advertised <= extractable, sorted(advertised - extractable)


def test_the_production_settings_advertise_only_what_can_be_extracted(tmp_path):
    """The production settings carry no list, so the code default applies."""
    import os
    import subprocess
    import sys

    from tests._paths import PACKAGE_DIR

    probe = (
        "from lasagna.core.config import get_rag_config;"
        "from lasagna.processing.document_processor import DocumentProcessor as D;"
        "print(sorted(set(get_rag_config().supported_extensions) - set(D.SUPPORTED_FORMATS)))"
    )
    env = {
        **os.environ,
        "RAG_SETTINGS_FILE": str(PACKAGE_DIR / "settings.lasagna_production.toml"),
        "RAG_DB_DIR": str(tmp_path / "dbs"),
        "RAG_UPLOAD_DIR": str(tmp_path / "up"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=env, timeout=300
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip().splitlines()[-1] == "[]"
