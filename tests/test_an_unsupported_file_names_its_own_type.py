"""An unsupported file is reported with its own type.

`detect_file_type` reused its extension variable as a loop variable, so the
error named the last supported format: ``x.exe``, a directory and a file with
no extension were all "Unsupported file type: .docx", on REST, CLI and MCP.
"""

from __future__ import annotations

import pytest

from lasagna.processing.document_processor import DocumentProcessor


@pytest.mark.parametrize("name, shown", [("x.exe", ".exe"), ("README", "no extension")])
def test_the_error_names_the_files_own_type(tmp_path, name, shown):
    path = tmp_path / name
    path.write_bytes(b"\x00\x01")
    with pytest.raises(ValueError) as refused:
        DocumentProcessor.detect_file_type(str(path))
    message = str(refused.value)
    assert f"Unsupported file type: {shown}" in message
    assert ".pdf" in message  # and what would have worked
