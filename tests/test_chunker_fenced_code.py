"""Fenced code is not Markdown structure, and a rule is not a passage.

The structure-aware chunker, which every built-in profile runs, read any line
starting with ``#`` as a header, shell comments inside a fenced code block
included: a README's install snippet was cut in two at the comment and its
second half filed under a "section" named after it. A horizontal rule standing
alone became a chunk of its own, and searching the README for its own curl
command returned ``---`` first, at score 1.0.
"""

from __future__ import annotations

import pytest

from lasagna.core.config import rag_config
from lasagna.processing.chunker.semantic import (
    SemanticChunker,
    StructureAwareChunkerAdapter,
    chunk_document,
)

#: What the built-in profiles run: headers and tables kept, no hierarchy.
STRUCTURE_AWARE = {
    "enable_hierarchical": False,
    "preserve_headers": True,
    "preserve_tables": True,
}

README = (
    "# Install\n\n"
    "Install the package, then check the installation.\n\n"
    "```bash\n"
    "# create a virtualenv first\n"
    "pip install lasagna-rag\n"
    "lasagna doctor\n"
    "```\n\n"
    "## Search\n\n"
    "Search what you uploaded.\n"
)


def _chunks(text: str):
    arm = rag_config.with_query_overrides(STRUCTURE_AWARE)
    return chunk_document(text, "doc", "session", "README.md", config=arm)


def test_a_comment_in_a_code_block_is_not_a_header():
    chunks = _chunks(README)
    headers = [meta.get("section_header") for _, meta in chunks]
    assert "create a virtualenv first" not in headers, headers
    assert headers == ["Install", "Search"], headers

    install = chunks[0][0]
    for line in (
        "```bash",
        "# create a virtualenv first",
        "pip install lasagna-rag",
        "lasagna doctor",
    ):
        assert line in install, f"{line!r} was split away from its code block: {chunks}"
    assert install.rstrip().endswith("```"), "the closing fence left its block"


def test_the_component_form_masks_code_too():
    # No embedding model: the text between headers is then chunked by size.
    sc = SemanticChunker(config=rag_config.with_query_overrides(STRUCTURE_AWARE))
    chunks = StructureAwareChunkerAdapter(semantic_chunker=sc).chunk(README)
    assert [c.section_header for c in chunks] == ["Install", "Search"]


def test_a_real_header_after_a_code_block_still_splits():
    text = "# A\n\nText.\n\n```\nx = 1\n```\n\n# B\n\nMore.\n\n```py\n# c\n```\n"
    assert [m.get("section_header") for _, m in _chunks(text)] == ["A", "B"]


@pytest.mark.parametrize("rule", ["---", "***", "___", "===", "- - -", "|---|---|"])
def test_a_horizontal_rule_is_not_a_chunk(rule):
    text = f"# Intro\n\nOpening text.\n\n## Divider\n\n{rule}\n\n## After\n\nClosing text.\n"
    chunks = _chunks(text)
    contents = [content for content, _ in chunks]
    assert contents == ["Opening text.", "Closing text."], contents
    assert [m["chunk_index"] for _, m in chunks] == [0, 1]
    assert {m["total_chunks"] for _, m in chunks} == {2}


def test_text_around_a_rule_is_kept_whole():
    """Only a chunk that is nothing but rule characters goes."""
    text = "# Intro\n\nOpening text.\n\n---\n\nStill the intro.\n"
    (content,) = [c for c, _ in _chunks(text)]
    assert "Opening text." in content and "Still the intro." in content
