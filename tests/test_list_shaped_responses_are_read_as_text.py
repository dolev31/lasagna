"""A model reply whose content is a list of blocks is read as its text.

Anthropic models with thinking or tool blocks, and OpenAI's Responses API,
return ``AIMessage.content`` as a list of blocks, not a string. Code that called
``.strip()`` on it crashed with "'list' object has no attribute 'strip'".
"""

from __future__ import annotations

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from lasagna.utils.model_utils import message_text

_BLOCKS = [{"type": "thinking", "thinking": "let me see"}, {"type": "text", "text": "Paris"}]


@pytest.mark.parametrize(
    "content,expected",
    [
        ("Paris", "Paris"),
        (_BLOCKS, "Paris"),
        (["Par", {"type": "text", "text": "is"}], "Paris"),
        ([], ""),
        (None, ""),
    ],
)
def test_message_text(content, expected):
    message = AIMessage(content="") if content is None else AIMessage(content=content)
    if content is None:
        message.content = None  # type: ignore[assignment]
    assert message_text(message) == expected


def test_a_plain_string_reply_is_its_own_text():
    assert message_text("already text") == "already text"


def _llm(content):
    return FakeMessagesListChatModel(responses=[AIMessage(content=content)])


async def test_chunk_refinement_reads_a_list_shaped_reply():
    from lasagna.agentic.agentic_rag import refine_chunk

    result = await refine_chunk(
        query="capital of France",
        chunk_content="Paris is the capital of France.",
        source_name="atlas.txt",
        chunk_index=0,
        llm=_llm([{"type": "text", "text": "  Paris is the capital.  "}]),
    )
    assert result[1] == "Paris is the capital."


async def test_crag_grading_reads_a_list_shaped_reply():
    from lasagna.agentic.agentic_rag import _grade_single_document

    index, grade = await _grade_single_document(
        query="q", document_text="d", doc_index=3, llm=_llm([{"type": "text", "text": "Correct"}])
    )
    assert (index, grade) == (3, "correct")
