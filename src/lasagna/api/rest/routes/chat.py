"""Chat completions in the OpenAI format, over a session's retrieval pipeline (#13).

- ``POST /api/v1/chat/completions``: search the session as ``/search/citations``
  does, then answer with the `cited` generator, at once or streamed as
  server-sent events.
- ``GET /api/v1/models``: the one model a client can name.

Any OpenAI client library or chat front end can then use a Lasagna session
with no glue code: point its base URL at ``.../api/v1`` and send the session
header. The session decides what is searched and how; ``model`` is echoed.
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from loguru import logger

from lasagna.api.rag_backend import RAGBackend
from lasagna.generation.strategies import cited_prompt
from lasagna.utils import degraded as _degraded
from ..dependencies import get_backend, get_session_id, get_user_id
from ..models import (
    ChatCompletionChoice,
    ChatCompletionMessage,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionUsage,
    CitationResponse,
    ErrorResponse,
    ModelCard,
    ModelList,
)
from .search import (
    _missing_llm,
    _nothing_to_search,
    _remember_citations,
    _within_deadline,
    citation_to_response,
    search_http_error,
)

router = APIRouter(tags=["chat"])

#: The one model ``GET /models`` lists. Any name is accepted and echoed.
MODEL_ID = "lasagna"

#: Answered without calling the LLM when the search found nothing: an answer
#: written from no passage would be the model's own, under Lasagna's name.
NO_PASSAGE = "No passage in this session matches the question."

_CHAT_ERRORS = {
    400: {
        "model": ErrorResponse,
        "description": "The request asks for what this route does not serve",
    },
    502: {"model": ErrorResponse, "description": "The LLM provider failed while answering"},
    503: {
        "model": ErrorResponse,
        "description": "No LLM is configured, a model is not downloaded, or the focus is unreadable",
    },
    504: {"model": ErrorResponse, "description": "The search ran past RAG_SEARCH_TIMEOUT"},
}


@router.get("/models", response_model=ModelList)
async def list_models() -> ModelList:
    """The model OpenAI clients may name; what is searched comes from the session."""
    return ModelList(data=[ModelCard(id=MODEL_ID)])


def _refuse_unserved(body: ChatCompletionRequest) -> None:
    """Refuse what would change what a client expects back, rather than drop it."""
    if body.n not in (None, 1):
        raise HTTPException(status_code=400, detail="Only n=1 is served: one answer per request.")
    if body.tools:
        raise HTTPException(status_code=400, detail="Tool calls are not served by this route.")
    if body.response_format not in (None, {"type": "text"}):
        raise HTTPException(
            status_code=400, detail="Only text answers are served (response_format)."
        )


def _question_and_instructions(body: ChatCompletionRequest) -> tuple:
    """The last user message, and the system messages joined as instructions."""
    question = next((m.text() for m in reversed(body.messages) if m.role == "user"), "")
    if not question.strip():
        raise HTTPException(
            status_code=400, detail="The conversation has no user message with text."
        )
    instructions = "\n".join(m.text() for m in body.messages if m.role == "system").strip()
    return question.strip(), instructions


def _chat_model(body: ChatCompletionRequest) -> Any:
    """The session's LLM at the request's temperature and length, or 503 when there is none."""
    from lasagna.providers import get_llm_provider

    kwargs: Dict[str, Any] = {}
    if body.temperature is not None:
        kwargs["temperature"] = body.temperature
    if body.max_tokens is not None:
        kwargs["max_tokens"] = body.max_tokens
    try:
        return get_llm_provider().get_llm(**kwargs)
    except Exception as exc:
        missing = _missing_llm(exc)
        if missing:
            raise HTTPException(status_code=503, detail=missing) from exc
        logger.error(f"Chat completion: the LLM could not be built: {exc}")
        raise HTTPException(status_code=502, detail="The LLM provider could not be reached.")


def _text(message: Any) -> str:
    """A chat model's reply or chunk as text, whether its content is a string or parts."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return content if isinstance(content, str) else str(content or "")


def _usage(message: Any) -> Optional[ChatCompletionUsage]:
    """Token counts when the provider reported them (LangChain's ``usage_metadata``)."""
    meta = getattr(message, "usage_metadata", None) or {}
    try:
        prompt, completion = int(meta["input_tokens"]), int(meta["output_tokens"])
    except (KeyError, TypeError, ValueError):
        return None
    return ChatCompletionUsage(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=prompt + completion
    )


def _event(payload: Dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/completions", response_model=ChatCompletionResponse, responses=_CHAT_ERRORS)
async def chat_completions(
    body: ChatCompletionRequest,
    session_id: str = Depends(get_session_id),
    user_id: str = Depends(get_user_id),
    backend: RAGBackend = Depends(get_backend),
):
    """Answer the last user message from the session's documents, in the OpenAI format.

    The search is the session's own, as on ``/search/citations``, under the
    same deadline, quota and errors. The `cited` generator writes the answer,
    and ``citations`` lists the passages it read: ``[n]`` in the answer is
    ``citations[n-1]``. ``degraded_stages`` names any search stage that fell
    back. With ``stream: true`` the answer arrives as ``chat.completion.chunk``
    events, the last of which carries the citations, then ``data: [DONE]``.
    A system message is passed to the generator as instructions; earlier turns
    are not used. A session with no documents, or a search that found nothing,
    is answered with a note and no LLM call.
    """
    _refuse_unserved(body)
    question, instructions = _question_and_instructions(body)
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    cited: List[CitationResponse] = []
    degraded: List[str] = []
    fixed: Optional[str] = _nothing_to_search(session_id)
    if fixed is None:
        try:
            with _degraded.collecting() as degraded:
                results, citations, _context = await _within_deadline(
                    backend.async_search_with_citations(
                        query=question,
                        session_id=session_id,
                        top_k=body.top_k,
                        document_ids=body.document_ids,
                    )
                )
        except Exception as exc:
            raise search_http_error(exc, session_id, "Chat completion search")
        cited = [citation_to_response(c) for c in citations]
        _remember_citations(session_id, user_id, cited, results)
        if not cited:
            fixed = NO_PASSAGE

    llm = None
    prompt = ""
    if fixed is None:
        llm = _chat_model(body)
        prompt = cited_prompt(question, [c.content for c in cited], instructions)

    if body.stream:
        return StreamingResponse(
            _stream(completion_id, created, body.model, llm, prompt, fixed, cited, degraded),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    usage = None
    if fixed is not None:
        answer = fixed
    else:
        try:
            reply = await llm.ainvoke(prompt)
        except Exception as exc:
            logger.error(f"Chat completion: the LLM provider failed: {exc}")
            raise HTTPException(status_code=502, detail="The LLM provider failed while answering.")
        answer = _text(reply).strip()
        usage = _usage(reply)
    return ChatCompletionResponse(
        id=completion_id,
        created=created,
        model=body.model,
        choices=[ChatCompletionChoice(message=ChatCompletionMessage(content=answer))],
        usage=usage,
        citations=cited,
        degraded_stages=degraded,
    )


async def _stream(
    completion_id: str,
    created: int,
    model: str,
    llm: Any,
    prompt: str,
    fixed: Optional[str],
    cited: List[CitationResponse],
    degraded: List[str],
) -> AsyncIterator[str]:
    """``chat.completion.chunk`` events, the citations on the last, then ``[DONE]``."""

    def chunk(delta: Dict[str, Any], finish_reason: Optional[str] = None, **extra) -> str:
        return _event(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
                **extra,
            }
        )

    yield chunk({"role": "assistant", "content": ""})
    try:
        if fixed is not None:
            yield chunk({"content": fixed})
        else:
            async for part in llm.astream(prompt):
                text = _text(part)
                if text:
                    yield chunk({"content": text})
    except Exception as exc:
        # The status line has been sent: the error goes in the stream, in the
        # shape OpenAI clients raise from.
        logger.error(f"Chat completion: the LLM provider failed while streaming: {exc}")
        yield _event(
            {
                "error": {
                    "message": "The LLM provider failed while answering.",
                    "type": "upstream_error",
                }
            }
        )
    else:
        yield chunk(
            {},
            finish_reason="stop",
            citations=[c.model_dump() for c in cited],
            degraded_stages=list(degraded),
        )
    yield "data: [DONE]\n\n"
