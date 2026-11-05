"""An OpenAI client can chat with a Lasagna session (#13).

`POST /api/v1/chat/completions` searches the session as `/search/citations`
does and answers with the `cited` generator, in the OpenAI format, at once or
streamed. These tests drive the real `openai` client library against the app
in process, with a stub search and a stub chat model: no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lasagna.api.rest.dependencies import get_backend
from lasagna.api.rest.routes import chat, search
from lasagna.core import model_downloads
from lasagna.core.quota import QuotaExceededError
from lasagna.utils import degraded

openai = pytest.importorskip("openai")

SESSION = "chat-session"
_CITATION = SimpleNamespace(
    citation_id="c-1",
    document_id="d-1",
    document_name="tides.md",
    chunk_id="k-1",
    chunk_index=0,
    page_number=None,
    score=0.9,
    content="Tides are caused by the moon.",
    highlights=None,
    highlight_ranges=None,
)


class _Backend:
    def __init__(self, citations=(_CITATION,), fail=None, fall_back=False):
        self.citations = list(citations)
        self.fail = fail
        self.fall_back = fall_back
        self.calls = []

    async def async_search_with_citations(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail:
            raise self.fail
        if self.fall_back:
            degraded.record("hybrid", "BM25 index empty; dense results only")
        return [], self.citations, "context"


class _Chat:
    """A LangChain-like chat model that answers from the prompt it was given."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.prompts = []

    async def ainvoke(self, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(
            content="The moon causes tides [1].",
            usage_metadata={"input_tokens": 40, "output_tokens": 7},
        )

    async def astream(self, prompt):
        self.prompts.append(prompt)
        for piece in ("The moon ", "causes tides ", "[1]."):
            yield SimpleNamespace(content=piece)


@pytest.fixture
def served(monkeypatch):
    """The chat and search routers over a stub backend and a stub LLM."""
    import lasagna.providers as providers

    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {SESSION: 1})
    models = []

    def get_llm(**kwargs):
        models.append(_Chat(**kwargs))
        return models[-1]

    monkeypatch.setattr(providers, "get_llm_provider", lambda: SimpleNamespace(get_llm=get_llm))

    def serve(backend=None):
        app = FastAPI()
        app.include_router(chat.router, prefix="/api/v1")
        backend = backend or _Backend()
        app.dependency_overrides[get_backend] = lambda: backend
        http = TestClient(app)
        client = openai.OpenAI(
            base_url="http://testserver/api/v1",
            api_key="unused",
            http_client=http,
            default_headers={"X-Session-ID": SESSION},
            max_retries=0,
        )
        return SimpleNamespace(client=client, http=http, backend=backend, models=models)

    return serve


def _ask(client, **kwargs):
    return client.chat.completions.create(
        model="lasagna", messages=[{"role": "user", "content": "What causes tides?"}], **kwargs
    )


def test_an_openai_client_gets_an_answer_with_its_citations(served):
    s = served()
    reply = _ask(s.client)
    assert reply.object == "chat.completion"
    assert reply.choices[0].message.content == "The moon causes tides [1]."
    assert reply.choices[0].finish_reason == "stop"
    assert reply.usage.total_tokens == 47
    # Lasagna's own fields come through the client untouched.
    extra = reply.model_dump()
    assert extra["citations"][0]["content"] == "Tides are caused by the moon."
    assert extra["degraded_stages"] == []
    # The answer was written from the passage, numbered as citations[0].
    assert "Document [1]: Tides are caused by the moon." in s.models[0].prompts[0]
    assert s.backend.calls[0]["query"] == "What causes tides?"


def test_a_streamed_answer_arrives_in_chunks_and_ends_with_the_citations(served):
    s = served()
    chunks = list(_ask(s.client, stream=True))
    text = "".join(c.choices[0].delta.content or "" for c in chunks)
    assert text == "The moon causes tides [1]."
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert chunks[-1].model_dump()["citations"][0]["citation_id"] == "c-1"


def test_the_raw_stream_is_server_sent_events_that_end_with_done(served):
    s = served()
    with s.http.stream(
        "POST",
        "/api/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "tides?"}], "stream": True},
        headers={"X-Session-ID": SESSION},
    ) as response:
        assert response.headers["content-type"].startswith("text/event-stream")
        lines = [line for line in response.iter_lines() if line]
    assert all(line.startswith("data: ") for line in lines)
    assert lines[-1] == "data: [DONE]"
    assert json.loads(lines[0][6:])["choices"][0]["delta"]["role"] == "assistant"


def test_the_system_message_and_the_sampling_settings_reach_the_model(served):
    s = served()
    s.client.chat.completions.create(
        model="anything",
        messages=[
            {"role": "system", "content": "Answer in French."},
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": [{"type": "text", "text": "What causes tides?"}]},
        ],
        temperature=0.3,
        max_tokens=64,
        extra_body={"top_k": 3, "document_ids": ["d-1"]},
    )
    assert s.models[0].kwargs == {"temperature": 0.3, "max_tokens": 64}
    assert "Answer in French." in s.models[0].prompts[0]
    assert "Earlier" not in s.models[0].prompts[0]
    assert s.backend.calls[0]["query"] == "What causes tides?"
    assert s.backend.calls[0]["top_k"] == 3
    assert s.backend.calls[0]["document_ids"] == ["d-1"]


def test_a_search_that_fell_back_says_so(served):
    s = served(_Backend(fall_back=True))
    assert _ask(s.client).model_dump()["degraded_stages"] == ["hybrid"]
    last = list(_ask(s.client, stream=True))[-1]
    assert last.model_dump()["degraded_stages"] == ["hybrid"]


def test_nothing_found_is_said_without_asking_the_model(served):
    s = served(_Backend(citations=()))
    assert _ask(s.client).choices[0].message.content == chat.NO_PASSAGE
    assert s.models == []


def test_a_session_with_no_documents_gets_the_note_search_gives(served, monkeypatch):
    monkeypatch.setattr(model_downloads, "sessions_with_documents", lambda: {})
    s = served()
    content = _ask(s.client).choices[0].message.content
    assert content == search._nothing_to_search(SESSION)
    assert s.backend.calls == [] and s.models == []


@pytest.mark.parametrize(
    "extra, says",
    [
        ({"n": 2}, "n=1"),
        ({"tools": [{"type": "function", "function": {"name": "f"}}]}, "Tool calls"),
        ({"response_format": {"type": "json_object"}}, "text answers"),
    ],
)
def test_what_is_not_served_is_refused_not_ignored(served, extra, says):
    s = served()
    with pytest.raises(openai.BadRequestError, match=says):
        _ask(s.client, **extra)


def test_a_conversation_without_a_user_question_is_refused(served):
    s = served()
    with pytest.raises(openai.BadRequestError, match="no user message"):
        s.client.chat.completions.create(
            model="lasagna", messages=[{"role": "system", "content": "Be brief."}]
        )


def test_no_llm_is_a_503_that_says_how_to_configure_one(served, monkeypatch):
    import lasagna.providers as providers

    def _missing():
        raise providers.LLMProviderNotConfigured("No LLMProvider configured: set OPENAI_API_KEY")

    s = served()
    monkeypatch.setattr(providers, "get_llm_provider", _missing)
    with pytest.raises(openai.InternalServerError) as caught:
        _ask(s.client)
    assert caught.value.status_code == 503
    assert "No LLMProvider configured" in str(caught.value)


def test_a_quota_is_a_429_as_on_search(served):
    s = served(_Backend(fail=QuotaExceededError("searches", 10, 10, SESSION)))
    with pytest.raises(openai.RateLimitError) as caught:
        _ask(s.client)
    assert caught.value.status_code == 429


def test_a_provider_that_fails_mid_stream_ends_the_stream_with_an_error(served, monkeypatch):
    s = served()

    async def broken(self, prompt):
        yield SimpleNamespace(content="The moon")
        raise ConnectionError("provider went away")

    monkeypatch.setattr(_Chat, "astream", broken)
    with pytest.raises(openai.APIError, match="LLM provider failed"):
        list(_ask(s.client, stream=True))


def test_the_models_list_names_the_one_model(served):
    s = served()
    assert [m.id for m in s.client.models.list()] == [chat.MODEL_ID]


def test_auth_applies_as_on_search(monkeypatch):
    """With RAG_REQUIRE_AUTH on, an unsigned session id is refused before any search."""
    from lasagna.api.rest import session_token
    from lasagna.api.rest.app import create_app

    monkeypatch.setenv("RAG_REQUIRE_AUTH", "true")
    monkeypatch.setenv("RAG_JWT_SECRET", "s" * 64)
    monkeypatch.setattr(session_token, "_fallback_secret", None)
    backend = _Backend()
    app = create_app()
    app.dependency_overrides[get_backend] = lambda: backend
    http = TestClient(app)
    body = {"messages": [{"role": "user", "content": "tides?"}]}
    unsigned = http.post("/api/v1/chat/completions", json=body, headers={"X-Session-ID": SESSION})
    forged = http.post("/api/v1/chat/completions", json=body, headers={"X-Session-Token": "x.y.z"})
    assert unsigned.status_code == 401
    assert forged.status_code == 401
    assert backend.calls == []
