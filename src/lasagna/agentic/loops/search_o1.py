"""Search-o1 (Li et al. 2025), with the pipeline's search in place of web search.

Li, Dong, Jin, Zhang, Zhu, Wu, Zhang, Dou. "Search-o1: Agentic
Search-Enhanced Large Reasoning Models". arXiv:2501.05366.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from loguru import logger

from lasagna.agentic.agentic_rag import AgenticRAGResult, RetrievedChunkData
from lasagna.agentic.loops._base import (
    LoopPipeline,
    LoopRun,
    cut_at,
    interleave,
    message_text,
    setting,
)

BEGIN_SEARCH_QUERY = "<|begin_search_query|>"
END_SEARCH_QUERY = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT = "<|end_search_result|>"

#: The authors' multi-hop QA instruction, verbatim from
#: github.com/sunnynexus/Search-o1 (scripts/prompts.py,
#: ``get_multiqa_search_o1_instruction``), with the search limit filled in.
SEARCH_O1_INSTRUCTION = (
    "You are a reasoning assistant with the ability to perform web searches to help "
    "you answer the user's question accurately. You have special tools:\n\n"
    "- To perform a search: write <|begin_search_query|> your query here <|end_search_query|>.\n"
    "Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|> ...search results... <|end_search_result|>.\n\n"
    "You can repeat the search process multiple times if necessary. The maximum number of search attempts is limited to {max_searches}.\n\n"
    "Once you have all the information you need, continue your reasoning.\n\n"
    "Example:\n"
    'Question: "Alice David is the voice of Lara Croft in a video game developed by which company?"\n'
    "Assistant thinking steps:\n"
    "- I need to find out who voices Lara Croft in the video game.\n"
    "- Then, I need to determine which company developed that video game.\n\n"
    "Assistant:\n"
    "<|begin_search_query|>Alice David Lara Croft voice<|end_search_query|>\n\n"
    "(System returns processed information from relevant web pages)\n\n"
    "Assistant thinks: The search results indicate that Alice David is the voice of Lara Croft in a specific video game. Now, I need to find out which company developed that game.\n\n"
    "Assistant:\n"
    "<|begin_search_query|>video game developed by Alice David Lara Croft<|end_search_query|>\n\n"
    "(System returns processed information from relevant web pages)\n\n"
    "Assistant continues reasoning with the new information...\n\n"
    "Remember:\n"
    "- Use <|begin_search_query|> to request a web search and end with <|end_search_query|>.\n"
    "- When done searching, continue your reasoning.\n\n"
)

#: The authors' open-QA task prompt for a reasoning model, verbatim
#: (``get_task_instruction_openqa`` with ``model_name='qwq'``).
SEARCH_O1_QUESTION = (
    "Please answer the following question. "
    "You should provide your final answer in the format \\boxed{{YOUR_ANSWER}}.\n\n"
    "Question:\n{question}\n\n"
)

#: The Reason-in-Documents prompt, verbatim
#: (``get_webpage_to_reasonchain_instruction``). The three input labels end in
#: two spaces, as the authors' do; written as ``  `` because the
#: trailing-whitespace hook strips them from the end of a source line.
REASON_IN_DOCUMENTS_PROMPT = """**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""

#: What the authors' loop inserts instead of a search the model may not make.
LIMIT_MESSAGE = (
    f"\n{BEGIN_SEARCH_RESULT}\nThe maximum search limit is exceeded. "
    f"You are not allowed to search.\n{END_SEARCH_RESULT}\n"
)
REPEAT_MESSAGE = (
    f"\n{BEGIN_SEARCH_RESULT}\nYou have searched this query. "
    f"Please refer to previous results.\n{END_SEARCH_RESULT}\n"
)

NO_INFORMATION = "No helpful information found."


def _open_query(text: str) -> Optional[str]:
    """The search query the reply ends on, or None when it ends without one.

    The authors stop generation at the end-of-query marker and search when the
    reply ends with it. A reply cut at that marker (``stop``, or ``cut_at``
    for a provider that ignores it) ends with an opened query and no closing
    marker, which is the same condition. An empty query ends the run, as
    ``extract_between`` returning "" does in the authors' code.
    """
    if BEGIN_SEARCH_QUERY not in text:
        return None
    query = text.rsplit(BEGIN_SEARCH_QUERY, 1)[1]
    if END_SEARCH_QUERY in query:
        return None
    return query.strip() or None


def _previous_reasoning(output: str) -> str:
    """The reasoning so far, as Reason-in-Documents reads it.

    A port of the authors' truncation: one numbered step per line, and past
    five steps only the first, the last three, and every step holding a search
    query or result, with "..." for each run left out. (The authors keep the
    last four pieces of a split that ends on an empty one, so three steps.)
    """
    lines = output.replace("\n\n", "\n").split("\n")
    numbered = "".join(f"Step {i + 1}: {step}\n\n" for i, step in enumerate(lines))
    steps = numbered.split("\n\n")
    if len(steps) <= 5:
        truncated = "\n\n".join(steps)
    else:
        truncated = ""
        for i, step in enumerate(steps):
            if (
                i == 0
                or i >= len(steps) - 4
                or BEGIN_SEARCH_QUERY in step
                or BEGIN_SEARCH_RESULT in step
            ):
                truncated += step + "\n\n"
            elif truncated[-len("\n\n...\n\n") :] != "\n\n...\n\n":
                truncated += "...\n\n"
    return truncated.strip("\n")


def _web_pages(chunks: Sequence[RetrievedChunkData]) -> str:
    """Passages laid out as the authors lay out web pages: a header, then JSON."""
    return "".join(
        f"**Web Page {i}:**\n"
        + json.dumps(
            {
                "id": i,
                "title": c.document_name or c.document_id or "untitled",
                "context": c.content.strip(),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
        for i, c in enumerate(chunks, start=1)
    )


def _final_information(reply: str) -> str:
    """The authors' ``extract_answer(mode='infogen')`` for a Search-o1 reply."""
    if "**Final Information**" in reply:
        return reply.split("**Final Information**")[-1].replace("\n", "").strip("```").strip()
    if "**Modified Reasoning Steps**" in reply:
        return reply.split("**Modified Reasoning Steps**")[-1].strip("```").strip()
    return NO_INFORMATION


def _boxed_answer(output: str) -> str:
    """The last ``\\boxed{...}``, as the authors' ``extract_answer(mode='qa')`` reads it."""
    matches = re.findall(r"\\boxed\{(.*)\}", output)
    if not matches:
        return ""
    answer = matches[-1]
    inner = re.findall(r"\\text\{(.*)\}", answer)
    if inner:
        answer = inner[-1]
    return answer.strip("()")


def _thinking_llm(llm: Any, update: Optional[Dict[str, Any]] = None) -> Any:
    """``llm``, asked to return its thinking when it would otherwise drop it.

    Ollama strips a reasoning model's ``<think>`` block from the reply, and
    returns it only when the request asks for it (``reasoning=True`` on
    ``ChatOllama``). Search-o1's queries are written in that thinking.
    ``update`` sets other ``ChatOllama`` fields (the context window and the
    output cap); other models are returned as they are.
    """
    if type(llm).__name__ != "ChatOllama":
        return llm
    fields = dict(update or {})
    if getattr(llm, "reasoning", False) is None:
        fields["reasoning"] = True
    return llm.model_copy(update=fields) if fields else llm


def _reasoning_and_text(response: Any) -> str:
    """The reply's thinking, when the provider returns it apart, then its text."""
    kwargs = getattr(response, "additional_kwargs", None) or {}
    reasoning = str(kwargs.get("reasoning_content") or "").strip()
    text = message_text(response)
    if not reasoning:
        return text
    return f"{reasoning}\n\n{text.lstrip()}" if text.strip() else reasoning


def _has_thinking(text: str) -> bool:
    return "<think>" in text or "</think>" in text


def _usage(
    prompt_tokens: Any, completion_tokens: Any, finish: Any, context_tokens: Optional[int]
) -> Dict[str, Any]:
    """One generation's token counts, and whether it reached a limit.

    ``finish`` is the server's reason for stopping ("length" when the output
    cap cut the reply). A prompt and reply that together fill the context
    window mean the server may have dropped the start of the prompt, which is
    the instruction that defines the search markers.
    """
    prompt = int(prompt_tokens) if prompt_tokens is not None else None
    completion = int(completion_tokens) if completion_tokens is not None else None
    filled = (
        context_tokens is not None
        and prompt is not None
        and prompt + (completion or 0) >= context_tokens
    )
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "finish": finish,
        "hit_output_cap": finish == "length",
        "hit_context": bool(filled),
    }


#: Qwen's chat template (ChatML), which Qwen3 and QwQ use: one user turn, then
#: the assistant turn the reasoning continues in. Ollama's raw mode applies no
#: template, so the loop applies it, as the authors' code does with
#: ``tokenizer.apply_chat_template`` before it appends each reply.
CHATML_TURN = "<|im_start|>user\n{content}<|im_end|>\n<|im_start|>assistant\n"


def _post_json(url: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """POST ``payload`` and return the JSON reply (one place, so tests replace it)."""
    import httpx

    response = httpx.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


async def _post(url: str, payload: Dict[str, Any], timeout: float = 3600.0) -> Dict[str, Any]:
    import asyncio

    return await asyncio.to_thread(_post_json, url, payload, timeout)


def _server_of(llm: Any) -> Optional[Tuple[str, str, str]]:
    """``(kind, base URL, model)`` when ``llm`` talks to Ollama or vLLM, else None."""
    if type(llm).__name__ == "ChatOllama":
        base = getattr(llm, "base_url", None) or os.environ.get(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        )
        return "ollama", str(base).rstrip("/"), str(getattr(llm, "model", ""))
    base = getattr(llm, "openai_api_base", None)
    model = getattr(llm, "model_name", None) or getattr(llm, "model", None)
    if not base or not model:
        return None
    base = str(base).rstrip("/")
    for env, kind in (("VLLM_BASE_URL", "vllm"), ("OLLAMA_BASE_URL", "ollama")):
        configured = os.environ.get(env, "").rstrip("/")
        if configured and _strip_v1(base) == _strip_v1(configured):
            return kind, base, str(model)
    return None


def _strip_v1(url: str) -> str:
    return url[: -len("/v1")] if url.endswith("/v1") else url


class _ChatGenerator:
    """The chat form: the reasoning so far goes back as text in a new user turn.

    A thinking model then starts a fresh ``<think>`` each turn and reads its
    earlier reasoning as the user's text. Used when no server that can
    continue a turn is known; ``trace["continuation"]`` says so.
    """

    kind = "off"

    def __init__(self, run: LoopRun, context_tokens: int, max_tokens: int) -> None:
        self.run = run
        self.context_tokens = context_tokens
        self.llm = _thinking_llm(run.llm, {"num_ctx": context_tokens, "num_predict": max_tokens})
        self.is_ollama = type(run.llm).__name__ == "ChatOllama"

    def _usage(self, response: Any) -> Dict[str, Any]:
        usage = getattr(response, "usage_metadata", None) or {}
        meta = getattr(response, "response_metadata", None) or {}
        finish = meta.get("done_reason") or meta.get("finish_reason")
        return _usage(
            usage.get("input_tokens"),
            usage.get("output_tokens"),
            finish,
            self.context_tokens if self.is_ollama else None,
        )

    async def reason(self, prompt: str, output: str) -> Tuple[str, Dict[str, Any]]:
        response = await self.run.ask(prompt + output, stop=[END_SEARCH_QUERY], llm=self.llm)
        return cut_at(_reasoning_and_text(response), [END_SEARCH_QUERY]), self._usage(response)

    async def read(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        response = await self.run.ask(prompt, llm=self.llm)
        return message_text(response), self._usage(response)


class _OllamaRawGenerator:
    """One assistant turn, continued: Ollama's ``/api/generate`` with ``raw``.

    The loop applies the model's chat template once (``CHATML_TURN``) and
    appends each reply and each search result to the same assistant turn, as
    ``run_search_o1.py`` does, so the model continues one chain of thought.
    ``num_ctx`` and ``num_predict`` set the context window and the output cap.
    """

    kind = "ollama"

    def __init__(
        self, run: LoopRun, base: str, model: str, context_tokens: int, max_tokens: int
    ) -> None:
        if not any(family in model.lower() for family in ("qwen", "qwq")):
            raise ValueError(
                f"search_o1 continues Ollama's raw mode with Qwen's chat template, "
                f"and {model!r} is not a Qwen model; set search_o1_continuation='off'"
            )
        self.run = run
        self.url = _strip_v1(base) + "/api/generate"
        self.model = model
        self.context_tokens = context_tokens
        self.options: Dict[str, Any] = {"num_ctx": context_tokens, "num_predict": max_tokens}
        temperature = getattr(run.llm, "temperature", None)
        if temperature is not None:
            self.options["temperature"] = temperature

    async def _generate(self, prompt: str, stop: Optional[List[str]]) -> Tuple[str, Dict]:
        options = dict(self.options, stop=stop) if stop else dict(self.options)
        started = time.time()
        reply = await _post(
            self.url,
            {
                "model": self.model,
                "prompt": prompt,
                "raw": True,
                "stream": False,
                "options": options,
            },
        )
        self.run.llm_calls += 1
        self.run.llm_ms += (time.time() - started) * 1000
        text = str(reply.get("response") or "")
        thinking = str(reply.get("thinking") or "")
        usage = _usage(
            reply.get("prompt_eval_count"),
            reply.get("eval_count"),
            reply.get("done_reason"),
            self.context_tokens,
        )
        return (thinking + text if thinking else text), usage

    async def reason(self, prompt: str, output: str) -> Tuple[str, Dict[str, Any]]:
        text, usage = await self._generate(
            CHATML_TURN.format(content=prompt) + output, [END_SEARCH_QUERY]
        )
        return cut_at(text, [END_SEARCH_QUERY]), usage

    async def read(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        return await self._generate(CHATML_TURN.format(content=prompt), None)


class _VLLMContinueGenerator:
    """One assistant turn, continued: vLLM's chat API with ``continue_final_message``.

    The request ends with the assistant message holding the reasoning so far,
    and vLLM renders it without closing the turn (``add_generation_prompt``
    off), so the model continues it. The server must run without
    ``--reasoning-parser``, which would move the thinking out of the text.
    The context window is the server's ``--max-model-len``.
    """

    kind = "vllm"

    def __init__(self, run: LoopRun, base: str, model: str, max_tokens: int) -> None:
        self.run = run
        base = base.rstrip("/")
        self.url = (base if base.endswith("/v1") else base + "/v1") + "/chat/completions"
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = getattr(run.llm, "temperature", None)

    async def _chat(
        self, messages: List[Dict[str, str]], extra: Dict[str, Any]
    ) -> Tuple[str, Dict]:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
            **extra,
        }
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        started = time.time()
        reply = await _post(self.url, payload)
        self.run.llm_calls += 1
        self.run.llm_ms += (time.time() - started) * 1000
        choice = (reply.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        thinking = str(message.get("reasoning_content") or "")
        text = str(message.get("content") or "")
        usage = reply.get("usage") or {}
        return (thinking + text if thinking else text), _usage(
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
            choice.get("finish_reason"),
            None,
        )

    async def reason(self, prompt: str, output: str) -> Tuple[str, Dict[str, Any]]:
        text, usage = await self._chat(
            [{"role": "user", "content": prompt}, {"role": "assistant", "content": output}],
            {
                "continue_final_message": True,
                "add_generation_prompt": False,
                "stop": [END_SEARCH_QUERY],
            },
        )
        return cut_at(text, [END_SEARCH_QUERY]), usage

    async def read(self, prompt: str) -> Tuple[str, Dict[str, Any]]:
        return await self._chat([{"role": "user", "content": prompt}], {})


def _generator(run: LoopRun, mode: str, context_tokens: int, max_tokens: int) -> Any:
    """The generator ``search_o1_continuation`` asks for.

    ``auto`` continues one turn when the loop's model is served by Ollama (a
    Qwen model) or vLLM, and falls back to the chat form, with a warning,
    otherwise. ``ollama`` and ``vllm`` require that server; ``off`` is the
    chat form.
    """
    if mode not in ("auto", "ollama", "vllm", "off"):
        raise ValueError(
            f"search_o1_continuation must be 'auto', 'ollama', 'vllm' or 'off', not {mode!r}"
        )
    server = None if mode == "off" else _server_of(run.llm)
    if mode in ("ollama", "vllm") and (server is None or server[0] != mode):
        raise ValueError(
            f"search_o1_continuation={mode!r}, but the loop's model is not served by {mode}"
        )
    if server is not None:
        kind, base, model = server
        if kind == "vllm":
            return _VLLMContinueGenerator(run, base, model, max_tokens)
        try:
            return _OllamaRawGenerator(run, base, model, context_tokens, max_tokens)
        except ValueError:
            if mode == "ollama":
                raise
    if mode == "auto":
        logger.warning(
            "[search_o1] the loop's model is not served by Ollama (a Qwen model) or vLLM, "
            "so each turn sends the reasoning so far as a new user message instead of "
            "continuing one chain of thought"
        )
    return _ChatGenerator(run, context_tokens, max_tokens)


class SearchO1Pipeline(LoopPipeline):
    """Search-o1: a reasoning model searches mid-reasoning; Reason-in-Documents reads.

    Li et al. 2025, "Search-o1: Agentic Search-Enhanced Large Reasoning
    Models" (arXiv:2501.05366), the full method (agentic search with
    Reason-in-Documents).

    The model reasons under the authors' instruction and writes a query
    between ``<|begin_search_query|>`` and ``<|end_search_query|>``.
    Generation stops at the end marker; the query is searched through the
    pipeline, and one LLM call with the authors' Reason-in-Documents prompt
    reads the passages, the reasoning so far and the query into a short
    account. That account goes back into the reasoning between
    ``<|begin_search_result|>`` and ``<|end_search_result|>``, and generation
    resumes. A query past the search limit, or one already searched, gets the
    authors' refusal message instead of a search. The run ends when a reply
    opens no query, or after ``search_o1_max_turns`` generations; the answer
    is the last ``\\boxed{...}`` in the reasoning, read as the authors' code
    reads it, and empty when there is none (``stop_reason`` says so).

    The prompts are the authors' (``sunnynexus/Search-o1``,
    ``scripts/prompts.py``) word for word: the multi-hop QA instruction, the
    open-QA question prompt for a reasoning model (their ``qwq`` variant) and
    the Reason-in-Documents prompt. What differs from the paper and its code,
    each forced by the setting rather than chosen:

    * Bing web search becomes the pipeline's own search, and a web page
      becomes a passage. Each passage is laid out as the authors lay out a
      page (``**Web Page i:**`` and a JSON record), with ``id``, ``title``
      and ``context`` only: a passage has no URL, site name, date or snippet,
      and its text is the context the authors cut from the fetched page
      around the snippet (``max_doc_len``), so no cut is made.
    * The authors apply the chat template once and append each reply and
      each search result to the same assistant turn, so the model continues
      one chain of thought. The loop does the same on the two local servers
      the issue names (``search_o1_continuation``, default ``auto``): Ollama's
      ``/api/generate`` in raw mode, with Qwen's chat template applied by the
      loop, and vLLM's chat API with ``continue_final_message``. These calls
      go to the server directly, so the experiment's LLM cache does not
      cover them. Any other model gets the chat form, with a warning: the
      reasoning so far goes back as text in a new user message each turn, as
      ``self_ask`` sends its transcript, and Ollama is asked to return the
      thinking it would otherwise strip (``_thinking_llm``).
    * The loop sets the context window (``search_o1_context_tokens``, 32768,
      Ollama's ``num_ctx``) and the output cap per generation
      (``search_o1_max_tokens``, 20480, the authors' cap for QwQ on QA).
      Ollama's default window is a few thousand tokens, and it drops the start
      of a prompt that does not fit, with no error, which is the instruction.
      ``loop_trace["generations"]`` records each generation's prompt and
      completion tokens, and ``hit_context`` and ``hit_output_cap`` say
      whether any reached a limit. On vLLM the window is the server's
      ``--max-model-len``.
    * A run in which no reply carried the model's thinking (no ``<think>``
      block, no reasoning field) logs a warning and records
      ``thinking_seen: false``. A server that strips the thinking, such as
      vLLM with ``--reasoning-parser``, hides the queries, and the run would
      otherwise read as a model answering from memory.
    * The instruction is the multi-hop one for every benchmark. The authors
      use it for HotpotQA, MuSiQue, Bamboogle and 2Wiki, and a single-hop
      variant (a one-search example) for NQ and TriviaQA; a loop does not
      know which benchmark it serves.
    * Sampling settings are the provider's, not the authors' (temperature
      0.7, top-p 0.8, top-k 20, repetition penalty 1.05 for QwQ).

    The bounds are the authors' for the multi-hop datasets: at most
    ``search_o1_max_searches`` searches (10; the code's default for
    HotpotQA, MuSiQue, Bamboogle and 2Wiki, 5 for single-hop QA),
    ``search_o1_max_turns`` generations (15) and
    ``search_o1_docs_per_search`` passages per search (10, the code's
    ``top_k``). The paper names no turn limit; the authors' code sets it.

    Settings: ``search_o1_max_searches``, ``search_o1_max_turns``,
    ``search_o1_docs_per_search``, ``search_o1_continuation``,
    ``search_o1_context_tokens``, ``search_o1_max_tokens``.
    """

    method = "search_o1"

    async def _loop(self, run: LoopRun) -> AgenticRAGResult:
        question = run.input.question
        max_searches = int(setting(run.config, "search_o1_max_searches"))
        max_turns = int(setting(run.config, "search_o1_max_turns"))
        docs_per_search = int(setting(run.config, "search_o1_docs_per_search"))

        prompt = SEARCH_O1_INSTRUCTION.format(
            max_searches=max_searches
        ) + SEARCH_O1_QUESTION.format(question=question)
        generator = _generator(
            run,
            str(setting(run.config, "search_o1_continuation")),
            int(setting(run.config, "search_o1_context_tokens")),
            int(setting(run.config, "search_o1_max_tokens")),
        )
        output = ""  # the model's reasoning, with the search results inserted
        searched: List[str] = []
        per_hop: List[List[RetrievedChunkData]] = []
        refused = 0
        generations: List[Dict[str, Any]] = []
        thinking_seen = False

        turns = 0
        stop_reason = "max_turns"
        while turns < max_turns:
            turns += 1
            text, usage = await generator.reason(prompt, output)
            generations.append({"step": "reason", "turn": turns, **usage})
            thinking_seen = thinking_seen or _has_thinking(text)
            query = _open_query(text)
            if query is None:
                output += text
                stop_reason = "answer"
                break
            # The authors keep the stop string in the output.
            output += text + END_SEARCH_QUERY
            if len(searched) >= max_searches:
                output += LIMIT_MESSAGE
                refused += 1
                continue
            if query in searched:
                output += REPEAT_MESSAGE
                refused += 1
                continue

            passages = run.search(query, docs_per_search)
            searched.append(query)
            per_hop.append(passages)
            reply, usage = await generator.read(
                REASON_IN_DOCUMENTS_PROMPT.format(
                    prev_reasoning=_previous_reasoning(output),
                    search_query=query,
                    document=_web_pages(passages),
                )
            )
            generations.append({"step": "read", "turn": turns, **usage})
            output += f"\n\n{BEGIN_SEARCH_RESULT}{_final_information(reply)}{END_SEARCH_RESULT}\n\n"

        answer = _boxed_answer(output)
        if not answer and stop_reason == "answer":
            stop_reason = "no_boxed_answer"
        if not thinking_seen:
            # A reasoning model whose thinking never reaches the loop writes its
            # queries where the loop cannot see them, and the run then reads as
            # answering from memory. Say so rather than score it quietly.
            logger.warning(
                "[search_o1] no reply carried the model's thinking (no <think> block and "
                "no reasoning field); a reasoning model's queries may have been hidden "
                "by the server, for example vLLM's --reasoning-parser"
            )

        logger.info(f"[search_o1] {len(searched)} search(es), {turns} turn(s), stop={stop_reason}")
        run.trace["searches"] = len(searched)
        run.trace["refused_searches"] = refused
        run.trace["turns"] = turns
        run.trace["continuation"] = generator.kind
        run.trace["thinking_seen"] = thinking_seen
        run.trace["generations"] = generations
        run.trace["hit_output_cap"] = any(g["hit_output_cap"] for g in generations)
        run.trace["hit_context"] = any(g["hit_context"] for g in generations)
        run.trace["transcript"] = output
        return run.result(
            answer,
            interleave(per_hop),
            stop_reason=stop_reason,
            steps=len(searched),
            max_steps=max_searches,
        )
