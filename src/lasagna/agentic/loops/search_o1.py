"""Search-o1 (Li et al. 2025), with the pipeline's search in place of web search.

Li, Dong, Jin, Zhang, Zhu, Wu, Zhang, Dou. "Search-o1: Agentic
Search-Enhanced Large Reasoning Models". arXiv:2501.05366.
"""

from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Sequence

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


def _thinking_llm(llm: Any) -> Any:
    """``llm``, asked to return its thinking when it would otherwise drop it.

    Ollama strips a reasoning model's ``<think>`` block from the reply, and
    returns it only when the request asks for it (``reasoning=True`` on
    ``ChatOllama``). Search-o1's queries are written in that thinking.
    """
    if type(llm).__name__ == "ChatOllama" and getattr(llm, "reasoning", False) is None:
        return llm.model_copy(update={"reasoning": True})
    return llm


def _reasoning_and_text(response: Any) -> str:
    """The reply's thinking, when the provider returns it apart, then its text."""
    kwargs = getattr(response, "additional_kwargs", None) or {}
    reasoning = str(kwargs.get("reasoning_content") or "").strip()
    text = message_text(response)
    if not reasoning:
        return text
    return f"{reasoning}\n\n{text.lstrip()}" if text.strip() else reasoning


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
    * The authors continue one assistant turn as a raw completion. A chat
      provider here gets the instruction, the question and the reasoning so
      far as one user message, as ``self_ask`` sends its transcript. The
      queries are written in the model's thinking, so the loop reads it: a
      ``reasoning_content`` the provider returns apart from the text comes
      first, and Ollama is asked to return it (``_thinking_llm``). A server
      that drops the thinking (vLLM's ``--reasoning-parser`` behind an
      OpenAI client that discards the field) hides the queries.
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
    ``search_o1_docs_per_search``.
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
        stops = [END_SEARCH_QUERY]
        reasoner = _thinking_llm(run.llm)
        output = ""  # the model's reasoning, with the search results inserted
        searched: List[str] = []
        per_hop: List[List[RetrievedChunkData]] = []
        refused = 0

        turns = 0
        stop_reason = "max_turns"
        while turns < max_turns:
            turns += 1
            response = await run.ask(prompt + output, stop=stops, llm=reasoner)
            text = cut_at(_reasoning_and_text(response), stops)
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
            reply = await run.complete(
                REASON_IN_DOCUMENTS_PROMPT.format(
                    prev_reasoning=_previous_reasoning(output),
                    search_query=query,
                    document=_web_pages(passages),
                )
            )
            output += f"\n\n{BEGIN_SEARCH_RESULT}{_final_information(reply)}{END_SEARCH_RESULT}\n\n"

        answer = _boxed_answer(output)
        if not answer and stop_reason == "answer":
            stop_reason = "no_boxed_answer"

        logger.info(f"[search_o1] {len(searched)} search(es), {turns} turn(s), stop={stop_reason}")
        run.trace["searches"] = len(searched)
        run.trace["refused_searches"] = refused
        run.trace["turns"] = turns
        return run.result(
            answer,
            interleave(per_hop),
            stop_reason=stop_reason,
            steps=len(searched),
            max_steps=max_searches,
        )
