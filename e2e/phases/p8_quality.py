"""Phase 8 — Quality: the generation half, with a real model.

Every other phase is hermetic. That is a virtue for speed and reproducibility,
and it is also why none of them touch what a RAG framework is actually for:
retrieving relevant context and generating an answer that is grounded in it.
Retrieval can be measured offline; grounding cannot.

So this phase makes real LLM calls. It ingests a tiny corpus with known facts,
asks questions whose answers appear in exactly one document, and checks that
the generated answer contains the fact and that the retrieved context is the
document that holds it.

**Without credentials it skips — loudly.** The report renders a skipped phase
as *unvalidated*, never as passed, and that distinction matters most here: a
green gate that silently never exercised generation would be claiming the
opposite of what it knows.

Credentials are read from the environment, and from a gitignored ``.env`` at
the repo root if present. Nothing here logs a key value; only which variables
were found.

**Cost.** Bounded by construction: a handful of short questions with a small
token ceiling. Token counts are recorded so the cost of a gate run is visible
rather than assumed.

**What this does NOT prove.** It is a smoke test of grounding, not a benchmark.
It says the pipeline can retrieve and generate coherently; it does not produce
a defensible NDCG or faithfulness number — that needs the full evaluation
harness over a real benchmark, which is a different job with a different
budget.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

from ..contract import PhaseResult, Severity

REPO_ROOT = Path(__file__).resolve().parents[2]

#: Generation model. Confirmed present on watsonx via the foundation-model
#: catalogue; ``openai/gpt-oss-120b`` is an open-weights model served there.
WATSONX_MODEL = "openai/gpt-oss-120b"

#: This model emits a ``reasoning_content`` field that consumes the token
#: budget before any answer text is produced. A small max_tokens yields an
#: empty answer that looks like a generation failure — it is not, it is a
#: budget that ran out during reasoning.
MAX_TOKENS = 400

#: Facts placed in separate documents so a correct retrieval is unambiguous:
#: each question is answerable from exactly one of them.
CORPUS: List[Tuple[str, str]] = [
    (
        "neptune.txt",
        "The Neptune Array is a deep-sea sensor network deployed in 2019. "
        "It operates at a depth of 2,840 metres and reports every 45 seconds.",
    ),
    (
        "cobalt.txt",
        "Project Cobalt is a materials research programme started in 2021. "
        "Its lead investigator is Dr. Amara Silva and it is funded through 2027.",
    ),
    (
        "larch.txt",
        "The Larch protocol compresses telemetry at a ratio of 12 to 1 "
        "and was ratified by the standards committee in March 2023.",
    ),
]

QUESTIONS: List[Tuple[str, str, str]] = [
    ("At what depth does the Neptune Array operate?", "2,840", "neptune.txt"),
    ("Who leads Project Cobalt?", "Silva", "cobalt.txt"),
    ("What compression ratio does the Larch protocol achieve?", "12", "larch.txt"),
]


def _load_dotenv() -> List[str]:
    """Read the gitignored .env into the environment. Returns names loaded.

    Values are never returned or logged — only variable names, so a report can
    say which credentials were present without leaking any of them.
    """
    path = REPO_ROOT / ".env"
    loaded: List[str] = []
    if not path.exists():
        return loaded
    try:
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip("'\"")
            if key and value and not os.environ.get(key):
                os.environ[key] = value
                loaded.append(key)
    except Exception:
        pass
    return loaded


def _select_platform() -> Optional[Tuple[str, str, str]]:
    """Pick a configured generation platform.

    Returns (platform, model, which env var proved it) or None.
    """
    if os.environ.get("WATSONX_APIKEY") or os.environ.get("WATSONX_API_KEY"):
        if os.environ.get("WATSONX_PROJECT_ID"):
            var = "WATSONX_APIKEY" if os.environ.get("WATSONX_APIKEY") else "WATSONX_API_KEY"
            return ("watsonx", WATSONX_MODEL, var)
    if os.environ.get("OPENAI_API_KEY"):
        return ("openai", "gpt-6-luna", "OPENAI_API_KEY")
    if os.environ.get("ANTHROPIC_API_KEY"):
        return ("anthropic", "claude-haiku-5-5", "ANTHROPIC_API_KEY")
    if os.environ.get("OPENROUTER_API_KEY"):
        return ("openrouter", "openai/gpt-6-luna", "OPENROUTER_API_KEY")
    return None


def run() -> PhaseResult:
    p = PhaseResult(key="p8", title="Quality (real LLM)")

    loaded = _load_dotenv()
    selected = _select_platform()

    if selected is None:
        p.skipped_reason = (
            "no generation credentials found. Set WATSONX_APIKEY + "
            "WATSONX_PROJECT_ID, OPENAI_API_KEY, ANTHROPIC_API_KEY or "
            "OPENROUTER_API_KEY (or put them in a gitignored .env at the repo "
            "root). The generation half of the pipeline is UNVALIDATED without "
            "this — retrieval alone says nothing about whether answers are "
            "grounded in what was retrieved."
        )
        return p

    platform, model, proof_var = selected
    p.measure(
        "credentials_loaded_from_dotenv",
        float(len(loaded)),
        "vars",
        note="names only; values never recorded",
    )

    import sys

    sys.path.insert(0, str(REPO_ROOT / "src"))

    # -- 1. the model answers at all -----------------------------------------
    llm = None
    t0 = time.monotonic()
    try:
        from lasagna.providers import get_llm_provider, reset_providers

        reset_providers()
        llm = get_llm_provider().get_chat_model(
            {"platform": platform, "model_name": model},
            temperature=0.0,
            max_tokens=MAX_TOKENS,
        )
        reply = llm.invoke("Reply with exactly: ready")
        text = (getattr(reply, "content", "") or "").strip()
        ok = bool(text)
    except Exception as exc:
        ok, text = False, f"{type(exc).__name__}: {exc}"
    p.measure("llm_handshake_s", round(time.monotonic() - t0, 2), "s")
    p.check(
        f"generation model responds ({platform}/{model})",
        ok,
        detail=f"credentials from {proof_var}; reply={text[:120]!r}",
        repro="python -m e2e.run --phase p8",
    )
    if not ok:
        # Everything below needs a working model. Skipping rather than failing
        # each one keeps the report honest about what was actually tested.
        for name in (
            "retrieval finds the document holding the answer",
            "generated answers are grounded in retrieved context",
        ):
            p.skip(name, "generation model did not respond; nothing downstream was run")
        return p

    # -- 2. ingest, retrieve, generate ---------------------------------------
    import tempfile

    workdir = tempfile.mkdtemp(prefix="lasagna-p8-")
    os.environ["RAG_DB_DIR"] = os.path.join(workdir, "dbs")
    os.environ["RAG_UPLOAD_DIR"] = os.path.join(workdir, "uploads")
    os.makedirs(os.environ["RAG_DB_DIR"], exist_ok=True)
    os.makedirs(os.environ["RAG_UPLOAD_DIR"], exist_ok=True)

    grounded = 0
    retrieved_right = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    query_times: List[float] = []
    ingest_s = 0.0
    failures: List[str] = []

    try:
        import asyncio

        from lasagna.sdk import KnowledgeClient

        async def drive() -> None:
            nonlocal grounded, retrieved_right, ingest_s
            nonlocal total_prompt_tokens, total_completion_tokens

            client = KnowledgeClient(session_id="e2e_p8")
            try:
                t_ing = time.monotonic()
                for name, body in CORPUS:
                    path = Path(workdir) / name
                    path.write_text(body)
                    res = await client.upload(str(path))
                    await client.documents.wait_for_processing(res.document_id)
                ingest_s = time.monotonic() - t_ing

                for question, expected_fact, source_doc in QUESTIONS:
                    t_q = time.monotonic()
                    results = await client.search_documents(question)
                    query_times.append(time.monotonic() - t_q)

                    hits = getattr(results, "results", []) or []
                    if hits:
                        top_name = getattr(getattr(hits[0], "document", None), "filename", "") or ""
                        if source_doc in top_name:
                            retrieved_right += 1

                    context = "\n\n".join(
                        (getattr(getattr(h, "chunk", None), "content", "") or "") for h in hits[:3]
                    )
                    prompt = (
                        "Answer the question using only the context. Be brief.\n\n"
                        f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
                    )
                    reply = llm.invoke(prompt)
                    answer = getattr(reply, "content", "") or ""
                    if expected_fact.lower() in answer.lower():
                        grounded += 1
                    else:
                        failures.append(
                            f"{question!r} -> {answer[:90]!r} (expected to contain "
                            f"{expected_fact!r})"
                        )

                    usage = getattr(reply, "usage_metadata", None) or {}
                    total_prompt_tokens += int(usage.get("input_tokens") or 0)
                    total_completion_tokens += int(usage.get("output_tokens") or 0)
            finally:
                await client.close()

        asyncio.run(drive())
        pipeline_ok = True
        pipeline_err = ""
    except Exception as exc:
        pipeline_ok, pipeline_err = False, f"{type(exc).__name__}: {exc}"

    # Metrics first, assertions after — a failed run's numbers are the ones
    # worth reading.
    p.measure("ingest_s", round(ingest_s, 2), "s")
    if query_times:
        p.measure("retrieval_p50_s", round(sorted(query_times)[len(query_times) // 2], 3), "s")
    p.measure("questions_asked", float(len(QUESTIONS)), "count", lower_is_better=False)
    p.measure("answers_grounded", float(grounded), "count", lower_is_better=False)
    p.measure("top1_retrieval_correct", float(retrieved_right), "count", lower_is_better=False)
    p.measure("prompt_tokens", float(total_prompt_tokens), "tokens")
    p.measure("completion_tokens", float(total_completion_tokens), "tokens")

    if not pipeline_ok:
        p.check(
            "retrieval and generation run end to end",
            False,
            detail=pipeline_err,
            repro="python -m e2e.run --phase p8",
        )
        return p

    p.check(
        "retrieval and generation run end to end",
        True,
        detail=f"{len(QUESTIONS)} questions over {len(CORPUS)} documents",
    )

    # Retrieval quality is a WARN: embedding models differ, and a top-1 miss on
    # a three-document corpus is a weak signal, not a broken package.
    p.check(
        "retrieval finds the document holding the answer",
        retrieved_right == len(QUESTIONS),
        severity=Severity.WARN,
        detail=f"top-1 correct on {retrieved_right}/{len(QUESTIONS)} questions",
        repro="python -m e2e.run --phase p8",
    )

    # Grounding is CRITICAL. If the model cannot restate a fact that is sitting
    # in the context it was handed, the generation path is not working, and no
    # amount of retrieval quality compensates.
    p.check(
        "generated answers are grounded in retrieved context",
        grounded == len(QUESTIONS),
        detail=(
            f"{grounded}/{len(QUESTIONS)} answers contained the expected fact"
            + ("; misses: " + " | ".join(failures) if failures else "")
        ),
        repro="python -m e2e.run --phase p8",
    )

    return p
