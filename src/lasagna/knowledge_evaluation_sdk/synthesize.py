"""Question-answer sets written by an LLM from your own documents.

``lasagna benchmark synthesize ./docs --n 200`` reads every supported file under
``./docs``, cuts it into passages with the ``fixed`` chunker, and asks an LLM for
three kinds of question:

* ``factoid``: answered by a short fact in one passage;
* ``multihop``: needs a fact from each of two passages in two documents;
* ``unanswerable``: on the subject of a passage, but not answered by it.

A second call to the same LLM checks each question: an answer must be supported
by its source passages, and an unanswerable question must not be answered by
its passage or by the passages that share the most words with it. Rows that
fail are dropped and counted. The output is the JSONL format of
``docs/own-data.md``, so ``lasagna benchmark run`` takes it directly.

Every call goes through ``get_llm_provider().get_llm(temperature=0)``, the
provider every other LLM stage uses, and every row records the prompt's id and
the sha256 of its text, the model and the seed.

**For development, not for claims.** A question an LLM wrote is phrased the way
that LLM phrases things, and a pipeline built around the same family of models
finds it easier than a person's question. Use a synthetic set to compare
configurations on your own documents while you build; report results on
human-written benchmarks.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

QUESTION_TYPES = ("factoid", "multihop", "unanswerable")

DEFAULT_MIX: Dict[str, float] = {"factoid": 0.6, "multihop": 0.3, "unanswerable": 0.1}

#: Passage size for question writing, in words. Long enough to hold a fact and
#: its subject, short enough that the source of an answer is specific.
DEFAULT_CHUNK_WORDS = 200

#: A trailing window shorter than this is a fragment, not a passage.
MIN_PASSAGE_WORDS = 20

#: Passages besides its own that an unanswerable question is checked against.
UNANSWERED_CHECK_NEIGHBORS = 3

#: For the dry run's token estimate, which says it is one.
_CHARS_PER_TOKEN = 4
_OUTPUT_TOKENS = {"write": 60, "check": 2}


@dataclass(frozen=True)
class Prompt:
    """A prompt template. Its id and the sha256 of its text go on every row."""

    id: str
    text: str

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


FACTOID_PROMPT = Prompt(
    "factoid-v1",
    """You are writing one test question for a search system, from the passage below.

Write a question that the passage answers with a short fact: a name, number, date, place or short phrase. The question must make sense to someone who has not seen the passage, so name what it is about instead of saying "the passage" or "this document". The answer must be stated in the passage.

Passage:
{passage}

Reply with one JSON object and nothing else:
{{"question": "...", "answer": "..."}}""",
)

MULTIHOP_PROMPT = Prompt(
    "multihop-v1",
    """You are writing one test question for a search system, from the two passages below. They come from different documents.

Write a question whose answer needs a fact from passage A and a fact from passage B, so that neither passage alone is enough. The question must make sense to someone who has not seen the passages, so name what it is about instead of saying "passage A". Keep the answer short.

Passage A:
{passage_a}

Passage B:
{passage_b}

Reply with one JSON object and nothing else:
{{"question": "...", "answer": "..."}}""",
)

UNANSWERABLE_PROMPT = Prompt(
    "unanswerable-v1",
    """You are writing one test question for a search system, from the passage below.

Write a question on the same subject that a reader of the passage might plausibly ask, but that the passage does not answer: it asks for a detail the passage never gives. Do not ask about an unrelated subject, and do not write a question with a false premise. The question must make sense to someone who has not seen the passage.

Passage:
{passage}

Reply with one JSON object and nothing else:
{{"question": "..."}}""",
)

SUPPORTED_CHECK_PROMPT = Prompt(
    "check-supported-v1",
    """You are checking a test question for a search system.

Passages:
{passages}

Question: {question}
Reference answer: {answer}

Is the reference answer a correct answer to the question, stated in or directly implied by the passages above? Reply with one word: yes or no.""",
)

UNANSWERED_CHECK_PROMPT = Prompt(
    "check-unanswered-v1",
    """You are checking a test question for a search system.

Passages:
{passages}

Question: {question}

Do the passages above answer the question, fully or in part? Reply with one word: yes or no.""",
)

_WRITE_PROMPTS = {
    "factoid": FACTOID_PROMPT,
    "multihop": MULTIHOP_PROMPT,
    "unanswerable": UNANSWERABLE_PROMPT,
}


# ---------------------------------------------------------------------------
# The plan: documents, passages, and which passages each question comes from
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    id: str  # "<doc_id>#<index>"
    doc_id: str  # path relative to the corpus folder
    text: str


@dataclass(frozen=True)
class _Item:
    """One question to write: its type and its source passages."""

    question_type: str
    passages: Tuple[Passage, ...]


@dataclass
class SynthesisPlan:
    """Everything decided before the first LLM call. ``--dry-run`` prints it."""

    corpus_dir: Path
    documents: List[str]
    #: ``(file, reason)`` for files found but not read.
    skipped: List[Tuple[str, str]]
    passages: List[Passage]
    counts: Dict[str, int]
    items: List[_Item]
    seed: int
    chunk_words: int

    @property
    def llm_calls(self) -> int:
        """One call to write each question and one to check it."""
        return 2 * len(self.items)

    def token_estimate(self) -> Tuple[int, int]:
        """``(input, output)`` tokens, estimated at four characters per token."""
        tokens_in = tokens_out = 0
        for item in self.items:
            write = _write_prompt(item)
            # The unanswerable check also reads the closest passages, which
            # depend on the question; counted as copies of the source passage.
            check_passages = list(item.passages)
            if item.question_type == "unanswerable":
                check_passages *= 1 + UNANSWERED_CHECK_NEIGHBORS
            check = _check_prompt(item.question_type, check_passages, "?" * 80, "?" * 40)
            tokens_in += (len(write) + len(check)) // _CHARS_PER_TOKEN
            tokens_out += _OUTPUT_TOKENS["write"] + _OUTPUT_TOKENS["check"]
        return tokens_in, tokens_out


def parse_mix(spec: Any) -> Dict[str, float]:
    """``"factoid=0.6,multihop=0.3,unanswerable=0.1"`` (or a dict) as weights.

    Types left out get 0. The weights must add up to 1.
    """
    if spec is None or spec == "":
        return dict(DEFAULT_MIX)
    if isinstance(spec, dict):
        pairs = list(spec.items())
    else:
        pairs = []
        for part in str(spec).split(","):
            if not part.strip():
                continue
            if "=" not in part:
                raise ValueError(
                    f"--mix takes type=share pairs, like factoid=0.6,multihop=0.3; got {part!r}"
                )
            key, value = part.split("=", 1)
            pairs.append((key.strip(), value.strip()))
    mix = {t: 0.0 for t in QUESTION_TYPES}
    for key, value in pairs:
        if key not in mix:
            from lasagna.core._text import did_you_mean

            raise ValueError(
                f"Unknown question type {key!r} in --mix{did_you_mean(key, QUESTION_TYPES)}. "
                f"Types: {', '.join(QUESTION_TYPES)}."
            )
        try:
            weight = float(value)
        except (TypeError, ValueError):
            raise ValueError(f"--mix share for {key} must be a number, got {value!r}") from None
        if weight < 0:
            raise ValueError(f"--mix share for {key} must not be negative, got {weight}")
        mix[key] = weight
    total = sum(mix.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"--mix shares must add up to 1, got {total:g}")
    return mix


def split_counts(n: int, mix: Dict[str, float]) -> Dict[str, int]:
    """``n`` divided by ``mix``, largest remainder first, ties in type order."""
    exact = {t: n * mix.get(t, 0.0) for t in QUESTION_TYPES}
    counts = {t: int(exact[t]) for t in QUESTION_TYPES}
    order = sorted(QUESTION_TYPES, key=lambda t: (-(exact[t] - counts[t]), QUESTION_TYPES.index(t)))
    for t in order[: n - sum(counts.values())]:
        counts[t] += 1
    return counts


def load_documents(corpus_dir: Any) -> Tuple[Dict[str, str], List[Tuple[str, str]]]:
    """``({doc_id: text}, skipped)`` for every readable file under ``corpus_dir``.

    ``.md`` and ``.txt`` are read as plain text; other formats go through
    ``DocumentProcessor.extract_text``, as an upload does. Hidden files are
    ignored. ``doc_id`` is the path relative to the folder.
    """
    from lasagna.processing.document_processor import DocumentProcessor

    root = Path(corpus_dir).expanduser()
    if not root.exists():
        raise ValueError(f"{corpus_dir}: no such file or folder")
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    documents: Dict[str, str] = {}
    skipped: List[Tuple[str, str]] = []
    for file in files:
        rel = file.name if root.is_file() else file.relative_to(root).as_posix()
        if any(part.startswith(".") for part in Path(rel).parts):
            continue
        suffix = file.suffix.lower()
        if suffix not in DocumentProcessor.SUPPORTED_FORMATS:
            skipped.append((rel, f"unsupported type {suffix or '(no extension)'}"))
            continue
        try:
            if suffix in (".md", ".txt"):
                text = DocumentProcessor.extract_text_from_txt(str(file))
            else:
                text = DocumentProcessor.extract_text(str(file))
        except Exception as exc:  # noqa: BLE001 - reported per file, not fatal
            skipped.append((rel, f"could not be read: {exc}"))
            continue
        if text.strip():
            documents[rel] = text
        else:
            skipped.append((rel, "no text"))
    return documents, skipped


def split_passages(
    documents: Dict[str, str], chunk_words: int = DEFAULT_CHUNK_WORDS
) -> List[Passage]:
    """Each document in fixed windows of ``chunk_words`` words, no overlap."""
    from lasagna.processing.chunker.splitters import FixedSizeChunker

    chunker = FixedSizeChunker(chunk_size=chunk_words, chunk_overlap=0, unit="words")
    passages: List[Passage] = []
    for doc_id, text in documents.items():
        kept = [c.content.strip() for c in chunker.chunk(text)]
        kept = [c for c in kept if len(c.split()) >= MIN_PASSAGE_WORDS]
        passages += [Passage(f"{doc_id}#{i}", doc_id, c) for i, c in enumerate(kept)]
    return passages


def plan_synthesis(
    corpus_dir: Any,
    n: int,
    mix: Any = None,
    seed: int = 42,
    *,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
) -> SynthesisPlan:
    """Read the documents and choose every question's source passages.

    Deterministic in ``seed``: the same folder and seed give the same plan.
    Reads files only; no LLM is involved.
    """
    if n < 1:
        raise ValueError(f"--n must be at least 1, got {n}")
    weights = parse_mix(mix)
    documents, skipped = load_documents(corpus_dir)
    if not documents:
        from lasagna.processing.document_processor import DocumentProcessor

        raise ValueError(
            f"{corpus_dir} has no document Lasagna can read (supported: "
            f"{', '.join(sorted(DocumentProcessor.SUPPORTED_FORMATS))})"
        )
    passages = split_passages(documents, chunk_words)
    if not passages:
        raise ValueError(
            f"The documents under {corpus_dir} have no passage of {MIN_PASSAGE_WORDS} words "
            f"or more to write a question from."
        )
    counts = split_counts(n, weights)
    with_passages = sorted({p.doc_id for p in passages})
    if counts["multihop"] and len(with_passages) < 2:
        raise ValueError(
            f"multihop questions need passages from two documents; {corpus_dir} has "
            f"{len(with_passages)}. Leave multihop out of --mix."
        )

    rng = random.Random(seed)
    items: List[_Item] = [_Item("factoid", (p,)) for p in _draw(passages, counts["factoid"], rng)]
    for first in _draw(passages, counts["multihop"], rng):
        items.append(_Item("multihop", (first, _partner(first, passages))))
    items += [_Item("unanswerable", (p,)) for p in _draw(passages, counts["unanswerable"], rng)]
    # Interleaved, so the first rows of the file have the requested mix too.
    rng.shuffle(items)
    return SynthesisPlan(
        corpus_dir=Path(corpus_dir),
        documents=sorted(documents),
        skipped=skipped,
        passages=passages,
        counts=counts,
        items=items,
        seed=seed,
        chunk_words=chunk_words,
    )


def _draw(passages: Sequence[Passage], k: int, rng: random.Random) -> List[Passage]:
    """``k`` passages, each used once before any is used twice."""
    out: List[Passage] = []
    while len(out) < k:
        batch = list(passages)
        rng.shuffle(batch)
        out += batch[: k - len(out)]
    return out


_WORD = re.compile(r"[a-z0-9]{4,}")


@lru_cache(maxsize=1 << 16)
def _words(text: str) -> frozenset:
    """Lowercase words of four or more characters: a crude, deterministic overlap."""
    return frozenset(_WORD.findall(text.lower()))


def _closest(text: str, candidates: Sequence[Passage], k: int) -> List[Passage]:
    """The ``k`` candidates sharing the most words with ``text``, in corpus order on ties."""
    words = _words(text)
    ranked = sorted(
        range(len(candidates)), key=lambda i: (-len(words & _words(candidates[i].text)), i)
    )
    return [candidates[i] for i in ranked[:k]]


def _partner(first: Passage, passages: Sequence[Passage]) -> Passage:
    """The passage of another document with the most words in common with ``first``.

    Two passages that share a subject can support a question that needs both;
    two chosen at random usually cannot.
    """
    others = [p for p in passages if p.doc_id != first.doc_id]
    return _closest(first.text, others, 1)[0]


# ---------------------------------------------------------------------------
# Writing and checking
# ---------------------------------------------------------------------------


@dataclass
class SynthesisResult:
    rows: List[Dict[str, Any]]
    kept: Dict[str, int] = field(default_factory=dict)
    #: Rows dropped, by reason: ``unparseable`` (the reply was not the JSON
    #: asked for), ``unsupported`` (the check found the answer unsupported),
    #: ``answered`` (an unanswerable question the passages answer) and
    #: ``unclear`` (the check replied neither yes nor no).
    dropped: Dict[str, int] = field(default_factory=dict)
    model: str = ""


def _write_prompt(item: _Item) -> str:
    template = _WRITE_PROMPTS[item.question_type].text
    if item.question_type == "multihop":
        return template.format(passage_a=item.passages[0].text, passage_b=item.passages[1].text)
    return template.format(passage=item.passages[0].text)


def _numbered(passages: Sequence[Passage]) -> str:
    return "\n\n".join(f"[{i}] {p.text}" for i, p in enumerate(passages, start=1))


def _check_prompt(question_type: str, passages: Sequence[Passage], question: str, answer: str):
    if question_type == "unanswerable":
        return UNANSWERED_CHECK_PROMPT.text.format(passages=_numbered(passages), question=question)
    return SUPPORTED_CHECK_PROMPT.text.format(
        passages=_numbered(passages), question=question, answer=answer
    )


def _reply_text(reply: Any) -> str:
    content = getattr(reply, "content", reply)
    if isinstance(content, list):  # content blocks, as some chat models return
        content = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def _parse_written(text: str, needs_answer: bool) -> Optional[Tuple[str, str]]:
    """``(question, answer)`` from the model's JSON reply, or None."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    question, answer = data.get("question"), data.get("answer", "")
    if not isinstance(question, str) or not question.strip():
        return None
    if needs_answer and (not isinstance(answer, str) or not answer.strip()):
        return None
    return question.strip(), (answer.strip() if needs_answer else "")


def _verdict(text: str) -> Optional[bool]:
    """True for yes, False for no, None for anything else."""
    match = re.match(r"\W*(yes|no)\b", text.strip().lower())
    return None if match is None else match.group(1) == "yes"


def _model_name(llm: Any) -> str:
    for attr in ("model_name", "model", "model_id", "deployment_name"):
        value = getattr(llm, attr, None)
        if isinstance(value, str) and value:
            return value
    return type(llm).__name__


def missing_llm_message(needs: str, instead: str) -> str:
    """What to set when an LLM is needed and none is configured.

    Shared by every command that needs one, so they all name the same keys.
    """
    from lasagna.providers import _API_KEY_TO_PLATFORM

    keys = ", ".join(dict.fromkeys(env for env, _ in _API_KEY_TO_PLATFORM))
    return (
        f"{needs} needs an LLM, and none is configured. Set one of these environment "
        f"variables: {keys}. MODEL_NAME picks the model. {instead}"
    )


def llm_configured() -> bool:
    """Whether an LLM provider can be built from the environment (no call is made)."""
    from lasagna.providers import LLMProviderNotConfigured, get_llm_provider

    try:
        get_llm_provider()
    except LLMProviderNotConfigured:
        return False
    return True


def _missing_llm_message() -> str:
    return missing_llm_message(
        "Writing and checking questions",
        "--model also picks the model, and --dry-run prints the plan without one.",
    )


def default_llm(model: Optional[str] = None) -> Any:
    """The configured LLM at temperature 0, as every other LLM stage gets it."""
    from lasagna.providers import LLMProviderNotConfigured, get_llm_provider

    try:
        provider = get_llm_provider()
    except LLMProviderNotConfigured as exc:
        raise LLMProviderNotConfigured(_missing_llm_message()) from exc
    if model:
        return provider.get_llm(model_name=model, temperature=0)
    return provider.get_llm(temperature=0)


def run_synthesis(
    plan: SynthesisPlan,
    llm: Any = None,
    *,
    model: Optional[str] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> SynthesisResult:
    """Write and check every question in ``plan``.

    ``llm`` is anything with ``invoke(prompt)`` returning text or an object with
    ``.content``; by default the configured provider's model at temperature 0.
    A failed call stops the run rather than leaving a hole in the set.
    """
    llm = llm if llm is not None else default_llm(model)
    model_name = model or _model_name(llm)
    width = max(4, len(str(len(plan.items))))
    result = SynthesisResult(
        rows=[],
        kept={t: 0 for t in QUESTION_TYPES},
        dropped={"unparseable": 0, "unsupported": 0, "answered": 0, "unclear": 0},
        model=model_name,
    )

    def ask(prompt: str) -> str:
        try:
            return _reply_text(llm.invoke(prompt))
        except Exception as exc:
            raise RuntimeError(
                f"The LLM call failed ({type(exc).__name__}: {exc}); nothing was written."
            ) from exc

    for index, item in enumerate(plan.items):
        if progress is not None:
            progress(index, len(plan.items))
        answerable = item.question_type != "unanswerable"
        written = _parse_written(ask(_write_prompt(item)), needs_answer=answerable)
        if written is None:
            result.dropped["unparseable"] += 1
            continue
        question, answer = written

        checked = list(item.passages)
        if not answerable:
            others = [p for p in plan.passages if p not in item.passages]
            checked += _closest(question, others, UNANSWERED_CHECK_NEIGHBORS)
        verdict = _verdict(ask(_check_prompt(item.question_type, checked, question, answer)))
        if verdict is None:
            result.dropped["unclear"] += 1
            continue
        if answerable and not verdict:
            result.dropped["unsupported"] += 1
            continue
        if not answerable and verdict:
            result.dropped["answered"] += 1
            continue

        prompt = _WRITE_PROMPTS[item.question_type]
        row: Dict[str, Any] = {"id": f"synth-{index + 1:0{width}d}", "question": question}
        if answerable:
            row["answer"] = answer
        row["contexts"] = [p.text for p in item.passages]
        row["metadata"] = {
            "question_type": item.question_type,
            "source_doc_ids": list(dict.fromkeys(p.doc_id for p in item.passages)),
            "prompt_id": prompt.id,
            "prompt_sha256": prompt.sha256,
            "model": model_name,
            "seed": plan.seed,
            "synthetic": True,
            "unanswerable": not answerable,
        }
        result.rows.append(row)
        result.kept[item.question_type] += 1
    if progress is not None:
        progress(len(plan.items), len(plan.items))
    return result


def synthesize(
    corpus_dir: Any,
    n: int,
    mix: Any = None,
    seed: int = 42,
    llm: Any = None,
    *,
    model: Optional[str] = None,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
) -> List[Dict[str, Any]]:
    """Up to ``n`` checked question rows from the documents in ``corpus_dir``.

    ``mix`` is a dict or ``"factoid=0.6,multihop=0.3,unanswerable=0.1"`` (the
    default). Rows the check rejects are dropped, so fewer than ``n`` can come
    back; :func:`run_synthesis` says how many and why. Each row is in the
    JSONL format of ``docs/own-data.md``.
    """
    plan = plan_synthesis(corpus_dir, n, mix, seed, chunk_words=chunk_words)
    return run_synthesis(plan, llm, model=model).rows


def write_jsonl(rows: Sequence[Dict[str, Any]], path: Any) -> Path:
    """One row per line, UTF-8, keys in the order written."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return out
